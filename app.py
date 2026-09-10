import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from train_model import train_and_select_best, predict_single, profile_dataset
from llm_explain import explain_prediction

st.set_page_config(page_title="AutoPrep — Synthetic + Balanced AutoML", layout="wide")

st.title("Data2Model  📊")
st.caption(
    "Limited data → profile → clean → synthetic augmentation → balance → AutoML → explain"
)

uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV to start.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head())

st.header("1. Choose Prediction Target🔬")

# The target is selected by the user for EVERY dataset.
# Never assume the first column is the target.
target_keywords = ["target", "label", "price", "salary", "sales", "output", "prediction", "class", "y"]
suggested_target = next(
    (col for col in df.columns if col.lower().strip() in target_keywords),
    df.columns[0]
)

target_col = st.selectbox(
    "🎯 TARGET FEATURE — What do you want the model to predict?",
    options=df.columns.tolist(),
    index=list(df.columns).index(suggested_target),
    help="Select the column you want the model to predict. This column is removed from the input features."
)

feature_cols_selected = [c for c in df.columns if c != target_col]
st.success(f"🎯 Prediction target: **{target_col}**")
st.write(f"**Input features ({len(feature_cols_selected)}):** {', '.join(feature_cols_selected)}")

profile = profile_dataset(df, target_col)

c1, c2, c3, c4 = st.columns(4)
c1.metric("AI-Readiness", f"{profile['readiness_score']}/100")
c2.metric("Missing", f"{profile['missing_pct']}%")
c3.metric("Duplicates", f"{profile['duplicate_pct']}%")
c4.metric("Imbalance ratio", str(profile["imbalance_ratio"] or "N/A"))

if profile["class_counts"]:
    st.write("Target distribution")
    st.bar_chart(pd.Series(profile["class_counts"]))

st.header("2. Exploratory Data Analysis🧪")
numeric_cols = df.select_dtypes(include="number").columns.tolist()

if numeric_cols:
    dist_col = st.selectbox("Numeric distribution", numeric_cols)
    fig, ax = plt.subplots()
    sns.histplot(df[dist_col], kde=True, ax=ax)
    st.pyplot(fig)

if len(numeric_cols) >= 2:
    fig2, ax2 = plt.subplots()
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

st.header("3. Data augmentation & balancing ⚖️ ")

col1, col2, col3 = st.columns(3)
with col1:
    use_synthetic = st.checkbox("Generate synthetic training data", value=True)
with col2:
    synthetic_min_rows = st.number_input(
        "Minimum training rows", min_value=50, max_value=10000, value=500, step=50
    )
with col3:
    use_balancing = st.checkbox("Balance imbalanced classes", value=True)

st.info(
    "Safety design: synthetic data and class balancing are applied only to the "
    "training split. The real test split stays untouched, so the final metrics "
    "measure performance on real data."
)

if st.button("Run Model Pipeline ⚙️ "):
    with st.spinner("Cleaning → generating data → balancing → AutoML..."):
        result = train_and_select_best(
            df,
            target_col,
            use_synthetic=use_synthetic,
            synthetic_min_rows=int(synthetic_min_rows),
            use_balancing=use_balancing,
        )
    st.session_state["result"] = result
    st.success(
        f"Target: {target_col} | Best model: {result['best_model_name']} | "
        f"Task: {result['task_type']}"
    )

if "result" in st.session_state:
    result = st.session_state["result"]

    st.header("4. What the system changed")
    st.info(f"🎯 The model is trained to predict **{result['target_col']}**. This target is never used as an input feature.")

    a, b, c, d = st.columns(4)
    a.metric("Synthetic rows added", result["synthetic_added"])
    b.metric("Balancing", result["balance_method"])
    c.metric("Real test rows", result["test_rows"])
    d.metric("Best model", result["best_model_name"])

    if result["synthetic_applied"]:
        st.success(
            f"🧪 Added {result['synthetic_added']} synthetic rows to the training set."
        )

    if result["balance_applied"]:
        st.success(
            f"⚖️ Class imbalance detected. Applied {result['balance_method']} "
            "to the training set."
        )
    else:
        st.info("⚖️ No significant class imbalance was detected, or balancing was disabled.")

    if result["task_type"] == "classification":
        st.subheader("Class distribution before vs after augmentation/balancing")
        before = pd.Series(result["train_class_counts_before"], name="Before")
        after = pd.Series(result["train_class_counts_after"], name="After")
        compare = pd.concat([before, after], axis=1).fillna(0).astype(int)
        st.dataframe(compare)

    st.header("5. AutoML comparison")
    metrics_df = pd.DataFrame(result["metrics"]).T
    st.dataframe(metrics_df)

    score_name = "F1" if result["task_type"] == "classification" else "R²"
    st.caption(
        f"Best model selected using {score_name}. "
        f"Tuned parameters: {result['best_params']}"
    )

    if result["feature_importance"]:
        st.subheader("Feature importance")
        imp = pd.DataFrame(
            result["feature_importance"].items(),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=False)
        st.bar_chart(imp.set_index("Feature"))

    st.header("6. Live prediction 🔍")
    input_dict = {}
    input_cols = st.columns(3)

    for i, col in enumerate(result["feature_cols"]):
        with input_cols[i % 3]:
            if col in result["encoders"]:
                options = list(result["encoders"][col].classes_)
                input_dict[col] = st.selectbox(col, options, key=f"pred_{col}")
            else:
                default_val = float(pd.to_numeric(df[col], errors="coerce").median())
                if pd.isna(default_val):
                    default_val = 0.0
                input_dict[col] = st.number_input(col, value=default_val, key=f"pred_{col}")

    if st.button("🔮 Predict"):
        prediction = predict_single(
            result["model"],
            result["encoders"],
            result["feature_cols"],
            input_dict,
            scaler=result["scaler"],
            scaled_cols=result["scaled_cols"],
        )

        st.metric(f"Predicted {result['target_col']}", str(prediction))
        st.write(
            explain_prediction(
                input_dict,
                prediction,
                task_type=result["task_type"],
                target_name=result["target_col"],
            )
        )

st.divider()
st.caption("Data2Model — Hackathon 2026")
