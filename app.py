"""
app.py
-------
Streamlit skeleton for the hackathon.

Right now it works with ANY CSV + any target column, so you can test the
whole flow today with a dummy dataset. When the real problem statement
drops, just upload the real data -- nothing else needs to change.

Run locally:
    streamlit run app.py

Deploy (free):
    Push this folder to GitHub, then deploy on share.streamlit.io
    (Streamlit Community Cloud) pointing at app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from train_model import train_and_select_best, predict_single, profile_dataset
from llm_explain import explain_prediction

st.set_page_config(page_title="AutoPrep — Data-to-Model Platform", layout="wide")

st.title("AutoPrep")
st.caption("Turning limited data into validated, ML-ready data — and finding the best model, automatically.")

# ---------------------------------------------------------------------------
# 1. Upload data
# ---------------------------------------------------------------------------
st.header("1. Upload your dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV to get started. (Use any dummy dataset now to test the app end-to-end.)")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
st.dataframe(df.head())

# ---------------------------------------------------------------------------
# 1b. Data Profiling — AI-Readiness Score (raw data, before cleaning)
# ---------------------------------------------------------------------------
st.header("Data profiling — AI-Readiness Score")
st.caption("This is calculated on the raw data, before any cleaning happens.")

raw_target_guess = df.columns[-1]
profile = profile_dataset(df, raw_target_guess)

score_col, detail_col = st.columns([1, 2])
with score_col:
    st.metric("Readiness score", f"{profile['readiness_score']} / 100")
with detail_col:
    st.write(f"Missing values: **{profile['missing_pct']}%**  |  "
             f"Duplicate rows: **{profile['duplicate_pct']}%**  |  "
             f"Outliers: **{profile['outlier_pct']}%**")

# ---------------------------------------------------------------------------
# 2. Explore the data (EDA)
# ---------------------------------------------------------------------------
st.header("2. Explore the data")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Missing values")
    st.write(df.isnull().sum())

with col2:
    st.subheader("Summary statistics")
    st.write(df.describe())

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if len(numeric_cols) >= 2:
    st.subheader("Correlation heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

if numeric_cols:
    st.subheader("Distribution of a column")
    dist_col = st.selectbox("Choose a numeric column to visualize", numeric_cols)
    fig2, ax2 = plt.subplots()
    sns.histplot(df[dist_col], kde=True, ax=ax2)
    st.pyplot(fig2)

# ---------------------------------------------------------------------------
# 3. Train a model
# ---------------------------------------------------------------------------
st.header("3. Train a model")

target_col = st.selectbox("Select the target column (what you want to predict)", df.columns)

if st.button("🚀 Run full pipeline (clean → engineer → AutoML)"):
    with st.spinner("Cleaning data, engineering features, training and tuning models..."):
        result = train_and_select_best(df, target_col)
    st.session_state["result"] = result
    st.success(f"Best model: {result['best_model_name']} ({result['task_type']})")

if "result" in st.session_state:
    result = st.session_state["result"]

    st.subheader("Before → after cleaning")
    before, after = result["profile_before"], result["profile_after"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Readiness score", f"{after['readiness_score']}/100", delta=round(after['readiness_score'] - before['readiness_score'], 1))
    c2.metric("Missing values", f"{after['missing_pct']}%", delta=round(after['missing_pct'] - before['missing_pct'], 2), delta_color="inverse")
    c3.metric("Duplicate rows", f"{after['duplicate_pct']}%", delta=round(after['duplicate_pct'] - before['duplicate_pct'], 2), delta_color="inverse")

    if result["smote_applied"]:
        st.info("Class imbalance detected — SMOTE was applied to balance the training data.")

    st.subheader("AutoML model comparison (with hyperparameter tuning)")
    metrics_df = pd.DataFrame(result["metrics"]).T
    st.dataframe(metrics_df)
    st.caption(f"Best model: **{result['best_model_name']}** — tuned parameters: `{result['best_params']}`")

    if result["feature_importance"]:
        st.subheader("Feature importance")
        importance_df = pd.DataFrame(
            result["feature_importance"].items(), columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Feature"))

    # -----------------------------------------------------------------------
    # 4. Live prediction
    # -----------------------------------------------------------------------
    st.header("4. Try a live prediction")
    st.caption("Enter values below and see the model predict in real time -- this is the part judges love.")

    input_dict = {}
    input_cols = st.columns(3)
    for i, col in enumerate(result["feature_cols"]):
        with input_cols[i % 3]:
            if col in result["encoders"]:
                options = list(result["encoders"][col].classes_)
                input_dict[col] = st.selectbox(col, options, key=f"input_{col}")
            else:
                default_val = float(df[col].median()) if col in df.columns else 0.0
                input_dict[col] = st.number_input(col, value=default_val, key=f"input_{col}")

    if st.button("🔮 Predict"):
        prediction = predict_single(
            result["model"], result["encoders"], result["feature_cols"], input_dict,
            scaler=result.get("scaler"), scaled_cols=result.get("scaled_cols"),
        )

        st.metric(label=f"Predicted {target_col}", value=str(prediction))

        with st.spinner("Generating explanation..."):
            explanation = explain_prediction(
                input_dict, prediction, task_type=result["task_type"], target_name=target_col
            )
        st.subheader("🗣️ In plain English")
        st.write(explanation)

st.divider()
st.caption("Built for Boot Camp & Hackathon 2026 — Sri Indu College of Engineering & Technology")
