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
from PIL import Image

from train_model import train_and_select_best, predict_single, profile_dataset
from llm_explain import explain_prediction, explain_image_prediction
from image_classify import (
    extract_zip_to_tempdir,
    cleanup_tempdir,
    collect_image_paths_and_labels,
    profile_image_dataset,
    train_image_classifier,
    predict_image,
)

st.set_page_config(page_title="Data2Model — Data-to-Model Platform", layout="wide")

st.title("Data2Model")
st.caption("Turning limited data into validated, ML-ready data — and finding the best model, automatically.")

# ---------------------------------------------------------------------------
# 0. Choose data type
# ---------------------------------------------------------------------------
data_mode = st.radio(
    "What kind of data are you working with?",
    ["Tabular data (CSV)", "Images (classification)"],
    horizontal=True,
)

if data_mode == "Images (classification)":
    # -----------------------------------------------------------------------
    # IMAGE CLASSIFICATION FLOW
    # -----------------------------------------------------------------------
    st.header("1. Upload your image dataset")
    st.caption(
        "Upload a .zip file where each subfolder is a class, e.g. "
        "`dataset.zip -> cat/*.jpg, dog/*.jpg`."
    )
    uploaded_zip = st.file_uploader("Upload a ZIP of labeled image folders", type=["zip"])

    if uploaded_zip is None:
        st.info("Upload a ZIP to get started.")
        st.stop()

    if "img_tmp_dir" not in st.session_state or st.session_state.get("img_zip_name") != uploaded_zip.name:
        # (Re)extract only when a new file is uploaded, not on every rerun
        if "img_tmp_dir" in st.session_state:
            cleanup_tempdir(st.session_state["img_tmp_dir"])
        st.session_state["img_tmp_dir"] = extract_zip_to_tempdir(uploaded_zip)
        st.session_state["img_zip_name"] = uploaded_zip.name
        st.session_state.pop("img_result", None)  # reset any previous training result

    paths, labels = collect_image_paths_and_labels(st.session_state["img_tmp_dir"])

    if not paths:
        st.error("No images found. Make sure the zip has one folder per class, each containing images.")
        st.stop()

    st.success(f"Found {len(paths)} images across {len(set(labels))} classes.")

    # -------------------------------------------------------------------
    # 1b. Data profiling — AI-Readiness Score (image version)
    # -------------------------------------------------------------------
    st.header("Data profiling — AI-Readiness Score")
    img_profile = profile_image_dataset(paths, labels)

    score_col, detail_col = st.columns([1, 2])
    with score_col:
        st.metric("Readiness score", f"{img_profile['readiness_score']} / 100")
    with detail_col:
        st.write(
            f"Classes: **{img_profile['num_classes']}**  |  "
            f"Corrupt images: **{img_profile['corrupt_images']}**  |  "
            f"Class balance ratio: **{img_profile['imbalance_ratio']}**"
        )

    st.subheader("Images per class")
    st.bar_chart(pd.Series(img_profile["class_counts"]))

    # -------------------------------------------------------------------
    # 2. Preview a few sample images
    # -------------------------------------------------------------------
    st.header("2. Preview samples")
    sample_cols = st.columns(5)
    for i, path in enumerate(paths[:5]):
        with sample_cols[i % 5]:
            st.image(path, caption=labels[i], use_container_width=True)

    # -------------------------------------------------------------------
    # 3. Train a classifier
    # -------------------------------------------------------------------
    st.header("3. Train a classifier")
    st.caption(
        "This uses transfer learning: a pretrained CNN turns each image into a "
        "feature vector, then a fast classifier is trained on top of that -- "
        "works well even with a small dataset."
    )

    if st.button("🚀 Extract features & train"):
        with st.spinner("Extracting image features and training classifiers..."):
            img_result = train_image_classifier(paths, labels)
        st.session_state["img_result"] = img_result
        st.success(f"Best model: {img_result['best_model_name']}")

    if "img_result" in st.session_state:
        img_result = st.session_state["img_result"]

        st.subheader("Model comparison")
        st.dataframe(pd.DataFrame(img_result["metrics"]).T)
        st.caption(f"Best model: **{img_result['best_model_name']}**")

        # -----------------------------------------------------------------
        # 4. Live prediction
        # -----------------------------------------------------------------
        st.header("4. Try a live prediction")
        target_name = st.text_input("What do these classes represent?", value="class")
        test_image = st.file_uploader("Upload an image to classify", type=["jpg", "jpeg", "png", "bmp", "webp"], key="predict_img")

        if test_image is not None:
            st.image(test_image, caption="Uploaded image", width=250)
            if st.button("🔮 Predict"):
                image_bytes = test_image.getvalue()
                label, confidence = predict_image(
                    img_result["model"], img_result["label_encoder"], image_bytes, scaler=img_result.get("scaler")
                )

                # The model only knows the classes it was trained on -- it has no
                # "none of these" option, so it will always pick one even for an
                # unrelated photo. A low confidence is the only signal we get that
                # the image may not actually match any trained class, so flag it
                # instead of presenting a low-confidence guess as a firm answer.
                num_classes = len(img_result["class_names"])
                chance_level = 100 / num_classes
                low_confidence_cutoff = min(chance_level + 20, 70)

                conf_str = f"{confidence}%" if confidence is not None else "n/a"

                if confidence is not None and confidence < low_confidence_cutoff:
                    st.warning(
                        f"⚠️ Low confidence ({conf_str}). This image may not actually match "
                        f"any of the trained classes ({', '.join(img_result['class_names'])}) -- "
                        "the model can only choose among what it was trained on, so a low "
                        "score here usually means 'none of these look right,' not a reliable answer."
                    )
                else:
                    st.metric(label=f"Predicted {target_name}", value=label, delta=f"confidence: {conf_str}")

                    with st.spinner("Generating explanation..."):
                        explanation = explain_image_prediction(label, confidence, target_name=target_name)
                    st.subheader("🗣️ In plain English")
                    st.write(explanation)

    st.divider()
    st.caption("Data-2-Model - Hackathon 2026")
    st.stop()

# ---------------------------------------------------------------------------
# TABULAR DATA FLOW (unchanged from before)
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
st.header("3. Data augmentation & balancing")

aug_col1, aug_col2, aug_col3 = st.columns(3)
with aug_col1:
    use_synthetic = st.checkbox("Generate synthetic training data", value=True)
with aug_col2:
    synthetic_min_rows = st.number_input(
        "Minimum training rows", min_value=50, max_value=10000, value=500, step=50
    )
with aug_col3:
    use_balancing = st.checkbox("Balance imbalanced classes", value=True)

st.info(
    "Safety design: synthetic data and class balancing are applied only to the "
    "training split. The real test split stays untouched, so the final metrics "
    "measure performance on real data."
)

st.header("4. Train a model")

target_col = st.selectbox("Select the target column (what you want to predict)", df.columns)

if st.button("🚀 Run full pipeline (clean → engineer → augment → AutoML)"):
    with st.spinner("Cleaning data, engineering features, augmenting, training and tuning models..."):
        result = train_and_select_best(
            df,
            target_col,
            use_synthetic=use_synthetic,
            synthetic_min_rows=int(synthetic_min_rows),
            use_balancing=use_balancing,
        )
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

    st.subheader("What the system changed")
    a, b, d = st.columns(3)
    a.metric("Synthetic rows added", result["synthetic_added"])
    b.metric("Balancing", result["balance_method"])
    d.metric("Real test rows", result["test_rows"])

    if result["synthetic_applied"]:
        st.success(f"🧪 Added {result['synthetic_added']} synthetic rows to the training set.")

    if result["balance_applied"]:
        st.success(f"⚖️ Class imbalance detected. Applied {result['balance_method']} to the training set.")

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
    # 5. Live prediction
    # -----------------------------------------------------------------------
    st.header("5. Try a live prediction")
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
st.caption("Data-2-Model - Hackathon 2026")
