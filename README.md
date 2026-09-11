# data2model — AutoPrep

**Turning limited data into validated, ML-ready data — and finding the best model, automatically.**

An end-to-end Data Science + AutoML platform built for our Boot Camp & Hackathon 2026 submission.
Upload any raw CSV, and the app automatically profiles it, cleans it, balances it, engineers features,
trains multiple ML models with hyperparameter tuning, and explains the best one in plain English.

## Live Demo
data-2-model.streamlit.app

## Pipeline

| Stage | What Happens |
|---|---|
| 1. Data Ingestion | Upload any CSV |
| 2. Data Profiling | Missing values, duplicates, outliers, imbalance -> AI-Readiness Score (0-100) |
| 3. Cleaning & Validation | Imputation, de-duplication, encoding |
| 4. Feature Engineering | Automatic scaling of numeric features |
| 5. AutoML Model Search | Trains Logistic/Linear Regression, Decision Tree, Random Forest with light hyperparameter tuning (GridSearchCV) + SMOTE for imbalanced classes |
| 6. Best Model + Report | Picks the best-performing model, shows metrics, and generates a plain-English explanation via LLM |

## Files
- `app.py` — Streamlit app (upload -> profile -> clean -> train -> predict -> explain)
- `train_model.py` — Full pipeline: profiling, cleaning, feature engineering, SMOTE, AutoML + tuning
- `llm_explain.py` — Turns a prediction into a plain-English explanation
- `requirements.txt` — All dependencies

## Setup

```bash
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## LLM Explanation Setup

1. Get an API key from [OpenAI](https://platform.openai.com/).
2. Set it as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-key-here"      # Mac/Linux
   setx OPENAI_API_KEY "your-key-here"         # Windows
   ```
3. Test it directly:
   ```bash
   python llm_explain.py
   ```
   If no key is set, the app still works — it falls back to a simple templated sentence so the demo never breaks.

## Deploying (free, via Streamlit Community Cloud)

1. Push this repo to GitHub (already done ✅).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Select this repo (`data2model`), branch `main`, main file `app.py`.
4. Under **Advanced settings -> Secrets**, add:
   ```
   OPENAI_API_KEY = "your-key-here"
   ```
5. Deploy. You'll get a live URL to put on your pitch slide.

## Team Roles

- **Data cleaning & profiling** — owns Stage 2-3
- **Model training & tuning** — owns Stage 4-5
- **Streamlit UI / app polish** — owns the interface
- **LLM explanation, deployment, pitch** — owns Stage 6 + demo

## Recommended Test Dataset

[Loan Prediction Dataset](https://www.kaggle.com) — has real missing values, categorical columns, and
class imbalance, so it properly demonstrates the readiness score, cleaning, and SMOTE balancing.

---
Built for Sri Indu College of Engineering & Technology — Boot Camp & Hackathon 2026.
