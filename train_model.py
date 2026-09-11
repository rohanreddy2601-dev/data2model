"""
train_model.py
---------------
Generic, reusable training pipeline for the hackathon.

Works with ANY dataset + target column:
- Auto-detects classification vs regression
- Cleans missing values
- Encodes categorical columns
- Trains a few standard models
- Picks the best one automatically
- Returns everything the Streamlit app needs to make predictions

Usage:
    from train_model import train_and_select_best

    result = train_and_select_best(df, target_col="Placed")
    result["model"]        -> the trained best model
    result["task_type"]    -> "classification" or "regression"
    result["metrics"]      -> dict of scores for every model tried
    result["encoders"]     -> dict of LabelEncoders used per column
    result["feature_cols"] -> list of feature column names, in order
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_absolute_error,
)

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False


def profile_dataset(df: pd.DataFrame, target_col: str = None) -> dict:
    """Stage 2 -- Data Profiling.

    Scans the RAW dataset (before any cleaning) and returns a readiness
    report: missing %, duplicate %, class imbalance, and a single
    0-100 'AI-Readiness Score'.
    """
    n_rows = len(df)
    missing_pct = round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2)
    duplicate_pct = round(df.duplicated().sum() / n_rows * 100, 2)

    imbalance_ratio = None
    if target_col and target_col in df.columns and not pd.api.types.is_numeric_dtype(df[target_col]):
        counts = df[target_col].value_counts()
        if len(counts) > 1:
            imbalance_ratio = round(counts.min() / counts.max(), 3)

    # Outlier check: % of numeric values beyond 3 std deviations
    numeric_df = df.select_dtypes(include="number")
    outlier_pct = 0.0
    if not numeric_df.empty:
        z_scores = (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0)
        outlier_pct = round((z_scores.abs() > 3).sum().sum() / numeric_df.size * 100, 2)

    # Simple weighted readiness score out of 100
    score = 100.0
    score -= min(missing_pct * 1.5, 40)      # missing data hurts most
    score -= min(duplicate_pct * 1.0, 20)
    score -= min(outlier_pct * 1.0, 15)
    if imbalance_ratio is not None and imbalance_ratio < 0.3:
        score -= 15  # severe class imbalance
    score = max(0, round(score, 1))

    return {
        "rows": n_rows,
        "columns": df.shape[1],
        "missing_pct": missing_pct,
        "duplicate_pct": duplicate_pct,
        "outlier_pct": outlier_pct,
        "imbalance_ratio": imbalance_ratio,
        "readiness_score": score,
    }


def clean_data(df: pd.DataFrame, target_col: str):
    """Handle missing values and encode categorical columns.

    Returns: cleaned dataframe, dict of encoders used (column -> LabelEncoder)
    """
    df = df.copy()

    # Drop rows where the target itself is missing -- can't use those
    df = df.dropna(subset=[target_col])

    # Remove exact duplicate rows (Stage 3: de-duplication)
    df = df.drop_duplicates().reset_index(drop=True)

    encoders = {}

    for col in df.columns:
        if col == target_col:
            # Leave the target column untouched -- sklearn classifiers handle
            # string labels natively, and this keeps predictions readable
            # (e.g. "Yes"/"No" instead of 1/0).
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            # Fill missing numeric values with the median
            df[col] = df[col].fillna(df[col].median())
        else:
            # Fill missing text values with the most common value
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    return df, encoders


def engineer_features(X: pd.DataFrame):
    """Stage 4 -- Feature Engineering.

    Automatically scales numeric features so columns on very different
    ranges (e.g. Age: 0-100 vs Income: 0-1,000,000) don't dominate the model.
    Returns the scaled dataframe and the fitted scaler (needed later for
    scaling new prediction inputs the same way).
    """
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    scaler = None
    X_scaled = X.copy()
    if numeric_cols:
        scaler = StandardScaler()
        X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    return X_scaled, scaler, numeric_cols


def detect_task_type(y: pd.Series) -> str:
    """Very simple heuristic: few unique values / non-numeric -> classification."""
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    unique_ratio = y.nunique() / len(y)
    if y.nunique() <= 15 or unique_ratio < 0.05:
        return "classification"
    return "regression"


def pad_with_synthetic_data(X_train: pd.DataFrame, y_train: pd.Series, task_type: str,
                             min_rows: int, random_state: int = 42):
    """Stage 3b -- Synthetic data generation.

    Hackathon datasets are often tiny (a few dozen rows), which makes
    train/test splits noisy and hyperparameter search unreliable. If the
    training set has fewer than `min_rows` rows, this pads it up by
    duplicating existing rows with a small amount of Gaussian noise added to
    numeric columns, so the duplicates aren't exact copies. For
    classification, duplicates are drawn proportionally to existing class
    frequencies so class balance isn't skewed by padding alone.

    Only ever applied to the TRAINING split -- the test set is never touched,
    so reported metrics still reflect performance on real data.

    Returns (X_padded, y_padded, synthetic_rows_added).
    """
    current_rows = len(X_train)
    if current_rows >= min_rows:
        return X_train, y_train, 0

    rng = np.random.RandomState(random_state)
    rows_needed = min_rows - current_rows

    if task_type == "classification":
        class_fracs = y_train.value_counts(normalize=True)
        sample_idx = []
        for cls, frac in class_fracs.items():
            n_cls = max(1, int(round(rows_needed * frac)))
            cls_indices = y_train[y_train == cls].index.to_numpy()
            sample_idx.extend(rng.choice(cls_indices, size=n_cls, replace=True))
        sample_idx = sample_idx[:rows_needed]
    else:
        sample_idx = rng.choice(y_train.index.to_numpy(), size=rows_needed, replace=True)

    X_synth = X_train.loc[sample_idx].reset_index(drop=True).copy()
    y_synth = y_train.loc[sample_idx].reset_index(drop=True).copy()

    # Jitter numeric columns slightly so synthetic rows aren't exact duplicates
    numeric_cols = X_synth.select_dtypes(include="number").columns
    for col in numeric_cols:
        col_std = X_train[col].std()
        if col_std and col_std > 0:
            noise = rng.normal(0, col_std * 0.05, size=len(X_synth))
            X_synth[col] = X_synth[col] + noise

    X_padded = pd.concat([X_train, X_synth], ignore_index=True)
    y_padded = pd.concat([y_train.reset_index(drop=True), y_synth], ignore_index=True)

    return X_padded, y_padded, len(X_synth)


def train_and_select_best(df: pd.DataFrame, target_col: str, test_size: float = 0.2,
                           random_state: int = 42, use_synthetic: bool = True,
                           synthetic_min_rows: int = 500, use_balancing: bool = True,
                           tune_hyperparams: bool = True):
    """Full pipeline: profile -> clean -> engineer features -> pad with synthetic
    data (if needed) -> balance classes -> AutoML search (with light
    hyperparameter tuning) -> return best model + report.
    """

    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' is not a column in the dataset.")

    # Stage 2: Profiling (on the RAW data, before touching it)
    profile_before = profile_dataset(df, target_col)

    task_type = detect_task_type(df[target_col])

    # Stage 3: Cleaning & validation
    df_clean, encoders = clean_data(df, target_col)
    profile_after = profile_dataset(df_clean, target_col)

    feature_cols = [c for c in df_clean.columns if c != target_col]
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    # Stage 4: Feature engineering (scaling)
    X_scaled, scaler, scaled_cols = engineer_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    # Stage 4b: Synthetic data padding (training split only -- test split stays real)
    synthetic_added = 0
    if use_synthetic:
        X_train, y_train, synthetic_added = pad_with_synthetic_data(
            X_train, y_train, task_type, synthetic_min_rows, random_state
        )
    synthetic_applied = synthetic_added > 0

    # Balance classes with SMOTE if classification + imbalance is significant
    smote_applied = False
    if task_type == "classification" and use_balancing and _SMOTE_AVAILABLE:
        counts = y_train.value_counts()
        if len(counts) > 1 and counts.min() / counts.max() < 0.5 and counts.min() >= 6:
            try:
                smote = SMOTE(random_state=random_state, k_neighbors=min(5, counts.min() - 1))
                X_train, y_train = smote.fit_resample(X_train, y_train)
                smote_applied = True
            except Exception:
                smote_applied = False

    # Stage 5: AutoML model search (+ light hyperparameter tuning)
    if task_type == "classification":
        candidates = {
            "LogisticRegression": (LogisticRegression(max_iter=1000), {"C": [0.1, 1, 10]}),
            "DecisionTree": (DecisionTreeClassifier(random_state=random_state), {"max_depth": [5, 10, None]}),
            "RandomForest": (RandomForestClassifier(random_state=random_state), {"n_estimators": [100, 200], "max_depth": [10, None]}),
        }
    else:
        candidates = {
            "LinearRegression": (LinearRegression(), {}),
            "DecisionTree": (DecisionTreeRegressor(random_state=random_state), {"max_depth": [5, 10, None]}),
            "RandomForest": (RandomForestRegressor(random_state=random_state), {"n_estimators": [100, 200], "max_depth": [10, None]}),
        }

    metrics = {}
    trained_models = {}
    best_params = {}

    for name, (model, param_grid) in candidates.items():
        if tune_hyperparams and param_grid:
            # Small, fast grid so this stays quick enough for a live demo
            search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
            search.fit(X_train, y_train)
            fitted_model = search.best_estimator_
            best_params[name] = search.best_params_
        else:
            fitted_model = model.fit(X_train, y_train)
            best_params[name] = "default"

        preds = fitted_model.predict(X_test)
        trained_models[name] = fitted_model

        if task_type == "classification":
            metrics[name] = {
                "accuracy": round(accuracy_score(y_test, preds), 4),
                "f1": round(f1_score(y_test, preds, average="weighted"), 4),
                "precision": round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
                "recall": round(recall_score(y_test, preds, average="weighted", zero_division=0), 4),
            }
        else:
            metrics[name] = {
                "r2": round(r2_score(y_test, preds), 4),
                "mae": round(mean_absolute_error(y_test, preds), 4),
            }

    # Pick the best model: highest accuracy (classification) or highest R2 (regression)
    score_key = "accuracy" if task_type == "classification" else "r2"
    best_name = max(metrics, key=lambda name: metrics[name][score_key])
    best_model = trained_models[best_name]

    # Feature importance if the model supports it (tree-based models do)
    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        feature_importance = dict(zip(feature_cols, best_model.feature_importances_.round(4)))

    return {
        "model": best_model,
        "best_model_name": best_name,
        "best_params": best_params.get(best_name),
        "task_type": task_type,
        "target_col": target_col,
        "metrics": metrics,
        "encoders": encoders,
        "feature_cols": feature_cols,
        "feature_importance": feature_importance,
        "scaler": scaler,
        "scaled_cols": scaled_cols,
        "smote_applied": smote_applied,
        "balance_applied": smote_applied,
        "balance_method": "SMOTE" if smote_applied else "None",
        "synthetic_applied": synthetic_applied,
        "synthetic_added": synthetic_added,
        "test_rows": len(X_test),
        "profile_before": profile_before,
        "profile_after": profile_after,
    }


def predict_single(model, encoders: dict, feature_cols: list, input_dict: dict,
                    scaler=None, scaled_cols=None):
    """Predict on a single new row of input coming from the Streamlit form.

    input_dict: {column_name: raw_value}
    scaler / scaled_cols: from train_and_select_best(), so new input is scaled
    the exact same way the training data was (Stage 4 consistency).
    """
    row = {}
    for col in feature_cols:
        val = input_dict.get(col)
        if col in encoders:
            # Encode using the same encoder used during training
            try:
                val = encoders[col].transform([str(val)])[0]
            except ValueError:
                # Unseen category at prediction time -> fall back to most common class
                val = 0
        row[col] = val

    X_new = pd.DataFrame([row], columns=feature_cols)

    if scaler is not None and scaled_cols:
        X_new[scaled_cols] = scaler.transform(X_new[scaled_cols])

    prediction = model.predict(X_new)[0]
    return prediction
