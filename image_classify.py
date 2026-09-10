"""
image_classify.py
------------------
Adds image classification to AutoPrep, using the SAME "upload data -> AutoML
-> live prediction" flow as train_model.py, just for images instead of CSVs.

How it works (classical computer-vision features, not a CNN):
    1. User uploads a .zip where each subfolder name is a class label, e.g.

        dataset.zip
        ├── cat/
        │   ├── img001.jpg
        │   └── img002.jpg
        └── dog/
            ├── img001.jpg
            └── img002.jpg

    2. Every image is resized and turned into a fixed-length numeric vector
       made of two classical, hand-crafted feature types:
         - a color histogram (captures overall color distribution)
         - HOG -- Histogram of Oriented Gradients (captures shape/edges)
       No neural network or pretrained weights are involved.
    3. Those vectors are scaled and fed into a lightweight classifier
       (LogisticRegression / RandomForest -- the same models already used
       for tabular data). This trains in seconds even on CPU.
    4. For live prediction, a new image is turned into a vector the same way
       and fed to the trained classifier.

Why this approach instead of a pretrained CNN (e.g. torch/torchvision):
    - Free hosting tiers (like Streamlit Community Cloud) cap memory around
      1GB. torch + torchvision + a downloaded model checkpoint can exceed
      that and get the app silently killed -- exactly what a CNN-based
      version of this file hit in production.
    - No model weights to download at startup -- one less thing that can
      fail on a flaky connection during a live demo.
    - Much lighter dependency footprint: just scikit-image + Pillow, both
      small, pure-Python-adjacent packages.
    - Trade-off: lower accuracy than a real CNN, especially on visually
      similar classes. Works well for visually distinct classes (cats vs
      dogs, several flower types) which covers most hackathon demos.

Setup:
    pip install scikit-image pillow
"""

import io
import os
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from skimage.feature import hog
from skimage.color import rgb2gray

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMG_SIZE = (128, 128)  # all images are resized to this before feature extraction


# ---------------------------------------------------------------------------
# 1. Loading a labeled image dataset from an uploaded ZIP
# ---------------------------------------------------------------------------
def extract_zip_to_tempdir(uploaded_zip_file) -> str:
    """Save an uploaded (Streamlit) zip file to disk and extract it.

    Returns the path to the extraction directory. Caller is responsible for
    cleaning it up (see cleanup_tempdir) once training is done.
    """
    tmp_dir = tempfile.mkdtemp(prefix="autoprep_images_")
    with zipfile.ZipFile(uploaded_zip_file) as zf:
        zf.extractall(tmp_dir)
    return tmp_dir


def cleanup_tempdir(tmp_dir: str):
    shutil.rmtree(tmp_dir, ignore_errors=True)


def collect_image_paths_and_labels(root_dir: str):
    """Walk a class-per-folder directory tree and collect (path, label) pairs.

    Skips macOS/zip junk (__MACOSX, .DS_Store) and non-image files.
    If the zip has one extra top-level wrapper folder (common when zipping
    a folder on Mac/Windows), it is transparently unwrapped.
    """
    # Unwrap a single top-level folder, e.g. dataset.zip -> dataset/cat, dataset/dog
    entries = [e for e in os.listdir(root_dir) if not e.startswith((".", "__MACOSX"))]
    if len(entries) == 1 and os.path.isdir(os.path.join(root_dir, entries[0])):
        root_dir = os.path.join(root_dir, entries[0])

    paths, labels = [], []
    for label in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir) or label.startswith((".", "__MACOSX")):
            continue
        for fname in os.listdir(label_dir):
            if fname.lower().endswith(VALID_EXTENSIONS):
                paths.append(os.path.join(label_dir, fname))
                labels.append(label)
    return paths, labels


def profile_image_dataset(paths: list, labels: list) -> dict:
    """Quick readiness report for an image dataset, mirroring profile_dataset()
    in train_model.py so the UI can show a similar "AI-Readiness" summary.
    """
    n = len(paths)
    class_counts = pd.Series(labels).value_counts().to_dict() if n else {}
    n_classes = len(class_counts)

    corrupt = 0
    for p in paths:
        try:
            with Image.open(p) as img:
                img.verify()
        except (UnidentifiedImageError, OSError):
            corrupt += 1

    imbalance_ratio = None
    if n_classes > 1:
        counts = list(class_counts.values())
        imbalance_ratio = round(min(counts) / max(counts), 3)

    score = 100.0
    if n < 20 * max(n_classes, 1):
        score -= 25  # very few images per class for transfer learning
    if corrupt:
        score -= min(corrupt / max(n, 1) * 100, 20)
    if imbalance_ratio is not None and imbalance_ratio < 0.3:
        score -= 15
    score = max(0, round(score, 1))

    return {
        "total_images": n,
        "num_classes": n_classes,
        "class_counts": class_counts,
        "corrupt_images": corrupt,
        "imbalance_ratio": imbalance_ratio,
        "readiness_score": score,
    }


# ---------------------------------------------------------------------------
# 2. Feature extraction (color histogram + HOG -- no neural network)
# ---------------------------------------------------------------------------
def _load_image(path_or_bytes) -> Image.Image:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(path_or_bytes))
    else:
        img = Image.open(path_or_bytes)
    return img.convert("RGB")


def _extract_feature_vector(img: Image.Image) -> np.ndarray:
    """Turn one image into a fixed-length numeric vector: a color histogram
    (16 bins x 3 channels) plus HOG shape features on the grayscale version.
    """
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H x W x 3, values in [0, 1]

    # Color histogram: distribution of pixel intensities per channel
    hist_features = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0.0, 1.0))
        hist_features.append(hist.astype(np.float32))
    color_hist = np.concatenate(hist_features)
    color_hist /= (color_hist.sum() + 1e-8)  # normalize so image size doesn't matter

    # HOG: captures shape/edges, robust to lighting differences
    gray = rgb2gray(arr)
    hog_features = hog(
        gray, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), feature_vector=True,
    )

    return np.concatenate([color_hist, hog_features]).astype(np.float32)


def embed_images(image_sources: list) -> np.ndarray:
    """Turn a list of image paths (or raw bytes) into an (N, D) feature matrix."""
    feats = [_extract_feature_vector(_load_image(src)) for src in image_sources]
    return np.vstack(feats)


def embed_single_image(image_source) -> np.ndarray:
    return _extract_feature_vector(_load_image(image_source))


# ---------------------------------------------------------------------------
# 3. Train + select best classifier on top of the embeddings
# ---------------------------------------------------------------------------
def train_image_classifier(paths: list, labels: list, test_size: float = 0.2, random_state: int = 42):
    """Full pipeline: embed images -> train a couple of classifiers on the
    embeddings -> pick the best by accuracy. Returns everything the app
    needs for live prediction, mirroring train_and_select_best()'s return
    shape in train_model.py.
    """
    if len(set(labels)) < 2:
        raise ValueError("Need at least 2 classes (folders) to train a classifier.")

    profile = profile_image_dataset(paths, labels)

    X = embed_images(paths)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Color histogram and HOG values live on different scales -- scale them
    # so neither dominates the other, especially for LogisticRegression.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=random_state),
    }

    metrics, trained = {}, {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        trained[name] = model
        metrics[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds, average="weighted"), 4),
            "precision": round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, average="weighted", zero_division=0), 4),
        }

    best_name = max(metrics, key=lambda n: metrics[n]["accuracy"])
    best_model = trained[best_name]

    return {
        "model": best_model,
        "best_model_name": best_name,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "metrics": metrics,
        "profile": profile,
        "class_names": list(label_encoder.classes_),
    }


def predict_image(model, label_encoder, image_source, scaler=None):
    """Predict the class of a single new image, returning (label, confidence)."""
    embedding = embed_single_image(image_source).reshape(1, -1)
    if scaler is not None:
        embedding = scaler.transform(embedding)
    pred_idx = model.predict(embedding)[0]
    label = label_encoder.inverse_transform([pred_idx])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(embedding)[0]
        confidence = round(float(np.max(proba)) * 100, 1)

    return label, confidence
