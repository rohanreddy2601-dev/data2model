"""
image_classify.py
------------------
Adds image classification to AutoPrep, using the SAME "upload data -> AutoML
-> live prediction" flow as train_model.py, just for images instead of CSVs.

How it works (transfer learning, not training a CNN from scratch):
    1. User uploads a .zip where each subfolder name is a class label, e.g.

        dataset.zip
        ├── cat/
        │   ├── img001.jpg
        │   └── img002.jpg
        └── dog/
            ├── img001.jpg
            └── img002.jpg

    2. Every image is passed through a small, FROZEN, ImageNet-pretrained
       CNN (MobileNetV3-Small) to turn it into a short numeric "embedding"
       vector. No weights are trained here -- this step is just feature
       extraction, so it's fast and works fine on CPU / free hosting tiers.
    3. A lightweight classifier (LogisticRegression / RandomForest -- the
       same models already used for tabular data) is trained on those
       embeddings. This works well even with a small number of images per
       class (tens, not thousands), which is realistic for a hackathon demo.
    4. For live prediction, a new image is embedded the same way and fed to
       the trained classifier.

Why this approach instead of training a CNN from scratch:
    - Works with very little data (a from-scratch CNN needs thousands of
      images per class to be any good).
    - Trains in seconds/minutes on CPU -- safe for a live demo.
    - Small dependency footprint: only torch + torchvision + Pillow.

Setup:
    pip install torch torchvision pillow
    (first run downloads the pretrained MobileNetV3-Small weights, ~10MB)
"""

import io
import os
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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
# 2. Feature extraction (frozen pretrained CNN)
# ---------------------------------------------------------------------------
_FEATURE_EXTRACTOR = None
_PREPROCESS = None


def get_feature_extractor():
    """Load (once) a frozen, pretrained MobileNetV3-Small with its
    classification head removed, so it outputs a feature embedding instead
    of ImageNet class scores. Cached as a module-level singleton so repeated
    calls (e.g. every Streamlit rerun) don't reload the weights.
    """
    global _FEATURE_EXTRACTOR, _PREPROCESS
    if _FEATURE_EXTRACTOR is None:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        model.classifier = torch.nn.Identity()  # strip the ImageNet head
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        _FEATURE_EXTRACTOR = model
        _PREPROCESS = weights.transforms()
    return _FEATURE_EXTRACTOR, _PREPROCESS


def _load_image(path_or_bytes) -> Image.Image:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(path_or_bytes))
    else:
        img = Image.open(path_or_bytes)
    return img.convert("RGB")


def embed_images(image_sources: list, batch_size: int = 16) -> np.ndarray:
    """Turn a list of image paths (or raw bytes) into an (N, D) embedding matrix."""
    model, preprocess = get_feature_extractor()
    all_feats = []

    with torch.no_grad():
        for i in range(0, len(image_sources), batch_size):
            batch = image_sources[i:i + batch_size]
            tensors = []
            for src in batch:
                img = _load_image(src)
                tensors.append(preprocess(img))
            batch_tensor = torch.stack(tensors)
            feats = model(batch_tensor)
            all_feats.append(feats.numpy())

    return np.vstack(all_feats)


def embed_single_image(image_source) -> np.ndarray:
    return embed_images([image_source])[0]


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
        "metrics": metrics,
        "profile": profile,
        "class_names": list(label_encoder.classes_),
    }


def predict_image(model, label_encoder, image_source):
    """Predict the class of a single new image, returning (label, confidence)."""
    embedding = embed_single_image(image_source).reshape(1, -1)
    pred_idx = model.predict(embedding)[0]
    label = label_encoder.inverse_transform([pred_idx])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(embedding)[0]
        confidence = round(float(np.max(proba)) * 100, 1)

    return label, confidence
