import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from data_utils import FEATURES, PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "tuned_best_model.joblib"
META_PATH = PROJECT_ROOT / "models" / "feature_meta.json"


def quality_category(score):
    """Task 16: numeric score -> category."""
    if score in (3, 4):
        return "Poor"
    if score == 5:
        return "Average"
    if score == 6:
        return "Good"
    if score in (7, 8):
        return "Excellent"
    return "Unknown"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run src/tune_model.py first.")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_meta():
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {META_PATH}. Run src/save_metadata.py first.")
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def predict_quality(values):
    """
    values: dict with the 11 feature names as keys.
    Returns predicted score, category, confidence (0-1) and all class probabilities.
    """
    missing = [f for f in FEATURES if f not in values]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    row = pd.DataFrame([[float(values[f]) for f in FEATURES]], columns=FEATURES)
    if row.isnull().any().any() or (row < 0).any().any():
        raise ValueError("All values must be valid, non-negative numbers.")

    model = load_model()
    proba = model.predict_proba(row)[0]
    classes = [int(c) for c in model.classes_]

    # Predicted class = class with the highest probability, so the shown
    # prediction and the shown confidence always refer to the same thing.
    best_idx = int(np.argmax(proba))
    predicted = classes[best_idx]

    return {
        "predicted_quality": predicted,
        "category": quality_category(predicted),
        "confidence": float(proba[best_idx]),
        "probabilities": {c: float(p) for c, p in zip(classes, proba)},
    }


def out_of_range_features(values):
    """Features whose value is outside the range seen in training data."""
    meta = load_meta()
    out = []
    for f in FEATURES:
        v = float(values[f])
        if v < meta[f]["min"] or v > meta[f]["max"]:
            out.append(f)
    return out