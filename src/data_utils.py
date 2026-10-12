from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "WineQT_clean.csv"

TARGET = "quality"
FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol",
]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_clean_data():
    if not CLEAN_PATH.exists():
        print(f"ERROR: Clean dataset not found at: {CLEAN_PATH}")
        print("Run src\\clean_data.py first.")
        sys.exit(1)
    return pd.read_csv(CLEAN_PATH)


def get_features_and_target(df):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    assert TARGET not in X.columns, "Target leaked into features!"
    assert "Id" not in X.columns, "Id must not be a feature!"
    return X, y


def get_train_test_split():
    """Unscaled, stratified 80/20 split. Scaling is done later, fitted on train only."""
    df = load_clean_data()
    X, y = get_features_and_target(df)
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )