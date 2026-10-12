from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "WineQT.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "WineQT_clean.csv"

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol",
]
TARGET = "quality"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    if not RAW_PATH.exists():
        print(f"ERROR: Dataset not found at: {RAW_PATH}")
        sys.exit(1)

    df = pd.read_csv(RAW_PATH)
    print(f"Original shape: {df.shape}")

    # 1. Drop Id (identifier, not a predictive feature)
    section("1. DROP Id")
    df = df.drop(columns=["Id"])
    print(f"Shape after dropping Id: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # 2. Missing values
    section("2. MISSING VALUES")
    missing = df.isnull().sum().sum()
    print(f"Total missing values: {missing}")
    if missing > 0:
        print("Missing values found. Stop and share this output.")
        sys.exit(1)
    print("No missing values, so no imputation needed.")

    # 3. Duplicates
    section("3. DUPLICATES")
    n_dup = df.duplicated().sum()
    print(f"Duplicate rows found: {n_dup}")
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    print(f"Shape after removing duplicates: {df.shape}")

    # 4. Invalid / unusual values
    section("4. INVALID VALUE CHECKS")
    checks = {
        "Negative values in any feature": (df[FEATURES] < 0).any(axis=1).sum(),
        "citric acid outside 0-1": (~df["citric acid"].between(0, 1)).sum(),
        "pH outside 0-14": (~df["pH"].between(0, 14)).sum(),
        "density <= 0": (df["density"] <= 0).sum(),
        "alcohol outside 0-100": (~df["alcohol"].between(0, 100)).sum(),
        "free SO2 > total SO2": (df["free sulfur dioxide"] > df["total sulfur dioxide"]).sum(),
        "quality outside 3-8": (~df[TARGET].between(3, 8)).sum(),
    }
    for name, count in checks.items():
        print(f"{name}: {count}")

    # 5. Data types
    section("5. DATA TYPES")
    print(df.dtypes)
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    print(f"\nNon-numeric columns: {non_numeric}")

    # 6. Final check and save
    section("6. FINAL CLEAN DATASET")
    print(f"Final shape: {df.shape}")
    print("\nQuality distribution after cleaning:")
    print(df[TARGET].value_counts().sort_index())
    print(f"\n'Id' present: {'Id' in df.columns}")

    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned dataset to: {CLEAN_PATH}")


if __name__ == "__main__":
    main()