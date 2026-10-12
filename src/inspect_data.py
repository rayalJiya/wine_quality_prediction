from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "WineQT.csv"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    # Show all columns in the terminal
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    section("1. FIRST 5 RECORDS")
    print(df.head().to_string())

    section("2. LAST 5 RECORDS")
    print(df.tail().to_string())

    section("3. 5 RANDOM RECORDS (random_state=42)")
    print(df.sample(5, random_state=42).to_string())

    section("4. SHAPE")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    section("5. COLUMN NAMES")
    print(list(df.columns))

    section("6. DATA TYPES")
    print(df.dtypes)

    section("7. UNIQUE VALUES PER COLUMN")
    print(df.nunique())

    section("8. QUALITY DISTRIBUTION (counts)")
    print(df["quality"].value_counts().sort_index())

    section("9. QUALITY DISTRIBUTION (percentage)")
    print((df["quality"].value_counts(normalize=True).sort_index() * 100).round(2))

    section("10. SUMMARY STATISTICS")
    print(df.describe().T.round(3).to_string())

    section("11. MISSING VALUES PER COLUMN")
    print(df.isnull().sum())
    print(f"\nTotal missing values: {df.isnull().sum().sum()}")

    section("12. DUPLICATE RECORDS")
    print(f"Fully duplicate rows (including Id): {df.duplicated().sum()}")
    features_only = df.drop(columns=["Id"])
    print(f"Duplicate rows ignoring Id: {features_only.duplicated().sum()}")


if __name__ == "__main__":
    main()