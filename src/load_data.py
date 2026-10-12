from pathlib import Path
import sys

import pandas as pd

# Build the path relative to this file, so it works no matter where you run it from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "WineQT.csv"


def main():
    # 1. Verify the file exists
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at: {DATA_PATH}")
        print("Place WineQT.csv inside the 'data' folder and try again.")
        sys.exit(1)

    print(f"File found: {DATA_PATH}\n")

    # 2. Load the dataset
    df = pd.read_csv(DATA_PATH)

    # 3. Print shape
    print(f"Shape: {df.shape}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

    # 4. Print column names
    print("Column names:")
    for col in df.columns:
        print(f"  - {col}")

    # 5. Print first five rows
    print("\nFirst five rows:")
    print(df.head().to_string())


if __name__ == "__main__":
    main()