from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "WineQT_clean.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

TARGET = "quality"
KEY_FEATURES = ["alcohol", "volatile acidity", "sulphates", "pH", "citric acid"]

sns.set_theme(style="whitegrid")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    if not CLEAN_PATH.exists():
        print(f"ERROR: Clean dataset not found at: {CLEAN_PATH}")
        print("Run src\\clean_data.py first.")
        sys.exit(1)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_PATH)
    features = [c for c in df.columns if c != TARGET]
    print(f"Loaded clean data: {df.shape}")

    # 1. Quality distribution
    section("1. QUALITY DISTRIBUTION")
    counts = df[TARGET].value_counts().sort_index()
    print(counts)
    print(f"\nMost common quality score: {counts.idxmax()} ({counts.max()} samples)")
    print(f"Least common quality score: {counts.idxmin()} ({counts.min()} samples)")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x=TARGET, color="steelblue", ax=ax)
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_title("Distribution of Wine Quality Scores")
    ax.set_xlabel("Quality")
    ax.set_ylabel("Number of samples")
    save(fig, "01_quality_distribution.png")

    # 2. Histograms of all features
    section("2. FEATURE HISTOGRAMS")
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.histplot(df[col], kde=True, bins=30, color="steelblue", ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("")
    for j in range(len(features), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Distribution of Chemical Features", fontsize=16)
    save(fig, "02_feature_histograms.png")

    # 3. Key features vs quality
    section("3. KEY FEATURES VS QUALITY")
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.flatten()
    for i, col in enumerate(KEY_FEATURES):
        sns.boxplot(data=df, x=TARGET, y=col, color="lightsteelblue", ax=axes[i])
        axes[i].set_title(f"{col} vs quality")
    axes[-1].axis("off")
    fig.suptitle("Relationship Between Key Features and Wine Quality", fontsize=16)
    save(fig, "03_key_features_vs_quality.png")

    print("\nMean value of each key feature per quality score:")
    print(df.groupby(TARGET)[KEY_FEATURES].mean().round(3).to_string())

    # 4. Outlier detection (IQR rule) + boxplots
    section("4. OUTLIERS (IQR RULE)")
    rows = []
    for col in features:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < low) | (df[col] > high)).sum()
        rows.append({
            "feature": col,
            "lower_bound": round(low, 3),
            "upper_bound": round(high, 3),
            "n_outliers": int(n_out),
            "pct": round(100 * n_out / len(df), 2),
        })
    outlier_df = pd.DataFrame(rows).sort_values("n_outliers", ascending=False)
    print(outlier_df.to_string(index=False))

    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.boxplot(x=df[col], color="lightcoral", ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("")
    for j in range(len(features), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Boxplots for Outlier Detection", fontsize=16)
    save(fig, "04_outlier_boxplots.png")

    print("\nEDA complete. Check the reports\\figures folder.")


if __name__ == "__main__":
    main()