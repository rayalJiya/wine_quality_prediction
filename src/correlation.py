from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "WineQT_clean.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET = "quality"
STRONG_WITH_TARGET = 0.30   # |r| threshold for "notable" link with quality
STRONG_BETWEEN_FEATURES = 0.60  # |r| threshold for feature-feature pairs

sns.set_theme(style="white")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    if not CLEAN_PATH.exists():
        print(f"ERROR: Clean dataset not found at: {CLEAN_PATH}")
        print("Run src\\clean_data.py first.")
        sys.exit(1)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_PATH)
    features = [c for c in df.columns if c != TARGET]
    print(f"Loaded clean data: {df.shape}")

    # 1. Correlation matrix (Pearson)
    section("1. CORRELATION MATRIX (Pearson)")
    corr = df.corr(numeric_only=True)
    print(corr.round(2).to_string())
    corr.round(4).to_csv(REPORT_DIR / "correlation_matrix.csv")

    # 2. Heatmap
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix of Wine Features and Quality")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {FIG_DIR / '05_correlation_heatmap.png'}")

    # 3. Correlation with quality
    section("2. CORRELATION WITH QUALITY (sorted by absolute value)")
    with_target = corr[TARGET].drop(TARGET)
    ranked = with_target.reindex(with_target.abs().sort_values(ascending=False).index)
    print(ranked.round(3).to_string())

    strong = ranked[ranked.abs() >= STRONG_WITH_TARGET]
    print(f"\nFeatures with |r| >= {STRONG_WITH_TARGET} with quality:")
    print(strong.round(3).to_string() if len(strong) else "None")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["seagreen" if v > 0 else "indianred" for v in ranked.values[::-1]]
    ax.barh(ranked.index[::-1], ranked.values[::-1], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Correlation of Each Feature with Quality")
    ax.set_xlabel("Pearson correlation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_correlation_with_quality.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {FIG_DIR / '06_correlation_with_quality.png'}")

    # 4. Feature-feature correlations
    section(f"3. FEATURE PAIRS WITH |r| >= {STRONG_BETWEEN_FEATURES}")
    pairs = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= STRONG_BETWEEN_FEATURES:
                pairs.append((a, b, round(r, 3)))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    if pairs:
        for a, b, r in pairs:
            print(f"{a}  <->  {b}: {r}")
    else:
        print("None")

    # 5. Spearman for comparison (quality is ordinal)
    section("4. SPEARMAN CORRELATION WITH QUALITY (for comparison)")
    spearman = df.corr(method="spearman", numeric_only=True)[TARGET].drop(TARGET)
    spearman = spearman.reindex(spearman.abs().sort_values(ascending=False).index)
    print(spearman.round(3).to_string())

    print("\nCorrelation analysis complete.")


if __name__ == "__main__":
    main()