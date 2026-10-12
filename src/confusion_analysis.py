import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data_utils import PROJECT_ROOT

REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
LABELS = [3, 4, 5, 6, 7, 8]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    pred_path = REPORT_DIR / "test_predictions.csv"
    comp_path = REPORT_DIR / "model_comparison.csv"
    for p in (pred_path, comp_path):
        if not p.exists():
            print(f"ERROR: {p} not found. Run src\\train_models.py first.")
            sys.exit(1)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pred_df = pd.read_csv(pred_path, index_col=0)
    comp = pd.read_csv(comp_path, index_col=0)

    best = comp.index[0]  # file is sorted by weighted F1, best first
    print(f"Best model (by weighted F1): {best}")

    y_true = pred_df["actual"]
    y_pred = pred_df[best]

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"actual_{l}" for l in LABELS],
                         columns=[f"pred_{l}" for l in LABELS])

    section("1. CONFUSION MATRIX (counts)")
    print(cm_df.to_string())
    cm_df.to_csv(REPORT_DIR / "confusion_matrix_best_model.csv")

    # Row-normalised (per-class recall view)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS,
                yticklabels=LABELS, ax=axes[0])
    axes[0].set_title(f"{best}: Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted quality")
    axes[0].set_ylabel("Actual quality")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=LABELS, yticklabels=LABELS, ax=axes[1])
    axes[1].set_title(f"{best}: Confusion Matrix (row-normalised)")
    axes[1].set_xlabel("Predicted quality")
    axes[1].set_ylabel("Actual quality")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_confusion_matrix_best_model.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {FIG_DIR / '07_confusion_matrix_best_model.png'}")

    section("2. PER-CLASS CORRECT PREDICTIONS")
    for i, label in enumerate(LABELS):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        pct = 100 * correct / total if total else 0
        print(f"Quality {label}: {correct} of {total} correct ({pct:.1f}%)")

    section("3. MOST FREQUENT CONFUSIONS (actual -> predicted)")
    errors = []
    for i, a in enumerate(LABELS):
        for j, p in enumerate(LABELS):
            if i != j and cm[i, j] > 0:
                errors.append((a, p, int(cm[i, j])))
    errors.sort(key=lambda t: t[2], reverse=True)
    for a, p, c in errors[:8]:
        print(f"Actual {a} -> Predicted {p}: {c} samples")

    section("4. DO ERRORS HAPPEN BETWEEN NEIGHBOURING SCORES?")
    total_err = sum(c for _, _, c in errors)
    off_by_one = sum(c for a, p, c in errors if abs(a - p) == 1)
    off_by_two = sum(c for a, p, c in errors if abs(a - p) == 2)
    off_more = total_err - off_by_one - off_by_two
    print(f"Total wrong predictions: {total_err}")
    if total_err:
        print(f"Off by exactly 1 level: {off_by_one} ({100 * off_by_one / total_err:.1f}%)")
        print(f"Off by 2 levels:        {off_by_two} ({100 * off_by_two / total_err:.1f}%)")
        print(f"Off by 3+ levels:       {off_more} ({100 * off_more / total_err:.1f}%)")

    print("\nPhase 8a complete.")


if __name__ == "__main__":
    main()