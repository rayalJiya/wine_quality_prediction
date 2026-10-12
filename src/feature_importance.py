import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from data_utils import FEATURES, PROJECT_ROOT, RANDOM_STATE, TARGET, get_train_test_split

REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

sns.set_theme(style="whitegrid")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = get_train_test_split()

    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"Random Forest test accuracy: {rf.score(X_test, y_test):.4f}")

    # 1. Impurity-based importance (computed from training)
    imp = pd.Series(rf.feature_importances_, index=FEATURES)

    # 2. Permutation importance on the TEST set (more reliable)
    perm = permutation_importance(
        rf, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    perm_s = pd.Series(perm.importances_mean, index=FEATURES)

    table = pd.DataFrame({
        "rf_importance": imp,
        "perm_importance_test": perm_s,
    }).sort_values("rf_importance", ascending=False)
    table["rf_rank"] = range(1, len(table) + 1)

    # 3. Compare with correlation analysis (from Phase 5)
    corr_path = REPORT_DIR / "correlation_matrix.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path, index_col=0)[TARGET].drop(TARGET)
        table["corr_with_quality"] = corr
        table["abs_corr_rank"] = table["corr_with_quality"].abs().rank(ascending=False).astype(int)
    else:
        print("Note: correlation_matrix.csv not found, skipping correlation comparison.")

    section("FEATURE IMPORTANCE (ranked, most to least important)")
    print(table.round(4).to_string())
    table.round(5).to_csv(REPORT_DIR / "feature_importance.csv")

    print(f"\nMost important feature: {table.index[0]}")
    print(f"Least important feature: {table.index[-1]}")
    print("Relatively low importance (bottom 3):", list(table.index[-3:]))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    order = table.sort_values("rf_importance")
    axes[0].barh(order.index, order["rf_importance"], color="steelblue")
    axes[0].set_title("Random Forest Feature Importance (impurity-based)")
    axes[0].set_xlabel("Importance")
    order2 = table.sort_values("perm_importance_test")
    axes[1].barh(order2.index, order2["perm_importance_test"], color="seagreen")
    axes[1].set_title("Permutation Importance (test set)")
    axes[1].set_xlabel("Mean drop in accuracy")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {FIG_DIR / '08_feature_importance.png'}")

    print("\nPhase 8b complete.")


if __name__ == "__main__":
    main()