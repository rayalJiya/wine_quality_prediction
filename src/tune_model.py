import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV

from data_utils import PROJECT_ROOT, get_train_test_split
from tuning_config import CV_FOLDS, GRIDS, get_cv, make_pipeline

REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    comp_path = REPORT_DIR / "model_comparison.csv"
    if not comp_path.exists():
        print("ERROR: run src\\train_models.py first.")
        sys.exit(1)

    MODEL_DIR.mkdir(exist_ok=True)
    best = pd.read_csv(comp_path, index_col=0).index[0]
    X_train, X_test, y_train, y_test = get_train_test_split()
    print(f"Tuning model: {best}")

    n_combos = 1
    for v in GRIDS[best].values():
        n_combos *= len(v)
    print(f"Grid size: {n_combos} combinations x {CV_FOLDS} folds = {n_combos * CV_FOLDS} fits")
    print("This can take a few minutes. Please wait...")

    grid = GridSearchCV(
        estimator=make_pipeline(best),
        param_grid=GRIDS[best],
        cv=get_cv(),
        scoring={"accuracy": "accuracy", "f1_weighted": "f1_weighted", "f1_macro": "f1_macro"},
        refit="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)  # test set is NOT used here

    section("BEST PARAMETERS")
    for k, v in grid.best_params_.items():
        print(f"{k}: {v}")

    i = grid.best_index_
    section("BEST CROSS-VALIDATION SCORES")
    print(f"CV accuracy:    {grid.cv_results_['mean_test_accuracy'][i]:.4f} "
          f"(+/- {grid.cv_results_['std_test_accuracy'][i]:.4f})")
    print(f"CV F1 weighted: {grid.cv_results_['mean_test_f1_weighted'][i]:.4f}")
    print(f"CV F1 macro:    {grid.cv_results_['mean_test_f1_macro'][i]:.4f}")

    # Final one-time test evaluation: original vs tuned
    section("FINAL TEST EVALUATION: ORIGINAL VS TUNED")
    original = make_pipeline(best).fit(X_train, y_train)
    tuned = grid.best_estimator_

    rows = []
    for label, model in [("Original", original), ("Tuned", tuned)]:
        p_tr, p_te = model.predict(X_train), model.predict(X_test)
        rows.append({
            "model": label,
            "train_accuracy": accuracy_score(y_train, p_tr),
            "test_accuracy": accuracy_score(y_test, p_te),
            "test_f1_weighted": f1_score(y_test, p_te, average="weighted", zero_division=0),
            "test_f1_macro": f1_score(y_test, p_te, average="macro", zero_division=0),
        })
    res = pd.DataFrame(rows).set_index("model").round(4)
    print(res.to_string())
    res.to_csv(REPORT_DIR / "tuning_comparison.csv")

    print(f"\nFinal TEST accuracy (tuned): {res.loc['Tuned', 'test_accuracy']:.4f}")
    print(f"Final TEST accuracy (original): {res.loc['Original', 'test_accuracy']:.4f}")
    print(f"Test set size = {len(y_test)}, so 1 sample = {100 / len(y_test):.2f}% accuracy.")

    section("PER-CLASS REPORT (tuned model, test set)")
    print(classification_report(y_test, tuned.predict(X_test), zero_division=0))

    # Top 5 configurations
    section("TOP 5 CONFIGURATIONS BY CV ACCURACY")
    cv_df = pd.DataFrame(grid.cv_results_)
    cols = ["rank_test_accuracy", "mean_test_accuracy", "std_test_accuracy", "params"]
    top = cv_df.sort_values("rank_test_accuracy")[cols].head(5)
    print(top.to_string(index=False))
    cv_df.to_csv(REPORT_DIR / "gridsearch_cv_results.csv", index=False)

    joblib.dump(tuned, MODEL_DIR / "tuned_best_model.joblib")
    print(f"\nSaved tuned model: {MODEL_DIR / 'tuned_best_model.joblib'}")
    print("\nPhase 9b complete.")


if __name__ == "__main__":
    main()