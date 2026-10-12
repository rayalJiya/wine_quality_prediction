import sys

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

from data_utils import PROJECT_ROOT, get_train_test_split
from tuning_config import EXPERIMENTS, get_cv, make_pipeline

REPORT_DIR = PROJECT_ROOT / "reports"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def cv_acc(model, X, y):
    return cross_val_score(model, X, y, cv=get_cv(), scoring="accuracy", n_jobs=-1).mean()


def main():
    comp_path = REPORT_DIR / "model_comparison.csv"
    if not comp_path.exists():
        print("ERROR: run src\\train_models.py first.")
        sys.exit(1)

    best = pd.read_csv(comp_path, index_col=0).index[0]
    X_train, X_test, y_train, y_test = get_train_test_split()
    print(f"Best model from Phase 7: {best}")

    base = make_pipeline(best)
    base_cv = cv_acc(base, X_train, y_train)
    print(f"Baseline 5-fold CV accuracy (train only): {base_cv:.4f}")

    # One-parameter-at-a-time experiments, judged by CV on TRAIN data only
    section("PARAMETER EXPERIMENTS (CV accuracy on training data)")
    rows = []
    chosen = {}
    for param, values in EXPERIMENTS[best].items():
        scores = []
        for v in values:
            model = clone(base).set_params(**{param: v})
            s = cv_acc(model, X_train, y_train)
            scores.append((v, s))
            rows.append({"parameter": param, "value": str(v), "cv_accuracy": round(s, 4)})
        chosen[param] = max(scores, key=lambda t: t[1])[0]  # ties keep first (baseline-first order)

    exp_df = pd.DataFrame(rows)
    print(exp_df.to_string(index=False))
    exp_df.to_csv(REPORT_DIR / "improvement_experiments.csv", index=False)

    section("EFFECT SIZE PER PARAMETER (max CV acc - min CV acc)")
    effect = (exp_df.groupby("parameter")["cv_accuracy"].agg(lambda s: s.max() - s.min())
              .sort_values(ascending=False).round(4))
    print(effect.to_string())
    print(f"\nParameter with greatest effect: {effect.index[0]}")

    # Combine best value of each parameter, compare with the original
    section("ORIGINAL VS IMPROVED MODEL")
    print("Chosen values (by CV on train):")
    for k, v in chosen.items():
        print(f"  {k} = {v}")

    improved = clone(base).set_params(**chosen)
    out = []
    for label, model in [("Original", clone(base)), ("Improved", improved)]:
        model.fit(X_train, y_train)
        p_tr, p_te = model.predict(X_train), model.predict(X_test)
        out.append({
            "model": label,
            "train_accuracy": accuracy_score(y_train, p_tr),
            "cv_accuracy": cv_acc(model, X_train, y_train),
            "test_accuracy": accuracy_score(y_test, p_te),
            "test_f1_weighted": f1_score(y_test, p_te, average="weighted", zero_division=0),
            "test_f1_macro": f1_score(y_test, p_te, average="macro", zero_division=0),
        })
    res = pd.DataFrame(out).set_index("model").round(4)
    res["train_test_gap"] = (res["train_accuracy"] - res["test_accuracy"]).round(4)
    print()
    print(res.to_string())
    res.to_csv(REPORT_DIR / "improvement_comparison.csv")

    print("\nNote: higher TRAIN accuracy alone does not mean a better model.")
    print("Check CV/test accuracy and the train-test gap.")
    print("\nPhase 9a complete.")


if __name__ == "__main__":
    main()