import sys
from datetime import date

import pandas as pd

from data_utils import PROJECT_ROOT, TARGET

REPORT_DIR = PROJECT_ROOT / "reports"
OUT_PATH = REPORT_DIR / "final_report.md"

REQUIRED = [
    "model_comparison.csv", "scaling_comparison.csv",
    "confusion_matrix_best_model.csv", "feature_importance.csv",
    "improvement_comparison.csv", "improvement_experiments.csv",
    "tuning_comparison.csv", "gridsearch_cv_results.csv",
    "correlation_matrix.csv",
]


def read(name, **kw):
    return pd.read_csv(REPORT_DIR / name, **kw)


def pct(x):
    return f"{x * 100:.1f}%"


def md_table(df, index_name=""):
    d = df.reset_index()
    if index_name:
        d = d.rename(columns={d.columns[0]: index_name})
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = []
    for _, r in d.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in r]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


def verdict(delta, tol=0.005):
    if delta > tol:
        return "improved"
    if delta < -tol:
        return "did not improve (slightly lower)"
    return "stayed essentially the same"


def main():
    missing = [f for f in REQUIRED if not (REPORT_DIR / f).exists()]
    if missing:
        print("These report files are missing. Run the matching scripts first:")
        for m in missing:
            print(f"  - reports\\{m}")
        print("\ntrain_models.py -> model_comparison, scaling_comparison")
        print("confusion_analysis.py -> confusion_matrix_best_model")
        print("feature_importance.py -> feature_importance")
        print("improve_model.py -> improvement_*")
        print("tune_model.py -> tuning_comparison, gridsearch_cv_results")
        print("correlation.py -> correlation_matrix")
        sys.exit(1)

    raw_path = PROJECT_ROOT / "data" / "WineQT.csv"
    clean_path = PROJECT_ROOT / "data" / "WineQT_clean.csv"
    raw_n = len(pd.read_csv(raw_path)) if raw_path.exists() else 1143
    clean = pd.read_csv(clean_path)
    n_clean = len(clean)
    class_counts = clean[TARGET].value_counts().sort_index()
    n_test = round(n_clean * 0.2)

    comp = read("model_comparison.csv", index_col=0)
    scal = read("scaling_comparison.csv", index_col=0)
    cm = read("confusion_matrix_best_model.csv", index_col=0)
    fi = read("feature_importance.csv", index_col=0)
    imp = read("improvement_comparison.csv", index_col=0)
    exp = read("improvement_experiments.csv")
    tun = read("tuning_comparison.csv", index_col=0)
    grid = read("gridsearch_cv_results.csv")

    best, worst = comp.index[0], comp.index[-1]
    best_row = comp.loc[best]

    # Confusion matrix facts
    labels = [int(i.split("_")[1]) for i in cm.index]
    vals = cm.values
    support = vals.sum(axis=1)
    recall = {l: (vals[i, i] / support[i] if support[i] else 0.0)
              for i, l in enumerate(labels)}
    hard = [l for l in labels if recall[l] < 0.30]
    errs = [(labels[i], labels[j], int(vals[i, j]))
            for i in range(len(labels)) for j in range(len(labels))
            if i != j and vals[i, j] > 0]
    errs.sort(key=lambda t: t[2], reverse=True)
    total_err = sum(e[2] for e in errs)
    off1 = sum(e[2] for e in errs if abs(e[0] - e[1]) == 1)
    top_conf = errs[0] if errs else None

    # Feature facts
    top_feat = fi.index[0]
    top_perm = fi["perm_importance_test"].idxmax()
    low_feats = list(fi.index[-3:])
    corr_line = ""
    if "corr_with_quality" in fi.columns:
        top_corr = fi["corr_with_quality"].abs().idxmax()
        corr_line = (f"By Pearson correlation the strongest link with quality is "
                     f"**{top_corr}** (r = {fi.loc[top_corr, 'corr_with_quality']:.2f}).")

    # Scaling
    scal_lines = []
    for m, r in scal.iterrows():
        d = r["test_acc_scaled"] - r["test_acc_unscaled"]
        scal_lines.append(
            f"- **{m}**: test accuracy {pct(r['test_acc_unscaled'])} (unscaled) vs "
            f"{pct(r['test_acc_scaled'])} (scaled), so scaling {verdict(d)}.")

    # Improvement and tuning
    effect = (exp.groupby("parameter")["cv_accuracy"]
              .agg(lambda s: s.max() - s.min()).sort_values(ascending=False))
    top_param = effect.index[0]
    d_imp = imp.loc["Improved", "test_accuracy"] - imp.loc["Original", "test_accuracy"]
    d_tun = tun.loc["Tuned", "test_accuracy"] - tun.loc["Original", "test_accuracy"]
    best_cfg = grid.sort_values("rank_test_accuracy").iloc[0]
    cv_best = best_cfg["mean_test_accuracy"]
    cv_std = best_cfg["std_test_accuracy"]
    best_params = str(best_cfg["params"])

    hard_txt = ", ".join(str(h) for h in hard) if hard else "none below 30% recall"

    report = f"""# Wine Quality Prediction: Final Analysis and Conclusion

Generated on {date.today().isoformat()} from the result files in `reports/`.

## Results summary

### Model comparison (test set, {n_test} samples)

{md_table(comp, "model")}

### Original vs tuned model ({best})

{md_table(tun, "model")}

---

# TASK 17: Final Model Analysis

**1. Most influential feature.** Random Forest importance ranks **{top_feat}** first. Permutation importance on the test set ranks **{top_perm}** first. {corr_line} The three least important features are {", ".join(low_feats)}.

**2. Best algorithm.** **{best}** performed best (weighted F1 = {best_row['f1_weighted']:.3f}, test accuracy = {pct(best_row['test_accuracy'])}). The weakest was **{worst}** (weighted F1 = {comp.loc[worst, 'f1_weighted']:.3f}).

**3. Why the best model won.** Wine quality depends on non-linear effects and interactions between chemical properties, and a tree ensemble can capture both without needing scaled inputs. Models with a simple or distance-based boundary struggle when the quality classes overlap this heavily. Note that the test set is small, so a gap of one or two percentage points between models can be noise.

**4. Did scaling help?**

{chr(10).join(scal_lines)}

Random Forest was not scaled because trees split on thresholds and are not affected by feature ranges.

**5. Did tuning help?** GridSearchCV (5-fold stratified CV, scaler inside the pipeline, test set untouched) chose `{best_params}` with a cross-validation accuracy of {pct(cv_best)} (std {cv_std:.3f}). On the test set, accuracy went from {pct(tun.loc['Original', 'test_accuracy'])} to {pct(tun.loc['Tuned', 'test_accuracy'])}, so the result {verdict(d_tun)}. One test sample equals {100 / n_test:.2f} percentage points, so small changes should not be over-interpreted. In the manual experiments the parameter with the largest effect on CV accuracy was **{top_param}**, and the combined "improved" model {verdict(d_imp)} on the test set. The train accuracy of the original model was {pct(imp.loc['Original', 'train_accuracy'])} against {pct(imp.loc['Original', 'test_accuracy'])} on test, which shows the overfitting gap. Higher training accuracy alone does not mean a better model.

**6. Difficult quality classes.** Classes with under 30% recall for {best}: **{hard_txt}**. Per-class recall: {", ".join(f"{l}: {pct(recall[l])}" for l in labels)}. {f"The most frequent confusion is actual {top_conf[0]} predicted as {top_conf[1]} ({top_conf[2]} samples)." if top_conf else ""} Of {total_err} wrong predictions, {off1} ({pct(off1 / total_err) if total_err else "0%"}) were off by exactly one level, so the model mostly confuses neighbouring scores.

**7. Dataset limitations.**
- Small: {n_clean} rows after cleaning ({raw_n - n_clean} duplicates removed from {raw_n}).
- Strong class imbalance: {", ".join(f"quality {k}: {v}" for k, v in class_counts.items())}. Classes 3 and 8 have too few samples for reliable precision or recall.
- Quality is a subjective human score, so identical wines can receive different labels (label noise).
- Only chemical properties are included. Grape variety, region, brand and aroma are missing.
- Only red wine samples, so results may not transfer to other wine types.
- The test set is small, so metrics have wide uncertainty.

---

# TASK 18: Final Conclusion

**Dataset understanding.** The dataset has {raw_n} wine samples with 11 chemical properties, an `Id` column and a target `quality` (scores 3 to 8). It is treated as multi-class classification.

**Data cleaning.** `Id` was dropped. There were no missing values. {raw_n - n_clean} duplicate rows (identical features and quality) were removed, leaving {n_clean} rows. No negative or physically invalid values were found. Outliers were kept because high sulfur dioxide or sugar values are plausible for wine.

**Exploratory Data Analysis.** Quality is concentrated in classes 5 and 6. Alcohol, volatile acidity and sulphates show clear patterns against quality, and several features contain outliers.

**Feature engineering and scaling.** All 11 features were used. StandardScaler was fitted on training data only, inside a Pipeline, for Logistic Regression, SVM and KNN, to avoid leakage.

**Models used.** Logistic Regression, Random Forest, SVM (RBF) and KNN, on a stratified 80/20 split with random_state=42.

**Model evaluation.** Accuracy, weighted and macro precision, recall and F1, per-class reports and confusion matrices were used, because accuracy alone hides poor results on rare classes.

**Best-performing model.** {best}, with {pct(best_row['test_accuracy'])} test accuracy and weighted F1 of {best_row['f1_weighted']:.3f}.

**Hyperparameter tuning.** Cross-validated GridSearchCV gave a CV accuracy of {pct(cv_best)}. Final test accuracy: {pct(tun.loc['Tuned', 'test_accuracy'])} (original: {pct(tun.loc['Original', 'test_accuracy'])}).

**Wine quality prediction system.** A Streamlit app takes the 11 chemical values and shows the predicted score, the category (3-4 Poor, 5 Average, 6 Good, 7-8 Excellent) and the confidence, which is the model's predicted probability for the chosen class. It warns on low confidence and on out-of-range inputs.

**Important findings.** {top_feat} is the most important feature by model importance. The model is reliable mainly for classes 5 and 6, and errors are mostly between neighbouring scores. Rare classes are hard to predict because of very few samples.

**Real-world applications.** Quick screening of wine batches in a winery, supporting quality control before expert tasting, and pricing or grading support. It should assist human tasters, not replace them.

**What I learned.** *(Write this part in your own words. Suggested points: preventing leakage with Pipelines and a held-out test set, why accuracy can mislead on imbalanced data, why tuning gives small gains on noisy labels, and how to build a Streamlit app around a saved model.)*
"""
    OUT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved: {OUT_PATH}")
    print(f"Best model: {best} | Worst: {worst}")
    print(f"Top feature: {top_feat} | Hard classes: {hard_txt}")


if __name__ == "__main__":
    main()