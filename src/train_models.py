import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_utils import PROJECT_ROOT, RANDOM_STATE, get_train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_models():
    """Return {name: (scaled_pipeline_or_model, unscaled_model_or_None)}."""
    return {
        "Logistic Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]),
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
            None,
        ),
        "SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(probability=True, random_state=RANDOM_STATE)),
            ]),
            SVC(probability=True, random_state=RANDOM_STATE),
        ),
        "KNN": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]),
            KNeighborsClassifier(n_neighbors=5),
        ),
    }


def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    metrics = {
        "train_accuracy": accuracy_score(y_train, pred_train),
        "test_accuracy": accuracy_score(y_test, pred_test),
        "precision_weighted": precision_score(y_test, pred_test, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, pred_test, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, pred_test, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_test, pred_test, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred_test, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, pred_test, average="macro", zero_division=0),
    }
    return metrics, pred_test


def main():
    X_train, X_test, y_train, y_test = get_train_test_split()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    results = []
    unscaled_results = []
    predictions = {}

    for name, (scaled_model, unscaled_model) in build_models().items():
        section(f"MODEL: {name}")
        metrics, pred = evaluate(scaled_model, X_train, X_test, y_train, y_test)
        results.append({"model": name, **metrics})
        predictions[name] = pred

        print(f"Train accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test accuracy:  {metrics['test_accuracy']:.4f}")
        print("\nPer-class report (test set):")
        print(classification_report(y_test, pred, zero_division=0))

        joblib.dump(scaled_model, MODEL_DIR / f"baseline_{name.replace(' ', '_').lower()}.joblib")

        if unscaled_model is not None:
            m_un, _ = evaluate(unscaled_model, X_train, X_test, y_train, y_test)
            unscaled_results.append({
                "model": name,
                "test_acc_unscaled": m_un["test_accuracy"],
                "test_acc_scaled": metrics["test_accuracy"],
                "f1_weighted_unscaled": m_un["f1_weighted"],
                "f1_weighted_scaled": metrics["f1_weighted"],
            })

    # Comparison table
    section("TASK 10: MODEL COMPARISON TABLE (test set)")
    res_df = pd.DataFrame(results).set_index("model").round(4)
    res_df = res_df.sort_values("f1_weighted", ascending=False)
    print(res_df.to_string())
    res_df.to_csv(REPORT_DIR / "model_comparison.csv")

    best = res_df.index[0]
    worst = res_df.index[-1]
    print(f"\nBest model (by weighted F1): {best}")
    print(f"Worst model (by weighted F1): {worst}")

    section("EFFECT OF SCALING (test set)")
    sc_df = pd.DataFrame(unscaled_results).set_index("model").round(4)
    print(sc_df.to_string())
    sc_df.to_csv(REPORT_DIR / "scaling_comparison.csv")

    # Save test predictions for the confusion matrix phase
    pred_df = pd.DataFrame(predictions, index=y_test.index)
    pred_df.insert(0, "actual", y_test)
    pred_df.to_csv(REPORT_DIR / "test_predictions.csv")

    print("\nPhase 7 complete.")


if __name__ == "__main__":
    main()