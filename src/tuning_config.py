from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_utils import RANDOM_STATE

CV_FOLDS = 5


def get_cv():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def make_pipeline(name):
    """Baseline settings are the same as in train_models.py."""
    if name == "Random Forest":
        return Pipeline([
            ("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ])
    if name == "Logistic Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ])
    if name == "SVM":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=RANDOM_STATE)),
        ])
    if name == "KNN":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ])
    raise ValueError(f"Unknown model: {name}")


# Task 13: one-parameter-at-a-time experiments (first value = baseline value)
EXPERIMENTS = {
    "Random Forest": {
        "model__n_estimators": [300, 100, 500, 800],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.5],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    },
    "SVM": {
        "model__C": [1, 0.1, 10, 100],
        "model__gamma": ["scale", 0.01, 0.1, 1],
        "model__class_weight": [None, "balanced"],
    },
    "KNN": {
        "model__n_neighbors": [5, 3, 7, 9, 11, 15, 21],
        "model__weights": ["uniform", "distance"],
        "model__p": [2, 1],
    },
    "Logistic Regression": {
        "model__C": [1, 0.01, 0.1, 10, 100],
        "model__class_weight": [None, "balanced"],
    },
}

# Task 14: GridSearchCV grids
GRIDS = {
    "Random Forest": {
        "model__n_estimators": [200, 500],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
        "model__class_weight": [None, "balanced"],
    },
    "SVM": {
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", 0.01, 0.1, 1],
        "model__class_weight": [None, "balanced"],
    },
    "KNN": {
        "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__class_weight": [None, "balanced"],
    },
}