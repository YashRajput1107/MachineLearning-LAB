"""Validation and leakage-safe model pipelines."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from .config import RANDOM_SEED


def validate(X_train, X_test):
    """Data-quality checks used in the EDA notebook."""
    return {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": X_train.shape[1],
        "columns_match": list(X_train.columns) == list(X_test.columns),
        "missing_train": int(X_train.isnull().sum().sum()),
        "missing_test": int(X_test.isnull().sum().sum()),
        "non_finite_train": int((~np.isfinite(X_train.values)).sum()),
    }


def make_pipeline(model_name):
    """Build a fresh estimator. Scalers live INSIDE the pipeline, so they are
    fitted only on training folds (no leakage). Random Forest needs no scaling.
    """
    if model_name == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ])
    if model_name == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)),
        ])
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
    raise ValueError(model_name)


def param_grid(model_name):
    """Small, sensible hyper-parameter grids (searched on training data only)."""
    if model_name == "knn":
        return {"clf__n_neighbors": [3, 5, 7, 11],
                "clf__weights": ["uniform", "distance"]}
    if model_name == "svm":
        return {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01]}
    if model_name == "random_forest":
        return {"n_estimators": [100, 200], "max_depth": [None, 20],
                "min_samples_leaf": [1, 2]}
    raise ValueError(model_name)
