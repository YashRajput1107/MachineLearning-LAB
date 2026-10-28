"""Define the four feature groups used in the experiments.

A: full (all 561)   B: accelerometer   C: gyroscope   D: reduced (SelectKBest)
Groups B and C are built from the documented UCI feature names; D is fitted on
the training data only.
"""
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from .config import ACC_TOKENS, GYRO_TOKENS, SELECTK, RANDOM_SEED


def accelerometer_features(feature_names):
    return [f for f in feature_names if any(t in f for t in ACC_TOKENS)]


def gyroscope_features(feature_names):
    return [f for f in feature_names if any(t in f for t in GYRO_TOKENS)]


def select_reduced(X_train, y_train, k=SELECTK):
    """Fit SelectKBest (mutual information) on TRAIN only; return selected names."""
    k = min(k, X_train.shape[1])
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=RANDOM_SEED),
        k=k,
    )
    selector.fit(X_train, y_train)
    cols = X_train.columns[selector.get_support()].tolist()
    return cols, selector


def get_feature_groups(X_train, y_train, feature_names, k=SELECTK):
    """Return {group_name: list_of_columns} for the four experiments."""
    reduced_cols, _ = select_reduced(X_train, y_train, k)
    return {
        "full": list(feature_names),
        "accelerometer": accelerometer_features(feature_names),
        "gyroscope": gyroscope_features(feature_names),
        "reduced": reduced_cols,
    }
