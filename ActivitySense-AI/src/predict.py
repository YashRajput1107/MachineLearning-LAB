"""Load trained models and make predictions (used by the Streamlit app)."""
import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR


def load_model(name):
    return joblib.load(MODELS_DIR / f"{name}.joblib")


def load_metadata():
    return joblib.load(MODELS_DIR / "metadata.joblib")


def predict_one(model, X_row):
    """X_row: a 1-row DataFrame with the model's expected columns (in any order
    that includes them). Returns (predicted_label, {class: probability}).
    """
    classes = list(model.classes_)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_row)[0]
        scores = {c: float(p) for c, p in zip(classes, proba)}
    else:  # fall back to decision scores, normalised for display only
        dec = np.atleast_2d(model.decision_function(X_row))[0]
        dec = dec - dec.min()
        scores = {c: float(v) for c, v in zip(classes, dec / dec.sum())}
    label = model.predict(X_row)[0]
    scores = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
    return label, scores


def validate_feature_csv(df, expected_cols):
    """Validate an uploaded pre-engineered-feature CSV.

    Returns (ok: bool, message: str, prepared_df_or_None).
    """
    missing = [c for c in expected_cols if c not in df.columns]
    if len(missing) == len(expected_cols):
        return (False,
                "None of the required feature columns were found. This looks like raw "
                "sensor data or an unrelated file. Upload a CSV with the dataset's "
                "pre-engineered feature columns.", None)
    if missing:
        return (False, f"Missing {len(missing)} required columns "
                       f"(e.g. {missing[:3]}).", None)
    sub = df[expected_cols]
    if not np.isfinite(sub.select_dtypes("number").values).all() \
            or sub.shape[1] != len(expected_cols):
        return (False, "Some values are missing or non-numeric.", None)
    return (True, f"OK - {len(df)} row(s) validated.", sub.reset_index(drop=True))
