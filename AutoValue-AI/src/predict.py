"""Load the saved pipeline, predict prices and compare them with asking prices."""
import json

import joblib

from .preprocessing import MODEL_PATH, METRICS_PATH, details_to_frame

DEAL_THRESHOLD_PCT = 5.0   # project-defined, not a market standard


def load_pipeline(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def load_metrics(path=METRICS_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found at {path}. Run `python -m src.train` first.")
    return json.loads(path.read_text())


def predict_price(details, pipeline=None):
    """Estimated resale price in rupees for one car (dict of input fields)."""
    pipeline = pipeline or load_pipeline()
    return float(pipeline.predict(details_to_frame(details))[0])


def compare_deal(predicted, asking, threshold=DEAL_THRESHOLD_PCT):
    """Compare an asking price with the model's estimate."""
    if predicted <= 0:
        raise ValueError("Predicted price must be positive.")
    difference = asking - predicted
    pct = difference / predicted * 100
    if pct < -threshold:
        label = "Potentially below estimate"
    elif pct > threshold:
        label = "Potentially above estimate"
    else:
        label = "Close to estimate"
    return {"predicted": predicted, "asking": asking, "difference": difference,
            "difference_pct": pct, "label": label}


def format_inr(amount):
    """Format a number with Indian digit grouping, e.g. 580000 -> '₹5,80,000'."""
    sign = "-" if amount < 0 else ""
    s = str(int(round(abs(amount))))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    return f"{sign}₹{s}"
