"""Shared helpers for the Streamlit app (caching, scoring, small UI pieces)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
import streamlit as st

from src import config as C
from src.data_loader import load_test
from src.predict import load_model, load_metadata, predict_one


@st.cache_resource
def get_models():
    models = {n: load_model(n) for n in C.MODEL_NAMES}
    models["manual"] = joblib.load(C.MODELS_DIR / "manual_input_model.joblib")
    return models, load_metadata()


@st.cache_data
def get_test_data():
    Xte, yte, yte_name, subj = load_test()
    return Xte, yte_name, subj


@st.cache_data
def get_metrics():
    return pd.read_csv(C.RESULTS_DIR / "metrics.csv")


def show_scores(scores, top=6):
    """Render class probabilities as a sorted bar chart + table."""
    s = pd.Series(scores).sort_values(ascending=False).head(top)
    st.bar_chart(s)
    st.dataframe((s * 100).round(1).rename("confidence %"), use_container_width=True)


def add_history(entry):
    st.session_state.setdefault("history", []).append(entry)


def predict_and_report(model, X_row, actual=None):
    label, scores = predict_one(model, X_row)
    c1, c2 = st.columns(2)
    c1.metric("Predicted activity", label)
    if actual is not None:
        c2.metric("Actual activity", actual,
                  delta="correct" if actual == label else "wrong",
                  delta_color="normal" if actual == label else "inverse")
    st.caption("Model-estimated class scores (not calibrated probabilities):")
    show_scores(scores)
    return label, scores
