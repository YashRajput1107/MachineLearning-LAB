"""ActivitySense AI - Streamlit dashboard.

Run from the project root:  streamlit run app/dashboard.py
Displays trained-model predictions and results - it does not retrain anything.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
import streamlit as st

from src import config as C
from src.predict import validate_feature_csv
from components import (get_models, get_test_data, get_metrics,
                        predict_and_report, add_history, show_scores)

st.set_page_config(page_title="ActivitySense AI", layout="wide")

models, meta = get_models()
Xte, yte_name, subj = get_test_data()
metrics = get_metrics()
best_name = meta["best_model"]
full_cols = meta["feature_cols_full"]

st.sidebar.title("ActivitySense AI")
st.sidebar.caption("UCI HAR | Yash Rajput - 2427010652")
page = st.sidebar.radio("Go to", ["Home", "Predict Activity", "Model Comparison",
                                  "Feature Analysis", "Prediction History"])

# ===========================================================================
if page == "Home":
    st.title("ActivitySense AI - Human Activity Recognition")
    st.write("""
    Recognises six human activities from smartphone accelerometer and gyroscope features,
    trained on the **UCI HAR** dataset (30 volunteers, 561 engineered features).
    """)
    c1, c2, c3 = st.columns(3)
    c1.metric("Train / test samples", f"{7352} / {2947}")
    c2.metric("Features", len(full_cols))
    c3.metric(f"Best model ({best_name})", f"{meta['test_accuracy']*100:.1f}% acc")
    st.subheader("Supported activities")
    st.write("  |  ".join(meta["labels"]))
    st.subheader("How prediction works")
    st.markdown("""
    1. Choose an input on **Predict Activity**: a held-out test example, a feature CSV, or manual entry.
    2. The chosen model returns the predicted activity and model-estimated class scores.
    3. For test examples the true label is shown too, so you can see if the prediction is correct.
    """)
    st.info("Scores are model estimates, not calibrated probabilities.")

# ===========================================================================
elif page == "Predict Activity":
    st.title("Predict Activity")
    mode = st.radio("Input method", ["Select a test example", "Upload a feature CSV",
                                     "Manual entry (demo)"], horizontal=True)

    if mode == "Select a test example":
        idx = st.selectbox("Test example index", list(range(len(Xte))), index=0)
        model_name = st.selectbox("Model", C.MODEL_NAMES,
                                  index=C.MODEL_NAMES.index(best_name))
        if st.button("Predict Activity", type="primary"):
            row = Xte[full_cols].iloc[[idx]]
            actual = yte_name.iloc[idx]
            label, scores = predict_and_report(models[model_name], row, actual=actual)
            add_history({"input": f"Test example {idx}", "predicted": label,
                         "actual": actual, "correct": label == actual})

    elif mode == "Upload a feature CSV":
        st.caption("Upload a CSV whose columns are the 561 pre-engineered features "
                   "(same as the dataset). Raw sensor time-series are not supported.")
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            df = pd.read_csv(up)
            ok, msg, prepared = validate_feature_csv(df, full_cols)
            (st.success if ok else st.error)(msg)
            if ok:
                preds = models[best_name].predict(prepared)
                out = pd.DataFrame({"row": range(len(preds)), "predicted_activity": preds})
                st.dataframe(out, use_container_width=True)
                st.download_button("Download predictions",
                                   out.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv")
                if len(preds) == 1:
                    _, scores = predict_and_report(models[best_name], prepared.iloc[[0]])

    else:  # manual entry
        st.caption("Enter a few interpretable features. This uses a **separate model** trained "
                   "only on these features - a demonstration, not the main model.")
        cols = st.columns(2)
        vals = {}
        for i, f in enumerate(meta["manual_features"]):
            lo, mid, hi = meta["manual_ranges"][f]
            vals[f] = cols[i % 2].slider(f, float(lo), float(hi), float(mid))
        if st.button("Predict Activity", type="primary"):
            row = pd.DataFrame([vals])
            label, scores = predict_and_report(models["manual"], row)
            add_history({"input": "Manual entry", "predicted": label,
                         "actual": None, "correct": None})

# ===========================================================================
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.subheader("All models x feature groups")
    st.dataframe(metrics.round(3), use_container_width=True)
    c1, c2 = st.columns(2)
    p = C.FIGURES_DIR / "model_comparison.png"
    if p.exists(): c1.image(str(p))
    p = C.FIGURES_DIR / "confusion_matrix.png"
    if p.exists(): c2.image(str(p))
    st.subheader("Per-model detail (full features)")
    m = st.selectbox("Model", C.MODEL_NAMES)
    row = metrics[(metrics.model == m) & (metrics.feature_group == "full")].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{row['accuracy']:.3f}")
    b.metric("Macro-F1", f"{row['macro_f1']:.3f}")
    c.metric("Train time", f"{row['train_time_s']}s")
    d.metric("Predict time", f"{row['predict_time_s']}s")

# ===========================================================================
elif page == "Feature Analysis":
    st.title("Feature Analysis")
    st.subheader("Accuracy by feature group")
    p = C.FIGURES_DIR / "feature_group_comparison.png"
    if p.exists(): st.image(str(p))
    st.write("Feature-group sizes:",
             {g: len(meta["feature_groups"][g]) for g in C.FEATURE_GROUPS})
    st.markdown("""
    - **Accelerometer** features alone are far more informative than **gyroscope** features alone.
    - A **reduced** 50-feature subset keeps most of the full-feature accuracy.
    """)
    st.subheader("Random Forest feature importance")
    p = C.FIGURES_DIR / "feature_importance.png"
    if p.exists(): st.image(str(p))

# ===========================================================================
elif page == "Prediction History":
    st.title("Prediction History (this session)")
    hist = st.session_state.get("history", [])
    if not hist:
        st.info("No predictions yet - make some on the Predict Activity page.")
    else:
        st.dataframe(pd.DataFrame(hist), use_container_width=True)
        correct = [h for h in hist if h["correct"] is True]
        known = [h for h in hist if h["correct"] is not None]
        if known:
            st.metric("Session accuracy on known examples",
                      f"{len(correct)}/{len(known)}")
