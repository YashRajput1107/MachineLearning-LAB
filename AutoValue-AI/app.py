"""AutoValue AI - Streamlit app.

Run from the project folder:  streamlit run app.py
The model is trained beforehand (python -m src.train); the app only loads it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find src/ from any working dir

import pandas as pd
import streamlit as st

from src.predict import compare_deal, format_inr, load_metrics, load_pipeline, predict_price, DEAL_THRESHOLD_PCT
from src.preprocessing import FIGURES_DIR, MODELS_DIR, validate_input

st.set_page_config(page_title="AutoValue AI", page_icon="🚗", layout="wide")


@st.cache_resource
def get_model():
    return load_pipeline()


@st.cache_data
def get_metrics():
    return load_metrics()


st.title("🚗 AutoValue AI")
st.caption("Used car price predictor & deal checker  |  Yash Rajput - 2427010652")

try:
    pipeline, metrics = get_model(), get_metrics()
except FileNotFoundError as exc:
    st.error(f"{exc}\n\nTrain the model first:\n\n```\npython -m src.download_data\npython -m src.train\n```")
    st.stop()

meta = metrics["app"]
test = metrics["test_metrics"]

tab_est, tab_deal, tab_info = st.tabs(["Price Estimator", "Deal Checker", "Model Insights"])


def pick(options, preferred):
    """Index of a sensible default option (falls back to the first one)."""
    return options.index(preferred) if preferred in options else 0

# ===========================================================================
with tab_est:
    st.subheader("Enter the car's details")
    c1, c2 = st.columns(2)
    brands = list(meta["brand_models"].keys())
    cats = meta["categories"]
    brand = c1.selectbox("Brand", brands, index=pick(brands, "Maruti"))
    models = meta["brand_models"][brand]
    model = c2.selectbox("Model", models, index=pick(models, "Maruti Swift"))
    year = c1.number_input("Manufacturing year", min_value=1950, max_value=2030, value=2015, step=1,
                           help=f"Supported range: {meta['year_min']}-{meta['year_max']} (training data)")
    km = c2.number_input("Kilometres driven", value=50000, step=1000)
    fuel = c1.selectbox("Fuel type", cats["fuel"], index=pick(cats["fuel"], "Petrol"))
    seller = c2.selectbox("Seller type", cats["seller_type"],
                          index=pick(cats["seller_type"], "Individual"))
    trans = c1.selectbox("Transmission", cats["transmission"],
                         index=pick(cats["transmission"], "Manual"))
    owner = c2.selectbox("Previous ownership", cats["owner"],
                         index=pick(cats["owner"], "First Owner"))

    if st.button("Predict Car Price", type="primary"):
        details = {"brand": brand, "model": model, "year": year, "km_driven": km, "fuel": fuel,
                   "seller_type": seller, "transmission": trans, "owner": owner}
        errors = validate_input(details, meta)
        if errors:
            for e in errors:
                st.error(e)
            st.session_state.pop("estimate", None)
        else:
            price = round(predict_price(details, pipeline))   # whole rupees
            st.session_state["estimate"] = price
            st.session_state["car"] = f"{year} {model} ({fuel}, {trans}, {km:,.0f} km)"

    if "estimate" in st.session_state:
        price = st.session_state["estimate"]
        st.markdown("#### Estimated resale price")
        st.markdown(f"# {format_inr(price)}")
        st.caption(st.session_state["car"])
        n = meta["model_counts"].get(model, 0)
        if n < 10:
            st.warning(f"Only {n} training example(s) for {model} - this estimate is less reliable.")
        st.info(
            f"**About accuracy:** on held-out test cars the model's average absolute error (MAE) was "
            f"**{format_inr(test['mae'])}**. Illustrative only - not a statistical prediction "
            f"interval: {format_inr(max(price - test['mae'], 0))} to {format_inr(price + test['mae'])}."
        )
        st.caption(f"Based on historical listings ({meta['year_min']}-{meta['year_max']}); "
                   "not a guaranteed current market value.")

# ===========================================================================
with tab_deal:
    st.subheader("Compare with the seller's asking price")
    if "estimate" not in st.session_state:
        st.info("First get an estimate on the **Price Estimator** tab.")
    else:
        est = st.session_state["estimate"]
        st.write(f"Car: **{st.session_state['car']}**  -  model estimate **{format_inr(est)}**")
        asking = st.number_input("Seller's asking price (₹)", value=int(est), step=10000)
        if asking <= 0:
            st.error("Asking price must be greater than zero.")
        else:
            deal = compare_deal(est, asking)
            a, b, c, d = st.columns(4)
            a.metric("Estimated price", format_inr(deal["predicted"]))
            b.metric("Asking price", format_inr(deal["asking"]))
            c.metric("Difference", format_inr(abs(deal["difference"])))
            d.metric("Difference vs estimate", f"{deal['difference_pct']:+.1f}%")
            msg = {"Potentially below estimate": st.success,
                   "Close to estimate": st.info,
                   "Potentially above estimate": st.warning}[deal["label"]]
            direction = "higher" if deal["difference"] > 0 else "lower"
            msg(f"**{deal['label']}** - the asking price is {abs(deal['difference_pct']):.1f}% "
                f"{direction} than the model's estimate." if deal["difference"] != 0
                else f"**{deal['label']}** - the asking price equals the model's estimate.")
            st.caption(f"Thresholds (±{DEAL_THRESHOLD_PCT:.0f}%) are a project design choice, not a "
                       "market standard. The model does not know the car's mechanical condition, "
                       "accident or service history, location, or post-2020 market changes - this is "
                       "not a buy/don't-buy recommendation.")

# ===========================================================================
with tab_info:
    st.subheader(f"Selected model: {metrics['best_model_label']}")
    st.caption(f"Chosen by {metrics['selection']}; evaluated once on {metrics['test_rows']} "
               "held-out test cars.")
    a, b, c = st.columns(3)
    a.metric("Test MAE", format_inr(test["mae"]))
    b.metric("Test RMSE", format_inr(test["rmse"]))
    c.metric("Test R²", f"{test['r2']:.3f}")

    st.markdown("**Model comparison**")
    comp = pd.read_csv(MODELS_DIR / "model_comparison.csv")
    show = pd.DataFrame({
        "Model": comp["model"].str.replace("_", " ").str.title(),
        "CV MAE (₹)": comp["cv_mae"].map(lambda v: f"{v:,.0f}"),
        "CV R²": comp["cv_r2"].round(3),
        "Test MAE (₹)": comp["test_mae"].map(lambda v: f"{v:,.0f}"),
        "Test RMSE (₹)": comp["test_rmse"].map(lambda v: f"{v:,.0f}"),
        "Test R²": comp["test_r2"].round(3),
    })
    st.dataframe(show, hide_index=True, use_container_width=True)

    g = metrics["grouped_check"]
    st.markdown("**Robustness check - car models unseen during training**")
    st.write(f"Grouped CV by car model: MAE {format_inr(g['mae'])}, R² {g['r2']:.3f} "
             f"(vs random CV MAE {format_inr(comp['cv_mae'].iloc[0])}). "
             "Estimates for rare or unfamiliar models are much less reliable.")

    l, r = st.columns(2)
    for col, (title, fname) in zip([l, r, l, r], [
            ("Model comparison (CV MAE)", "model_comparison_mae.png"),
            ("Actual vs predicted (test set)", "actual_vs_predicted.png"),
            ("Residuals", "residuals.png"),
            ("Permutation importance", "permutation_importance.png")]):
        p = FIGURES_DIR / fname
        if p.exists():
            col.markdown(f"**{title}**")
            col.image(str(p), use_container_width=True)

    st.markdown("""
**Dataset:** CarDekho *Car details v3* (Kaggle, nehalbirla) - historical listings, 1983-2020.
7 input fields are used; `name` is split into brand and model, giving 8 model features.

**Limitations:** historical data (no post-2020 cars or prices); condition, accident history, service
records and location are not modelled; rare car models have few examples. Treat estimates as
educational, not as a market valuation.
""")
