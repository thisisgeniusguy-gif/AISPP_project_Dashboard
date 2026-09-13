"""
app.py
======
Deliverable 3 - Assignment 2, Part D (Dataset 8: E-commerce Delivery Time Prediction)

Streamlit application that loads the trained preprocessing + regression pipeline from
`model.pkl` and predicts `Delivery_Time_Hours` for a single order.

The app does **no** preprocessing of its own: scaling and one-hot encoding live inside the
saved pipeline, so training and serving cannot drift apart.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Constants that mirror the training notebook's feature contract
# --------------------------------------------------------------------------- #
MODEL_PATH = Path("model.pkl")
META_PATH = Path("model_metadata.pkl")

NUM_FEATURES = ["Order_Value", "Package_Weight_Kg", "Warehouse_Distance_Km",
                "Items_in_Order", "Warehouse_Processing_Hours",
                "Courier_Load_Index", "Traffic_Index"]
CAT_FEATURES = ["Product_Category", "Shipping_Mode"]
FEATURE_ORDER = NUM_FEATURES + CAT_FEATURES

FALLBACK_CATEGORIES = {
    "Product_Category": ["Apparel", "Beauty", "Books", "Electronics",
                         "Groceries", "Home_Kitchen"],
    "Shipping_Mode": ["Express", "Same_Day", "Standard"],
}

# Training-data envelope. Inputs far outside this are extrapolation, not prediction.
TRAINED_RANGES = {
    "Order_Value": (199.0, 90_000.0),
    "Package_Weight_Kg": (0.1, 25.0),
    "Warehouse_Distance_Km": (2.0, 1500.0),
    "Items_in_Order": (1, 15),
    "Warehouse_Processing_Hours": (0.3, 13.0),
    "Courier_Load_Index": (1.0, 10.0),
    "Traffic_Index": (1.0, 10.0),
}

PRETTY = {
    "Order_Value": "Order value",
    "Package_Weight_Kg": "Package weight",
    "Warehouse_Distance_Km": "Warehouse distance",
    "Items_in_Order": "Items in order",
    "Warehouse_Processing_Hours": "Warehouse processing time",
    "Courier_Load_Index": "Courier load",
    "Traffic_Index": "Traffic congestion",
    "Product_Category": "Product category",
    "Shipping_Mode": "Shipping mode",
}

st.set_page_config(page_title="Delivery Time Predictor",
                   page_icon="📦", layout="wide")


# --------------------------------------------------------------------------- #
# Model loading (once per session, not per interaction)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the trained model...")
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    pipeline = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH) if META_PATH.exists() else {}
    return pipeline, meta


pipeline, meta = load_model()

if pipeline is None:
    st.error(
        "**`model.pkl` not found.** Run the training notebook `model_training.ipynb` "
        "(or `python generate_data.py` followed by the notebook) so that the saved "
        "pipeline sits next to `app.py`."
    )
    st.stop()

CATEGORIES = meta.get("categories", FALLBACK_CATEGORIES)
TEST_METRICS = meta.get("train_metrics", {"r2_test": 0.968, "mae_test": 1.23, "rmse_test": 1.52})
MAE = float(TEST_METRICS.get("mae_test", 1.23))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_row(values: dict) -> pd.DataFrame:
    """Single-row frame in exactly the column order the pipeline was fitted on."""
    return pd.DataFrame([values])[FEATURE_ORDER]


def validate(v: dict) -> tuple[list[str], list[str]]:
    """Return (blocking errors, non-blocking warnings)."""
    errors, warnings = [], []

    # Hard physical / logical impossibilities - block the prediction
    if v["Order_Value"] <= 0:
        errors.append("Order value must be greater than zero.")
    if v["Package_Weight_Kg"] <= 0:
        errors.append("Package weight must be greater than zero kg.")
    if v["Warehouse_Distance_Km"] <= 0:
        errors.append("Warehouse distance must be greater than zero km.")
    if v["Items_in_Order"] < 1:
        errors.append("An order must contain at least one item.")
    if v["Warehouse_Processing_Hours"] < 0:
        errors.append("Warehouse processing hours cannot be negative.")
    if not 1 <= v["Courier_Load_Index"] <= 10:
        errors.append("Courier load index must be between 1 and 10.")
    if not 1 <= v["Traffic_Index"] <= 10:
        errors.append("Traffic index must be between 1 and 10.")

    # Business-logic sanity checks
    if v["Items_in_Order"] > 1 and v["Package_Weight_Kg"] < 0.05 * v["Items_in_Order"]:
        warnings.append(
            f"{v['Items_in_Order']} items weighing only {v['Package_Weight_Kg']:.2f} kg "
            "in total looks like a data-entry error - check the weight."
        )
    if v["Shipping_Mode"] == "Same_Day" and v["Warehouse_Distance_Km"] > 100:
        warnings.append(
            f"Same-day service on a {v['Warehouse_Distance_Km']:.0f} km lane was never "
            "observed in training (same-day lanes were under ~60 km). The prediction is an "
            "extrapolation - treat it as indicative only."
        )
    if v["Shipping_Mode"] == "Express" and v["Warehouse_Distance_Km"] > 900:
        warnings.append(
            "Express service beyond ~900 km is outside the trained lane mix - "
            "verify the service is actually available on this route."
        )

    # Outside the trained envelope
    for f, (lo, hi) in TRAINED_RANGES.items():
        if not lo <= v[f] <= hi:
            warnings.append(
                f"{PRETTY[f]} ({v[f]:,.2f}) is outside the trained range "
                f"{lo:,.0f}-{hi:,.0f}; accuracy degrades outside this envelope."
            )
    return errors, warnings


def decompose(row: pd.DataFrame) -> pd.DataFrame:
    """
    Hour-by-hour attribution of the prediction.

    The pipeline standardises numerics and one-hot encodes categories, so
    `transformed_value * coefficient` is the contribution of that feature relative to an
    average order in the reference category. Contributions plus the intercept reproduce
    the prediction exactly.
    """
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    x_t = np.asarray(pre.transform(row)).ravel()
    names = pre.get_feature_names_out()
    contrib = x_t * model.coef_

    out = []
    for name, c in zip(names, contrib):
        block, raw = name.split("__", 1)
        if block == "num":
            label = PRETTY.get(raw, raw)
        else:
            feature = next((f for f in CAT_FEATURES if raw.startswith(f + "_")), raw)
            level = raw[len(feature) + 1:] if feature != raw else raw
            label = f"{PRETTY.get(feature, feature)}: {level.replace('_', ' ')}"
        if abs(c) > 1e-9:
            out.append({"Driver": label, "Impact (hours)": float(c)})

    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.reindex(df["Impact (hours)"].abs().sort_values(ascending=False).index)


def counterfactual(values: dict, **overrides) -> float:
    v = dict(values)
    v.update(overrides)
    return float(pipeline.predict(build_row(v))[0])


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("📦 E-commerce Delivery Time Predictor")
st.markdown(
    "Estimates **door-step delivery time in hours** for an order at the moment of dispatch, "
    "using route, warehouse, parcel and service-level attributes. Built on a multiple linear "
    "regression pipeline trained on ~1,190 historical orders."
)
c1, c2, c3 = st.columns(3)
c1.metric("Test R²", f"{TEST_METRICS['r2_test']:.3f}")
c2.metric("Mean absolute error", f"{MAE:.2f} h")
c3.metric("Test RMSE", f"{TEST_METRICS['rmse_test']:.2f} h")
st.divider()

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
st.sidebar.header("Order inputs")

st.sidebar.subheader("Service and product")
shipping_mode = st.sidebar.selectbox(
    "Shipping mode", CATEGORIES["Shipping_Mode"],
    index=CATEGORIES["Shipping_Mode"].index("Standard")
    if "Standard" in CATEGORIES["Shipping_Mode"] else 0,
    help="Service level sold to the customer.")
product_category = st.sidebar.selectbox(
    "Product category", CATEGORIES["Product_Category"],
    help="Handling profile differs by category (fragile, serialised, cold-chain).")

st.sidebar.subheader("Order and parcel")
order_value = st.sidebar.number_input(
    "Order value (₹)", min_value=1.0, max_value=500_000.0, value=2_500.0, step=100.0,
    help="Basket value. Has almost no effect on delivery speed - kept for completeness.")
items_in_order = st.sidebar.number_input(
    "Items in order", min_value=1, max_value=50, value=3, step=1)
package_weight = st.sidebar.number_input(
    "Package weight (kg)", min_value=0.01, max_value=100.0, value=2.50, step=0.25)

st.sidebar.subheader("Route and network")
distance_km = st.sidebar.number_input(
    "Warehouse distance (km)", min_value=0.5, max_value=2_000.0, value=250.0, step=10.0,
    help="Road distance from the fulfilling warehouse to the delivery pincode.")
processing_hours = st.sidebar.slider(
    "Warehouse processing time (hours)", 0.0, 24.0, 4.0, 0.25,
    help="Dwell time from order drop to manifest/handover.")
courier_load = st.sidebar.slider(
    "Courier load index (1-10)", 1.0, 10.0, 6.0, 0.1,
    help="Courier capacity saturation on the route. 1 = idle, 10 = fully saturated.")
traffic_index = st.sidebar.slider(
    "Traffic index (1-10)", 1.0, 10.0, 6.5, 0.1,
    help="Expected road congestion on the lane. 1 = clear, 10 = severe.")

values = {
    "Order_Value": float(order_value),
    "Package_Weight_Kg": float(package_weight),
    "Warehouse_Distance_Km": float(distance_km),
    "Items_in_Order": int(items_in_order),
    "Warehouse_Processing_Hours": float(processing_hours),
    "Courier_Load_Index": float(courier_load),
    "Traffic_Index": float(traffic_index),
    "Product_Category": product_category,
    "Shipping_Mode": shipping_mode,
}

st.sidebar.divider()
predict_clicked = st.sidebar.button("Predict delivery time", type="primary",
                                    use_container_width=True)

# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
errors, warns = validate(values)

if errors:
    st.subheader("Input validation failed")
    for e in errors:
        st.error(e)
    st.info("Correct the highlighted inputs in the sidebar; no prediction is produced from "
            "impossible values.")
    st.stop()

if not predict_clicked:
    st.info("Set the order attributes in the sidebar, then select **Predict delivery time**.")
    st.stop()

for w in warns:
    st.warning(w)

row = build_row(values)
prediction = float(pipeline.predict(row)[0])
prediction = max(prediction, 0.5)  # a delivery cannot take negative time

# Uncertainty widens with journey length (documented mild heteroscedasticity in Part A)
band = MAE * (1.0 + 0.6 * min(values["Warehouse_Distance_Km"] / 1500.0, 1.0))
low, high = max(prediction - 1.96 * band, 0.5), prediction + 1.96 * band
eta = dt.datetime.now() + dt.timedelta(hours=prediction)

st.subheader("Predicted delivery time")
m1, m2, m3 = st.columns([1.1, 1, 1])
m1.metric("Expected delivery", f"{prediction:.1f} Hours",
          help="Conditional mean predicted by the regression model.")
m2.metric("Promise window", f"{low:.1f} - {high:.1f} Hours",
          help="Approximate 95% window; it widens on longer lanes.")
m3.metric("Equivalent", f"{prediction / 24:.1f} days")
st.caption(f"If dispatched now, expected arrival ≈ **{eta:%a %d %b, %H:%M}**. "
           f"Recommended customer-facing promise: **{np.ceil(high):.0f} hours** "
           "(the upper bound, not the point estimate).")

st.divider()

# --------------------------------------------------------------------------- #
# Managerial insight
# --------------------------------------------------------------------------- #
st.subheader("What is driving this prediction?")
st.caption(
    "Each bar is hours added or removed **relative to an average order** shipped Express in "
    "the reference category (Apparel). The bars plus the model baseline equal the prediction."
)

contrib = decompose(row)
left, right = st.columns([1.25, 1])

with left:
    if not contrib.empty:
        top = contrib.head(9).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6.2, 3.9))
        colors = ["#c0392b" if v > 0 else "#1a7a4c" for v in top["Impact (hours)"]]
        ax.barh(top["Driver"], top["Impact (hours)"], color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Hours added (+) / saved (-)")
        ax.set_title("Driver contribution to this order")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

with right:
    st.dataframe(
        contrib.assign(**{"Impact (hours)": contrib["Impact (hours)"].round(2)}),
        hide_index=True, height=330,
    )

# ---- Narrative interpretation ------------------------------------------------
pushers = contrib[contrib["Impact (hours)"] > 0].head(3)
savers = contrib[contrib["Impact (hours)"] < 0].head(2)

bullets = []
for _, r in pushers.iterrows():
    bullets.append(f"**{r['Driver']}** is adding **{r['Impact (hours)']:+.1f} h**.")
for _, r in savers.iterrows():
    bullets.append(f"**{r['Driver']}** is saving **{abs(r['Impact (hours)']):.1f} h**.")

st.markdown("#### Read for the operations manager")
st.markdown("\n".join(f"- {b}" for b in bullets))

st.markdown(
    f"""
Structural drivers versus controllable drivers, for this specific order:

- **Distance ({values['Warehouse_Distance_Km']:,.0f} km)** is a *network-design* lever, not an
  execution one. At roughly **+1.5 hours per 100 km**, fulfilling this order from a node
  100 km closer would save about **1.5 hours** - the business case for forward stocking.
- **Warehouse processing ({values['Warehouse_Processing_Hours']:.2f} h)** passes through
  **hour for hour**: every hour saved inside the warehouse is an hour off the customer
  promise, and this is fully inside the team's control today.
- **Traffic ({values['Traffic_Index']:.1f}/10) and courier load
  ({values['Courier_Load_Index']:.1f}/10)** are conditions to *plan around*, worth roughly
  **+0.53 h** and **+0.34 h per index point**. They belong in the promise shown at checkout,
  not in a post-hoc apology.
- **Order value (₹{values['Order_Value']:,.0f})** has effectively **no** effect on delivery
  speed once weight, item count and category are accounted for - so there is no
  delivery-speed reason to treat high-value baskets differently.
"""
)

# ---- Actionable what-if scenarios -------------------------------------------
st.markdown("#### What-if levers")
scenarios = []

for mode in CATEGORIES["Shipping_Mode"]:
    if mode != values["Shipping_Mode"]:
        delta = counterfactual(values, Shipping_Mode=mode) - prediction
        scenarios.append((f"Switch shipping mode to {mode.replace('_', ' ')}", delta))

if values["Warehouse_Processing_Hours"] >= 1:
    scenarios.append(("Cut warehouse dwell time by 1 hour",
                      counterfactual(values, Warehouse_Processing_Hours=values[
                          "Warehouse_Processing_Hours"] - 1) - prediction))
if values["Warehouse_Distance_Km"] > 100:
    scenarios.append(("Fulfil from a warehouse 100 km closer",
                      counterfactual(values, Warehouse_Distance_Km=values[
                          "Warehouse_Distance_Km"] - 100) - prediction))
if values["Courier_Load_Index"] > 2:
    scenarios.append(("Relieve courier load by 2 index points",
                      counterfactual(values, Courier_Load_Index=values[
                          "Courier_Load_Index"] - 2) - prediction))

scen_df = (pd.DataFrame(scenarios, columns=["Lever", "Change in delivery time (hours)"])
           .sort_values("Change in delivery time (hours)"))
scen_df["Change in delivery time (hours)"] = scen_df[
    "Change in delivery time (hours)"].round(2)
st.dataframe(scen_df, hide_index=True)
st.caption("Negative values shorten the delivery; each lever is evaluated holding every "
           "other attribute of this order constant.")

# --------------------------------------------------------------------------- #
with st.expander("Model details, assumptions and limitations"):
    st.markdown(
        f"""
**Model.** scikit-learn `Pipeline`: `ColumnTransformer`
(`StandardScaler` on 7 numeric predictors, `OneHotEncoder(drop='first')` on
`Product_Category` and `Shipping_Mode`) followed by `LinearRegression`. Trained on an 80/20
split of ~1,190 cleaned orders; loaded here from `model.pkl` - the app never re-trains and
never preprocesses independently of the pipeline.

**Accuracy.** Test R² **{TEST_METRICS['r2_test']:.3f}**, MAE **{MAE:.2f} h**,
RMSE **{TEST_METRICS['rmse_test']:.2f} h**; ~82% of held-out orders landed within ±2 h.

**Reference levels.** Coefficients for categories are read against `Product_Category = Apparel`
and `Shipping_Mode = Express`.

**Limitations to communicate before anyone acts on a number.**
1. Predicts **normal-course** delivery time. Six exception cases (customs holds, failed
   first attempts) were removed during training, so the model will *under*-predict an order
   that gets stuck. Exception risk needs its own model.
2. Errors grow with journey length (mild heteroscedasticity, confirmed by Breusch-Pagan), so
   the promise window above widens with distance instead of using one fixed tolerance.
3. Same-day service beyond ~60 km and express beyond ~900 km were not represented in training;
   such inputs are flagged as extrapolation.
4. Coefficients are conditional associations from observational data, not proven causal
   effects. The distance and dwell-time levers are strong enough to act on; treat the smaller
   category effects as directional.
"""
    )
