"""
generate_data.py
================
Deliverable 1 - Assignment 2 (Dataset 8: E-commerce Delivery Time Prediction)

Generates a realistic synthetic dataset (`data.csv`) for predicting
`Delivery_Time_Hours` from order, package, warehouse and route features.

Design notes
------------
1.  Column names follow the Dataset Specification document exactly
    (snake_case, e.g. `Order_ID`, `Order_Value`, `Warehouse_Distance_Km`).
2.  The target is built from an explicit, *linear* structural equation so that
    multiple linear regression is a genuinely appropriate model, plus Gaussian
    noise whose scale grows mildly with the expected duration (this creates
    realistic, detectable-but-mild heteroscedasticity).
3.  `Package_Weight_Kg`, `Items_in_Order` and `Order_Value` are deliberately
    correlated (bigger baskets are heavier and cost more) so that the
    multicollinearity / VIF diagnostics in Part A have something real to find.
4.  A small, configurable amount of realistic "dirt" is injected on purpose
    (missing values, exact duplicates, impossible values, inconsistent category
    spellings, a few operational outliers) so that the data-quality section of
    Part A is a real exercise rather than a formality. Set INJECT_DIRT = False
    for a perfectly clean file.

Run:
    python generate_data.py
Output:
    data.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
N_ROWS = 1_200          # clean rows before duplicate injection (>= 1,000 required)
INJECT_DIRT = True      # intentional data-quality issues for Part A
OUTPUT_PATH = "data.csv"

rng = np.random.default_rng(RANDOM_SEED)

# Product categories: probability, weight profile (kg), value profile (INR)
CATEGORIES = {
    "Electronics":   dict(p=0.20, w_mu=0.9,  w_sd=0.7, v_mu=9.2, v_sd=0.7, t_eff=+1.10),
    "Apparel":       dict(p=0.24, w_mu=-0.4, w_sd=0.5, v_mu=7.4, v_sd=0.6, t_eff=-0.30),
    "Home_Kitchen":  dict(p=0.18, w_mu=1.3,  w_sd=0.8, v_mu=8.0, v_sd=0.7, t_eff=+0.80),
    "Groceries":     dict(p=0.16, w_mu=1.5,  w_sd=0.5, v_mu=6.9, v_sd=0.5, t_eff=-1.50),
    "Books":         dict(p=0.12, w_mu=0.2,  w_sd=0.4, v_mu=6.4, v_sd=0.5, t_eff=-0.60),
    "Beauty":        dict(p=0.10, w_mu=-0.7, w_sd=0.5, v_mu=7.1, v_sd=0.6, t_eff=-0.20),
}

# Shipping mode service-level effect on delivery duration (hours)
MODE_EFFECT = {"Standard": 5.0, "Express": 0.0, "Same_Day": -2.5}

# Structural coefficients for the target (hours)
BETA = dict(
    intercept=2.20,
    distance_km=0.0150,       # ~0.90 h per additional 60 km
    processing_hours=1.00,    # warehouse dwell time passes straight through
    traffic_index=0.55,       # per point on a 1-10 congestion scale
    courier_load_index=0.38,  # per point on a 1-10 courier saturation scale
    items_in_order=0.18,      # extra pick/pack + handling per line item
    package_weight_kg=0.045,  # heavier parcels are slower to handle
    order_value=0.000020,     # near-zero: high-value orders are not inherently slow
)

NOISE_BASE_SD = 0.80          # irreducible operational variability (hours)
NOISE_SCALE_SD = 0.035        # mild variance growth with expected duration


# --------------------------------------------------------------------------- #
# Feature generation
# --------------------------------------------------------------------------- #
def sample_shipping_mode(distance_km: np.ndarray) -> np.ndarray:
    """Service level is offered conditional on distance (realistic constraint)."""
    modes = np.empty(distance_km.shape, dtype=object)
    for i, d in enumerate(distance_km):
        if d < 60:
            modes[i] = rng.choice(["Standard", "Express", "Same_Day"], p=[0.40, 0.30, 0.30])
        elif d < 400:
            modes[i] = rng.choice(["Standard", "Express"], p=[0.62, 0.38])
        elif d < 900:
            modes[i] = rng.choice(["Standard", "Express"], p=[0.78, 0.22])
        else:
            modes[i] = "Standard"
    return modes


def generate_clean_frame(n: int) -> pd.DataFrame:
    cat_names = list(CATEGORIES)
    cat_probs = np.array([CATEGORIES[c]["p"] for c in cat_names])
    cat_probs = cat_probs / cat_probs.sum()
    category = rng.choice(cat_names, size=n, p=cat_probs)

    # --- Order size: a latent "basket size" drives items, weight and value ---
    basket = rng.gamma(shape=2.0, scale=1.0, size=n)                 # latent size
    items = np.clip(np.round(1 + basket * 1.4 + rng.normal(0, 0.6, n)), 1, 15).astype(int)

    w_mu = np.array([CATEGORIES[c]["w_mu"] for c in category])
    w_sd = np.array([CATEGORIES[c]["w_sd"] for c in category])
    weight = np.exp(w_mu + w_sd * rng.normal(0, 1, n)) * (0.30 + 0.70 * items / 3.0)
    weight = np.clip(weight, 0.05, 45.0)

    v_mu = np.array([CATEGORIES[c]["v_mu"] for c in category])
    v_sd = np.array([CATEGORIES[c]["v_sd"] for c in category])
    order_value = np.exp(v_mu + v_sd * rng.normal(0, 1, n)) * (0.35 + 0.65 * items / 3.0)
    order_value = np.clip(order_value, 199.0, 250_000.0)

    # --- Route / network features -----------------------------------------
    # Mixture: intra-city, regional, national lanes
    lane = rng.choice([0, 1, 2], size=n, p=[0.42, 0.38, 0.20])
    distance = np.where(
        lane == 0, rng.uniform(2, 55, n),
        np.where(lane == 1, rng.uniform(55, 450, n), rng.uniform(450, 1500, n)),
    )

    traffic = np.clip(rng.normal(5.4, 1.9, n), 1.0, 10.0)
    courier_load = np.clip(rng.normal(5.8, 1.7, n) + 0.35 * (traffic - 5.4), 1.0, 10.0)

    # Warehouse dwell time grows with order complexity and courier saturation
    processing = (
        rng.gamma(shape=2.2, scale=1.15, size=n)
        + 0.22 * items
        + 0.30 * (courier_load - 5.8)
    )
    processing = np.clip(processing, 0.3, 26.0)

    mode = sample_shipping_mode(distance)

    # --- Structural equation for the target -------------------------------
    mu = (
        BETA["intercept"]
        + BETA["distance_km"] * distance
        + BETA["processing_hours"] * processing
        + BETA["traffic_index"] * traffic
        + BETA["courier_load_index"] * courier_load
        + BETA["items_in_order"] * items
        + BETA["package_weight_kg"] * weight
        + BETA["order_value"] * order_value
        + np.array([MODE_EFFECT[m] for m in mode])
        + np.array([CATEGORIES[c]["t_eff"] for c in category])
    )

    sd = NOISE_BASE_SD + NOISE_SCALE_SD * mu          # mild heteroscedasticity
    delivery_hours = np.clip(mu + rng.normal(0, sd), 0.75, None)

    df = pd.DataFrame(
        {
            "Order_ID": [f"ORD-{100000 + i}" for i in range(n)],
            "Product_Category": category,
            "Shipping_Mode": mode,
            "Order_Value": np.round(order_value, 2),
            "Package_Weight_Kg": np.round(weight, 3),
            "Warehouse_Distance_Km": np.round(distance, 1),
            "Items_in_Order": items,
            "Warehouse_Processing_Hours": np.round(processing, 2),
            "Courier_Load_Index": np.round(courier_load, 2),
            "Traffic_Index": np.round(traffic, 2),
            "Delivery_Time_Hours": np.round(delivery_hours, 2),
        }
    )
    return df


# --------------------------------------------------------------------------- #
# Intentional data-quality issues (so Part A has real work to do)
# --------------------------------------------------------------------------- #
def inject_dirt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # 1. Missing values (sensor / integration gaps)
    for col, frac in [("Traffic_Index", 0.020),
                      ("Order_Value", 0.015),
                      ("Package_Weight_Kg", 0.010),
                      ("Warehouse_Processing_Hours", 0.008)]:
        idx = rng.choice(n, size=max(1, int(frac * n)), replace=False)
        df.loc[idx, col] = np.nan

    # 2. Inconsistent category labels (free-text / multi-source ingestion)
    cat_variants = {"Electronics": "electronics", "Apparel": "APPAREL ",
                    "Home_Kitchen": " Home_Kitchen", "Groceries": "groceries"}
    for clean, dirty in cat_variants.items():
        pool = df.index[df["Product_Category"] == clean]
        if len(pool):
            idx = rng.choice(pool, size=max(1, int(0.06 * len(pool))), replace=False)
            df.loc[idx, "Product_Category"] = dirty

    mode_variants = {"Standard": "standard", "Express": "EXPRESS",
                     "Same_Day": "Same Day"}
    for clean, dirty in mode_variants.items():
        pool = df.index[df["Shipping_Mode"] == clean]
        if len(pool):
            idx = rng.choice(pool, size=max(1, int(0.05 * len(pool))), replace=False)
            df.loc[idx, "Shipping_Mode"] = dirty

    # 3. Impossible values (negative measures, zero items, out-of-scale index)
    df.loc[rng.choice(n, 5, replace=False), "Package_Weight_Kg"] = -1.5
    df.loc[rng.choice(n, 4, replace=False), "Warehouse_Distance_Km"] = -12.0
    df.loc[rng.choice(n, 4, replace=False), "Items_in_Order"] = 0
    df.loc[rng.choice(n, 3, replace=False), "Traffic_Index"] = 99.0
    df.loc[rng.choice(n, 3, replace=False), "Delivery_Time_Hours"] = -4.0

    # 4. Operational outliers (customs hold / failed first attempt)
    idx = rng.choice(n, 6, replace=False)
    df.loc[idx, "Delivery_Time_Hours"] = np.round(
        df.loc[idx, "Delivery_Time_Hours"].fillna(30) * 3.2 + 40, 2)

    # 5. Exact duplicate rows (double ingestion of the same order)
    dup = df.sample(n=15, random_state=RANDOM_SEED)
    df = pd.concat([df, dup], ignore_index=True)

    return df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


# --------------------------------------------------------------------------- #
def main() -> None:
    df = generate_clean_frame(N_ROWS)
    if INJECT_DIRT:
        df = inject_dirt(df)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {OUTPUT_PATH}: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumns:", list(df.columns))
    print("\nHead:")
    print(df.head(5).to_string(index=False))
    print("\nTarget summary (Delivery_Time_Hours):")
    print(df["Delivery_Time_Hours"].describe().round(2).to_string())
    print(f"\nMissing values injected: {int(df.isna().sum().sum())}")
    print(f"Duplicate rows present:  {int(df.duplicated().sum())}")


if __name__ == "__main__":
    main()
