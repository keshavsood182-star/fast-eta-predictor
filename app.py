"""FAST ETA PREDICTOR - Streamlit front end for the trained Zomato ETA model.

The trained pipeline (preprocessing + Random Forest) is loaded as-is from
``zomato_fast_eta_model.joblib``. This app never retrains, tunes or replaces it:
it only builds the 20 raw model features and calls ``model.predict``.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "zomato_fast_eta_model.joblib"
METADATA_PATH = BASE_DIR / "zomato_fast_eta_metadata.json"

EARTH_RADIUS_KM = 6371.0
GPS_MIN_ABS_COORD = 0.1      # any |coordinate| < 0.1 => GPS treated as invalid
MAX_VALID_DISTANCE_KM = 100.0  # distances above this are treated as invalid

AGE_RANGE = (15, 50)
RATING_RANGE = (1.0, 5.0)

# Raw category values exactly as the fitted OneHotEncoder knows them.
# (Verified against the loaded pipeline; see README "Verification".)
CATEGORIES = {
    "Weather_conditions": ["Cloudy", "Fog", "Sandstorms", "Stormy", "Sunny", "Windy"],
    "Road_traffic_density": ["Low", "Medium", "High", "Jam"],
    "Type_of_order": ["Buffet", "Drinks", "Meal", "Snack"],
    "Type_of_vehicle": ["bicycle", "electric_scooter", "motorcycle", "scooter"],
    "Festival": ["No", "Yes"],
    "City": ["Metropolitian", "Semi-Urban", "Urban"],
}
# Display-only labels. The raw value (left) is what is passed to the model.
DISPLAY_LABELS = {
    "Metropolitian": "Metropolitan",  # spelling in the training data is preserved internally
    "electric_scooter": "Electric scooter",
    "bicycle": "Bicycle",
    "motorcycle": "Motorcycle",
    "scooter": "Scooter",
}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

logger = logging.getLogger("fast_eta")

st.set_page_config(
    page_title="Fast ETA Predictor",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Small, targeted CSS: enlarge the headline metric only.
st.markdown(
    """
    <style>
      div[data-testid="stMetricValue"] { font-weight: 700; }
      .st-key-eta-card div[data-testid="stMetricValue"] { font-size: 3.4rem; }
      .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading trained model...")
def load_model(path: str = str(MODEL_PATH)):
    """Load the trusted, pre-trained pipeline directly with joblib."""
    return joblib.load(path)


@st.cache_data
def load_metadata(path: str = str(METADATA_PATH)) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_artifacts():
    """Load model + metadata, or stop with a clear, non-technical error."""
    missing = [p.name for p in (MODEL_PATH, METADATA_PATH) if not p.exists()]
    if missing:
        st.error(
            "**Required file(s) missing:** " + ", ".join(f"`{m}`" for m in missing)
            + ".\n\nPlace them in the same folder as `app.py` and restart the app."
        )
        st.stop()
    try:
        metadata = load_metadata()
        model = load_model()
    except Exception:  # noqa: BLE001 - shown to user as a friendly message
        logger.exception("Failed to load model artifacts")
        st.error(
            "The model could not be loaded. This usually means the installed library "
            "versions differ from those used for training. Install the versions in "
            "`requirements.txt` (notably scikit-learn 1.6.1) and restart."
        )
        st.stop()

    features = metadata["features"]
    fitted = getattr(model, "feature_names_in_", None)
    if fitted is not None and list(fitted) != list(features):
        st.error("The model's feature schema does not match the metadata file. "
                 "Please check that both files come from the same export.")
        st.stop()
    return model, metadata


# --------------------------------------------------------------------------- #
# Feature engineering (mirrors the development logic)
# --------------------------------------------------------------------------- #
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km (Earth radius 6371 km)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def resolve_gps(gps_available: bool, coords: tuple[float, float, float, float]) -> dict:
    """Apply the GPS / distance validity rules.

    Returns the four coordinates, distance_km, GPS_valid, distance_valid and a
    human-readable ``note`` (None when everything is fine).
    """
    nan = float("nan")
    invalid = {
        "coords": (nan, nan, nan, nan), "distance_km": nan,
        "GPS_valid": 0, "distance_valid": 0,
    }
    if not gps_available:
        return {**invalid, "note": "GPS not provided - the model receives missing location values."}

    if any(abs(c) < GPS_MIN_ABS_COORD for c in coords):
        return {**invalid, "note": "At least one coordinate is within 0.1 of zero, so the GPS set is "
                                   "treated as invalid and passed to the model as missing."}

    dist = haversine_km(*coords)
    if dist <= MAX_VALID_DISTANCE_KM:
        return {"coords": coords, "distance_km": dist, "GPS_valid": 1,
                "distance_valid": 1, "note": None}
    return {"coords": coords, "distance_km": nan, "GPS_valid": 1, "distance_valid": 0,
            "note": f"The calculated distance ({dist:,.1f} km) exceeds {MAX_VALID_DISTANCE_KM:.0f} km, "
                    "so the distance is treated as invalid and passed to the model as missing."}


def build_features(inp: dict, gps: dict, feature_order: list[str]) -> pd.DataFrame:
    """Create the one-row DataFrame of raw model features, in the model's column order."""
    lat_r, lon_r, lat_d, lon_d = gps["coords"]
    hour_known = inp["order_hour"] is not None
    row = {
        "Delivery_person_Age": float(inp["age"]),
        "Delivery_person_Ratings": float(inp["rating"]),
        "Vehicle_condition": float(inp["vehicle_condition"]),
        "multiple_deliveries": float(inp["multiple_deliveries"]),
        "Restaurant_latitude": lat_r,
        "Restaurant_longitude": lon_r,
        "Delivery_location_latitude": lat_d,
        "Delivery_location_longitude": lon_d,
        "distance_km": gps["distance_km"],
        "order_hour": float(inp["order_hour"]) if hour_known else float("nan"),
        "order_day_of_week": float(inp["day_of_week"]),
        "GPS_valid": float(gps["GPS_valid"]),
        "distance_valid": float(gps["distance_valid"]),
        "order_time_missing": 0.0 if hour_known else 1.0,
        "Weather_conditions": inp["weather"],
        "Road_traffic_density": inp["traffic"],
        "Type_of_order": inp["order_type"],
        "Type_of_vehicle": inp["vehicle"],
        "Festival": inp["festival"],
        "City": inp["city"],
    }
    return pd.DataFrame([row])[feature_order]


def validate_inputs(age: float, rating: float, gps_available: bool, coords) -> list[str]:
    errors = []
    if not AGE_RANGE[0] <= age <= AGE_RANGE[1]:
        errors.append(f"Delivery person age must be between {AGE_RANGE[0]} and {AGE_RANGE[1]}.")
    if not RATING_RANGE[0] <= rating <= RATING_RANGE[1]:
        errors.append(f"Delivery person rating must be between {RATING_RANGE[0]:g} and {RATING_RANGE[1]:g}.")
    if gps_available:
        lat_r, lon_r, lat_d, lon_d = coords
        if not (-90 <= lat_r <= 90 and -90 <= lat_d <= 90):
            errors.append("Latitude values must be between -90 and 90.")
        if not (-180 <= lon_r <= 180 and -180 <= lon_d <= 180):
            errors.append("Longitude values must be between -180 and 180.")
    return errors


def label(value: str) -> str:
    return DISPLAY_LABELS.get(value, value)


# --------------------------------------------------------------------------- #
# Load artifacts
# --------------------------------------------------------------------------- #
model, metadata = load_artifacts()
FEATURES = metadata["features"]
TEST = metadata["test_metrics"]
VAL = metadata["validation_metrics"]

# --------------------------------------------------------------------------- #
# Sidebar: inputs
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Order details")
    if sklearn.__version__ != metadata.get("scikit_learn_version"):
        st.warning(
            f"scikit-learn {sklearn.__version__} is installed but the model was trained with "
            f"{metadata.get('scikit_learn_version')}. Predictions may be unreliable or fail."
        )

    st.subheader("Courier")
    age = st.number_input("Delivery person age", min_value=0, max_value=100, value=30, step=1,
                          help=f"Accepted range: {AGE_RANGE[0]}-{AGE_RANGE[1]}.")
    rating = st.number_input("Delivery person rating", min_value=0.0, max_value=10.0, value=4.7,
                             step=0.1, format="%.1f",
                             help=f"Accepted range: {RATING_RANGE[0]:g}-{RATING_RANGE[1]:g}.")
    vehicle = st.selectbox("Type of vehicle", CATEGORIES["Type_of_vehicle"],
                           index=2, format_func=label)
    vehicle_condition = st.selectbox("Vehicle condition", [0, 1, 2, 3], index=1,
                                     help="Vehicle-condition code as used in the training data "
                                          "(ordinal codes; their exact meaning is not documented here).")
    multiple_deliveries = st.selectbox("Multiple deliveries", [0, 1, 2, 3], index=1,
                                       help="Number of additional deliveries carried at the same time.")

    st.subheader("Order & conditions")
    order_type = st.selectbox("Type of order", CATEGORIES["Type_of_order"], index=3)
    weather = st.selectbox("Weather conditions", CATEGORIES["Weather_conditions"], index=4)
    traffic = st.selectbox("Road traffic density", CATEGORIES["Road_traffic_density"], index=0)
    festival = st.selectbox("Festival", CATEGORIES["Festival"], index=0)
    city = st.selectbox("City", CATEGORIES["City"], index=0, format_func=label)

    st.subheader("Timing")
    hour_known = st.checkbox("Order hour is known", value=True)
    hour = st.slider("Order hour (0-23)", 0, 23, 19, disabled=not hour_known)
    day_name = st.selectbox("Day of week", DAYS, index=4)

    st.subheader("Location (GPS)")
    gps_available = st.checkbox("GPS available", value=True)
    c1, c2 = st.columns(2)
    lat_r = c1.number_input("Restaurant lat", value=12.9716, format="%.6f", disabled=not gps_available)
    lon_r = c2.number_input("Restaurant lon", value=77.5946, format="%.6f", disabled=not gps_available)
    lat_d = c1.number_input("Delivery lat", value=12.9916, format="%.6f", disabled=not gps_available)
    lon_d = c2.number_input("Delivery lon", value=77.6146, format="%.6f", disabled=not gps_available)

    submitted = st.button("PREDICT ETA", type="primary", use_container_width=True)

    # Live distance preview
if gps_available:
    _g = resolve_gps(True, (lat_r, lon_r, lat_d, lon_d))
    if not math.isnan(_g["distance_km"]):
        st.caption(f"Haversine distance: **{_g['distance_km']:.2f} km**")
    elif _g["note"]:
        st.caption(f"Distance unavailable: {_g['note']}")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🛵 Fast ETA Predictor")
st.caption("Machine-learning delivery-time estimation")

# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
current_sig = repr((age, rating, vehicle, vehicle_condition, multiple_deliveries, order_type, weather,
                    traffic, festival, city, hour_known, hour, day_name, gps_available,
                    lat_r, lon_r, lat_d, lon_d))
if submitted:
    coords = (lat_r, lon_r, lat_d, lon_d)
    errors = validate_inputs(age, rating, gps_available, coords)
    if errors:
        st.session_state.pop("result", None)
        for e in errors:
            st.error(e)
    else:
        try:
            gps = resolve_gps(gps_available, coords)
            inputs = {
                "age": age, "rating": rating, "vehicle_condition": vehicle_condition,
                "multiple_deliveries": multiple_deliveries,
                "order_hour": hour if hour_known else None,
                "day_of_week": DAYS.index(day_name),
                "weather": weather, "traffic": traffic, "order_type": order_type,
                "vehicle": vehicle, "festival": festival, "city": city,
            }
            features = build_features(inputs, gps, FEATURES)
            prediction = float(model.predict(features)[0])
            if not np.isfinite(prediction):
                raise ValueError("Non-finite prediction")
            st.session_state["result"] = {
                "eta": prediction, "distance_km": gps["distance_km"], "note": gps["note"],
                "traffic": traffic, "vehicle": vehicle, "order_type": order_type,
                "weather": weather, "city": city, "festival": festival,
                "day": day_name, "hour": hour if hour_known else None,
                "sig": current_sig,
            }
        except Exception:  # noqa: BLE001
            logger.exception("Prediction failed")
            st.session_state.pop("result", None)
            st.error("Sorry, the prediction could not be completed. Please check your inputs and try again.")

result = st.session_state.get("result")

st.subheader("Estimated delivery time")
with st.container(border=True, key="eta-card"):
    if result is None:
        st.info("Enter the order details in the sidebar and press **PREDICT ETA**.")
    else:
        st.metric("Estimated Delivery Time", f"{round(result['eta'])} minutes")
        st.caption(f"Raw model output: {result['eta']:.1f} min")
        if result.get("sig") != current_sig:
            st.info("Inputs have changed since this prediction. Press **PREDICT ETA** to update it.")
        if result["note"]:
            st.warning(result["note"])
        st.caption("This is a model estimate, not a guaranteed delivery time.")

if result is not None:
    d1, d2, d3, d4 = st.columns(4)
    dist = result["distance_km"]
    d1.metric("Haversine distance", "N/A" if np.isnan(dist) else f"{dist:.2f} km")
    d2.metric("Traffic condition", result["traffic"])
    d3.metric("Vehicle type", label(result["vehicle"]))
    d4.metric("Order type", result["order_type"])
    hour_txt = "unknown" if result["hour"] is None else f"{result['hour']:02d}:00"
    st.caption(f"{result['day']}, hour {hour_txt} · {result['weather']} weather · "
               f"{label(result['city'])} · Festival: {result['festival']}")

st.divider()

# --------------------------------------------------------------------------- #
# Model performance
# --------------------------------------------------------------------------- #
st.header("Model performance")
st.markdown("**FINAL CHRONOLOGICAL TEST-SET PERFORMANCE**")
p1, p2, p3, p4 = st.columns(4)
p1.metric("RMSE", f"{TEST['rmse_minutes']:.3f} min")
p2.metric("MAE", f"{TEST['mae_minutes']:.3f} min")
p3.metric("R²", f"{TEST['r2']:.3f}")
p4.metric("Within ±5 min", f"{TEST['within_5_minutes_percent']:.2f}%")
st.caption(
    f"{TEST['within_5_minutes_percent']:.2f}% of test predictions were within ±5 minutes of the actual "
    "delivery time. The test set contains later, unseen dates and was not used for model selection."
)
with st.expander("Validation-set results (used during model selection)"):
    st.table(pd.DataFrame(
        {"Validation": [f"{VAL['rmse_minutes']:.3f} min", f"{VAL['mae_minutes']:.3f} min",
                        f"{VAL['r2']:.3f}", f"{VAL['within_5_minutes_percent']:.2f}%"],
         "Final test": [f"{TEST['rmse_minutes']:.3f} min", f"{TEST['mae_minutes']:.3f} min",
                        f"{TEST['r2']:.3f}", f"{TEST['within_5_minutes_percent']:.2f}%"]},
        index=["RMSE", "MAE", "R²", "Within ±5 min"],
    ))

st.divider()

# --------------------------------------------------------------------------- #
# Engineering experiment
# --------------------------------------------------------------------------- #
st.header("Engineering experiment: road-network distance")
st.markdown(
    "Road-network distance was investigated **separately** and is **not** used in the production model, "
    "which uses straight-line Haversine distance."
)
e1, e2, e3, e4, e5 = st.columns(5)
e1.metric("Mean Haversine", "8.81 km")
e2.metric("Mean road distance", "12.63 km")
e3.metric("Mean ratio", "1.50x")
e4.metric("Median ratio", "1.42x")
e5.metric("Successful routes", "99 / 100")
st.markdown(
    "In this pilot, road-network routes were substantially longer than straight-line distances. "
    "However, a separate road-distance model was **not** fully trained and evaluated on the final test set, "
    "so no road-distance error metrics or accuracy comparison are reported, and this result does not show "
    "that road distance would be more predictive."
)

st.divider()

# --------------------------------------------------------------------------- #
# About
# --------------------------------------------------------------------------- #
st.header("About the model")
hp = metadata["hyperparameters"]
st.markdown(
    f"""
- **Model:** tuned Random Forest Regressor (`n_estimators={hp['n_estimators']}`, `max_depth={hp['max_depth']}`,
  `min_samples_leaf={hp['min_samples_leaf']}`, `min_samples_split={hp['min_samples_split']}`,
  `max_features={hp['max_features']}`, `random_state={hp['random_state']}`), packaged with its preprocessing in one pipeline.
- **Distance:** Haversine (straight-line) distance between restaurant and delivery location.
- **Methodology:** chronological train / validation / test split, so the test set contains later dates than the training data.
- **Inputs:** {len(FEATURES)} features that are available at the moment an order is placed.
- **No leakage:** `Time_Order_picked` was excluded because it occurs after the order is placed.
- **No IDs:** identifier columns were not used as predictive features.
- **Tuning:** a lightweight 3-fold cross-validation search.
- **Missing data:** if GPS or the order hour is unavailable, the model receives missing values plus
  validity indicators (`GPS_valid`, `distance_valid`, `order_time_missing`) and its built-in imputers fill in typical values.
"""
)
st.caption(
    "The model predicts expected delivery time from historical data for a specific period. "
    "It does not guarantee arrival time and does not support causal conclusions."
)
