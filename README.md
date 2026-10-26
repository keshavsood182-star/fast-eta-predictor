# 🛵 Fast ETA Predictor

Machine-learning delivery-time estimation. A Streamlit web application wrapped around a **pre-trained** Random Forest pipeline that predicts total food-delivery time (minutes) at the moment an order is placed.

> **The application never trains, tunes, or replaces the model.** It loads `zomato_fast_eta_model.joblib` with `joblib.load()` and calls `model.predict()`.

---

## Problem statement

Given conditions that are known when an order is placed (courier, vehicle, weather, traffic, order type, city, time, and optionally restaurant/delivery GPS), estimate how many minutes the delivery will take.

## Model

| Item | Value |
|---|---|
| Model | Tuned Random Forest Regressor (inside a scikit-learn `Pipeline` with preprocessing) |
| `n_estimators` | 150 |
| `max_depth` | 15 |
| `min_samples_leaf` | 1 |
| `min_samples_split` | 2 |
| `max_features` | 0.5 |
| `random_state` | 42 |
| Distance feature | Haversine (straight-line) |
| Target | `Time_taken (min)` |
| Tuning | Lightweight 3-fold cross-validation search |

The persisted pipeline contains a `ColumnTransformer`: numeric columns → median imputer; categorical columns → most-frequent imputer + one-hot encoder (`handle_unknown="ignore"`); followed by the Random Forest. The app passes **raw** features straight into it.

## Features (20, available at order-placement time)

**Numeric (14):** `Delivery_person_Age`, `Delivery_person_Ratings`, `Vehicle_condition`, `multiple_deliveries`, `Restaurant_latitude`, `Restaurant_longitude`, `Delivery_location_latitude`, `Delivery_location_longitude`, `distance_km`, `order_hour`, `order_day_of_week`, `GPS_valid`, `distance_valid`, `order_time_missing`

**Categorical (6):** `Weather_conditions`, `Road_traffic_density`, `Type_of_order`, `Type_of_vehicle`, `Festival`, `City`

`Time_Order_picked` is **not** used: it occurs after the order is placed and would leak information. ID columns are not used as predictive features.

## Data methodology

- A **chronological** train / validation / test split was used: validation and test sets contain later dates than the data used for fitting.
- The test set was **not** used for model selection.
- Dataset size, source, and date range are not recorded in the files supplied with this app, so they are not stated here.

## Results

### Validation (used during model selection)

| Metric | Value |
|---|---|
| RMSE | 4.058 min |
| MAE | 3.219 min |
| R² | 0.817 |
| Within ±5 min | 80.26% |

### Final chronological test set

| Metric | Value |
|---|---|
| **RMSE** | **4.032 minutes** |
| **MAE** | **3.214 minutes** |
| **R²** | **0.811** |
| **Within ±5 min** | **80.38%** |

**80.38% of test predictions were within ±5 minutes of the actual delivery time.** (This is not a generic "accuracy" figure.)

## Feature engineering (applied in `app.py`)

**Haversine distance** – Earth radius 6371 km, standard formula between restaurant and delivery coordinates.

**GPS validity**
- If *any* of the four coordinates has `abs(value) < 0.1`: `GPS_valid = 0`, all four coordinates → `NaN`, `distance_km = NaN`, `distance_valid = 0`.
- Otherwise `GPS_valid = 1` and the Haversine distance is computed. If it is `<= 100 km`, `distance_valid = 1`; otherwise `distance_km = NaN` and `distance_valid = 0`.
- If the user says GPS is unavailable, the same "invalid" values are passed.

**Order time** – known hour: `order_hour` (0–23), `order_time_missing = 0`. Unknown: `order_hour = NaN`, `order_time_missing = 1`.

**Day of week** – Monday = 0 … Sunday = 6.

When values are `NaN`, the pipeline's own imputers fill them with training-set medians / most-frequent categories. For example, the fitted median `distance_km` is ≈ 9.19 km, so a missing-GPS prediction behaves as if a typical distance was supplied.

## Road-distance engineering experiment

Road-network distance was investigated **separately** and is **not part of the production model**. Pilot results:

| Measure | Value |
|---|---|
| Mean Haversine distance | 8.81 km |
| Mean road-network distance | 12.63 km |
| Mean road / Haversine ratio | 1.50× |
| Median road / Haversine ratio | 1.42× |
| Successful routes | 99 / 100 |

Road-network routes were substantially longer than straight-line distances in the pilot. However, a separate road-distance model was **not** fully trained and evaluated on the final test set, so no road-distance error metric or accuracy comparison is reported, and this experiment does not show that road distance is more predictive. The app makes no external routing API calls.

## Project structure

```
fast-eta-predictor/
├── app.py                          # Streamlit application
├── requirements.txt                # Pinned dependencies
├── README.md
├── .gitignore
├── zomato_fast_eta_model.joblib    # Trained pipeline (your artifact, ~98 MB)
└── zomato_fast_eta_metadata.json   # Feature schema + metrics (your artifact)
```

## Reproducibility / environment

The persisted model must be loaded with compatible versions:

| Package | Version |
|---|---|
| Python | 3.13.16 (training environment) |
| scikit-learn | 1.6.1 |
| pandas | 2.2.3 |
| numpy | 2.1.3 |
| joblib | 1.6.0 |

`requirements.txt` additionally pins `scipy==1.14.1`, `threadpoolctl==3.5.0` and `streamlit==1.66.1`. The app shows a warning if the installed scikit-learn version differs from the one recorded in the metadata. Do not upgrade scikit-learn or numpy casually.

## Run locally

```bash
# 1. Put app.py, requirements.txt, and both model files in one folder, then:
cd fast-eta-predictor

# 2. Create and activate a virtual environment (Python 3.13 recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Deployment

### Streamlit Community Cloud
1. Push the folder (including both model files) to a GitHub repository.
2. The model file is ~98 MB: under GitHub's 100 MiB hard limit, but above the 50 MB warning threshold. If a push is rejected or you prefer, track it with Git LFS: `git lfs install && git lfs track "*.joblib"`, then commit `.gitattributes`.
3. In [share.streamlit.io](https://share.streamlit.io) choose **New app**, select the repo/branch, set the main file to `app.py`.
4. Under **Advanced settings**, select **Python 3.13** (or the closest available version compatible with the pinned packages).
5. Deploy and check the logs for installation errors.

### Docker / any VM or container host
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```
```bash
docker build -t fast-eta .
docker run -p 8501:8501 fast-eta
```

## Input validation & error handling

Age must be 15–50 and rating 1–5; coordinates must be valid latitude/longitude values. Invalid GPS (zero-ish coordinates, or distance > 100 km) is handled by passing the validity indicators described above. A missing model or metadata file, a failed model load, or a failed prediction results in a plain-language message; technical details go to the server log, not the UI. There is no feature for uploading or loading arbitrary model files.

## Verification performed on the supplied artifacts

- The 20 feature names in the app match the metadata JSON and the loaded pipeline's `feature_names_in_` (same order).
- Category values in the app match the fitted `OneHotEncoder` categories (including the training-data spelling `Metropolitian`, shown to users as "Metropolitan").
- Hyperparameters in the loaded model match the metadata.
- Displayed performance figures are read from the metadata JSON.
- The app's Haversine output was checked against an independent implementation.

**Not verifiable from the supplied files:** the allowed value sets for `Vehicle_condition` and `multiple_deliveries` (the app offers 0–3 based on the usual form of this dataset), the dataset size/date range, and the exact training-time Python 3.13.16 behaviour (the app was tested here under Python 3.12 with the pinned library versions).

## Limitations

- The historical dataset represents a specific period; performance may change under different operational conditions.
- Haversine distance is not the actual courier route distance.
- Traffic and weather variables are historical **categorical** data, not live feeds.
- Historical data does not support causal claims about operational changes (e.g. "reducing traffic would cut ETA by X").
- The model predicts **expected** delivery time, not a guaranteed arrival time. No prediction or confidence interval is provided; the Random Forest is not calibrated for intervals in this project.
- Road-network distance was explored separately but is not part of the final model.
- When GPS or order hour is missing, imputed typical values are used, so those estimates are less specific.
