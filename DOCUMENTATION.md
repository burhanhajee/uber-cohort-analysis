# Technical Documentation
## Uber Pro-Driver Blueprint: Segmentation & Churn Prediction Engine

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Sources](#2-data-sources)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Feature Engineering (Script 01)](#4-feature-engineering-script-01)
5. [Clustering Model (Script 02)](#5-clustering-model-script-02)
6. [Churn Prediction Model (Script 03)](#6-churn-prediction-model-script-03)
7. [Notebooks](#7-notebooks)
8. [Model Artifacts](#8-model-artifacts)
9. [Known Design Decisions & Bug Fixes](#9-known-design-decisions--bug-fixes)
10. [Dependencies](#10-dependencies)

---

## 1. Project Overview

This project analyzes synthetic Uber driver behavioral data to:

1. Engineer meaningful behavioral features from raw event logs
2. Segment drivers into 4 distinct personas using unsupervised learning
3. Predict which drivers are likely to churn using supervised learning
4. Surface actionable retention insights (cancellation frustration > low earnings as churn driver)

**Tech stack:** DuckDB (SQL feature engineering), scikit-learn (clustering + preprocessing), XGBoost (classification), MLflow (experiment tracking), Pandas/NumPy (data wrangling), Seaborn/Plotly (visualization).

---

## 2. Data Sources

All data is synthetically generated. Raw files live in `data/raw/`.

### profile_data.csv
| Column | Type | Description |
|---|---|---|
| `driver_id` | string | Unique driver identifier |
| `signup_date` | date | Account creation date |
| `vehicle_dispatchability` | string | Comma-separated trip types vehicle qualifies for |
| `avg_rating` | float | Average passenger rating (1–5) |
| `current_tier` | string | Uber Pro tier: Blue, Gold, Platinum, Diamond |
| `Churned` | int | Target label: 1 = churned, 0 = active |

**Size:** 581 rows. Churn distribution: ~68.8% active, ~31.2% churned.

### activity_logs.csv
| Column | Type | Description |
|---|---|---|
| `session_id` | string | Unique session identifier |
| `driver_id` | string | Foreign key to profiles |
| `session_start` | timestamp | When driver went online |
| `session_end` | timestamp | When driver went offline |

**Size:** ~59,225 rows. Validated: `session_end > session_start` for all rows.

### trip_logs.csv
| Column | Type | Description |
|---|---|---|
| `trip_id` | string | Unique trip identifier |
| `driver_id` | string | Foreign key to profiles |
| `request_time` | timestamp | When trip was requested |
| `pickup_time` | timestamp | When passenger was picked up |
| `trip_duration_seconds` | int | Duration of the trip |
| `fare` | float | Total fare charged (surge-inclusive) |
| `tip` | float | Passenger tip |
| `surge_multiplier` | float | Surge multiplier applied (1.0 = no surge) |
| `pickup_lat/long` | float | Pickup coordinates |
| `dropoff_lat/long` | float | Dropoff coordinates |
| `trip_type` | string | UberX, Comfort, Electric, Premier, UberBlack |

**Size:** ~522,888 rows. Validated: `pickup_time > request_time` for all rows.

### incentive_logs.csv
| Column | Type | Description |
|---|---|---|
| `incentive_id` | string | Unique incentive identifier |
| `driver_id` | string | Foreign key to profiles |
| `incentive_type` | string | Quest or Boost |
| `offer_date` | date | When incentive was offered |
| `status` | string | completed, pending, expired, failed |
| `bonus_amount` | float | Bonus payout if completed |

**Size:** ~15,228 rows.

### interaction_logs.csv
| Column | Type | Description |
|---|---|---|
| `event_id` | string | Unique event identifier |
| `driver_id` | string | Foreign key to profiles |
| `event_timestamp` | timestamp | When event occurred |
| `event_type` | string | app_open, destination_set, heatmap_view, trip_accepted, trip_cancelled, trip_ignored |

**Size:** ~1.2M rows.

---

## 3. Pipeline Architecture

```
data/raw/
    ├── profile_data.csv
    ├── activity_logs.csv
    ├── trip_logs.csv
    ├── incentive_logs.csv
    └── interaction_logs.csv
           │
           ▼
scripts/01_feature_engineering.py  (DuckDB SQL)
           │
           ▼
data/processed/training_data.csv   (581 drivers × 18 columns)
           │
           ▼
scripts/02_cluster_model.py        (K-Means, k=4)
           │
           ▼
data/processed/training_data_with_clusters.csv
models/kmeans_model.joblib
models/scaler.joblib
           │
           ▼
scripts/03_prediction_model.py     (XGBoost + MLflow)
           │
           ▼
models/churn_model_optimized.json
mlruns/ (MLflow experiment logs)
```

---

## 4. Feature Engineering (Script 01)

**File:** `scripts/01_feature_engineering.py`
**Input:** 5 raw CSV files from `data/raw/`
**Output:** `data/processed/training_data.csv`
**Engine:** DuckDB in-memory SQL

Each feature is computed as a separate temporary table, then joined back to the `profiles` table as the spine.

### Feature Definitions

#### A. avg_earnings_per_hour_online (EPHO)
```
EPHO = SUM(fare + tip) / SUM(session_duration_hours)
```
Measures how efficiently a driver converts online time into earnings. Drivers with low EPHO but high online hours signal poor strategy (idling, inefficient zones).

#### B. trip_utilization_rate
```
utilization = LEAST(SUM(trip_duration_seconds) / SUM(session_duration_seconds), 1.0)
```
Fraction of online time spent actively on trips. Clamped to 1.0 to handle timestamp misalignments between trip and session logs. High utilization = efficient dispatch acceptance.

#### C. surge_reliance_score
```
surge_premium = fare - (fare / surge_multiplier)
surge_score = SUM(surge_premium) / SUM(fare / surge_multiplier)
```
Percentage premium earned above base fare from surge pricing. A score of 0.40 means the driver earned 40% on top of their base fare purely from surge. Denominator uses base fare (not total fare) so the score is not artificially suppressed for high-surge drivers.

#### D. premium_trip_ratio
```
premium_ratio = COUNT(trips WHERE type IN ('UberBlack', 'Premier')) / COUNT(all trips)
```
Proportion of trips in premium categories. Combined with `is_premium_capable` (binary, derived from `vehicle_dispatchability`) to distinguish drivers who can vs. those who choose premium trips.

#### E. peak_hour_driver_score
```
peak_score = COUNT(trips during peak windows) / COUNT(all trips)
```
Peak windows (America/New_York timezone):
- Mon–Thu: 07:00–10:00, 16:00–19:00 (commute)
- Friday: 07:00–09:00, 16:00–23:59 (commute + nightlife)
- Sat/Sun: 22:00–03:00 (bar/event rush)
- Sunday: 11:00–14:00 (brunch)

#### F. session_regularity
```
regularity = STDDEV(gap_between_consecutive_sessions_hours)
```
Standard deviation of inter-session gaps. Lower = more consistent schedule. Drivers with only one session get regularity=0 (handled via COALESCE on the NULL from STDDEV with a single input).

#### G. session_count
```
session_count = COUNT(sessions per driver)
```
Volume proxy; preserves the signal that new/about-to-churn drivers have few sessions, which `session_regularity` alone cannot capture.

#### H. quest_completion_rate
```
quest_rate = COUNT(Quest WHERE status='completed') / COUNT(Quest offers)
```
Measures how reliably a driver completes offered Quests. Only counts `incentive_type='Quest'` rows.

#### I. incentive_reliance_pct
```
reliance = SUM(bonus WHERE status='completed') / (bonus + fare + tip)
```
Proportion of total earnings from completed incentives. Filtered to `status='completed'` to align with `quest_completion_rate` — both measure the same completed-incentive population.

#### J. pro_tier_rank
```
Blue=1, Gold=2, Platinum=3, Diamond=4
```
Ordinal encoding of Uber Pro tier. Higher = more experienced/committed. Direction is intentional: higher rank → higher number → meaningful Euclidean distances in K-Means.

#### K. cancellation_rate
```
cancel_rate = COUNT(trip_cancelled) / COUNT(trip_accepted + trip_cancelled)
```
Post-acceptance bail rate. Denominator counts only events where cancellation was possible (accepted or cancelled). Bounded [0,1] by construction and independent of `acceptance_rate`.

#### L. acceptance_rate
```
accept_rate = COUNT(trip_accepted) / COUNT(trip_accepted + trip_ignored)
```
Fraction of dispatched trips the driver accepts. Note: this feature is excluded from clustering due to 96% correlation with `trip_utilization_rate`, but retained in the churn model.

---

## 5. Clustering Model (Script 02)

**File:** `scripts/02_cluster_model.py`
**Input:** `data/processed/training_data.csv`
**Output:** `data/processed/training_data_with_clusters.csv`, `models/kmeans_model.joblib`, `models/scaler.joblib`

### Feature Selection
6 features used (out of 11):

```python
features_to_cluster = [
    'avg_earnings_per_hour_online',
    'trip_utilization_rate',
    'surge_reliance_score',
    'premium_trip_ratio',
    'quest_completion_rate',
    'cancellation_rate'
]
```

`acceptance_rate` is dropped (96% Pearson correlation with `trip_utilization_rate` — verified in EDA). Including it would give the utilization dimension double weight and distort cluster geometry.

### Preprocessing
StandardScaler (`mean=0, std=1`) applied to all 6 features. Scaler is fitted on the full dataset and saved to `models/scaler.joblib` for use at inference time.

### Model
- **Algorithm:** K-Means
- **k=4** (determined via Elbow Method in EDA notebook)
- `n_init=10` (10 independent centroid initializations, best result kept)
- `random_state=42` (reproducibility)

### Cluster Profiles

| Cluster | Persona | EPHO | Utilization | Surge Reliance | Premium Ratio | Quest Rate | Cancel Rate |
|---|---|---|---|---|---|---|---|
| 0 | Casual / At-Risk | Low | Low | Low | Low | Low | High |
| 1 | Quest Grinder | Medium | Medium | Low | Low | High | Medium |
| 2 | Premium Specialist | High | Low | Low | High | Low | Low |
| 3 | Pro-Optimizer | High | High | High | Low | Medium | Low |

*Exact values vary by run; use `02_cluster_model.py` output or notebook 03 for current values.*

---

## 6. Churn Prediction Model (Script 03)

**File:** `scripts/03_prediction_model.py`
**Input:** `data/processed/training_data_with_clusters.csv`
**Output:** `models/churn_model_optimized.json`, MLflow run logs

### Feature Matrix (9 features)
```python
features = [
    'avg_earnings_per_hour_online',
    'trip_utilization_rate',
    'surge_reliance_score',
    'premium_trip_ratio',
    'quest_completion_rate',
    'cancellation_rate',   # 6 from clustering
    'acceptance_rate',     # retained here (not in clustering)
    'pro_tier_rank',
    'cluster_label'        # categorical, XGBoost native encoding
]
```

`cluster_label` is cast to `category` dtype so XGBoost handles it natively via `enable_categorical=True`.

### Class Imbalance
Churn rate ~31%. Handled by setting:
```python
scale_pos_weight = count(non-churned) / count(churned)
```
This up-weights the minority class (churners) during training, improving recall on churners.

### Hyperparameter Tuning
- **Method:** `RandomizedSearchCV` (50 iterations)
- **CV:** 5-fold `StratifiedKFold` (preserves class ratio in each fold)
- **Scoring:** ROC AUC (robust to class imbalance vs. accuracy)
- **Parallelism:** `n_jobs=-1`

Search space:
```python
{
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'gamma': [0, 0.1, 0.5, 1],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'reg_lambda': [0.1, 1.0, 5.0, 10.0]
}
```

### Evaluation
- Train/test split: 80/20, `random_state=42`
- Metrics logged: Test Accuracy, Test ROC AUC, CV Best ROC AUC
- Classification report printed to stdout
- **Achieved:** ~0.61 Recall on churners

### MLflow Tracking
Experiment name: `Uber_Driver_Churn_Prediction`
Logged per run: all best hyperparameters, `scale_pos_weight`, tuning method, test metrics, and the model artifact.
View runs: `mlflow ui` (runs at `http://localhost:5000` by default)

---

## 7. Notebooks

### 01_data_validation.ipynb
Purpose: Sanity-check raw data before any transformation.

Key checks performed:
- Row counts and dtypes for all 5 tables
- Null value audit (all clean)
- Churn distribution (~31% churned)
- Temporal logic: `session_end > session_start`, `pickup_time > request_time`
- Categorical coverage: all 4 tiers, 5 trip types, 6 event types present

### 02_eda_post_feature_engineering.ipynb
Purpose: Understand feature distributions and inform modeling decisions.

Key findings:
- Most features are multi-modal → good for K-Means separation
- `acceptance_rate` ↔ `trip_utilization_rate`: 96% Pearson correlation → drop acceptance_rate for clustering
- Outliers in `cancellation_rate` and `surge_reliance_score` are real signal, not noise → retain
- High earners do not universally stay → churn is multidimensional

### 03_model_analysis_and_storytelling.ipynb
Purpose: Persona profiling, model interpretation, and business narrative.

Key outputs:
- **Snake Plot:** Normalized (0–1) feature profiles across 4 clusters — visualizes strategic differences
- **Confusion Matrix:** XGBoost test set performance
- **Feature Importance Bar Chart:** `avg_earnings_per_hour_online` is #1 predictor (0.24), followed by `trip_utilization_rate` (0.13) and `quest_completion_rate` (0.11); `cancellation_rate` ranks 5th (0.10)
- **Churn Rate by Persona:** At-Risk cluster ~60%
- **Radar Chart:** Multi-dimensional persona comparison
- **Earnings Composition:** Stacked bar — organic (fare+tip) vs. inorganic (incentive) earnings by persona

---

## 8. Model Artifacts

| File | Size | Description |
|---|---|---|
| `models/scaler.joblib` | ~1 KB | StandardScaler fitted on 6 clustering features |
| `models/kmeans_model.joblib` | ~3 KB | K-Means model (k=4) |
| `models/churn_model.json` | ~236 KB | Initial XGBoost model |
| `models/churn_model_optimized.json` | ~396 KB | Tuned XGBoost model (final) |

To load and use at inference:
```python
import joblib
import xgboost as xgb

scaler = joblib.load("models/scaler.joblib")
kmeans = joblib.load("models/kmeans_model.joblib")

model = xgb.XGBClassifier(enable_categorical=True)
model.load_model("models/churn_model_optimized.json")
```

---

## 9. Known Design Decisions & Bug Fixes

All fixes below were identified during development and are documented inline in `scripts/01_feature_engineering.py`.

| Feature | Issue | Fix Applied |
|---|---|---|
| `trip_utilization_rate` | Minor timestamp misalignments produced values >1.0 | Clamped with `LEAST(..., 1.0)` |
| `surge_reliance_score` | Denominator used surge-inflated fare, suppressing scores for high-surge drivers | Denominator changed to `fare / surge_multiplier` (base fare equivalent) |
| `peak_hour_driver_score` | Silent operator precedence bug caused ~15/24 hours to count as "peak" on weekends | Added explicit parentheses; tightened peak windows to realistic high-demand slots |
| `session_regularity` | Single-session drivers produced NULL from STDDEV | `COALESCE(STDDEV(...), 0)` — single-session drivers have zero gap variance by definition |
| `incentive_reliance_pct` | Summed ALL bonus amounts regardless of status, inflating scores for failed quests | Added `WHERE status = 'completed'` filter |
| `pro_tier_rank` | Diamond=0, Blue=3 — inverted relative to business meaning | Reversed: Diamond=4, Blue=1; renamed from `pro_tier_status` to `pro_tier_rank` |
| `cancellation_rate` | Used `cancellations / acceptances` — not a standard rate, can exceed 1.0 | Changed to `cancellations / (accepted + cancelled)` — true post-acceptance bail rate |
| `use_label_encoder` | Deprecated XGBoost parameter causing warnings in newer versions | Removed; `enable_categorical=True` is the modern replacement |

---

## 10. Dependencies

```
pandas          # DataFrame operations
numpy           # Numerical computing
faker           # Synthetic data generation
pyarrow         # Columnar memory format (DuckDB backend)
duckdb          # In-memory SQL engine for feature engineering
scikit-learn    # StandardScaler, KMeans, model selection utilities
xgboost         # Gradient boosted tree classifier
mlflow          # Experiment tracking and model registry
streamlit       # (available — for future dashboard deployment)
notebook        # Jupyter notebook runtime
ipykernel       # IPython kernel
seaborn         # Statistical data visualization
plotly          # Interactive charts
```

Install all: `pip install -r requirements.txt`

Tested on Python 3.10+.
