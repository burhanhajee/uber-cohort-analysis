# The Uber Pro-Driver Blueprint
### A Multi-Dimensional Segmentation & Churn Prediction Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Classification-orange)
![Sklearn](https://img.shields.io/badge/Sklearn-KMeans-yellow)
![DuckDB](https://img.shields.io/badge/DuckDB-SQL-lightgrey)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-red)

---

## The Business Case

**Problem:** High driver churn reduces network liquidity and increases acquisition costs. Standard retention strategies (generic "Do 50 trips" Quests) are inefficient because they treat all drivers the same.

**Objective:** Move beyond simple "hours online" metrics to identify distinct **Driver Personas** based on strategy, efficiency, and frustration signals. Use these personas to predict churn and tailor retention spend.

**Outcome:** Identified 4 distinct driver personas and a "Leaky Bucket" segment with a **60% churn rate**, primarily driven by cancellation frustration rather than low earnings.

---

## Key Insights

### The 4 Driver Personas (K-Means, k=4)

| Persona | Strategy | Churn Rate |
|---|---|---|
| **Pro-Optimizer** | High utilization, surge-chasing — treats driving as a business | ~7% |
| **Quest Grinder** | High volume, incentive-dependent — chases every Quest | ~23% |
| **Premium Specialist** | Low volume, cherry-picks UberBlack/Premier trips | ~5% |
| **Casual / At-Risk** | Low earnings, high cancellations — disengaged | ~60% |

### Churn Drivers: It's Not Just Money

XGBoost feature importance reveals that **`avg_earnings_per_hour_online` is the #1 churn predictor** (importance score: 0.24) — nearly 2× the weight of the next feature. Drivers churn primarily because the economics don't work for them. `trip_utilization_rate` and `quest_completion_rate` follow as secondary signals, with `cancellation_rate` fifth.

**Strategic Recommendation:** Retention budget for At-Risk drivers should focus on **earnings efficiency coaching** — positioning strategy, surge awareness, and peak-hour scheduling — rather than generic cash bonuses.

---

## Technical Methodology

### Phase 1: Feature Engineering (DuckDB)

Raw logs (~1.2M events) are transformed into 11 behavioral features per driver using SQL window functions in DuckDB. Feature categories:

- **Efficiency:** `avg_earnings_per_hour_online`, `trip_utilization_rate`
- **Strategy:** `surge_reliance_score`, `premium_trip_ratio`, `is_premium_capable`
- **Commitment:** `peak_hour_driver_score`, `session_regularity`, `session_count`
- **Incentives:** `quest_completion_rate`, `incentive_reliance_pct`
- **Frustration:** `cancellation_rate`, `acceptance_rate`
- **Loyalty:** `pro_tier_rank`

### Phase 2: Unsupervised Learning (K-Means Clustering)

- **Algorithm:** K-Means (k=4, validated via Elbow Method)
- **Feature Selection:** 6 features used; `acceptance_rate` dropped (96% correlated with `trip_utilization_rate`)
- **Preprocessing:** StandardScaler normalization
- **Output:** 4 behaviorally distinct driver clusters

### Phase 3: Supervised Learning (Churn Prediction)

- **Algorithm:** XGBoost Classifier
- **Tuning:** RandomizedSearchCV (50 iterations, 5-fold StratifiedKFold, ROC AUC scoring)
- **Class Imbalance:** Handled via `scale_pos_weight` (churn rate ~30%)
- **Performance:** 0.61 Recall on churners (vs. 0.50 baseline)
- **Tracking:** MLflow experiment logging

---

## Project Structure

```
uber-cohort-analysis/
├── data/
│   ├── raw/                        # Synthetic source logs
│   │   ├── profile_data.csv        # 581 drivers, includes churn label
│   │   ├── activity_logs.csv       # 59K driving sessions
│   │   ├── trip_logs.csv           # 522K trips
│   │   ├── incentive_logs.csv      # 15K Quest/incentive records
│   │   └── interaction_logs.csv    # 1.2M app events
│   └── processed/
│       ├── training_data.csv       # Engineered features (581 × 18)
│       └── training_data_with_clusters.csv
├── notebooks/
│   ├── 01_data_validation.ipynb              # Data integrity & null checks
│   ├── 02_eda_post_feature_engineering.ipynb # Distributions, correlations, outliers
│   └── 03_model_analysis_and_storytelling.ipynb  # Personas, importance, visuals
├── scripts/
│   ├── 01_feature_engineering.py   # DuckDB SQL transformation pipeline
│   ├── 02_cluster_model.py         # K-Means training & artifact saving
│   └── 03_prediction_model.py      # XGBoost training, tuning, MLflow logging
├── models/
│   ├── kmeans_model.joblib
│   ├── scaler.joblib
│   ├── churn_model.json
│   └── churn_model_optimized.json
├── requirements.txt
└── README.md
```

---

## How to Run

**Prerequisites:** Python 3.10+, install dependencies with `pip install -r requirements.txt`

Run the pipeline in order:

```bash
# Step 1: Engineer features from raw logs
python scripts/01_feature_engineering.py

# Step 2: Train clustering model and assign driver personas
python scripts/02_cluster_model.py

# Step 3: Train and optimize the churn prediction model
python scripts/03_prediction_model.py
```

Then explore the notebooks in order for analysis and storytelling.

To view MLflow experiment results:
```bash
mlflow ui
```

---

## Data

All data is **synthetically generated** using Python's `Faker` library to simulate realistic Uber driver behavior patterns. No real driver data is used.

**Scale:** 581 drivers, ~59K sessions, ~523K trips, ~1.2M app events.
