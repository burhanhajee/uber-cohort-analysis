# 🚖 The Uber Pro-Driver Blueprint
### A Multi-Dimensional Segmentation & Churn Prediction Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Classification-orange)
![Sklearn](https://img.shields.io/badge/Sklearn-KMeans-yellow)
![DuckDB](https://img.shields.io/badge/DuckDB-SQL-lightgrey)

## 📉 The Business Case
**Problem:** High driver churn reduces network liquidity and increases acquisition costs. Standard retention strategies (generic "Do 50 trips" Quests) are inefficient because they treat all drivers the same.

**Objective:** Move beyond simple "hours online" metrics to identify distinct **Driver Personas** based on strategy, efficiency, and frustration signals. Use these personas to predict churn and tailor retention spend.

**Outcome:** Identified 4 distinct driver personas and a "Leaky Bucket" segment with a **60% churn rate**, primarily driven by cancellation frustration rather than low earnings.

---

## 📊 Key Insights & Visuals

### 1. The "Snake Plot": Unmasking the Personas
*We used K-Means Clustering (k=4) to segment drivers based on 6 behavioral features. The "Snake Plot" below standardizes these behaviors to show strategic differences.*


* **The Pro-Optimizer (Red):** High utilization, high surge reliance. They treat driving as a business. **Churn Risk: Low (7%).**
* **The Quest Grinder (Blue):** High volume, low efficiency. They chase incentives blindly. **Churn Risk: Medium (23%).**
* **The Premium Specialist (Green):** Low volume, extreme earnings/hour. They cherry-pick high-value trips. **Churn Risk: Low (5%).**
* **The Casual / At-Risk (Orange):** Low earnings, **High Cancellation Rate**. **Churn Risk: Critical (60%).**

### 2. The Churn Drivers: It's Not Just Money
*Using XGBoost feature importance, we discovered that frustration drives churn more than earnings.*


* **#1 Predictor:** `cancellation_rate`. Drivers who frequently cancel are signaling frustration with the dispatch system.
* **Strategic Recommendation:** Retention budget should not be spent on cash bonuses for "At-Risk" drivers. Instead, invest in **Re-Onboarding/Education** to reduce cancellations and frustration.

---

## 🛠️ Technical Methodology

### Phase 1: Feature Engineering (DuckDB)
Raw trip logs and activity timestamps were transformed into behavioral features using SQL-based window functions in **DuckDB**.
* **Metric Example:** `Earnings Per Hour Online (EPHO)` = Total Fare / (Session End - Session Start).
* **Metric Example:** `Surge Reliance Score` = % of earnings derived specifically from surge multipliers.

### Phase 2: Unsupervised Learning (Clustering)
* **Algorithm:** K-Means Clustering.
* **Validation:** Used the **Elbow Method** to determine optimal $k=4$.
* **Profiling:** Visualized standardized cluster centers (Snake Plot) to name the personas.

### Phase 3: Supervised Learning (Churn Prediction)
* **Algorithm:** XGBoost Classifier.
* **Optimization:** Addressed class imbalance (Churn ~30%) using `scale_pos_weight`. Tuned `max_depth` to 3 to prevent overfitting.
* **Performance:** Achieved **0.61 Recall** on Churners (capturing 61% of at-risk drivers) vs. a baseline of 0.50.

---

## 📂 Project Structure

```bash
uber-driver-blueprint/
├── data/
│   ├── raw/                   # Synthetic logs (Trips, Activities)
│   └── processed/             # SQL-engineered features (training_data.csv)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb         # Data Integrity & Distributions
│   ├── 02_feature_validation.ipynb           # Correlation & Hypothesis Testing
│   └── 03_model_analysis_and_storytelling.ipynb  # THE CORE ANALYSIS (Snake Plots, ROI)
├── scripts/
│   ├── 01_feature_engineering.py   # DuckDB transformation pipeline
│   ├── 02_train_cluster_model.py   # K-Means training & saving
│   └── 03_train_churn_model.py     # XGBoost training & evaluation
├── models/                    # Saved .joblib and .json artifacts
├── reports/                   # Generated plots (PNGs)
├── requirements.txt
└── README.md