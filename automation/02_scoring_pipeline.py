"""
Driver Scoring Pipeline.

Loads saved model artifacts (scaler, K-Means, XGBoost) and scores a
DataFrame of drivers, returning persona + churn probability for each.

Cluster Label Stability Fix (Appendix C.3):
    K-Means integer labels are arbitrary — label 0 in one run may not be
    "Casual" in another. This module derives persona names from the actual
    centroid feature profiles (inverse-transformed to original scale) using
    business-rule-based assignment. The mapping is re-derived every call so
    it remains correct even if the saved model is replaced with a re-fit.
"""
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from automation.config import (
    SCALER_PATH, KMEANS_PATH, CHURN_MODEL_PATH,
    CLUSTERING_FEATURES, MODEL_FEATURES, CHURN_RISK_THRESHOLD,
    PREMIUM_RATIO_THRESHOLD, QUEST_RATE_THRESHOLD, MIN_UTILIZATION_FOR_PRO,
    CASUAL_PERSONA_NAME,
)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_models():
    """Load all three model artifacts from disk."""
    if not all(os.path.exists(p) for p in [SCALER_PATH, KMEANS_PATH, CHURN_MODEL_PATH]):
        raise FileNotFoundError(
            "One or more model files not found. Run scripts 02 and 03 first."
        )
    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    churn_model = xgb.XGBClassifier(enable_categorical=True)
    churn_model.load_model(CHURN_MODEL_PATH)
    return scaler, kmeans, churn_model


# ---------------------------------------------------------------------------
# Cluster Label → Persona Name  (stability fix)
# ---------------------------------------------------------------------------

def build_persona_map(scaler, kmeans) -> dict[int, str]:
    """
    Derive a stable cluster_id → persona_name mapping by inverse-transforming
    K-Means cluster centroids and applying business-rule classification.

    Rules (applied in priority order):
      1. premium_trip_ratio > threshold  →  Premium Specialist
      2. surge above cluster mean AND utilization > floor  →  Pro-Optimizer
      3. quest_completion_rate > threshold  →  Quest Grinder
      4. otherwise  →  Casual / At-Risk
    """
    centroids_scaled   = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    df_c = pd.DataFrame(centroids_original, columns=CLUSTERING_FEATURES)

    mean_surge = df_c['surge_reliance_score'].mean()
    persona_map: dict[int, str] = {}

    for cluster_id, row in df_c.iterrows():
        if row['premium_trip_ratio'] > PREMIUM_RATIO_THRESHOLD:
            name = "Premium Specialist"
        elif (row['surge_reliance_score'] > mean_surge and
              row['trip_utilization_rate'] > MIN_UTILIZATION_FOR_PRO):
            name = "Pro-Optimizer"
        elif row['quest_completion_rate'] > QUEST_RATE_THRESHOLD:
            name = "Quest Grinder"
        else:
            name = CASUAL_PERSONA_NAME
        persona_map[int(cluster_id)] = name

    return persona_map


# ---------------------------------------------------------------------------
# Main Scoring Function
# ---------------------------------------------------------------------------

def score_drivers(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Score a DataFrame of driver feature vectors.

    Required input columns: all columns in CLUSTERING_FEATURES + acceptance_rate
                            + pro_tier_rank.
    Added output columns:
        cluster_label       — integer from K-Means
        persona             — stable human-readable name
        churn_probability   — XGBoost P(churn)
        is_high_risk        — bool, True if churn_probability >= threshold
    """
    scaler, kmeans, churn_model = load_models()
    persona_map = build_persona_map(scaler, kmeans)

    df = df_raw.copy()

    # Step 1: Scale clustering features
    X_cluster = df[CLUSTERING_FEATURES].fillna(0)
    X_scaled  = scaler.transform(X_cluster)

    # Step 2: Assign cluster labels and stable persona names
    df['cluster_label'] = kmeans.predict(X_scaled)
    df['persona']       = df['cluster_label'].map(persona_map)

    # Step 3: Prepare feature matrix for churn model
    # Cast with explicit categories matching what was used during training (0–3).
    # XGBoost requires the category dtype to have the same codes in train and score.
    n_clusters = kmeans.n_clusters
    df['cluster_label'] = pd.Categorical(df['cluster_label'],
                                         categories=list(range(n_clusters)))

    missing = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing:
        raise KeyError(f"Input DataFrame is missing columns required by the churn model: {missing}")

    X_model = df[MODEL_FEATURES]

    # Step 4: Predict churn probability
    df['churn_probability'] = churn_model.predict_proba(X_model)[:, 1].round(4)
    df['is_high_risk']      = df['churn_probability'] >= CHURN_RISK_THRESHOLD

    # Restore cluster_label to int for downstream use
    df['cluster_label'] = df['cluster_label'].cat.codes

    return df


if __name__ == "__main__":
    from automation.config import TRAINING_DATA_PATH
    df = pd.read_csv(TRAINING_DATA_PATH)
    scored = score_drivers(df)
    print(scored[['driver_id', 'persona', 'churn_probability', 'is_high_risk']].head(10))
    print("\nPersona distribution:")
    print(scored['persona'].value_counts())
    print("\nHigh-risk count:", scored['is_high_risk'].sum())
