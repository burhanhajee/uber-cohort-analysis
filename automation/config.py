"""
Central configuration for the Uber Pro-Driver Automation Pipeline.

All paths, thresholds, and credentials live here.
Real SMTP credentials should be injected via environment variables —
never committed to source control.
"""
import os

# ---------------------------------------------------------------------------
# Model Artifact Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCALER_PATH         = os.path.join(BASE_DIR, "models", "scaler.joblib")
KMEANS_PATH         = os.path.join(BASE_DIR, "models", "kmeans_model.joblib")
CHURN_MODEL_PATH    = os.path.join(BASE_DIR, "models", "churn_model_optimized.json")
TRAINING_DATA_PATH  = os.path.join(BASE_DIR, "data", "processed", "training_data_with_clusters.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
LOGS_DIR            = os.path.join(BASE_DIR, "automation", "logs")

# ---------------------------------------------------------------------------
# Feature Definitions  (order must match model training)
# ---------------------------------------------------------------------------
CLUSTERING_FEATURES = [
    'avg_earnings_per_hour_online',
    'trip_utilization_rate',
    'surge_reliance_score',
    'premium_trip_ratio',
    'quest_completion_rate',
    'cancellation_rate',
]

MODEL_FEATURES = [
    'avg_earnings_per_hour_online',
    'trip_utilization_rate',
    'surge_reliance_score',
    'premium_trip_ratio',
    'quest_completion_rate',
    'cancellation_rate',
    'acceptance_rate',
    'pro_tier_rank',
    'cluster_label',
]

# ---------------------------------------------------------------------------
# Business Thresholds
# ---------------------------------------------------------------------------
CHURN_RISK_THRESHOLD = 0.5   # Probability >= this → flagged as high-risk
AB_TEST_RATIO        = 0.5   # Fraction of at-risk Casual drivers who receive a nudge
CASUAL_PERSONA_NAME  = "Casual / At-Risk"

# Persona identification thresholds (inverse-transformed centroid rules)
PREMIUM_RATIO_THRESHOLD  = 0.35   # premium_trip_ratio above this → Premium Specialist
QUEST_RATE_THRESHOLD     = 0.50   # quest_completion_rate above this → Quest Grinder
MIN_UTILIZATION_FOR_PRO  = 0.40   # trip_utilization_rate floor for Pro-Optimizer

# ---------------------------------------------------------------------------
# Email / SMTP  (Mailtrap sandbox by default for safe portfolio demos)
# ---------------------------------------------------------------------------
# Mailtrap free plan: sign up at https://mailtrap.io, copy inbox credentials below.
# For production, swap host/port/user/pass via environment variables.
SMTP_HOST  = os.getenv("SMTP_HOST",  "sandbox.smtp.mailtrap.io")
SMTP_PORT  = int(os.getenv("SMTP_PORT", "2525"))
SMTP_USER  = os.getenv("SMTP_USER",  "")   # Set in env or Streamlit secrets
SMTP_PASS  = os.getenv("SMTP_PASS",  "")   # Set in env or Streamlit secrets

SENDER_NAME    = "Uber Pro Ops"
SENDER_EMAIL   = "ops@uber-pro-demo.com"
OPERATOR_EMAIL = os.getenv("OPERATOR_EMAIL", "ops@uber-pro-demo.com")

# ---------------------------------------------------------------------------
# Weekly Pipeline
# ---------------------------------------------------------------------------
NEW_DRIVERS_PER_WEEK = 30    # Synthetic cohort size for weekly simulation
RANDOM_SEED          = None  # None = different cohort each run; int = reproducible
