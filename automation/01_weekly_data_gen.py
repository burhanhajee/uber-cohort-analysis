"""
Weekly Synthetic Driver Cohort Generator.

Generates a new cohort of driver feature vectors each week, simulating
newly onboarded or re-scored drivers. Draws from distributions observed
in the training data rather than re-running the full raw-log pipeline.

Output: data/processed/weekly_cohort_YYYY-MM-DD.csv
"""
import pandas as pd
import numpy as np
from datetime import date
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from automation.config import PROCESSED_DATA_PATH, TRAINING_DATA_PATH, NEW_DRIVERS_PER_WEEK, RANDOM_SEED


def _fit_distributions(df: pd.DataFrame) -> dict:
    """
    Capture mean/std of each feature from training data so new cohorts
    stay within realistic bounds.
    """
    stats = {}
    continuous = [
        'avg_earnings_per_hour_online', 'trip_utilization_rate',
        'surge_reliance_score', 'premium_trip_ratio',
        'quest_completion_rate', 'cancellation_rate',
        'acceptance_rate', 'peak_hour_driver_score',
        'session_regularity', 'incentive_reliance_pct', 'avg_rating',
    ]
    for col in continuous:
        if col in df.columns:
            stats[col] = {'mean': df[col].mean(), 'std': df[col].std(),
                          'min': df[col].min(), 'max': df[col].max()}
    return stats


def generate_weekly_cohort(n: int = NEW_DRIVERS_PER_WEEK,
                            seed: int | None = RANDOM_SEED,
                            save: bool = True) -> pd.DataFrame:
    """
    Generate n synthetic driver records.

    Returns a DataFrame in the same schema as training_data_with_clusters.csv
    but WITHOUT cluster_label or Churned (those are assigned by the scoring pipeline).
    """
    rng = np.random.default_rng(seed)

    # Load training data to calibrate distributions
    ref = pd.read_csv(TRAINING_DATA_PATH)
    stats = _fit_distributions(ref)

    def _clipped_normal(col, size, low=0.0, high=1.0):
        m, s = stats[col]['mean'], stats[col]['std']
        return np.clip(rng.normal(m, s, size), low, high)

    n_drivers = n

    # --- Core behavioural features ---
    epho = np.clip(
        rng.lognormal(
            mean=np.log(stats['avg_earnings_per_hour_online']['mean']),
            sigma=0.5,
            size=n_drivers
        ), 3.0, 80.0
    )

    utilization    = _clipped_normal('trip_utilization_rate', n_drivers)
    surge          = _clipped_normal('surge_reliance_score', n_drivers)
    quest_rate     = _clipped_normal('quest_completion_rate', n_drivers)
    cancel_rate    = _clipped_normal('cancellation_rate', n_drivers)
    acceptance     = _clipped_normal('acceptance_rate', n_drivers)
    peak_score     = _clipped_normal('peak_hour_driver_score', n_drivers)
    incentive_pct  = _clipped_normal('incentive_reliance_pct', n_drivers)

    # Premium ratio: bimodal — most drivers ~0, some specialists ~0.8
    premium_specialist_mask = rng.random(n_drivers) < 0.15
    premium_ratio = np.where(
        premium_specialist_mask,
        np.clip(rng.normal(0.75, 0.12, n_drivers), 0.4, 1.0),
        np.clip(rng.exponential(0.04, n_drivers), 0.0, 0.35)
    )

    # Session regularity: right-skewed
    session_reg   = np.clip(rng.exponential(40, n_drivers), 0, 300)
    session_count = rng.integers(1, 200, n_drivers)

    # Pro tier: categorical 1–4
    tier_probs = [0.45, 0.30, 0.15, 0.10]   # Blue, Gold, Platinum, Diamond
    pro_tier   = rng.choice([1, 2, 3, 4], size=n_drivers, p=tier_probs)

    # Premium capability: correlated with premium_ratio
    is_premium_capable = (premium_ratio > 0.3).astype(int)

    # Ratings: realistic 4.0–5.0
    avg_rating = np.clip(rng.normal(4.75, 0.15, n_drivers), 4.0, 5.0)

    # Driver IDs  (uuid-style)
    import uuid
    driver_ids = [str(uuid.uuid4()) for _ in range(n_drivers)]

    today = date.today().isoformat()

    df = pd.DataFrame({
        'driver_id':                    driver_ids,
        'signup_date':                  today,
        'avg_rating':                   avg_rating.round(2),
        'is_premium_capable':           is_premium_capable,
        'pro_tier_rank':                pro_tier,
        'avg_earnings_per_hour_online': epho.round(4),
        'trip_utilization_rate':        utilization.round(4),
        'surge_reliance_score':         surge.round(4),
        'premium_trip_ratio':           premium_ratio.round(4),
        'quest_completion_rate':        quest_rate.round(4),
        'cancellation_rate':            cancel_rate.round(4),
        'acceptance_rate':              acceptance.round(4),
        'peak_hour_driver_score':       peak_score.round(4),
        'session_regularity':           session_reg.round(2),
        'session_count':                session_count,
        'incentive_reliance_pct':       incentive_pct.round(4),
        'week':                         today,
    })

    if save:
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        out_path = os.path.join(PROCESSED_DATA_PATH, f"weekly_cohort_{today}.csv")
        df.to_csv(out_path, index=False)
        print(f"Weekly cohort saved → {out_path}  ({len(df)} drivers)")

    return df


if __name__ == "__main__":
    cohort = generate_weekly_cohort()
    print(cohort[['driver_id', 'avg_earnings_per_hour_online',
                   'cancellation_rate', 'quest_completion_rate']].head())
