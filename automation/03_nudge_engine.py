"""
Gamified CRM Nudge Engine.

Generates personalised HTML email nudges for Casual / At-Risk drivers.
Each nudge is tailored to the driver's primary weakness signal using two
behavioural-economics principles:

  Goal-Gradient Effect  — progress bars showing how close they are to the
                          next Uber Pro tier or a weekly earnings milestone
  Social Proof          — "Pro-Optimizer drivers in your city earn X/hr"
                          benchmarks to make the gap feel bridgeable
  Loss Framing          — where appropriate, surfaces what they are leaving
                          on the table vs. a concrete fix

Three nudge variants (selected by the driver's worst-performing signal, in priority order
matching XGBoost feature importance — earnings is #1 predictor at 0.24):
  A. LOW_EARNINGS  — avg_earnings_per_hour_online < 15  (primary: #1 churn driver)
  B. HIGH_CANCEL   — cancellation_rate > 0.15           (secondary: #5 churn driver)
  C. GOAL_GRADIENT — default / near-tier upgrade (most motivating when economics are borderline)
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import date
from string import Template

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from automation.config import (
    CASUAL_PERSONA_NAME, TRAINING_DATA_PATH
)

# ---------------------------------------------------------------------------
# Benchmark statistics  (derived once from training data at import time)
# ---------------------------------------------------------------------------

def _load_benchmarks() -> dict:
    """Pull Pro-Optimizer median stats to use as social-proof benchmarks."""
    try:
        df = pd.read_csv(TRAINING_DATA_PATH)
        pro = df[df.get('persona', df.get('cluster_label', pd.Series([]))) == 3]
        if pro.empty:
            # Fall back to top quartile by earnings
            pro = df[df['avg_earnings_per_hour_online'] >= df['avg_earnings_per_hour_online'].quantile(0.75)]
        return {
            'pro_epho_median':    round(pro['avg_earnings_per_hour_online'].median(), 2),
            'pro_util_median':    round(pro['trip_utilization_rate'].median() * 100, 1),
            'pro_cancel_median':  round(pro['cancellation_rate'].median() * 100, 1),
        }
    except Exception:
        return {'pro_epho_median': 28.0, 'pro_util_median': 55.0, 'pro_cancel_median': 3.0}


_BENCHMARKS = _load_benchmarks()


# ---------------------------------------------------------------------------
# Tier system
# ---------------------------------------------------------------------------

TIERS = {1: 'Blue', 2: 'Gold', 3: 'Platinum', 4: 'Diamond'}
TIER_THRESHOLDS = {1: 0, 2: 100, 3: 250, 4: 500}   # Lifetime trips (simplified)

def _tier_progress_html(pro_tier_rank: int, session_count: int) -> str:
    """Render a CSS progress bar showing distance to next Uber Pro tier."""
    current_trips = session_count  # Using session_count as a proxy for lifetime trips
    next_rank = min(pro_tier_rank + 1, 4)
    if pro_tier_rank == 4:
        return f"<p>🏆 You're already at <b>Diamond</b> — the top 1% of drivers.</p>"

    current_threshold = TIER_THRESHOLDS[pro_tier_rank]
    next_threshold    = TIER_THRESHOLDS[next_rank]
    trips_since_tier  = max(0, current_trips - current_threshold)
    trips_needed      = next_threshold - current_threshold
    pct               = min(int(trips_since_tier / trips_needed * 100), 100)

    return f"""
    <div style="margin:16px 0;">
      <p style="margin-bottom:6px; font-size:14px;">
        <b>{TIERS[pro_tier_rank]}</b> → <b>{TIERS[next_rank]}</b>
        &nbsp;({trips_needed - trips_since_tier} more trips to unlock)
      </p>
      <div style="background:#e0e0e0; border-radius:6px; height:18px; width:100%;">
        <div style="background:#1fb954; border-radius:6px; height:18px; width:{pct}%;
                    text-align:center; color:#fff; font-size:12px; line-height:18px;">
          {pct}%
        </div>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# Nudge variant selection
# ---------------------------------------------------------------------------

def _select_variant(row: pd.Series) -> str:
    """
    Pick the most relevant nudge based on the driver's worst signal.
    Priority order mirrors XGBoost feature importance:
      1. LOW_EARNINGS  — earnings efficiency is the #1 churn predictor (importance 0.24)
      2. HIGH_CANCEL   — cancellation rate is a compounding secondary signal (#5 at 0.10)
      3. GOAL_GRADIENT — default for drivers who are borderline on both metrics
    """
    if row.get('avg_earnings_per_hour_online', 99) < 15:
        return 'LOW_EARNINGS'
    if row.get('cancellation_rate', 0) > 0.15:
        return 'HIGH_CANCEL'
    return 'GOAL_GRADIENT'


# ---------------------------------------------------------------------------
# HTML email templates
# ---------------------------------------------------------------------------

_HEADER = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>$subject</title>
<style>
  body  { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background:#f4f4f4; margin:0; padding:0; }
  .wrap { max-width:600px; margin:24px auto; background:#fff;
          border-radius:10px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,.12); }
  .hdr  { background:#000; color:#fff; padding:24px 32px; }
  .hdr h1 { margin:0; font-size:22px; }
  .hdr p  { margin:4px 0 0; font-size:13px; color:#aaa; }
  .body { padding:28px 32px; color:#222; font-size:15px; line-height:1.6; }
  .stat { background:#f9f9f9; border-left:4px solid $accent;
          border-radius:4px; padding:12px 16px; margin:16px 0; }
  .cta  { display:inline-block; margin-top:20px; padding:12px 28px;
          background:$accent; color:#fff; border-radius:6px;
          text-decoration:none; font-weight:600; font-size:15px; }
  .ftr  { background:#f4f4f4; text-align:center; padding:16px;
          font-size:12px; color:#999; }
</style>
</head>
<body><div class="wrap">
<div class="hdr">
  <h1>$title</h1>
  <p>$subtitle</p>
</div>
<div class="body">
"""

_FOOTER = """
</div>
<div class="ftr">
  This message was sent to you as part of the Uber Pro retention programme.<br>
  <a href="#" style="color:#999;">Unsubscribe</a>
</div>
</div></body></html>
"""


def _build_high_cancel(row: pd.Series) -> tuple[str, str]:
    """Nudge A — address high post-acceptance cancellation rate."""
    cancel_pct = round(row.get('cancellation_rate', 0) * 100, 1)
    subject = "Your cancellation rate is affecting your Pro status"

    body_html = f"""
    <p>Hi there,</p>
    <p>We noticed something worth a quick look: your post-acceptance cancellation
       rate is currently <b>{cancel_pct}%</b>. The average for top-tier
       Pro-Optimizer drivers in your city is just
       <b>{_BENCHMARKS['pro_cancel_median']}%</b>.</p>
    <div class="stat">
      <b>Why it matters:</b> Each cancelled trip after acceptance reduces your
      Acceptance Score — which directly affects your Uber Pro tier and access to
      higher-demand dispatch zones.
    </div>
    <p><b>3 quick fixes drivers use:</b></p>
    <ol>
      <li>Check the destination before accepting — use Destination Mode to avoid
          routes that don't work for you.</li>
      <li>Review peak-hour heatmaps before going online — positioning yourself
          near demand means the trips you get are worth taking.</li>
      <li>If a trip looks problematic, decline it at dispatch rather than
          post-acceptance — a lower acceptance rate is less costly than a
          higher cancellation rate.</li>
    </ol>
    {_tier_progress_html(int(row.get('pro_tier_rank', 1)),
                         int(row.get('session_count', 10)))}
    <a class="cta" href="#">View My Driver Dashboard</a>
    """
    return subject, body_html


def _build_low_earnings(row: pd.Series) -> tuple[str, str]:
    """Nudge B — address low earnings per hour online."""
    epho     = round(row.get('avg_earnings_per_hour_online', 0), 2)
    gap      = round(_BENCHMARKS['pro_epho_median'] - epho, 2)
    util_pct = round(row.get('trip_utilization_rate', 0) * 100, 1)

    subject = f"You're earning ${epho}/hr — here's how top drivers earn ${_BENCHMARKS['pro_epho_median']}/hr"

    body_html = f"""
    <p>Hi there,</p>
    <p>Right now you're averaging <b>${epho} per hour online</b>. Pro-Optimizer
       drivers — the top earning segment — average
       <b>${_BENCHMARKS['pro_epho_median']}/hr</b>.
       That's a <b>${gap}/hr gap</b> — worth hundreds of dollars a month.</p>
    <div class="stat">
      <b>Your trip utilization rate: {util_pct}%</b><br>
      Top earners average {_BENCHMARKS['pro_util_median']}%. Every percentage
      point here is money you're leaving behind while sitting idle.
    </div>
    <p><b>The playbook top drivers use:</b></p>
    <ul>
      <li>🗺️ <b>Position before demand:</b> Open the heatmap 15 minutes before
          commute windows (7–9am, 4–7pm weekdays) and move toward the yellow/red
          zones before trips start flowing.</li>
      <li>⚡ <b>Chase surge strategically:</b> Drivers with high surge reliance
          earn 20–40% premiums. Surge zones are predictable — events, airports,
          Friday nights.</li>
      <li>✅ <b>Complete your Quests:</b> Your quest completion rate is
          {round(row.get('quest_completion_rate', 0) * 100, 1)}%. Each completed
          Quest adds a bonus directly on top of organic earnings.</li>
    </ul>
    {_tier_progress_html(int(row.get('pro_tier_rank', 1)),
                         int(row.get('session_count', 10)))}
    <a class="cta" href="#">See Earnings Breakdown</a>
    """
    return subject, body_html


def _build_goal_gradient(row: pd.Series) -> tuple[str, str]:
    """Nudge C — goal-gradient + social proof for drivers near a milestone."""
    tier_name = TIERS.get(int(row.get('pro_tier_rank', 1)), 'Blue')
    epho      = round(row.get('avg_earnings_per_hour_online', 0), 2)
    subject   = f"You're closer to {TIERS.get(int(row.get('pro_tier_rank', 1)) + 1, 'Diamond')} than you think"

    body_html = f"""
    <p>Hi there,</p>
    <p>You're currently at <b>Uber Pro {tier_name}</b>. Moving to the next tier
       unlocks higher-priority dispatch, cashback on fuel and EV charging,
       and exclusive Quest bonuses.</p>
    {_tier_progress_html(int(row.get('pro_tier_rank', 1)),
                         int(row.get('session_count', 10)))}
    <div class="stat">
      <b>What drivers at the next tier earn:</b> Pro-Optimizer and Platinum/Diamond
      drivers in your market average <b>${_BENCHMARKS['pro_epho_median']}/hr</b>
      vs. your current <b>${epho}/hr</b>.
      The difference is mostly strategy, not hours.
    </div>
    <p><b>Your fastest path forward:</b></p>
    <ul>
      <li>🕐 Drive during <b>peak windows</b> — Mon–Thu 7–10am and 4–7pm,
          Friday evenings, and Sat/Sun late night. These windows offer 2–3×
          the trip density.</li>
      <li>✅ <b>Complete your active Quest</b> — bonus trips count toward your
          Pro tier progress AND add guaranteed income on top of fares.</li>
      <li>📍 <b>Reduce cancellations</b> — your current rate is
          {round(row.get('cancellation_rate', 0) * 100, 1)}%.
          Even cutting it in half will improve your tier score within a week.</li>
    </ul>
    <a class="cta" href="#">View My Progress</a>
    """
    return subject, body_html


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_VARIANT_BUILDERS = {
    'HIGH_CANCEL':   _build_high_cancel,
    'LOW_EARNINGS':  _build_low_earnings,
    'GOAL_GRADIENT': _build_goal_gradient,
}

_ACCENT_COLORS = {
    'HIGH_CANCEL':   '#e74c3c',
    'LOW_EARNINGS':  '#f39c12',
    'GOAL_GRADIENT': '#1fb954',
}


def generate_nudge(row: pd.Series) -> dict:
    """
    Generate a full nudge record for a single driver row.

    Returns:
        {
            'driver_id':   str,
            'variant':     str  ('HIGH_CANCEL' | 'LOW_EARNINGS' | 'GOAL_GRADIENT'),
            'subject':     str,
            'html':        str  (full HTML email),
            'generated_at': str (ISO date),
        }
    """
    variant = _select_variant(row)
    builder = _VARIANT_BUILDERS[variant]
    accent  = _ACCENT_COLORS[variant]

    subject, body_html = builder(row)

    # Build full HTML
    header = Template(_HEADER).safe_substitute(
        subject=subject,
        accent=accent,
        title="Uber Pro Driver Insights",
        subtitle=f"Week of {date.today().strftime('%B %d, %Y')}",
    )
    full_html = header + body_html + _FOOTER

    return {
        'driver_id':    row.get('driver_id', 'unknown'),
        'variant':      variant,
        'subject':      subject,
        'html':         full_html,
        'generated_at': date.today().isoformat(),
    }


def generate_nudges_for_cohort(df: pd.DataFrame) -> list[dict]:
    """
    Generate nudges for all Casual/At-Risk high-risk drivers in a scored cohort.
    Applies A/B split: only the 'nudge' group receives emails (ab_group == 'nudge').
    """
    target = df[
        (df['persona'] == CASUAL_PERSONA_NAME) &
        (df['is_high_risk'] == True) &
        (df.get('ab_group', pd.Series(['nudge'] * len(df))) == 'nudge')
    ]
    return [generate_nudge(row) for _, row in target.iterrows()]


if __name__ == "__main__":
    import importlib
    from automation.config import TRAINING_DATA_PATH
    sp     = importlib.import_module("automation.02_scoring_pipeline")
    df     = pd.read_csv(TRAINING_DATA_PATH)
    scored = sp.score_drivers(df)
    casual_at_risk = scored[
        (scored['persona'] == CASUAL_PERSONA_NAME) & scored['is_high_risk']
    ]
    if not casual_at_risk.empty:
        sample = casual_at_risk.iloc[0]
        nudge = generate_nudge(sample)
        print(f"Variant: {nudge['variant']}")
        print(f"Subject: {nudge['subject']}")
        print("\n--- HTML Preview (first 500 chars) ---")
        print(nudge['html'][:500])
