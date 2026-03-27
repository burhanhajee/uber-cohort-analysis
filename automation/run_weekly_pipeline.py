"""
Weekly Pipeline Orchestrator.

Runs all five automation steps in sequence:
  1. Generate a new synthetic driver cohort
  2. Score each driver  (persona + churn probability)
  3. Apply A/B split on Casual / At-Risk high-risk drivers
  4. Generate personalised nudge emails for the nudge group
  5. Send nudge emails + ops report via SMTP (Mailtrap by default)

Usage:
  python automation/run_weekly_pipeline.py

Scheduling:
  Windows Task Scheduler  — point at python.exe with this script as argument
  cron (Linux/Mac)        — 0 8 * * 1  python /path/to/run_weekly_pipeline.py
  GitHub Actions          — schedule: [{cron: '0 8 * * 1'}] with a workflow step
"""
import sys
import os
import numpy as np
import pandas as pd
import importlib
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automation.config import (
    CASUAL_PERSONA_NAME, AB_TEST_RATIO, RANDOM_SEED,
    PROCESSED_DATA_PATH,
)

# Dynamic imports so the numeric module names (01_, 02_, etc.) work
_gen    = importlib.import_module("automation.01_weekly_data_gen")
_score  = importlib.import_module("automation.02_scoring_pipeline")
_nudge  = importlib.import_module("automation.03_nudge_engine")
_report = importlib.import_module("automation.04_report_builder")
_disp   = importlib.import_module("automation.05_dispatcher")


def run_pipeline(verbose: bool = True) -> dict:
    """
    Execute the full weekly pipeline.

    Returns a summary dict with counts and the generated ops report HTML.
    """
    results = {}
    today   = date.today().isoformat()

    # ------------------------------------------------------------------
    # Step 1: Generate weekly cohort
    # ------------------------------------------------------------------
    if verbose: print("\n[1/5] Generating weekly driver cohort…")
    cohort = _gen.generate_weekly_cohort(save=True)
    results['cohort_size'] = len(cohort)
    if verbose: print(f"      {len(cohort)} new driver records generated.")

    # ------------------------------------------------------------------
    # Step 2: Score drivers
    # ------------------------------------------------------------------
    if verbose: print("[2/5] Scoring drivers (persona + churn probability)…")
    scored = _score.score_drivers(cohort)
    results['persona_counts'] = scored['persona'].value_counts().to_dict()
    results['high_risk_count'] = int(scored['is_high_risk'].sum())
    if verbose:
        print(f"      High-risk drivers flagged: {results['high_risk_count']}")
        print(f"      Persona breakdown:\n{scored['persona'].value_counts().to_string()}")

    # ------------------------------------------------------------------
    # Step 3: A/B split on Casual / At-Risk
    # ------------------------------------------------------------------
    if verbose: print("[3/5] Applying A/B split to Casual / At-Risk drivers…")
    rng          = np.random.default_rng(RANDOM_SEED)
    casual_mask  = (scored['persona'] == CASUAL_PERSONA_NAME) & scored['is_high_risk']
    scored.loc[casual_mask, 'ab_group'] = rng.choice(
        ['nudge', 'control'],
        size=int(casual_mask.sum()),
        p=[AB_TEST_RATIO, 1 - AB_TEST_RATIO],
    )
    nudge_n   = int((scored.get('ab_group', pd.Series([])) == 'nudge').sum())
    control_n = int((scored.get('ab_group', pd.Series([])) == 'control').sum())
    results['nudge_group_size']   = nudge_n
    results['control_group_size'] = control_n
    if verbose: print(f"      Nudge: {nudge_n} | Control: {control_n}")

    # ------------------------------------------------------------------
    # Step 4: Generate nudges
    # ------------------------------------------------------------------
    if verbose: print("[4/5] Generating personalised nudge emails…")
    nudges = _nudge.generate_nudges_for_cohort(scored)
    results['nudges_generated'] = len(nudges)
    if verbose: print(f"      {len(nudges)} nudges generated.")

    # ------------------------------------------------------------------
    # Step 5: Dispatch emails
    # ------------------------------------------------------------------
    if verbose: print("[5/5] Dispatching emails…")

    sent_count = 0
    for nudge_record in nudges:
        # In a real system, look up the driver's email from a CRM database.
        # Here we use a fake address so the email goes to Mailtrap.
        fake_email = f"{nudge_record['driver_id'][:8]}@driver.uber-demo.com"
        success    = _disp.send_driver_nudge(fake_email, nudge_record)
        if success:
            sent_count += 1

    # Mark nudge_sent flag on scored df (for the ops report)
    scored['nudge_sent'] = False
    if nudges:
        sent_ids = {n['driver_id'] for n in nudges}
        scored.loc[scored['driver_id'].isin(sent_ids), 'nudge_sent'] = True

    # Build and send ops report
    report_html = _report.build_ops_report(scored, run_date=today)
    _disp.send_ops_report(report_html)

    # Save scored cohort for audit trail
    out_path = os.path.join(PROCESSED_DATA_PATH, f"scored_cohort_{today}.csv")
    scored.to_csv(out_path, index=False)

    results['nudges_sent']   = sent_count
    results['report_html']   = report_html
    results['scored_df']     = scored
    results['nudge_records'] = nudges
    results['run_date']      = today

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Weekly pipeline complete — {today}")
        print(f"  Cohort: {results['cohort_size']} drivers")
        print(f"  High-risk: {results['high_risk_count']}")
        print(f"  Nudges sent: {sent_count}")
        print(f"{'='*55}\n")

    return results


if __name__ == "__main__":
    run_pipeline(verbose=True)
