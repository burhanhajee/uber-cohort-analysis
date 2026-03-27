"""
Page 3 — CRM Automation Pipeline.

Demonstrates the full weekly automation:
  1. Generate a synthetic weekly driver cohort
  2. Score all drivers (persona + churn probability)
  3. Apply A/B split on Casual / At-Risk high-risk drivers
  4. Preview gamified nudge emails per driver
  5. Send to Mailtrap sandbox (or log to file if SMTP not configured)
  6. Display the weekly ops HTML report
  7. Show delivery log

Email validation for portfolio reviewers:
  - Configure Mailtrap credentials in .streamlit/secrets.toml
  - Every "sent" email lands in the shared Mailtrap inbox — fully rendered,
    fully inspectable HTML. Share the inbox URL with recruiters/reviewers.
  - If SMTP is not configured, emails are previewed in-app only.
"""
import streamlit as st
import pandas as pd
import numpy as np
import importlib
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="CRM Pipeline", page_icon="📬", layout="wide")

# Dynamic imports
_gen    = importlib.import_module("automation.01_weekly_data_gen")
_score  = importlib.import_module("automation.02_scoring_pipeline")
_nudge  = importlib.import_module("automation.03_nudge_engine")
_report = importlib.import_module("automation.04_report_builder")
_disp   = importlib.import_module("automation.05_dispatcher")

from automation.config import (
    CASUAL_PERSONA_NAME, AB_TEST_RATIO, SMTP_USER, LOGS_DIR
)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("📬 CRM Automation Pipeline")
st.markdown(
    "Run the full weekly pipeline: generate a driver cohort, score everyone, "
    "preview nudge emails, and send them to an email sandbox you can open in your browser."
)

# SMTP status banner
if SMTP_USER:
    st.success("✅ SMTP configured — emails will be sent to Mailtrap sandbox.", icon="📧")
else:
    st.warning(
        "SMTP not configured. Emails will be previewed here but not sent. "
        "Add `SMTP_USER` and `SMTP_PASS` to `.streamlit/secrets.toml` to enable sending.",
        icon="⚙️",
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Step 1 — Generate cohort
# ---------------------------------------------------------------------------

st.subheader("Step 1 — Generate Weekly Driver Cohort")

col_n, col_seed = st.columns(2)
n_drivers = col_n.slider("Cohort size (drivers)", 10, 100, 30, 5)
use_seed  = col_seed.checkbox("Fixed seed (reproducible demo)", value=True)
seed      = 42 if use_seed else None

if st.button("🔄 Generate New Cohort", type="primary"):
    with st.spinner("Generating synthetic driver cohort…"):
        cohort = _gen.generate_weekly_cohort(n=n_drivers, seed=seed, save=False)
    st.session_state['cohort'] = cohort
    st.session_state['scored'] = None
    st.session_state['nudges'] = None
    st.success(f"Generated {len(cohort)} driver records.")

if 'cohort' not in st.session_state:
    st.session_state['cohort'] = None
if 'scored' not in st.session_state:
    st.session_state['scored'] = None
if 'nudges' not in st.session_state:
    st.session_state['nudges'] = None

if st.session_state['cohort'] is not None:
    with st.expander("Preview raw cohort (first 5 rows)"):
        st.dataframe(
            st.session_state['cohort'][
                ['driver_id', 'avg_earnings_per_hour_online',
                 'cancellation_rate', 'quest_completion_rate', 'pro_tier_rank']
            ].head(),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Step 2 — Score drivers
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Step 2 — Score Drivers")

if st.session_state['cohort'] is not None and st.button("📊 Score All Drivers"):
    with st.spinner("Running scoring pipeline…"):
        scored = _score.score_drivers(st.session_state['cohort'])

        # A/B split
        rng          = np.random.default_rng(seed)
        casual_mask  = (scored['persona'] == CASUAL_PERSONA_NAME) & scored['is_high_risk']
        if casual_mask.sum() > 0:
            scored.loc[casual_mask, 'ab_group'] = rng.choice(
                ['nudge', 'control'],
                size=int(casual_mask.sum()),
                p=[AB_TEST_RATIO, 1 - AB_TEST_RATIO],
            )
        else:
            scored['ab_group'] = np.nan

    st.session_state['scored'] = scored
    st.success("Scoring complete.")

if st.session_state['scored'] is not None:
    scored = st.session_state['scored']

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Scored", len(scored))
    col_m2.metric("High-Risk Flagged", int(scored['is_high_risk'].sum()))
    col_m3.metric("Casual / At-Risk", int((scored['persona'] == CASUAL_PERSONA_NAME).sum()))
    col_m4.metric("Nudge Group", int((scored.get('ab_group', pd.Series([])) == 'nudge').sum()))

    with st.expander("Full scored results table"):
        display_cols = ['driver_id', 'persona', 'churn_probability', 'is_high_risk', 'ab_group']
        display_cols = [c for c in display_cols if c in scored.columns]
        st.dataframe(
            scored[display_cols].sort_values('churn_probability', ascending=False),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Step 3 — Nudge preview
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Step 3 — Preview & Send Nudge Emails")

if st.session_state['scored'] is not None:
    scored = st.session_state['scored']

    # Identify nudge-group drivers
    nudge_mask = (
        (scored['persona'] == CASUAL_PERSONA_NAME) &
        (scored['is_high_risk']) &
        (scored.get('ab_group', pd.Series(['nudge'] * len(scored))) == 'nudge')
    )
    nudge_drivers = scored[nudge_mask]

    if nudge_drivers.empty:
        st.info("No Casual / At-Risk high-risk drivers in the nudge group. "
                "Try a larger cohort or regenerate.", icon="ℹ️")
    else:
        st.markdown(f"**{len(nudge_drivers)} drivers** in the nudge group:")

        driver_options = nudge_drivers['driver_id'].tolist()
        selected_id    = st.selectbox(
            "Select a driver to preview their nudge email",
            options=driver_options,
            format_func=lambda x: x[:16] + "…",
        )

        selected_row  = nudge_drivers[nudge_drivers['driver_id'] == selected_id].iloc[0]
        nudge_preview = _nudge.generate_nudge(selected_row)

        # Variant badge
        variant_meta = {
            'LOW_EARNINGS':  ("#f39c12", "Low Earnings Per Hour  [#1 churn signal]"),
            'HIGH_CANCEL':   ("#e74c3c", "High Cancellation Rate  [secondary signal]"),
            'GOAL_GRADIENT': ("#1fb954", "Tier Upgrade Goal-Gradient"),
        }
        v_color, v_desc = variant_meta.get(nudge_preview['variant'], ("#888", ""))

        col_badge, col_subj = st.columns([1, 4])
        col_badge.markdown(
            f"<span style='background:{v_color}22; color:{v_color}; "
            f"padding:4px 10px; border-radius:12px; font-weight:600; font-size:13px;'>"
            f"{nudge_preview['variant']}</span>",
            unsafe_allow_html=True,
        )
        col_subj.markdown(f"**Subject:** {nudge_preview['subject']}")

        with st.expander("📧 Rendered Email Preview", expanded=True):
            st.components.v1.html(nudge_preview['html'], height=680, scrolling=True)

        # Generate all nudges and send
        if st.button("🚀 Generate & Send All Nudge Emails"):
            with st.spinner("Generating nudges for all drivers in nudge group…"):
                all_nudges = _nudge.generate_nudges_for_cohort(scored)

            sent, skipped = 0, 0
            for nudge_r in all_nudges:
                fake_email = f"{nudge_r['driver_id'][:8]}@driver.uber-demo.com"
                if _disp.send_driver_nudge(fake_email, nudge_r):
                    sent += 1
                else:
                    skipped += 1

            st.session_state['nudges'] = all_nudges

            if sent > 0:
                st.success(f"✅ {sent} nudge emails sent to Mailtrap sandbox!")
            if skipped > 0:
                st.info(f"{skipped} nudges previewed only (SMTP not configured).")
else:
    st.info("Run Step 2 first to score the cohort.", icon="👆")

# ---------------------------------------------------------------------------
# Step 4 — Ops Report
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Step 4 — Weekly Ops Report")

if st.session_state['scored'] is not None:
    scored = st.session_state['scored']

    if 'nudges' in st.session_state and st.session_state['nudges']:
        sent_ids = {n['driver_id'] for n in st.session_state['nudges']}
        scored   = scored.copy()
        scored['nudge_sent'] = scored['driver_id'].isin(sent_ids)

    if st.button("📋 Generate Ops Report"):
        report_html = _report.build_ops_report(scored)

        with st.expander("📊 Rendered Ops Report", expanded=True):
            st.components.v1.html(report_html, height=900, scrolling=True)

        if st.button("📧 Send Ops Report to Operator Email"):
            ok = _disp.send_ops_report(report_html)
            if ok:
                st.success("Ops report sent!")
            else:
                st.info("Report previewed — SMTP not configured for sending.")
else:
    st.info("Run Steps 1 and 2 first.", icon="👆")

# ---------------------------------------------------------------------------
# Step 5 — Delivery Log
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Step 5 — Dispatch Log")

log_entries = _disp.load_dispatch_log()
if log_entries:
    log_df = pd.DataFrame(log_entries)
    st.dataframe(log_df, use_container_width=True)
else:
    st.info("No dispatches logged yet. Send some emails first.", icon="📋")

# ---------------------------------------------------------------------------
# Mailtrap setup instructions
# ---------------------------------------------------------------------------

with st.expander("⚙️ How to connect a real email sandbox (Mailtrap)"):
    st.markdown("""
**What is Mailtrap?**
Mailtrap is a free fake SMTP inbox. Emails sent to it are *captured, not delivered*.
You get a real email UI where you can open, inspect, and forward emails — ideal for demos.

**Setup (5 minutes):**
1. Sign up at [mailtrap.io](https://mailtrap.io) (free)
2. Go to **Inboxes → SMTP Settings**
3. Copy your **Username** and **Password**
4. Create `.streamlit/secrets.toml` in the project root:
```toml
SMTP_USER = "your-mailtrap-username"
SMTP_PASS = "your-mailtrap-password"
SMTP_HOST = "sandbox.smtp.mailtrap.io"
SMTP_PORT = "2525"
OPERATOR_EMAIL = "you@example.com"
```
5. Restart `streamlit run app.py`

**For portfolio reviewers:**
Share the Mailtrap inbox link with a read-only token so they can open every email
that was "sent" in the demo — zero real delivery risk.

**For Streamlit Cloud deployment:**
Add the same keys under **Settings → Secrets** in the Streamlit Cloud dashboard.
""")
