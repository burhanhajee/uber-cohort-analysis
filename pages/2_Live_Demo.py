"""
Page 2 — Live Scoring Demo.

Visitors adjust sliders for a hypothetical driver and see:
  - Which persona they'd be assigned to
  - Their predicted churn probability (with a gauge chart)
  - Which CRM nudge variant they'd receive
  - The full rendered nudge email HTML
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Live Demo", page_icon="🎮", layout="wide")

# Dynamic imports for numerically-prefixed modules
_score = importlib.import_module("automation.02_scoring_pipeline")
_nudge = importlib.import_module("automation.03_nudge_engine")

# ---------------------------------------------------------------------------
# Persona visual config
# ---------------------------------------------------------------------------

PERSONA_CONFIG = {
    "Casual / At-Risk":   {"color": "#E74C3C", "icon": "⚠️",
                           "desc": "Low engagement, high cancellations. Prime churn candidate."},
    "Quest Grinder":      {"color": "#3498DB", "icon": "🏃",
                           "desc": "Volume-driven, incentive-dependent. Moderate churn risk."},
    "Premium Specialist": {"color": "#9B59B6", "icon": "💎",
                           "desc": "Cherry-picks premium trips. Low volume, high quality."},
    "Pro-Optimizer":      {"color": "#2ECC71", "icon": "🏆",
                           "desc": "Peak-hour, surge-aware. Treats driving like a business."},
}

# ---------------------------------------------------------------------------
# Gauge chart helper
# ---------------------------------------------------------------------------

def churn_gauge(prob: float) -> go.Figure:
    color = "#2ECC71" if prob < 0.35 else ("#F39C12" if prob < 0.55 else "#E74C3C")
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = prob * 100,
        title = {"text": "Churn Probability (%)"},
        gauge = {
            "axis":  {"range": [0, 100], "tickwidth": 1},
            "bar":   {"color": color},
            "steps": [
                {"range": [0,  35], "color": "#eafaf1"},
                {"range": [35, 55], "color": "#fef9e7"},
                {"range": [55, 100], "color": "#fdecea"},
            ],
            "threshold": {
                "line":  {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
        number = {"suffix": "%", "font": {"size": 36}},
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
    return fig

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.title("🎮 Live Scoring Demo")
st.markdown("Adjust the sliders to define a hypothetical driver — the trained model scores them instantly.")
st.markdown("---")

left, right = st.columns([1, 1])

# ---------------------------------------------------------------------------
# Left panel — input sliders
# ---------------------------------------------------------------------------

with left:
    st.subheader("Driver Feature Input")

    epho        = st.slider("Avg Earnings per Hour Online ($)",        3.0,  60.0, 12.0, 0.5,
                             help="Total (fare + tip) ÷ total hours online")
    utilization = st.slider("Trip Utilization Rate",                   0.05,  1.0, 0.35, 0.01,
                             help="Fraction of online time spent actively on trips")
    surge       = st.slider("Surge Reliance Score",                    0.0,   0.8, 0.08, 0.01,
                             help="Premium earned above base fare from surge (0.4 = 40% premium)")
    premium     = st.slider("Premium Trip Ratio",                      0.0,   1.0, 0.0,  0.01,
                             help="Fraction of trips that are UberBlack or Premier")
    quest_rate  = st.slider("Quest Completion Rate",                   0.0,   1.0, 0.3,  0.01,
                             help="Fraction of offered Quests completed")
    cancel_rate = st.slider("Cancellation Rate",                       0.0,   0.5, 0.12, 0.01,
                             help="Post-acceptance cancellation rate")
    accept_rate = st.slider("Acceptance Rate",                         0.1,   1.0, 0.65, 0.01,
                             help="Fraction of dispatched trips accepted")
    tier_rank   = st.select_slider("Uber Pro Tier",
                                   options=[1, 2, 3, 4],
                                   value=1,
                                   format_func=lambda x: {1:"Blue",2:"Gold",3:"Platinum",4:"Diamond"}[x])
    session_count = st.slider("Session Count (lifetime)",              1, 300, 25, 1,
                               help="Total driving sessions (proxy for experience)")

# ---------------------------------------------------------------------------
# Score the driver
# ---------------------------------------------------------------------------

driver_row = pd.DataFrame([{
    'driver_id':                    'demo-driver',
    'avg_earnings_per_hour_online': epho,
    'trip_utilization_rate':        utilization,
    'surge_reliance_score':         surge,
    'premium_trip_ratio':           premium,
    'quest_completion_rate':        quest_rate,
    'cancellation_rate':            cancel_rate,
    'acceptance_rate':              accept_rate,
    'pro_tier_rank':                tier_rank,
    'session_count':                session_count,
}])

try:
    scored = _score.score_drivers(driver_row)
    row    = scored.iloc[0]
    persona     = row['persona']
    churn_prob  = float(row['churn_probability'])
    is_at_risk  = bool(row['is_high_risk'])
    score_ok    = True
except Exception as e:
    score_ok = False
    st.error(f"Scoring error: {e}")

# ---------------------------------------------------------------------------
# Right panel — results
# ---------------------------------------------------------------------------

with right:
    if score_ok:
        cfg = PERSONA_CONFIG.get(persona, {"color": "#888", "icon": "❓", "desc": ""})

        st.subheader("Model Output")

        # Persona card
        st.markdown(
            f"""
            <div style="background:{cfg['color']}22; border-left:5px solid {cfg['color']};
                        border-radius:8px; padding:16px 20px; margin-bottom:16px;">
              <h3 style="margin:0; color:{cfg['color']};">{cfg['icon']} {persona}</h3>
              <p style="margin:6px 0 0; color:#444;">{cfg['desc']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Churn gauge
        st.plotly_chart(churn_gauge(churn_prob), use_container_width=True)

        if is_at_risk:
            st.error(f"⚠️ High-risk driver — churn probability: **{churn_prob:.1%}**")
        else:
            st.success(f"✅ Low risk — churn probability: **{churn_prob:.1%}**")

# ---------------------------------------------------------------------------
# CRM nudge preview (full width, below columns)
# ---------------------------------------------------------------------------

if score_ok and persona == "Casual / At-Risk":
    st.markdown("---")
    st.subheader("📬 CRM Nudge Preview")
    st.markdown("This driver would receive the following gamified re-engagement email:")

    nudge_row = scored.iloc[0].copy()
    nudge_row['session_count'] = session_count
    nudge_record = _nudge.generate_nudge(nudge_row)

    variant_colors = {
        'LOW_EARNINGS':  ("#f39c12", "Earnings per hour below threshold  [#1 churn signal]"),
        'HIGH_CANCEL':   ("#e74c3c", "Post-acceptance cancellation rate too high  [secondary]"),
        'GOAL_GRADIENT': ("#1fb954", "Goal-gradient: tier upgrade progress"),
    }
    vc, vdesc = variant_colors.get(nudge_record['variant'], ("#888", ""))

    col_v1, col_v2 = st.columns([1, 3])
    col_v1.markdown(
        f"<span style='background:{vc}22; color:{vc}; padding:4px 10px; "
        f"border-radius:12px; font-weight:600; font-size:13px;'>"
        f"{nudge_record['variant']}</span>",
        unsafe_allow_html=True,
    )
    col_v2.markdown(f"*{vdesc}*")

    st.markdown(f"**Subject:** {nudge_record['subject']}")

    with st.expander("View rendered email HTML", expanded=True):
        st.components.v1.html(nudge_record['html'], height=700, scrolling=True)

elif score_ok and persona != "Casual / At-Risk":
    st.markdown("---")
    st.info(
        f"This driver is a **{persona}** — outside the Casual / At-Risk CRM campaign target. "
        "No nudge would be sent. Nudges are targeted exclusively at the highest-risk persona.",
        icon="ℹ️",
    )
