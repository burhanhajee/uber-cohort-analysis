"""
Page 1 — Business Report.

Interactive version of the full analytical report:
  1. Executive Summary
  2. The 4 Driver Personas  (radar chart)
  3. Churn Analysis         (feature importance + churn by persona)
  4. The Driver Maturity Gap
  5. Earnings Composition
  6. Strategic Action Plan
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Business Report", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df   = pd.read_csv(os.path.join(base, "data", "processed",
                                    "training_data_with_clusters.csv"))
    cluster_names = {
        0: "Casual / At-Risk",
        1: "Quest Grinder",
        2: "Premium Specialist",
        3: "Pro-Optimizer",
    }
    df['persona'] = df['cluster_label'].map(cluster_names)
    return df

@st.cache_resource
def load_churn_model():
    base  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(os.path.join(base, "models", "churn_model_optimized.json"))
    return model

df    = load_data()
model = load_churn_model()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("📊 Business Report")
st.markdown("### Uber Pro-Driver Blueprint: Segmentation & Churn Analysis")
st.markdown("---")

# ---------------------------------------------------------------------------
# Section 1 — Executive Summary
# ---------------------------------------------------------------------------

st.header("1. Executive Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Total Drivers", f"{len(df):,}")
col2.metric("Overall Churn Rate", f"{df['Churned'].mean():.1%}")
col3.metric("At-Risk Cluster Churn Rate", f"{df[df['persona']=='Casual / At-Risk']['Churned'].mean():.1%}")

st.markdown("""
**The Leaky Bucket:** Nearly a third of drivers churn within the observation period.
But churn is not evenly distributed — it concentrates in one segment.
The **Casual / At-Risk** persona churns at 4× the rate of the best-performing segments.

**The primary finding:** The #1 churn predictor is **earnings per hour online** (XGBoost
importance: 0.24) — nearly twice the weight of any other feature. Drivers who cannot
make the economics work leave. `trip_utilization_rate` and `quest_completion_rate` are
the next strongest signals. Retention spend on this group should target
**earnings efficiency coaching** — positioning, peak-hour strategy, and surge awareness.
""")

# ---------------------------------------------------------------------------
# Section 2 — The 4 Driver Personas (Radar Chart)
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("2. The 4 Driver Personas")

metrics = [
    'acceptance_rate', 'trip_utilization_rate', 'surge_reliance_score',
    'quest_completion_rate', 'cancellation_rate', 'avg_earnings_per_hour_online',
]
metric_labels = [
    'Acceptance Rate', 'Trip Utilization', 'Surge Reliance',
    'Quest Completion', 'Cancellation Rate', 'Avg Earnings/Hr',
]

cluster_means = df.groupby('cluster_label')[metrics].mean()
min_vals      = df[metrics].min()
max_vals      = df[metrics].max()
norm          = (cluster_means - min_vals) / (max_vals - min_vals)

persona_info = {
    0: ("Casual / At-Risk",    '#E74C3C', "Low everything. High cancellations. Prime churn candidate."),
    1: ("Quest Grinder",       '#3498DB', "Volume-driven. Lives for Quest bonuses. Moderate churn risk."),
    2: ("Premium Specialist",  '#9B59B6', "Cherry-picks UberBlack/Premier. Low volume, high fare quality."),
    3: ("Pro-Optimizer",       '#2ECC71', "Peak-hour, surge-aware. Treats driving like a business."),
}

fig_radar = go.Figure()
for cluster_id in [1, 2, 3, 0]:
    if cluster_id not in norm.index:
        continue
    r       = norm.loc[cluster_id].values.tolist() + [norm.loc[cluster_id].values[0]]
    theta   = metric_labels + [metric_labels[0]]
    name, color, _ = persona_info[cluster_id]
    lw      = 4 if cluster_id == 0 else 2

    fig_radar.add_trace(go.Scatterpolar(
        r=r, theta=theta, fill='toself', name=name,
        line=dict(color=color, width=lw), opacity=0.85,
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                        gridcolor="#E0E0E0"),
        angularaxis=dict(tickfont=dict(size=12)),
        bgcolor="white",
    ),
    legend=dict(orientation="v", x=1.05, y=0.5),
    title="Driver Persona Fingerprints (Normalised 0–1)",
    height=500,
)

st.plotly_chart(fig_radar, use_container_width=True)

# Persona summary table
summary_rows = []
for cid, (name, color, desc) in persona_info.items():
    sub    = df[df['cluster_label'] == cid]
    n      = len(sub)
    churn  = sub['Churned'].mean()
    summary_rows.append({"Persona": name, "Count": n,
                          "Churn Rate": f"{churn:.1%}", "Strategy": desc})
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Section 3 — Churn Analysis
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("3. Churn Analysis")

tab_importance, tab_churn_rate = st.tabs(["Feature Importance", "Churn Rate by Persona"])

with tab_importance:
    features = [
        'avg_earnings_per_hour_online', 'trip_utilization_rate', 'surge_reliance_score',
        'premium_trip_ratio', 'quest_completion_rate', 'cancellation_rate',
        'acceptance_rate', 'pro_tier_rank', 'cluster_label',
    ]
    df_model = df.copy()
    df_model['cluster_label'] = df_model['cluster_label'].astype('category')
    X = df_model[features]
    y = df_model['Churned']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    importance = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importance})\
              .sort_values('Importance', ascending=True)

    fig_fi = px.bar(
        fi_df, x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale='Viridis',
        title="Why Do Drivers Quit? (XGBoost Feature Importance)",
    )
    fig_fi.update_layout(coloraxis_showscale=False, height=420)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.info(
        "**Key finding:** `avg_earnings_per_hour_online` is the #1 churn predictor "
        "(importance: 0.24) — nearly 2× the next feature. Drivers churn primarily because "
        "the economics don't work for them. `cancellation_rate` and `trip_utilization_rate` "
        "are secondary compounding signals.",
        icon="💡",
    )

with tab_churn_rate:
    churn_by_persona = (df.groupby('persona')['Churned']
                          .mean().reset_index()
                          .sort_values('Churned', ascending=False))
    churn_by_persona.columns = ['Persona', 'Churn Rate']

    fig_churn = px.bar(
        churn_by_persona, x='Persona', y='Churn Rate',
        color='Churn Rate', color_continuous_scale='Reds',
        text=churn_by_persona['Churn Rate'].apply(lambda x: f"{x:.1%}"),
        title="Risk Analysis: Which Personas Are We Losing?",
        range_y=[0, 1],
    )
    fig_churn.update_traces(textposition='outside')
    fig_churn.update_layout(coloraxis_showscale=False, height=420)
    st.plotly_chart(fig_churn, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4 — Driver Maturity Gap
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("4. The Driver Maturity Gap")
st.markdown("""
Not all low earners are equally at risk. The chart below compares three personas
on the four dimensions that separate strategic from reactive drivers.
""")

target_clusters = [0, 1, 3]
df_gap = df[df['cluster_label'].isin(target_clusters)].copy()
gap_map = {0: 'Casual (The Rookie)', 3: 'Pro-Optimizer (The Strategist)',
           1: 'Quest Grinder (The Hustler)'}
df_gap['Persona'] = df_gap['cluster_label'].map(gap_map)

plot_metrics  = ['trip_utilization_rate', 'surge_reliance_score',
                 'quest_completion_rate', 'avg_earnings_per_hour_online']
friendly_lbls = ['Asset Utilization', 'Market Intelligence',
                 'Goal Achievement', 'Earnings Efficiency (normalized)']

df_gap['norm_earnings'] = df_gap['avg_earnings_per_hour_online'] / df_gap['avg_earnings_per_hour_online'].max()
plot_cols = ['trip_utilization_rate', 'surge_reliance_score', 'quest_completion_rate', 'norm_earnings']

melted = df_gap.melt(id_vars='Persona', value_vars=plot_cols,
                     var_name='Metric', value_name='Score')
melted['Metric'] = melted['Metric'].map(dict(zip(plot_cols, friendly_lbls)))

fig_gap = px.bar(
    melted.groupby(['Persona', 'Metric'])['Score'].mean().reset_index(),
    x='Metric', y='Score', color='Persona', barmode='group',
    color_discrete_map={
        'Casual (The Rookie)':          '#95A5A6',
        'Quest Grinder (The Hustler)':  '#E74C3C',
        'Pro-Optimizer (The Strategist)': '#27AE60',
    },
    title="The Driver Maturity Curve: Strategic Metrics",
)
fig_gap.update_layout(height=420, yaxis_title="Performance Score (0–1)")
st.plotly_chart(fig_gap, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 5 — Earnings Composition
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("5. Earnings Composition")
st.markdown("Quest Grinders are incentive-dependent — removing Quests would collapse their earnings.")

raw_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "raw")

@st.cache_data
def load_earnings_data():
    trips_path     = os.path.join(raw_path, "trip_logs.csv")
    incentive_path = os.path.join(raw_path, "incentive_logs.csv")
    if not (os.path.exists(trips_path) and os.path.exists(incentive_path)):
        return None
    trips     = pd.read_csv(trips_path)
    incentives = pd.read_csv(incentive_path)
    trips['organic'] = trips['fare'] + trips['tip']
    organic_df  = trips.groupby('driver_id')['organic'].sum().reset_index()
    inorganic_df = (incentives[incentives['status'] == 'completed']
                    .groupby('driver_id')['bonus_amount'].sum().reset_index())
    merged = (df[['driver_id', 'cluster_label', 'persona']]
              .merge(organic_df, on='driver_id', how='left')
              .merge(inorganic_df, on='driver_id', how='left')
              .fillna(0))
    merged['total']    = merged['organic'] + merged['bonus_amount']
    merged['pct_org']  = merged['organic']       / merged['total'].replace(0, np.nan) * 100
    merged['pct_inorg'] = merged['bonus_amount'] / merged['total'].replace(0, np.nan) * 100
    return merged

earn_df = load_earnings_data()

if earn_df is not None:
    earn_agg = (earn_df[earn_df['cluster_label'].isin([1, 3])]
                .groupby('persona')[['pct_org', 'pct_inorg']].mean()
                .reset_index()
                .melt(id_vars='persona', var_name='Component', value_name='Pct'))
    earn_agg['Component'] = earn_agg['Component'].map(
        {'pct_org': 'Organic (Fare + Tip)', 'pct_inorg': 'Inorganic (Quest Bonuses)'}
    )
    fig_earn = px.bar(
        earn_agg, x='persona', y='Pct', color='Component',
        color_discrete_map={'Organic (Fare + Tip)': '#27ae60',
                            'Inorganic (Quest Bonuses)': '#e74c3c'},
        title="Earnings Composition: Organic vs. Inorganic (%)",
        text=earn_agg['Pct'].apply(lambda x: f"{x:.1f}%"),
    )
    fig_earn.update_traces(textposition='inside', textfont_color='white')
    fig_earn.update_layout(height=420, yaxis_title="% of Total Earnings",
                           barmode='stack')
    st.plotly_chart(fig_earn, use_container_width=True)
else:
    st.warning("Raw trip/incentive log files not found in data/raw/ — "
               "earnings composition chart unavailable.")

# ---------------------------------------------------------------------------
# Section 6 — Strategic Action Plan
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("6. Strategic Action Plan")

st.markdown("""
#### Persona-Level Retention Strategy

| Persona | Primary Risk | Recommended Intervention |
|---|---|---|
| **Casual / At-Risk** | Low earning efficiency (primary), low utilization + high cancellations (secondary) | Earnings coaching: positioning, peak-hour strategy, surge awareness |
| **Quest Grinder** | Incentive dependency (not self-sustaining) | Gradual Quest difficulty ramping + organic earnings coaching |
| **Premium Specialist** | Low volume (vulnerable to demand shifts) | Guaranteed minimum dispatch windows for premium trip types |
| **Pro-Optimizer** | Near-zero churn — protect, not retain | Exclusive Diamond perks + peer recognition programmes |
""")

st.markdown("---")
st.subheader("Casual Driver Action Plan (The 30-Day Re-Engagement Sprint)")

st.markdown("""
The Casual / At-Risk segment accounts for **33% of all drivers** but **~60% churn**.
The intervention focuses on three mechanics:

**Week 1 — Re-onboarding**
- Trigger: driver flagged as high-risk (churn probability ≥ 0.5)
- Action: Send personalised nudge identifying their *specific* weak signal —
  the primary target is low EPHO (heatmap coaching, surge positioning, peak-hour windows);
  secondary targets are high cancellation rate and low utilization
- Channel: In-app push notification + email (gamified CRM via this automation system)

**Week 2–3 — Goal-Gradient Acceleration**
- Show progress bar toward next Uber Pro tier in every nudge
- Unlock a "micro-Quest" (10 trips, high bonus) visible *only* to this segment
- Social proof: "Pro-Optimizer drivers in your zone earn $X/hr — here's their 3-step approach"

**Week 4 — A/B Evaluation**
- Compare 30-day retention: nudge group vs. control group
- If nudge group retention lifts ≥5pp → scale campaign to all at-risk Casual drivers
- If not → test alternative nudge variant (cancel-rate focus vs. earnings focus)

> See **Page 3 (CRM Pipeline)** to run this campaign and preview the actual emails.
""")
