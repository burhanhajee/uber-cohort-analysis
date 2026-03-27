"""
Uber Pro-Driver Blueprint — Streamlit Portfolio App.

Entry point for the multi-page Streamlit application.
Navigate between pages using the sidebar.

Pages:
  1. Business Report  — interactive version of the full analysis report
  2. Live Demo        — score a single driver in real-time
  3. CRM Pipeline     — run the weekly automation, preview nudge emails

Deployment:
  streamlit run app.py
  Or: push to GitHub and connect to https://share.streamlit.io (free).
"""
import streamlit as st

st.set_page_config(
    page_title  = "Uber Pro-Driver Blueprint",
    page_icon   = "🚗",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------

st.title("🚗 The Uber Pro-Driver Blueprint")
st.markdown("### A Multi-Dimensional Segmentation & Churn Prediction Engine")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Drivers Analysed", "581")
col2.metric("Raw Events Processed", "~1.8M")
col3.metric("At-Risk Cluster Churn Rate", "~60%")
col4.metric("Churn Model Recall", "0.61")

st.markdown("---")

st.markdown("""
## The Business Case

**Problem:** High driver churn reduces network liquidity and increases acquisition costs.
Standard retention strategies (generic *"Do 50 trips"* Quests) are inefficient because
they treat all drivers the same.

**Approach:**
1. Engineer **12 behavioural features** from 1.8M raw events using DuckDB SQL
2. Segment drivers into **4 distinct personas** using K-Means clustering
3. Predict which drivers are likely to churn using **XGBoost** (ROC AUC: ~0.78)
4. Automate **gamified CRM nudges** targeted at the highest-risk persona

**Key finding:** `avg_earnings_per_hour_online` is the **#1 churn predictor** (importance: 0.24).
Low earning efficiency is the primary churn driver — drivers who can't make the economics work leave.
`cancellation_rate` and `trip_utilization_rate` are secondary signals that compound the risk.
""")

st.markdown("---")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### The 4 Driver Personas")
    st.markdown("""
| Persona | Strategy | Churn Rate |
|---|---|---|
| **Pro-Optimizer** | High utilization + surge-chasing | ~7% |
| **Quest Grinder** | Volume + incentive-dependent | ~23% |
| **Premium Specialist** | Cherry-picks UberBlack / Premier | ~5% |
| **Casual / At-Risk** | Low earnings, high cancellations | ~60% |
""")

with col_b:
    st.markdown("### Tech Stack")
    st.markdown("""
| Layer | Technology |
|---|---|
| Feature Engineering | DuckDB (in-memory SQL) |
| Clustering | scikit-learn K-Means (k=4) |
| Churn Prediction | XGBoost + RandomizedSearchCV |
| Experiment Tracking | MLflow |
| CRM Automation | Python + SMTP (Mailtrap sandbox) |
| Portfolio App | Streamlit |
""")

st.markdown("---")

st.markdown("""
### Navigate this app

Use the **sidebar** (← left) to explore:

- **📊 Business Report** — Full interactive analysis: persona fingerprints, churn drivers,
  feature importance, driver maturity gap, and strategic recommendations
- **🎮 Live Demo** — Adjust driver behaviour sliders and see real-time persona assignment
  + churn probability from the trained model
- **📬 CRM Pipeline** — Generate a new synthetic driver cohort, run the full scoring
  pipeline, preview gamified nudge emails, and send them to a sandbox inbox you can open
  in your browser
""")

st.info(
    "**Portfolio note:** All data is synthetic (Faker library). "
    "No real driver information is used or stored. "
    "Email delivery uses Mailtrap sandbox — emails are inspectable but never delivered to real inboxes.",
    icon="ℹ️",
)
