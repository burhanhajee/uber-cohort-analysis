"""
Weekly Operator Report Builder.

Generates an HTML ops report summarising:
  - Pipeline run metadata (date, cohort size)
  - Churn risk distribution across personas
  - A/B test status for Casual driver nudge campaign
  - Top 10 highest-risk drivers (driver_id + churn probability)
  - Nudges dispatched this week vs. control group size
"""
import pandas as pd
import numpy as np
from datetime import date
from string import Template
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from automation.config import CASUAL_PERSONA_NAME


# ---------------------------------------------------------------------------
# HTML scaffolding
# ---------------------------------------------------------------------------

_BASE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Uber Pro — Weekly Ops Report</title>
<style>
  body  { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background:#f0f0f0; margin:0; padding:24px; color:#222; }
  .wrap { max-width:700px; margin:0 auto; }
  .card { background:#fff; border-radius:10px; padding:24px 28px;
          margin-bottom:20px; box-shadow:0 2px 6px rgba(0,0,0,.10); }
  h1    { font-size:24px; margin:0 0 4px; }
  h2    { font-size:16px; margin:0 0 14px; color:#555; font-weight:500; }
  h3    { font-size:15px; margin:16px 0 8px; }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  th    { background:#f4f4f4; text-align:left; padding:8px 12px;
          border-bottom:2px solid #ddd; font-weight:600; }
  td    { padding:8px 12px; border-bottom:1px solid #eee; }
  tr:last-child td { border-bottom:none; }
  .badge { display:inline-block; padding:2px 8px; border-radius:12px;
           font-size:12px; font-weight:600; }
  .red    { background:#fdecea; color:#c0392b; }
  .orange { background:#fef3e2; color:#d35400; }
  .green  { background:#eafaf1; color:#1e8449; }
  .blue   { background:#eaf0fb; color:#1a5276; }
  .gray   { background:#f4f4f4; color:#555; }
  .stat-row { display:flex; gap:16px; flex-wrap:wrap; }
  .stat-box { flex:1; min-width:120px; background:#f9f9f9; border-radius:8px;
              padding:14px; text-align:center; }
  .stat-box .num { font-size:28px; font-weight:700; }
  .stat-box .lbl { font-size:12px; color:#888; margin-top:2px; }
  .bar-wrap { margin:6px 0; }
  .bar-bg   { background:#e0e0e0; border-radius:4px; height:14px; width:100%; }
  .bar-fg   { border-radius:4px; height:14px; }
  .hdr { background:#000; color:#fff; padding:24px 28px; border-radius:10px;
         margin-bottom:20px; }
  .hdr h1 { color:#fff; }
  .hdr p  { color:#aaa; margin:4px 0 0; font-size:13px; }
</style>
</head>
<body><div class="wrap">
$content
</div></body></html>"""


def _badge(text: str, color: str) -> str:
    return f'<span class="badge {color}">{text}</span>'


def _bar(value: float, color: str = "#1fb954") -> str:
    pct = min(int(value * 100), 100)
    return (f'<div class="bar-wrap"><div class="bar-bg">'
            f'<div class="bar-fg" style="width:{pct}%;background:{color};"></div>'
            f'</div></div>')


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _header_section(run_date: str, cohort_size: int) -> str:
    return f"""
    <div class="hdr">
      <h1>Uber Pro — Weekly Ops Report</h1>
      <p>Week of {run_date} &nbsp;|&nbsp; Cohort: {cohort_size} drivers scored</p>
    </div>"""


def _summary_cards(df: pd.DataFrame) -> str:
    total      = len(df)
    high_risk  = int(df['is_high_risk'].sum())
    nudge_sent = int(df.get('nudge_sent', pd.Series([False] * total)).sum())
    casual_n   = int((df['persona'] == CASUAL_PERSONA_NAME).sum())

    return f"""
    <div class="card">
      <h2>Pipeline Summary</h2>
      <div class="stat-row">
        <div class="stat-box">
          <div class="num">{total}</div>
          <div class="lbl">Drivers Scored</div>
        </div>
        <div class="stat-box">
          <div class="num" style="color:#c0392b;">{high_risk}</div>
          <div class="lbl">High-Risk Flags</div>
        </div>
        <div class="stat-box">
          <div class="num" style="color:#d35400;">{casual_n}</div>
          <div class="lbl">Casual / At-Risk</div>
        </div>
        <div class="stat-box">
          <div class="num" style="color:#1fb954;">{nudge_sent}</div>
          <div class="lbl">Nudges Sent</div>
        </div>
      </div>
    </div>"""


def _persona_breakdown(df: pd.DataFrame) -> str:
    persona_colors = {
        "Casual / At-Risk":   ("#e74c3c", "red"),
        "Quest Grinder":      ("#3498db", "blue"),
        "Premium Specialist": ("#9b59b6", "blue"),
        "Pro-Optimizer":      ("#1fb954", "green"),
    }

    rows = ""
    by_persona = df.groupby('persona').agg(
        n=('driver_id', 'count'),
        churn_mean=('churn_probability', 'mean'),
        high_risk=('is_high_risk', 'sum'),
    ).reset_index().sort_values('churn_mean', ascending=False)

    for _, r in by_persona.iterrows():
        bar_color, badge_color = persona_colors.get(r['persona'], ("#888", "gray"))
        rows += f"""
        <tr>
          <td>{_badge(r['persona'], badge_color)}</td>
          <td>{int(r['n'])}</td>
          <td>
            {_bar(r['churn_mean'], bar_color)}
            <small>{r['churn_mean']:.1%}</small>
          </td>
          <td>{int(r['high_risk'])}</td>
        </tr>"""

    return f"""
    <div class="card">
      <h2>Churn Risk by Persona</h2>
      <table>
        <tr><th>Persona</th><th>Count</th><th>Avg Churn Risk</th><th>High-Risk</th></tr>
        {rows}
      </table>
    </div>"""


def _ab_test_section(df: pd.DataFrame) -> str:
    if 'ab_group' not in df.columns:
        return ""

    casual = df[df['persona'] == CASUAL_PERSONA_NAME]
    nudge_n   = int((casual['ab_group'] == 'nudge').sum())
    control_n = int((casual['ab_group'] == 'control').sum())

    return f"""
    <div class="card">
      <h2>A/B Test — Casual Driver Re-Engagement Campaign</h2>
      <p>Casual / At-Risk drivers flagged as high-risk are randomly split:</p>
      <table>
        <tr><th>Group</th><th>Size</th><th>Action</th></tr>
        <tr>
          <td>{_badge("Nudge", "orange")}</td>
          <td>{nudge_n}</td>
          <td>Received personalised CRM email this week</td>
        </tr>
        <tr>
          <td>{_badge("Control", "gray")}</td>
          <td>{control_n}</td>
          <td>No intervention — baseline churn rate tracking</td>
        </tr>
      </table>
      <p style="font-size:13px;color:#888;margin-top:12px;">
        Compare 30-day retention between groups to measure nudge lift.
      </p>
    </div>"""


def _top_risk_drivers(df: pd.DataFrame, n: int = 10) -> str:
    top = (df[df['is_high_risk']]
           .sort_values('churn_probability', ascending=False)
           .head(n))

    if top.empty:
        return ""

    rows = ""
    for _, r in top.iterrows():
        short_id = str(r['driver_id'])[:8] + "…"
        risk_pct = f"{r['churn_probability']:.1%}"
        persona_badge_colors = {
            "Casual / At-Risk": "red", "Quest Grinder": "blue",
            "Premium Specialist": "blue", "Pro-Optimizer": "green",
        }
        badge_color = persona_badge_colors.get(r.get('persona', ''), "gray")
        rows += f"""
        <tr>
          <td><code>{short_id}</code></td>
          <td>{_badge(r.get('persona', 'Unknown'), badge_color)}</td>
          <td><b style="color:#c0392b;">{risk_pct}</b></td>
        </tr>"""

    return f"""
    <div class="card">
      <h2>Top {n} Highest-Risk Drivers</h2>
      <table>
        <tr><th>Driver ID</th><th>Persona</th><th>Churn Probability</th></tr>
        {rows}
      </table>
    </div>"""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_ops_report(df_scored: pd.DataFrame,
                     run_date: str | None = None) -> str:
    """
    Build a full HTML weekly ops report from a scored driver DataFrame.

    Expected columns: driver_id, persona, is_high_risk, churn_probability,
                      [optional] ab_group, nudge_sent.
    Returns: HTML string.
    """
    run_date = run_date or date.today().strftime("%B %d, %Y")

    content = (
        _header_section(run_date, len(df_scored))
        + _summary_cards(df_scored)
        + _persona_breakdown(df_scored)
        + _ab_test_section(df_scored)
        + _top_risk_drivers(df_scored)
    )

    return Template(_BASE).safe_substitute(content=content)


if __name__ == "__main__":
    from automation.config import TRAINING_DATA_PATH
    import importlib
    sp = importlib.import_module("automation.02_scoring_pipeline")
    df = pd.read_csv(TRAINING_DATA_PATH)
    scored = sp.score_drivers(df)

    # Simulate A/B split
    rng   = np.random.default_rng(42)
    casual_mask = scored['persona'] == CASUAL_PERSONA_NAME
    scored.loc[casual_mask, 'ab_group'] = rng.choice(
        ['nudge', 'control'], size=casual_mask.sum(), p=[0.5, 0.5]
    )
    scored['nudge_sent'] = (
        (scored['persona'] == CASUAL_PERSONA_NAME) &
        (scored['is_high_risk']) &
        (scored.get('ab_group', 'nudge') == 'nudge')
    )

    html = build_ops_report(scored)
    out  = os.path.join(os.path.dirname(__file__), "logs", "sample_report.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved → {out}")
