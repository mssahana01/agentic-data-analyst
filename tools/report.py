"""
report.py — Report/Dashboard generator for the Agentic Data Analyst.

This is the "glue" tool: it runs cleaning -> EDA -> modeling in sequence
and produces ONE polished, self-contained HTML report a human can open
in a browser. This is the artifact that used to take a data analyst
hours to hand-build in Power BI/Excel.
"""

import os
import sys
import datetime
import pandas as pd
import plotly.express as px
import plotly.io as pio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.cleaning import clean_data
from tools.eda import run_eda
from tools.modeling import train_model

PLOTLY_JS = "cdn"  # loaded once, shared across all embedded charts


def _fig_to_div(fig) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def generate_report(raw_df: pd.DataFrame, target_column: str, output_path: str = "outputs/report.html") -> str:
    """
    Runs the full pipeline (clean -> EDA -> model) on `raw_df` and writes
    a single polished HTML report to `output_path`. Returns the path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cleaned_df, clean_report = clean_data(raw_df)
    eda_report = run_eda(cleaned_df, output_dir=os.path.join(os.path.dirname(output_path), "charts"))
    model, model_report, _ = train_model(cleaned_df, target_column=target_column)

    charts_html = []

    # Correlation heatmap
    numeric_cols = eda_report.numeric_columns
    if len(numeric_cols) >= 2:
        corr_fig = px.imshow(eda_report.correlations, text_auto=True, color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Correlation Heatmap")
        charts_html.append(_fig_to_div(corr_fig))

    # Feature importance
    if model_report.feature_importances:
        imp_series = pd.Series(model_report.feature_importances).sort_values(ascending=True).tail(10)
        imp_fig = px.bar(x=imp_series.values, y=imp_series.index, orientation="h",
                          title=f"Feature Importance — predicting {target_column}",
                          labels={"x": "importance", "y": "feature"})
        charts_html.append(_fig_to_div(imp_fig))

    # A couple of distribution charts
    for col in numeric_cols[:2]:
        dist_fig = px.histogram(cleaned_df, x=col, title=f"Distribution of {col}", marginal="box")
        charts_html.append(_fig_to_div(dist_fig))

    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Data Analysis Report</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 960px; margin: 40px auto;
          padding: 0 20px; color: #1a1a1a; background: #fafafa; }}
  h1 {{ font-size: 28px; margin-bottom: 0; }}
  .timestamp {{ color: #888; font-size: 13px; margin-top: 4px; }}
  .card {{ background: white; border: 1px solid #e5e5e5; border-radius: 10px; padding: 24px; margin: 20px 0; }}
  .card h2 {{ margin-top: 0; font-size: 18px; }}
  .metric-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .metric {{ background: #f3f4f6; border-radius: 8px; padding: 12px 18px; min-width: 120px; }}
  .metric .label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
  .metric .value {{ font-size: 22px; font-weight: 600; }}
  pre {{ white-space: pre-wrap; font-family: inherit; line-height: 1.6; }}
  .chart-wrap {{ margin: 12px 0; }}
</style>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
  <h1>Automated Data Analysis Report</h1>
  <div class="timestamp">Generated {generated_at}</div>

  <div class="card">
    <h2>1. Data Cleaning</h2>
    <pre>{clean_report.summary()}</pre>
  </div>

  <div class="card">
    <h2>2. Exploratory Analysis</h2>
    <pre>{eda_report.summary()}</pre>
  </div>

  <div class="card">
    <h2>3. Model Performance</h2>
    <div class="metric-row">
      {''.join(f'<div class="metric"><div class="label">{k}</div><div class="value">{v}</div></div>' for k, v in model_report.metrics.items())}
    </div>
    <pre style="margin-top:16px">{model_report.summary()}</pre>
  </div>

  <div class="card">
    <h2>4. Charts</h2>
    {''.join(f'<div class="chart-wrap">{c}</div>' for c in charts_html)}
  </div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    return output_path


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)
    n = 300
    quantity = rng.integers(1, 500, n).astype(float)
    region = rng.choice(["US", "EU", "APAC"], size=n)
    supplier = rng.choice(["Acme Corp", "Globex", "Initech"], size=n)
    region_effect = pd.Series(region).map({"US": 5, "EU": 0, "APAC": -3}).values
    unit_cost = 60 - 0.02 * quantity + region_effect + rng.normal(0, 4, n)

    df = pd.DataFrame({
        "supplier": supplier,
        "unit_cost": unit_cost,
        "quantity": quantity,
        "region": region,
    })

    path = generate_report(df, target_column="unit_cost")
    print(f"Report written to: {path}")