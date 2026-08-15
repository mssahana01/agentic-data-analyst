"""
eda.py — Exploratory Data Analysis tool for the Agentic Data Analyst.

Takes a (cleaned) DataFrame and automatically produces:
  - summary statistics
  - missing value overview
  - correlation matrix
  - a handful of Plotly charts saved as standalone HTML files

Like cleaning.py, this returns a report object the agent can narrate.
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, field


@dataclass
class EDAReport:
    shape: tuple
    numeric_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    summary_stats: pd.DataFrame = None
    correlations: pd.DataFrame = None
    top_correlated_pairs: list = field(default_factory=list)  # [(col1, col2, corr)]
    categorical_top_values: dict = field(default_factory=dict)  # col -> {value: count}
    chart_paths: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Analyzed {self.shape[0]} rows x {self.shape[1]} columns "
            f"({len(self.numeric_columns)} numeric, {len(self.categorical_columns)} categorical).",
        ]
        if self.top_correlated_pairs:
            pairs_str = ", ".join(
                f"{a} & {b} (r={c:.2f})" for a, b, c in self.top_correlated_pairs
            )
            lines.append(f"Strongest correlations: {pairs_str}")
        for col, counts in self.categorical_top_values.items():
            top_val = next(iter(counts))
            lines.append(f"Most common '{col}': {top_val} ({counts[top_val]} rows)")
        if self.chart_paths:
            lines.append(f"Generated {len(self.chart_paths)} charts: {', '.join(self.chart_paths)}")
        return "\n".join(lines)


def run_eda(df: pd.DataFrame, output_dir: str = "outputs/charts", top_n_categories: int = 5) -> EDAReport:
    """
    Runs automated EDA on a cleaned DataFrame and saves charts as HTML files.
    """
    os.makedirs(output_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    report = EDAReport(shape=df.shape, numeric_columns=numeric_cols, categorical_columns=categorical_cols)
    report.summary_stats = df[numeric_cols].describe().round(2) if numeric_cols else pd.DataFrame()

    # Correlations
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(2)
        report.correlations = corr

        # find top correlated pairs (excluding self-correlation)
        pairs = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                pairs.append((col1, col2, corr.loc[col1, col2]))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        report.top_correlated_pairs = pairs[:3]

        # correlation heatmap
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Correlation Heatmap")
        path = os.path.join(output_dir, "correlation_heatmap.html")
        fig.write_html(path)
        report.chart_paths.append(path)

    # Distribution plots for numeric columns (max 4, to keep it digestible)
    for col in numeric_cols[:4]:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}", marginal="box")
        path = os.path.join(output_dir, f"distribution_{col}.html")
        fig.write_html(path)
        report.chart_paths.append(path)

    # Bar charts for categorical columns (max 3)
    for col in categorical_cols[:3]:
        counts = df[col].value_counts().head(top_n_categories)
        report.categorical_top_values[col] = counts.to_dict()
        fig = px.bar(x=counts.index, y=counts.values, title=f"Top {col} values",
                     labels={"x": col, "y": "count"})
        path = os.path.join(output_dir, f"top_values_{col}.html")
        fig.write_html(path)
        report.chart_paths.append(path)

    return report


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.cleaning import clean_data

    # Reuse the same synthetic messy dataset as cleaning.py's self-test
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "supplier": rng.choice(["Acme Corp", "Globex", "Initech", None], size=n, p=[0.4, 0.3, 0.25, 0.05]),
        "unit_cost": [f"{v:,.2f}" if rng.random() > 0.1 else None for v in rng.normal(50, 15, n)],
        "quantity": rng.integers(1, 500, n).astype(float),
        "region": rng.choice(["US", "EU", "APAC"], size=n),
        "mostly_empty_col": [None] * int(n * 0.7) + list(rng.normal(0, 1, n - int(n * 0.7))),
    })
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    df.loc[0, "quantity"] = 999999

    cleaned, clean_report = clean_data(df)
    eda_report = run_eda(cleaned)

    print("--- Cleaning ---")
    print(clean_report.summary())
    print("\n--- EDA ---")
    print(eda_report.summary())
    print("\nSummary stats:")
    print(eda_report.summary_stats)