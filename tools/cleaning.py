"""
cleaning.py — Data cleaning tool for the Agentic Data Analyst.

This is designed to be called BOTH as a standalone function and as a
"tool" an LLM agent can invoke (see agents/orchestrator.py later).
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class CleaningReport:
    """Keeps track of everything the cleaner did, so the agent can explain it later."""
    original_shape: tuple
    final_shape: tuple
    dropped_duplicate_rows: int = 0
    columns_dropped_high_missing: list = field(default_factory=list)
    numeric_columns_imputed: dict = field(default_factory=dict)  # col -> fill value used
    categorical_columns_imputed: list = field(default_factory=list)
    dtype_fixes: dict = field(default_factory=dict)  # col -> new dtype
    outlier_columns_capped: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Started with {self.original_shape[0]} rows x {self.original_shape[1]} cols, "
            f"ended with {self.final_shape[0]} rows x {self.final_shape[1]} cols.",
        ]
        if self.dropped_duplicate_rows:
            lines.append(f"Removed {self.dropped_duplicate_rows} duplicate rows.")
        if self.columns_dropped_high_missing:
            lines.append(f"Dropped columns >50% missing: {self.columns_dropped_high_missing}")
        if self.numeric_columns_imputed:
            lines.append(f"Imputed numeric columns (median): {list(self.numeric_columns_imputed.keys())}")
        if self.categorical_columns_imputed:
            lines.append(f"Imputed categorical columns (mode): {self.categorical_columns_imputed}")
        if self.dtype_fixes:
            lines.append(f"Fixed dtypes: {self.dtype_fixes}")
        if self.outlier_columns_capped:
            lines.append(f"Capped outliers (IQR method) in: {self.outlier_columns_capped}")
        return "\n".join(lines)


def clean_data(df: pd.DataFrame, missing_threshold: float = 0.5) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Autonomously cleans a raw, messy DataFrame.

    Steps:
      1. Drop exact duplicate rows
      2. Drop columns that are more than `missing_threshold` missing
      3. Try to fix obvious dtype issues (numbers stored as strings, date strings)
      4. Impute remaining missing values (median for numeric, mode for categorical)
      5. Cap extreme outliers using the IQR method (doesn't drop rows, just caps values)

    Returns the cleaned DataFrame plus a CleaningReport explaining every decision.
    """
    df = df.copy()
    report = CleaningReport(original_shape=df.shape, final_shape=df.shape)

    # 1. Duplicates
    before = len(df)
    df = df.drop_duplicates()
    report.dropped_duplicate_rows = before - len(df)

    # 2. Drop high-missing columns
    missing_frac = df.isna().mean()
    to_drop = missing_frac[missing_frac > missing_threshold].index.tolist()
    if to_drop:
        df = df.drop(columns=to_drop)
        report.columns_dropped_high_missing = to_drop

    # 3. Dtype fixes — try to coerce object columns that are secretly numeric
    for col in df.select_dtypes(include=["object", "string"]).columns:
        coerced = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        # if >90% of non-null values convert cleanly, treat it as numeric
        non_null = df[col].notna().sum()
        if non_null > 0 and coerced.notna().sum() / non_null > 0.9:
            df[col] = coerced
            report.dtype_fixes[col] = "numeric"

    # 4. Impute
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            report.numeric_columns_imputed[col] = round(float(fill_val), 3)

    for col in df.select_dtypes(include=["object", "string"]).columns:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            report.categorical_columns_imputed.append(col)

    # 5. Outlier capping (IQR)
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            df[col] = df[col].clip(lower, upper)
            report.outlier_columns_capped.append(col)

    report.final_shape = df.shape
    return df, report


if __name__ == "__main__":
    # Quick self-test with a synthetic messy "procurement" dataset
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "supplier": rng.choice(["Acme Corp", "Globex", "Initech", None], size=n, p=[0.4, 0.3, 0.25, 0.05]),
        "unit_cost": [f"{v:,.2f}" if rng.random() > 0.1 else None for v in rng.normal(50, 15, n)],
        "quantity": rng.integers(1, 500, n).astype(float),
        "region": rng.choice(["US", "EU", "APAC"], size=n),
        "mostly_empty_col": [None] * int(n * 0.7) + list(rng.normal(0, 1, n - int(n * 0.7))),
    })
    # inject some duplicates and an outlier
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    df.loc[0, "quantity"] = 999999

    cleaned, report = clean_data(df)
    print(report.summary())
    print("\nCleaned sample:")
    print(cleaned.head())