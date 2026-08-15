"""
modeling.py — Modeling tool for the Agentic Data Analyst.

Takes a cleaned DataFrame + a target column, automatically figures out
whether it's a regression or classification problem, trains an XGBoost
model, and reports performance + feature importance.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score,
)


@dataclass
class ModelingReport:
    task_type: str = ""  # "regression" or "classification"
    target_column: str = ""
    n_train: int = 0
    n_test: int = 0
    metrics: dict = field(default_factory=dict)
    feature_importances: dict = field(default_factory=dict)  # feature -> importance score

    def summary(self) -> str:
        lines = [
            f"Trained an XGBoost {self.task_type} model to predict '{self.target_column}' "
            f"({self.n_train} train rows, {self.n_test} test rows).",
        ]
        metrics_str = ", ".join(f"{k}={v}" for k, v in self.metrics.items())
        lines.append(f"Performance: {metrics_str}")
        if self.feature_importances:
            top_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
            feat_str = ", ".join(f"{f} ({v:.2f})" for f, v in top_features)
            lines.append(f"Top predictive features: {feat_str}")
        return "\n".join(lines)


def train_model(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """
    Automatically trains an XGBoost model on `df` to predict `target_column`.

    - If the target is numeric with many unique values -> regression
    - If the target is categorical or has few unique values -> classification
    - Categorical FEATURE columns are one-hot encoded automatically

    Returns (trained_model, ModelingReport).
    """
    df = df.copy()
    if target_column not in df.columns:
        raise ValueError(f"'{target_column}' not found in columns: {list(df.columns)}")

    y_raw = df[target_column]
    X_raw = df.drop(columns=[target_column])

    # Decide task type
    is_numeric_target = pd.api.types.is_numeric_dtype(y_raw)
    n_unique = y_raw.nunique()
    task_type = "regression" if (is_numeric_target and n_unique > 15) else "classification"

    # Encode features: one-hot for categoricals
    X = pd.get_dummies(X_raw, drop_first=True)

    label_encoder = None
    if task_type == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    report = ModelingReport(
        task_type=task_type,
        target_column=target_column,
        n_train=len(X_train),
        n_test=len(X_test),
    )

    if task_type == "regression":
        model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        report.metrics = {
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
            "MAE": round(float(mean_absolute_error(y_test, preds)), 3),
            "R2": round(float(r2_score(y_test, preds)), 3),
        }
    else:
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=random_state, eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        average = "binary" if len(set(y)) == 2 else "weighted"
        report.metrics = {
            "Accuracy": round(float(accuracy_score(y_test, preds)), 3),
            "F1": round(float(f1_score(y_test, preds, average=average)), 3),
        }

    report.feature_importances = dict(zip(X.columns, model.feature_importances_.tolist()))

    return model, report, label_encoder


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.cleaning import clean_data

    # Synthetic procurement dataset, this time with a target worth predicting
    rng = np.random.default_rng(42)
    n = 300
    quantity = rng.integers(1, 500, n).astype(float)
    region = rng.choice(["US", "EU", "APAC"], size=n)
    supplier = rng.choice(["Acme Corp", "Globex", "Initech"], size=n)
    # unit_cost depends on quantity + region, with noise, so the model has something real to learn
    region_effect = pd.Series(region).map({"US": 5, "EU": 0, "APAC": -3}).values
    unit_cost = 60 - 0.02 * quantity + region_effect + rng.normal(0, 4, n)

    df = pd.DataFrame({
        "supplier": supplier,
        "unit_cost": unit_cost,
        "quantity": quantity,
        "region": region,
    })

    cleaned, _ = clean_data(df)
    model, report, _ = train_model(cleaned, target_column="unit_cost")
    print(report.summary())