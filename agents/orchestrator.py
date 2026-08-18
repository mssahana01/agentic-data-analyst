"""
orchestrator.py — The agent brain.

Instead of hardcoding "clean -> EDA -> model -> report" like report.py does,
this gives an LLM (Claude) access to each step as a callable TOOL, and lets
the LLM decide which tools to call, in what order, based on a natural
language request. This is what makes it "agentic" rather than a script.

Tools operate on file paths (not DataFrames directly) because that's what
an LLM can pass as arguments in a tool call.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
load_dotenv() 

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from tools.cleaning import clean_data
from tools.eda import run_eda
from tools.modeling import train_model
from tools.report import generate_report


# ---------------------------------------------------------------------------
# Tool wrappers — each one reads a CSV, does its job, writes output, and
# returns a short text summary. Keeping I/O as file paths keeps this simple
# and lets you inspect every intermediate artifact yourself.
# ---------------------------------------------------------------------------

@tool
def clean_dataset(csv_path: str) -> str:
    """Cleans a raw CSV file: removes duplicates, drops mostly-empty columns,
    fixes numbers stored as text, imputes missing values, and caps outliers.
    Writes the cleaned file to outputs/cleaned.csv and returns a summary."""
    df = pd.read_csv(csv_path)
    cleaned, report = clean_data(df)
    out_path = "outputs/cleaned.csv"
    os.makedirs("outputs", exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    return f"Cleaned data saved to {out_path}.\n{report.summary()}"


@tool
def explore_dataset(csv_path: str) -> str:
    """Runs exploratory data analysis on a CSV: summary stats, correlations,
    and saves charts to outputs/charts/. Returns a text summary of findings."""
    df = pd.read_csv(csv_path)
    report = run_eda(df)
    return report.summary()


@tool
def train_predictive_model(csv_path: str, target_column: str) -> str:
    """Trains an XGBoost model on a CSV to predict `target_column`.
    Automatically picks regression or classification. Returns performance
    metrics and top predictive features."""
    df = pd.read_csv(csv_path)
    _, report, _ = train_model(df, target_column=target_column)
    return report.summary()


@tool
def generate_full_report(csv_path: str, target_column: str) -> str:
    """Runs the ENTIRE pipeline (clean -> EDA -> model) on a raw CSV and
    writes one polished HTML report to outputs/report.html. Use this when
    the user wants a complete end-to-end analysis, not just one step."""
    df = pd.read_csv(csv_path)
    path = generate_report(df, target_column=target_column)
    return f"Full report generated at {path}. Open it in a browser to view."


TOOLS = [clean_dataset, explore_dataset, train_predictive_model, generate_full_report]

SYSTEM_PROMPT = """You are an autonomous data analyst agent. You have four tools:
- clean_dataset: cleans messy raw data
- explore_dataset: runs EDA and generates charts
- train_predictive_model: trains an XGBoost model on a target column
- generate_full_report: runs the entire pipeline end-to-end and writes a report

Decide which tool(s) to call based on what the user asks. If they want a full
analysis, prefer generate_full_report. If they only want one step (e.g. "just
clean this data"), call only that tool. Always explain what you did and why in
plain language after the tool results come back."""


def build_agent():
    """Builds and returns a ready-to-invoke LangGraph agent."""
    model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    agent = create_react_agent(model, TOOLS, prompt=SYSTEM_PROMPT)
    return agent


def run_agent(user_message: str):
    """Convenience function: runs the agent on a single message and prints
    the full trace of what it decided to do."""
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})

    for msg in result["messages"]:
        role = msg.__class__.__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\n[AGENT DECIDED TO CALL] {tc['name']}({tc['args']})")
        elif role == "ToolMessage":
            print(f"[TOOL RESULT] {msg.content[:300]}")
        elif role == "AIMessage" and msg.content:
            print(f"\n[AGENT SAYS]\n{msg.content}")

    return result


if __name__ == "__main__":
    import numpy as np

    # Build the same synthetic dataset used in earlier steps and save it as a CSV
    rng = np.random.default_rng(42)
    n = 300
    quantity = rng.integers(1, 500, n).astype(float)
    region = rng.choice(["US", "EU", "APAC"], size=n)
    supplier = rng.choice(["Acme Corp", "Globex", "Initech"], size=n)
    region_effect = pd.Series(region).map({"US": 5, "EU": 0, "APAC": -3}).values
    unit_cost = 60 - 0.02 * quantity + region_effect + rng.normal(0, 4, n)

    df = pd.DataFrame({
        "supplier": supplier, "unit_cost": unit_cost,
        "quantity": quantity, "region": region,
    })
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/sample_procurement.csv", index=False)

    run_agent(
        "I have a raw dataset at data/sample_procurement.csv. "
        "Give me a full analysis and try to predict unit_cost."
    )