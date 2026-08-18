"""
chat_app.py — Streamlit chat UI for the Agentic Data Analyst.

Run with: streamlit run app/chat_app.py

This wraps agents/orchestrator.py in a web chat interface: upload a CSV,
tell the agent what you want in plain English, and watch it decide which
tools to call.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

from agents.orchestrator import build_agent

st.set_page_config(page_title="Agentic Data Analyst", page_icon="📊", layout="wide")

st.title("📊 Agentic Data Analyst")
st.caption("Upload a dataset, tell the agent what you want, and watch it decide which tools to call.")

# --- Session state setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}
if "csv_path" not in st.session_state:
    st.session_state.csv_path = None
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()

# --- Sidebar: upload + report link ---
with st.sidebar:
    st.header("1. Upload your data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        os.makedirs("data", exist_ok=True)
        save_path = os.path.join("data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.csv_path = save_path
        st.success(f"Saved to {save_path}")

    if st.session_state.csv_path:
        st.info(f"Active file:\n`{st.session_state.csv_path}`")

    st.divider()
    st.header("2. View results")
    report_path = "outputs/report.html"
    if os.path.exists(report_path):
        if st.button("🔍 View latest report"):
            st.session_state.show_report = True
    else:
        st.caption("No report generated yet — ask the agent for a full analysis.")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# --- Show embedded report if requested ---
if st.session_state.get("show_report") and os.path.exists("outputs/report.html"):
    with open("outputs/report.html") as f:
        html = f.read()
    components.html(html, height=800, scrolling=True)
    if st.button("← Back to chat"):
        st.session_state.show_report = False
        st.rerun()
    st.stop()

# --- Chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("e.g. 'Give me a full analysis, predict unit_cost' or 'Just clean this data'")

if user_input:
    if not st.session_state.csv_path:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({
            "role": "assistant",
            "content": "I don't have a dataset yet — please upload a CSV in the sidebar first."
        })
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Give the agent the file path as context along with the user's message
    full_prompt = f"The dataset is at {st.session_state.csv_path}. {user_input}"

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🤔 Thinking...")

        tool_log = []
        final_text = ""

        for step in st.session_state.agent.stream(
            {"messages": [{"role": "user", "content": full_prompt}]},
            stream_mode="values",
        ):
            last_msg = step["messages"][-1]

            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    tool_log.append(f"🔧 Calling `{tc['name']}`...")
                    placeholder.markdown("\n\n".join(tool_log))

            elif last_msg.__class__.__name__ == "AIMessage" and last_msg.content:
                final_text = last_msg.content

        display_text = ("\n\n".join(tool_log) + "\n\n---\n\n" + final_text) if tool_log else final_text
        placeholder.markdown(display_text)

    st.session_state.messages.append({"role": "assistant", "content": display_text})

    if os.path.exists("outputs/report.html"):
        st.rerun()