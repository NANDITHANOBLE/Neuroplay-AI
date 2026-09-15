"""Analytics dashboard — aggregate gameplay statistics visualization."""

import sys
from pathlib import Path

import plotly.express as px
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import API_BASE_URL  # noqa: E402

st.title("📈 Analytics Dashboard")

if "username" not in st.session_state or not st.session_state.get("username"):
    st.warning("Please enter a username on the home page first.")
    st.stop()

username = st.session_state.username

resp = requests.get(f"{API_BASE_URL}/analytics/{username}", timeout=10)
data = resp.json()

if data.get("total_rounds", 0) == 0:
    st.info("No gameplay data yet. Play some matches first!")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Total Matches", data["total_matches"])
col2.metric("Total Rounds", data["total_rounds"])
col3.metric("Win Rate", f"{data['win_rate']:.1%}")

st.markdown("---")

move_labels = {"0": "Rock", "1": "Paper", "2": "Scissors"}
move_dist = data["move_distribution"]
move_df_data = {
    "Move": [move_labels[str(k)] for k in move_dist],
    "Count": list(move_dist.values()),
}

col_a, col_b = st.columns(2)
with col_a:
    fig_moves = px.bar(
        move_df_data, x="Move", y="Count", title="Your Move Distribution", color="Move"
    )
    st.plotly_chart(fig_moves, use_container_width=True)

with col_b:
    result_counts = data["result_counts"]
    fig_results = px.pie(
        names=list(result_counts.keys()),
        values=list(result_counts.values()),
        title="Win / Loss / Draw Breakdown",
        color=list(result_counts.keys()),
        color_discrete_map={"win": "#2ca02c", "loss": "#d62728", "draw": "#7f7f7f"},
    )
    st.plotly_chart(fig_results, use_container_width=True)

st.markdown("### Win Rate Trend Over Time")
timeline = data["timeline"]
rolling_window = 10
results_binary = [1 if t["result"] == "win" else 0 for t in timeline]
rolling_win_rate = [
    sum(results_binary[max(0, i - rolling_window + 1) : i + 1])
    / len(results_binary[max(0, i - rolling_window + 1) : i + 1])
    for i in range(len(results_binary))
]

fig_trend = px.line(
    x=list(range(1, len(rolling_win_rate) + 1)),
    y=rolling_win_rate,
    labels={"x": "Round (across all matches)", "y": "Rolling Win Rate"},
    title=f"Rolling Win Rate (window={rolling_window})",
)
st.plotly_chart(fig_trend, use_container_width=True)
