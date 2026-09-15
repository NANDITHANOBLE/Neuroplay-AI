"""Psychology dashboard — behavioral pattern classification and explainability."""

import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import API_BASE_URL  # noqa: E402

st.title("🧬 Psychology Dashboard")

if "username" not in st.session_state or not st.session_state.get("username"):
    st.warning("Please enter a username on the home page first.")
    st.stop()

username = st.session_state.username

resp = requests.get(f"{API_BASE_URL}/psychology/{username}", timeout=10)
data = resp.json()

if data.get("pattern") == "insufficient_data":
    st.info(
        f"Only {data.get('total_rounds_analyzed', 0)} rounds played so far. "
        "Play at least 10 rounds to unlock your psychology profile!"
    )
    st.stop()

PATTERN_DESCRIPTIONS = {
    "win_stay_lose_shift": (
        "🔁 **Win-Stay, Lose-Shift** — You tend to repeat moves after winning "
        "and switch after losing. This is the most common human RPS heuristic!"
    ),
    "frequency_biased": (
        "⚖️ **Frequency Biased** — You have a clear favorite move you play "
        "more often than the others."
    ),
    "cyclic": ("🔄 **Cyclic** — Your moves follow a fairly predictable repeating pattern."),
    "unpredictable": (
        "🎲 **Unpredictable** — Your moves are close to random. You're a tough "
        "opponent for pattern-based prediction!"
    ),
}

pattern = data["pattern"]
st.markdown(f"## Your Dominant Pattern: {pattern.replace('_', ' ').title()}")
st.markdown(PATTERN_DESCRIPTIONS.get(pattern, ""))

col1, col2, col3 = st.columns(3)
col1.metric("Confidence", f"{data['confidence']:.1%}")
col2.metric("LZ Complexity", f"{data['lz_complexity']:.3f}")
col3.metric("Rounds Analyzed", data["total_rounds_analyzed"])

st.markdown("---")
st.markdown("### Move Distribution")
move_labels = {"0": "🪨 Rock", "1": "📄 Paper", "2": "✂️ Scissors"}
move_dist = data.get("move_distribution", {})
for move_id, count in move_dist.items():
    st.write(f"{move_labels.get(str(move_id), move_id)}: {count} times")

st.markdown("---")
st.markdown("### Why This Matters")
st.info(
    "NeuroPlay-AI's neural network uses patterns like these to predict your "
    "next move. The more predictable your pattern, the more the AI can "
    "anticipate you — try mixing up your strategy to keep it guessing!"
)
