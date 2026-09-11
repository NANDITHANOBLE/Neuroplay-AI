"""Keyboard-based gameplay page."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_client import play_move, start_game  # noqa: E402

MOVE_LABELS = {0: "🪨 Rock", 1: "📄 Paper", 2: "✂️ Scissors"}

st.title("🎮 Keyboard Gameplay")

if "username" not in st.session_state or not st.session_state.get("username"):
    st.warning("Please enter a username on the home page first.")
    st.stop()

if "match_id" not in st.session_state:
    if st.button("Start New Match"):
        result = start_game(st.session_state.username, mode="keyboard")
        st.session_state.match_id = result["match_id"]
        st.session_state.score_user = 0
        st.session_state.score_ai = 0
        st.session_state.round_history = []
        st.rerun()
    st.stop()

st.info(f"Match ID: {st.session_state.match_id}")

col1, col2, col3 = st.columns(3)
move_choice = None
with col1:
    if st.button("🪨 Rock", use_container_width=True):
        move_choice = 0
with col2:
    if st.button("📄 Paper", use_container_width=True):
        move_choice = 1
with col3:
    if st.button("✂️ Scissors", use_container_width=True):
        move_choice = 2

if move_choice is not None:
    result = play_move(st.session_state.match_id, move_choice)
    st.session_state.score_user = result["running_score_user"]
    st.session_state.score_ai = result["running_score_ai"]
    st.session_state.round_history.append(result)

    result_emoji = {"win": "🎉", "loss": "😔", "draw": "🤝"}[result["result"]]
    st.markdown(
        f"### You played {MOVE_LABELS[result['player_move']]} | "
        f"AI played {MOVE_LABELS[result['ai_move']]} | "
        f"**{result['result'].upper()}** {result_emoji}"
    )

st.markdown("---")
score_col1, score_col2 = st.columns(2)
score_col1.metric("Your Score", st.session_state.get("score_user", 0))
score_col2.metric("AI Score", st.session_state.get("score_ai", 0))

if st.session_state.get("round_history"):
    st.markdown("### Round History")
    for r in reversed(st.session_state.round_history[-10:]):
        st.text(
            f"Round {r['round_number']}: "
            f"You={MOVE_LABELS[r['player_move']]} | "
            f"AI={MOVE_LABELS[r['ai_move']]} | {r['result']}"
        )

if st.button("End Match / New Match"):
    del st.session_state.match_id
    st.rerun()
