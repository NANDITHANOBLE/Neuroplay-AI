"""
NeuroPlay-AI Streamlit application entry point.
Run: streamlit run frontend/streamlit_app/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402
from api_client import check_health  # noqa: E402

st.set_page_config(
    page_title="NeuroPlay-AI",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 NeuroPlay-AI")
st.subheader("Adaptive Human Behavior Prediction and Strategy Intelligence Platform")

if check_health():
    st.success("✅ Backend API is online.")
else:
    st.error(
        "❌ Backend API is not reachable. "
        "Start it with: `uvicorn backend.app.main:app --reload --port 8000`"
    )

st.markdown(
    """
    ### Welcome!
    Use the sidebar to navigate:
    - **🎮 Play** — Keyboard-based Rock-Paper-Scissors against the AI
    - **📷 Webcam** — Play using hand gestures (MediaPipe-powered)
    - **🏆 Leaderboard** — See top players by win rate

    NeuroPlay-AI predicts your next move using a trained neural network,
    explains its reasoning via SHAP, and detects when you change strategy
    mid-game using concept drift detection.
    """
)

if "username" not in st.session_state:
    st.session_state.username = st.text_input("Enter your username to begin:", value="player1")
else:
    st.info(f"Playing as: **{st.session_state.username}**")
