"""Webcam-based gameplay page using MediaPipe gesture detection."""

import sys
from pathlib import Path

import cv2
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_client import play_move, start_game  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from neuroplay.constants import MOVE_NAMES  # noqa: E402
from neuroplay.cv.webcam_capture import WebcamGestureDetector  # noqa: E402

st.title("📷 Webcam Gameplay")

if "username" not in st.session_state or not st.session_state.get("username"):
    st.warning("Please enter a username on the home page first.")
    st.stop()

if "wc_match_id" not in st.session_state:
    if st.button("Start New Webcam Match"):
        result = start_game(st.session_state.username, mode="webcam")
        st.session_state.wc_match_id = result["match_id"]
        st.rerun()
    st.stop()

st.info(f"Match ID: {st.session_state.wc_match_id}")
st.warning("Show your hand gesture, then click 'Capture Move' to submit.")

frame_placeholder = st.empty()

if "detector" not in st.session_state:
    st.session_state.detector = WebcamGestureDetector()

frame, gesture = st.session_state.detector.read_frame()
if frame is not None:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")

    if gesture and gesture.move is not None:
        st.success(f"Detected: {MOVE_NAMES[gesture.move]} ({gesture.confidence:.0%} confidence)")
        if st.button("Capture Move"):
            result = play_move(st.session_state.wc_match_id, int(gesture.move))
            st.json(result)
    else:
        st.info("No clear gesture detected. Adjust your hand position.")

if st.button("Stop Webcam"):
    st.session_state.detector.release()
    del st.session_state.detector
    del st.session_state.wc_match_id
    st.rerun()
