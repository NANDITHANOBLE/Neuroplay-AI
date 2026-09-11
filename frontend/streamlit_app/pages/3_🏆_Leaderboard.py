"""Leaderboard page."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_client import get_leaderboard  # noqa: E402

st.title("🏆 Leaderboard")

data = get_leaderboard(limit=20)
entries = data.get("entries", [])

if not entries:
    st.info("No leaderboard data yet. Play some matches!")
else:
    df = pd.DataFrame(entries)
    df["win_rate"] = (df["win_rate"] * 100).round(1).astype(str) + "%"
    df.index = df.index + 1
    st.dataframe(df, use_container_width=True)
