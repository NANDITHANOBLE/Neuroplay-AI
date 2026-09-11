"""
Thin HTTP client wrapping all FastAPI backend calls.
The Streamlit app NEVER imports neuroplay.models/db directly — everything
goes through the Phase 17 API, keeping a single source of truth.
"""

import requests
from config import API_BASE_URL

TIMEOUT = 10


def start_game(username: str, mode: str = "keyboard") -> dict:
    resp = requests.post(
        f"{API_BASE_URL}/game/start",
        json={"username": username, "mode": mode},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def play_move(match_id: int, player_move: int, reaction_time_ms: int | None = None) -> dict:
    resp = requests.post(
        f"{API_BASE_URL}/game/play",
        json={
            "match_id": match_id,
            "player_move": player_move,
            "reaction_time_ms": reaction_time_ms,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def get_match_history(match_id: int) -> dict:
    resp = requests.get(f"{API_BASE_URL}/game/{match_id}/history", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_leaderboard(limit: int = 10) -> dict:
    resp = requests.get(f"{API_BASE_URL}/leaderboard", params={"limit": limit}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def check_health() -> bool:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False
