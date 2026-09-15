"""Tests for FastAPI endpoints using TestClient (no live server needed)."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_start_game_creates_match(client):
    response = client.post("/game/start", json={"username": "pytest_user", "mode": "keyboard"})
    assert response.status_code == 200
    data = response.json()
    assert "match_id" in data
    assert data["user_id"] is not None


def test_play_move_returns_valid_result(client):
    start_resp = client.post("/game/start", json={"username": "pytest_user2", "mode": "keyboard"})
    match_id = start_resp.json()["match_id"]

    play_resp = client.post(
        "/game/play",
        json={"match_id": match_id, "player_move": 0, "reaction_time_ms": 300},
    )
    assert play_resp.status_code == 200
    data = play_resp.json()
    assert data["result"] in ["win", "loss", "draw"]
    assert data["round_number"] == 1
