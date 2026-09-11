"""
Pydantic schemas for request/response validation across all API endpoints.
"""

from pydantic import BaseModel


class StartGameRequest(BaseModel):
    username: str
    mode: str = "keyboard"  # "keyboard" | "webcam"


class StartGameResponse(BaseModel):
    match_id: int
    user_id: int
    message: str


class PlayMoveRequest(BaseModel):
    match_id: int
    player_move: int  # 0=Rock, 1=Paper, 2=Scissors
    reaction_time_ms: int | None = None


class PlayMoveResponse(BaseModel):
    round_number: int
    player_move: int
    ai_move: int
    result: str
    predicted_next_move: int | None
    prediction_confidence: float | None
    drift_warning: bool
    running_score_user: int
    running_score_ai: int


class MatchHistoryRow(BaseModel):
    round_number: int
    player_move: int
    ai_move: int
    result: str


class MatchHistoryResponse(BaseModel):
    match_id: int
    rounds: list[MatchHistoryRow]
    final_score_user: int
    final_score_ai: int


class LeaderboardEntry(BaseModel):
    username: str
    total_matches: int
    total_wins: int
    win_rate: float


class LeaderboardResponse(BaseModel):
    entries: list[LeaderboardEntry]


class ExplanationResponse(BaseModel):
    move_id: int
    predicted_move: int
    feature_attributions: dict[str, float]
