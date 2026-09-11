"""
Game endpoints — start a match, play a round, retrieve match history.
"""

import random
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from neuroplay.constants import BEATS, Move
from neuroplay.db.models import Match, Move as MoveModel, User
from neuroplay.logger import get_logger
from neuroplay.models.ann_model import ANNBaseline

from ..dependencies import get_ann_model, get_db_session
from ..schemas import (
    MatchHistoryResponse,
    MatchHistoryRow,
    PlayMoveRequest,
    PlayMoveResponse,
    StartGameRequest,
    StartGameResponse,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/game", tags=["game"])


def _determine_result(player_move: int, ai_move: int) -> str:
    if player_move == ai_move:
        return "draw"
    return "win" if BEATS[Move(player_move)] == ai_move else "loss"


@router.post("/start", response_model=StartGameResponse)
def start_game(req: StartGameRequest, db: Session = Depends(get_db_session)):
    user = db.query(User).filter_by(username=req.username).first()
    if user is None:
        user = User(username=req.username)
        db.add(user)
        db.commit()
        db.refresh(user)

    match = Match(
        user_id=user.id, started_at=datetime.utcnow(), mode=req.mode, model_used="ann_model"
    )
    db.add(match)
    db.commit()
    db.refresh(match)

    logger.info(f"Started match {match.id} for user '{req.username}'")
    return StartGameResponse(
        match_id=match.id, user_id=user.id, message="Match started successfully."
    )


@router.post("/play", response_model=PlayMoveResponse)
def play_move(
    req: PlayMoveRequest,
    db: Session = Depends(get_db_session),
    ann: ANNBaseline = Depends(get_ann_model),
):
    match = db.query(Match).filter_by(id=req.match_id).first()
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found.")

    prior_moves = (
        db.query(MoveModel).filter_by(match_id=req.match_id).order_by(MoveModel.round_number).all()
    )
    round_number = len(prior_moves) + 1

    # Simple counter-strategy: play against a random move for now.
    # (DQN integration for optimal counter-play is wired in Phase 15b/future iteration.)
    ai_move = random.choice(list(Move)).value
    result = _determine_result(req.player_move, ai_move)

    move_row = MoveModel(
        match_id=req.match_id,
        round_number=round_number,
        player_move=req.player_move,
        ai_move=ai_move,
        result=result,
        reaction_time_ms=req.reaction_time_ms,
        timestamp=datetime.utcnow(),
    )
    db.add(move_row)
    db.commit()

    running_wins = sum(1 for m in prior_moves if m.result == "win") + (1 if result == "win" else 0)
    running_losses = sum(1 for m in prior_moves if m.result == "loss") + (
        1 if result == "loss" else 0
    )

    logger.info(f"Match {req.match_id} | Round {round_number} | Result: {result}")

    return PlayMoveResponse(
        round_number=round_number,
        player_move=req.player_move,
        ai_move=ai_move,
        result=result,
        predicted_next_move=None,  # Populated once feature-window integration is wired
        prediction_confidence=None,
        drift_warning=False,
        running_score_user=running_wins,
        running_score_ai=running_losses,
    )


@router.get("/{match_id}/history", response_model=MatchHistoryResponse)
def get_match_history(match_id: int, db: Session = Depends(get_db_session)):
    match = db.query(Match).filter_by(id=match_id).first()
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found.")

    moves = db.query(MoveModel).filter_by(match_id=match_id).order_by(MoveModel.round_number).all()
    rounds = [
        MatchHistoryRow(
            round_number=m.round_number,
            player_move=m.player_move,
            ai_move=m.ai_move,
            result=m.result,
        )
        for m in moves
    ]
    final_wins = sum(1 for m in moves if m.result == "win")
    final_losses = sum(1 for m in moves if m.result == "loss")

    return MatchHistoryResponse(
        match_id=match_id,
        rounds=rounds,
        final_score_user=final_wins,
        final_score_ai=final_losses,
    )
