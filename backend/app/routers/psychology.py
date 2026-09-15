"""
Psychology classification endpoint — analyzes a user's real gameplay history
to classify their dominant behavioral pattern (Feature F5).
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from neuroplay.db.models import Match, Move as MoveModel, User
from neuroplay.eda.complexity import lempel_ziv_complexity

from ..dependencies import get_db_session

router = APIRouter(prefix="/psychology", tags=["psychology"])


def _classify_pattern(moves: list[int], results: list[str]) -> dict:
    if len(moves) < 10:
        return {"pattern": "insufficient_data", "lz_complexity": 0.0, "confidence": 0.0}

    lz_score = lempel_ziv_complexity(moves)

    move_counts = {0: 0, 1: 0, 2: 0}
    for m in moves:
        move_counts[m] += 1
    max_freq = max(move_counts.values()) / len(moves)

    win_stay_matches = 0
    lose_shift_matches = 0
    win_or_loss_count = 0
    for i in range(1, len(moves)):
        if results[i - 1] == "win":
            win_or_loss_count += 1
            if moves[i] == moves[i - 1]:
                win_stay_matches += 1
        elif results[i - 1] == "loss":
            win_or_loss_count += 1
            if moves[i] != moves[i - 1]:
                lose_shift_matches += 1

    wsls_rate = (
        (win_stay_matches + lose_shift_matches) / win_or_loss_count
        if win_or_loss_count > 0
        else 0.0
    )

    if max_freq > 0.45:
        pattern, confidence = "frequency_biased", max_freq
    elif lz_score < 0.5:
        pattern, confidence = "cyclic", 1 - lz_score
    elif wsls_rate > 0.6:
        pattern, confidence = "win_stay_lose_shift", wsls_rate
    else:
        pattern, confidence = "unpredictable", lz_score

    return {
        "pattern": pattern,
        "lz_complexity": round(lz_score, 4),
        "confidence": round(confidence, 4),
        "move_distribution": move_counts,
        "wsls_rate": round(wsls_rate, 4),
    }


@router.get("/{username}")
def get_psychology_profile(username: str, db: Session = Depends(get_db_session)):
    user = db.query(User).filter_by(username=username).first()
    if user is None:
        return {"error": "User not found"}

    match_ids = [m.id for m in db.query(Match).filter_by(user_id=user.id).all()]
    moves_query = (
        db.query(MoveModel)
        .filter(MoveModel.match_id.in_(match_ids))
        .order_by(MoveModel.match_id, MoveModel.round_number)
        .all()
    )

    moves = [m.player_move for m in moves_query]
    results = [m.result for m in moves_query]

    classification = _classify_pattern(moves, results)
    return {"username": username, "total_rounds_analyzed": len(moves), **classification}
