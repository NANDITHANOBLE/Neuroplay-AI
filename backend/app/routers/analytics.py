"""
Analytics endpoint — aggregate statistics across all matches for a user,
feeding the Phase 20 Analytics Dashboard.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from neuroplay.db.models import Match, Move as MoveModel, User

from ..dependencies import get_db_session

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/{username}")
def get_user_analytics(username: str, db: Session = Depends(get_db_session)):
    user = db.query(User).filter_by(username=username).first()
    if user is None:
        return {"error": "User not found", "matches": []}

    matches = db.query(Match).filter_by(user_id=user.id).all()
    match_ids = [m.id for m in matches]

    moves = db.query(MoveModel).filter(MoveModel.match_id.in_(match_ids)).all()

    move_distribution = {0: 0, 1: 0, 2: 0}
    result_counts = {"win": 0, "loss": 0, "draw": 0}
    timeline = []

    for m in sorted(moves, key=lambda x: (x.match_id, x.round_number)):
        move_distribution[m.player_move] += 1
        result_counts[m.result] += 1
        timeline.append(
            {
                "match_id": m.match_id,
                "round_number": m.round_number,
                "result": m.result,
                "player_move": m.player_move,
            }
        )

    total_rounds = len(moves)
    win_rate = result_counts["win"] / total_rounds if total_rounds > 0 else 0.0

    return {
        "username": username,
        "total_matches": len(matches),
        "total_rounds": total_rounds,
        "win_rate": round(win_rate, 4),
        "move_distribution": move_distribution,
        "result_counts": result_counts,
        "timeline": timeline,
    }
