"""
Leaderboard endpoint — ranks users by win rate.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from neuroplay.db.models import Match, User

from ..dependencies import get_db_session
from ..schemas import LeaderboardEntry, LeaderboardResponse

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


@router.get("", response_model=LeaderboardResponse)
def get_leaderboard(db: Session = Depends(get_db_session), limit: int = 10):
    users = db.query(User).all()
    entries = []

    for user in users:
        matches = db.query(Match).filter_by(user_id=user.id).all()
        if not matches:
            continue
        total_matches = len(matches)
        total_wins = sum(m.final_score_user for m in matches)
        total_rounds = sum(m.final_score_user + m.final_score_ai for m in matches)
        win_rate = total_wins / total_rounds if total_rounds > 0 else 0.0

        entries.append(
            LeaderboardEntry(
                username=user.username,
                total_matches=total_matches,
                total_wins=total_wins,
                win_rate=round(win_rate, 4),
            )
        )

    entries.sort(key=lambda e: e.win_rate, reverse=True)
    return LeaderboardResponse(entries=entries[:limit])
