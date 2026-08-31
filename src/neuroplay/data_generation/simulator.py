"""
Match simulator — plays out synthetic persona bots against a simple opponent
and logs every round directly into the database via the Phase 3 ORM models.
"""

import random  # noqa: I001
from datetime import datetime, timedelta

from neuroplay.constants import BEATS, Move
from neuroplay.data_generation.personas import BasePersona
from neuroplay.db.models import Match
from neuroplay.db.models import Move as MoveModel
from neuroplay.db.models import User
from neuroplay.db.session import get_session
from neuroplay.logger import get_logger

logger = get_logger(__name__)


def _determine_result(player_move: Move, ai_move: Move) -> str:
    if player_move == ai_move:
        return "draw"
    return "win" if BEATS[player_move] == ai_move else "loss"


def simulate_match(
    persona: BasePersona,
    user_id: int,
    num_rounds: int = 100,
    model_used: str = "synthetic_baseline",
) -> int:
    """
    Simulates one full match for a given persona against a random-move opponent,
    logging all rounds to the database. Returns the created match_id.
    """
    session = get_session()
    try:
        match = Match(
            user_id=user_id,
            started_at=datetime.utcnow(),
            mode="synthetic",
            model_used=model_used,
        )
        session.add(match)
        session.flush()  # assigns match.id without full commit

        wins, losses = 0, 0
        timestamp = datetime.utcnow()

        for round_number in range(1, num_rounds + 1):
            player_move = persona.next_move()
            ai_move = random.choice(list(Move))
            result = _determine_result(player_move, ai_move)

            if result == "win":
                wins += 1
            elif result == "loss":
                losses += 1

            move_row = MoveModel(
                match_id=match.id,
                round_number=round_number,
                player_move=int(player_move),
                ai_move=int(ai_move),
                result=result,
                reaction_time_ms=random.randint(200, 800),  # synthetic plausible value
                timestamp=timestamp + timedelta(seconds=round_number),
            )
            session.add(move_row)

            persona.record(player_move, ai_move)

        match.ended_at = timestamp + timedelta(seconds=num_rounds)
        match.final_score_user = wins
        match.final_score_ai = losses

        session.commit()
        logger.info(
            f"Simulated match {match.id} | persona={persona.name} | "
            f"rounds={num_rounds} | wins={wins} losses={losses}"
        )
        return match.id
    finally:
        session.close()


def ensure_synthetic_user(username: str = "synthetic_generator") -> int:
    """Gets or creates the placeholder user account that owns all synthetic matches."""
    session = get_session()
    try:
        user = session.query(User).filter_by(username=username).first()
        if user is None:
            user = User(username=username)
            session.add(user)
            session.commit()
            logger.info(f"Created synthetic user '{username}' (id={user.id})")
        return user.id
    finally:
        session.close()
