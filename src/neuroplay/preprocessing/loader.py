"""
Loads raw match/move data from the database into pandas DataFrames.
"""

import pandas as pd
from sqlalchemy import select

from neuroplay.db.models import Match, Move
from neuroplay.db.session import get_session
from neuroplay.logger import get_logger

logger = get_logger(__name__)


def load_moves_dataframe() -> pd.DataFrame:
    """
    Loads all moves joined with their parent match metadata into a single
    flat DataFrame, ordered by match_id then round_number.
    """
    session = get_session()
    try:
        stmt = (
            select(
                Move.id.label("move_id"),
                Move.match_id,
                Move.round_number,
                Move.player_move,
                Move.ai_move,
                Move.result,
                Move.reaction_time_ms,
                Move.timestamp,
                Match.model_used,
                Match.mode,
                Match.user_id,
            )
            .join(Match, Move.match_id == Match.id)
            .order_by(Move.match_id, Move.round_number)
        )
        rows = session.execute(stmt).all()
        df = pd.DataFrame(
            rows,
            columns=[
                "move_id",
                "match_id",
                "round_number",
                "player_move",
                "ai_move",
                "result",
                "reaction_time_ms",
                "timestamp",
                "model_used",
                "mode",
                "user_id",
            ],
        )
        logger.info(f"Loaded {len(df)} move rows across {df['match_id'].nunique()} matches.")
        return df
    finally:
        session.close()
