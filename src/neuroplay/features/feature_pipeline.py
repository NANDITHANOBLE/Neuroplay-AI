"""
CLI entry point for the full feature engineering pipeline.
Run: python -m neuroplay.features.feature_pipeline
"""

import pandas as pd

from neuroplay.config import settings
from neuroplay.features.complexity_features import compute_rolling_lz_complexity
from neuroplay.features.rolling_features import (
    compute_last_move_onehot,
    compute_move_frequency,
    compute_reaction_time_features,
    compute_rolling_win_rate,
    compute_round_position,
)
from neuroplay.features.streak_features import compute_streak_features
from neuroplay.logger import get_logger
from neuroplay.preprocessing.loader import load_moves_dataframe

logger = get_logger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["match_id", "round_number"]).reset_index(drop=True)
    df = compute_streak_features(df)
    df = compute_rolling_win_rate(df)
    df = compute_move_frequency(df)
    df = compute_reaction_time_features(df)
    df = compute_last_move_onehot(df)
    df = compute_round_position(df)
    df = compute_rolling_lz_complexity(df)
    return df


def main() -> None:
    logger.info("Starting feature engineering pipeline...")

    raw_df = load_moves_dataframe()
    featured_df = engineer_features(raw_df)

    processed_dir = settings.data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / "features_full.parquet"
    featured_df.to_parquet(output_path, index=False)

    logger.info(
        f"✅ Feature engineering complete. {len(featured_df.columns)} columns, "
        f"{len(featured_df)} rows. Saved to {output_path}"
    )


if __name__ == "__main__":
    main()
