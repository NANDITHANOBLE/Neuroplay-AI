"""
Self-play data export — pulls real gameplay logged via the FastAPI backend
(Phase 17) into the same feature-engineered format as Phase 7's synthetic
data, closing the adaptive learning loop.
Run: python -m neuroplay.db.export_selfplay_data
"""

from neuroplay.config import settings
from neuroplay.features.feature_pipeline import engineer_features
from neuroplay.logger import get_logger
from neuroplay.preprocessing.loader import load_moves_dataframe

logger = get_logger(__name__)

EXCLUDED_USERNAME = "synthetic_generator"


def main() -> None:
    logger.info("Loading all moves from database...")
    raw_df = load_moves_dataframe()

    real_gameplay_df = raw_df[raw_df["mode"] != "synthetic"]
    logger.info(
        f"Found {len(real_gameplay_df)} real gameplay rows "
        f"across {real_gameplay_df['match_id'].nunique()} matches "
        f"(excluding synthetic Phase 4 data)."
    )

    if real_gameplay_df.empty:
        logger.warning("No real gameplay data found yet. Play some games via the API first!")
        return

    featured_df = engineer_features(real_gameplay_df)

    output_dir = settings.data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "selfplay_features.parquet"
    featured_df.to_parquet(output_path, index=False)

    logger.info(f"✅ Self-play data exported to {output_path}")
    logger.info(
        "This data can be merged with features_full.parquet to retrain "
        "models on real human behavior (future iteration)."
    )


if __name__ == "__main__":
    main()
