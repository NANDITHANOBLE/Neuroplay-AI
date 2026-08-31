"""
CLI entry point for the full preprocessing pipeline.
Run: python -m neuroplay.preprocessing.preprocess_pipeline
"""

from neuroplay.config import settings
from neuroplay.logger import get_logger
from neuroplay.preprocessing.loader import load_moves_dataframe
from neuroplay.preprocessing.splitter import split_by_match
from neuroplay.preprocessing.windowing import create_windows

logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting preprocessing pipeline...")

    raw_df = load_moves_dataframe()
    windowed_df = create_windows(raw_df)
    train_df, val_df, test_df = split_by_match(windowed_df)

    processed_dir = settings.data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)

    logger.info(f"✅ Preprocessing complete. Files saved to {processed_dir}")


if __name__ == "__main__":
    main()
