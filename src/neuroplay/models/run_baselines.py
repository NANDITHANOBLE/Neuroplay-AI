"""
CLI entry point to train and evaluate all baseline models.
Run: python -m neuroplay.models.run_baselines
"""

import pandas as pd

from neuroplay.config import settings
from neuroplay.logger import get_logger
from neuroplay.models.evaluation import run_baseline_comparison

logger = get_logger(__name__)


def main() -> None:
    processed_dir = settings.data_dir / "processed"

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    logger.info(f"Loaded train ({len(train_df)} rows) and test ({len(test_df)} rows).")

    comparison_df = run_baseline_comparison(train_df, test_df)

    logger.info("\n" + comparison_df.to_string(index=False))

    output_path = processed_dir / "baseline_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"✅ Baseline comparison saved to {output_path}")


if __name__ == "__main__":
    main()
