"""
Splits windowed data into train/val/test sets by match_id (prevents leakage),
stratified by persona (model_used) to keep class balance across splits.
"""

import numpy as np
import pandas as pd

from neuroplay.logger import get_logger

logger = get_logger(__name__)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO is the remainder (0.15)

RANDOM_SEED = 42


def split_by_match(windowed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data at the match_id level, stratified by model_used (persona),
    so no single match's rounds are split across train/val/test.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    match_personas = (
        windowed_df[["match_id", "model_used"]].drop_duplicates().reset_index(drop=True)
    )

    train_ids, val_ids, test_ids = [], [], []

    for persona, group in match_personas.groupby("model_used"):
        match_ids = group["match_id"].to_numpy().copy()
        rng.shuffle(match_ids)

        n = len(match_ids)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_ids.extend(match_ids[:n_train])
        val_ids.extend(match_ids[n_train : n_train + n_val])
        test_ids.extend(match_ids[n_train + n_val :])

    train_df = windowed_df[windowed_df["match_id"].isin(train_ids)].reset_index(drop=True)
    val_df = windowed_df[windowed_df["match_id"].isin(val_ids)].reset_index(drop=True)
    test_df = windowed_df[windowed_df["match_id"].isin(test_ids)].reset_index(drop=True)

    logger.info(
        f"Split complete — train: {len(train_df)} rows ({len(train_ids)} matches), "
        f"val: {len(val_df)} rows ({len(val_ids)} matches), "
        f"test: {len(test_df)} rows ({len(test_ids)} matches)."
    )
    return train_df, val_df, test_df
