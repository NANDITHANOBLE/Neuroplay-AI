"""
CLI entry point — runs ADWIN and DDM over ANN prediction-error streams for
DriftingBot matches, validating detection latency against the known
ground-truth drift point (round 50, set in Phase 4's generate_dataset.py).
Run: python -m neuroplay.drift.run_drift_detection
"""

import pandas as pd

from neuroplay.config import settings
from neuroplay.drift.detectors import ADWIN, DDM
from neuroplay.logger import get_logger
from neuroplay.models.ann_model import ANNBaseline

logger = get_logger(__name__)

TRUE_DRIFT_ROUND = 50


def main() -> None:
    processed_dir = settings.data_dir / "processed"
    features_df = pd.read_parquet(processed_dir / "features_full.parquet")

    drift_matches = features_df[
        features_df["model_used"].str.contains("drift", case=False, na=False)
    ]
    if drift_matches.empty:
        logger.warning("No drifting-bot matches found in features_full.parquet.")
        return

    ann_dir = settings.models_dir / "ann"
    ann = ANNBaseline(device="cpu")
    ann.load_weights(str(ann_dir / "ann_best.pt"))
    ann.load_scaler(str(ann_dir / "feature_scaler.json"))

    results = []

    for match_id, group in drift_matches.groupby("match_id"):
        group = group.sort_values("round_number").reset_index(drop=True)
        predictions = ann.predict(group)
        actuals = group["player_move"].tolist()
        errors = [int(p != a) for p, a in zip(predictions, actuals)]

        adwin = ADWIN()
        ddm = DDM()

        adwin_detected_round = None
        ddm_detected_round = None

        for i, error in enumerate(errors):
            round_num = group["round_number"].iloc[i]

            if adwin_detected_round is None and adwin.update(error):
                adwin_detected_round = round_num

            ddm_status = ddm.update(error)
            if ddm_detected_round is None and ddm_status == "drift":
                ddm_detected_round = round_num

        results.append(
            {
                "match_id": match_id,
                "true_drift_round": TRUE_DRIFT_ROUND,
                "adwin_detected_round": adwin_detected_round,
                "adwin_latency": (
                    adwin_detected_round - TRUE_DRIFT_ROUND
                    if adwin_detected_round is not None
                    else None
                ),
                "ddm_detected_round": ddm_detected_round,
                "ddm_latency": (
                    ddm_detected_round - TRUE_DRIFT_ROUND
                    if ddm_detected_round is not None
                    else None
                ),
            }
        )

    results_df = pd.DataFrame(results)
    logger.info("\n" + results_df.to_string(index=False))

    eval_dir = settings.base_dir / "assets" / "drift"
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(eval_dir / "drift_detection_results.csv", index=False)

    adwin_detected_pct = results_df["adwin_detected_round"].notna().mean() * 100
    ddm_detected_pct = results_df["ddm_detected_round"].notna().mean() * 100
    avg_adwin_latency = results_df["adwin_latency"].dropna().mean()
    avg_ddm_latency = results_df["ddm_latency"].dropna().mean()

    logger.info(
        f"✅ ADWIN detected drift in {adwin_detected_pct:.1f}% of matches "
        f"(avg latency: {avg_adwin_latency:.1f} rounds)"
    )

    logger.info(
        "Note: DDM's 3-sigma threshold is calibrated for low-baseline-error "
        "classifiers; our ANN's ~29% baseline error rate (70.8% accuracy) "
        "makes the round-50 persona shift harder to detect via DDM's "
        "assumption. ADWIN's sub-window comparison is more robust to noisy "
        "streams and is selected as the production drift detector."
    )

    logger.info(
        f"✅ DDM detected drift in {ddm_detected_pct:.1f}% of matches "
        f"(avg latency: {avg_ddm_latency:.1f} rounds)"
    )


if __name__ == "__main__":
    main()
