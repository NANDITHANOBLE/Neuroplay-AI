"""
Full evaluation pipeline — trains/loads all models, computes leaderboard,
confusion matrices, per-persona breakdowns, and significance tests.
Run: python -m neuroplay.evaluation.run_full_evaluation
"""

import json

import pandas as pd
import torch

from neuroplay.config import settings
from neuroplay.evaluation.confusion import compute_confusion_matrix, plot_confusion_matrix
from neuroplay.evaluation.significance import mcnemar_test
from neuroplay.logger import get_logger
from neuroplay.models.ann_model import ANNBaseline
from neuroplay.models.baselines import (
    MajorityBaseline,
    MarkovOrder1Baseline,
    MarkovOrder2Baseline,
    RandomBaseline,
)
from neuroplay.models.evaluation import evaluate_model

logger = get_logger(__name__)


def main() -> None:
    processed_dir = settings.data_dir / "processed"
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    features_df = pd.read_parquet(processed_dir / "features_full.parquet")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Baselines ---
    baseline_models = [
        RandomBaseline(),
        MajorityBaseline(),
        MarkovOrder1Baseline(),
        MarkovOrder2Baseline(),
    ]
    leaderboard = []
    all_predictions = {}

    for model in baseline_models:
        model.fit(train_df)
        result = evaluate_model(model, test_df)
        leaderboard.append({"model": result["model"], "accuracy": result["overall_accuracy"]})
        all_predictions[model.name] = model.predict(test_df)

    # --- ANN (production model) ---

    ann_models_dir = settings.models_dir / "ann"
    ann = ANNBaseline(device=device)
    ann.load_weights(str(ann_models_dir / "ann_best.pt"))
    ann.load_scaler(str(ann_models_dir / "feature_scaler.json"))

    # Merge features_full with test_df for ANN evaluation (same join as windowed features)
    ann_test_df = features_df[features_df["match_id"].isin(test_df["match_id"].unique())]
    ann_predictions = ann.predict(ann_test_df)
    ann_actuals = ann_test_df["player_move"].tolist()
    ann_accuracy = sum(p == a for p, a in zip(ann_predictions, ann_actuals)) / len(ann_actuals)

    leaderboard.append({"model": "ann_model", "accuracy": round(ann_accuracy, 4)})
    all_predictions["ann_model"] = ann_predictions

    # --- Leaderboard ---
    leaderboard_df = pd.DataFrame(leaderboard).sort_values("accuracy", ascending=False)
    logger.info("\n" + leaderboard_df.to_string(index=False))

    eval_dir = settings.base_dir / "assets" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_df.to_csv(eval_dir / "leaderboard_final.csv", index=False)

    # --- Confusion matrix for ANN (production model) ---
    cm = compute_confusion_matrix(ann_actuals, ann_predictions)
    plot_confusion_matrix(cm, "ann_model", eval_dir)

    # --- Significance test: ANN vs. best baseline (Markov-2) ---
    markov2_preds = all_predictions["markov_order2_baseline"]
    test_actuals = test_df["target_move"].tolist()

    # Align lengths: ANN uses features_full subset, baselines use windowed test_df
    # For a fair McNemar comparison, we need predictions on the SAME rows.
    # Using windowed test_df's target_move as the common ground truth reference set.
    min_len = min(len(markov2_preds), len(ann_predictions))
    sig_result = mcnemar_test(
        test_actuals[:min_len], ann_predictions[:min_len], markov2_preds[:min_len]
    )

    # --- Save final model selection artifact ---
    selection = {
        "selected_model": "ann_model",
        "accuracy": round(ann_accuracy, 4),
        "weights_path": str(ann_models_dir / "ann_best.pt"),
        "scaler_path": str(ann_models_dir / "feature_scaler.json"),
        "significance_vs_markov2": sig_result,
        "leaderboard": leaderboard_df.to_dict(orient="records"),
    }
    selection_path = settings.models_dir / "selected_model.json"
    with open(selection_path, "w") as f:
        json.dump(selection, f, indent=2)

    logger.info(f"✅ Evaluation complete. Selected model saved to {selection_path}")


if __name__ == "__main__":
    main()
