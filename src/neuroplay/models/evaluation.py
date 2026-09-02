"""
Shared evaluation harness for all NeuroPlay-AI models (baseline through Transformer).
Computes top-1 accuracy, implied win-rate, and per-persona breakdowns.
"""

import pandas as pd

from neuroplay.constants import BEATS, Move
from neuroplay.logger import get_logger
from neuroplay.models.base_model import BaseMovePredictor

logger = get_logger(__name__)


def _counter_move(predicted_move: int) -> int:
    """Returns the move that beats the predicted move (i.e. the AI's optimal counter)."""
    for move, beaten in BEATS.items():
        if beaten == predicted_move:
            return int(move)
    raise ValueError(f"Invalid move: {predicted_move}")


def evaluate_model(model: BaseMovePredictor, test_df: pd.DataFrame) -> dict:
    """
    Evaluates a fitted model on the test set. Returns overall and per-persona
    accuracy + implied win-rate (if the AI always played the counter move).
    """
    predictions = model.predict(test_df)
    actuals = test_df["target_move"].tolist()

    correct = sum(p == a for p, a in zip(predictions, actuals))
    accuracy = correct / len(actuals) if actuals else 0.0

    ai_moves = [_counter_move(p) for p in predictions]
    wins = sum(BEATS[Move(ai_move)] == actual for ai_move, actual in zip(ai_moves, actuals))
    win_rate = wins / len(actuals) if actuals else 0.0

    results = {
        "model": model.name,
        "overall_accuracy": round(accuracy, 4),
        "overall_win_rate": round(win_rate, 4),
        "per_persona": {},
    }

    eval_df = test_df.copy()
    eval_df["prediction"] = predictions
    eval_df["ai_move"] = ai_moves
    eval_df["correct"] = eval_df["prediction"] == eval_df["target_move"]
    eval_df["ai_won"] = eval_df.apply(
        lambda r: BEATS[Move(r["ai_move"])] == r["target_move"], axis=1
    )

    for persona, group in eval_df.groupby("model_used"):
        results["per_persona"][persona] = {
            "accuracy": round(group["correct"].mean(), 4),
            "win_rate": round(group["ai_won"].mean(), 4),
            "n_samples": len(group),
        }

    logger.info(
        f"[{model.name}] Overall accuracy: {results['overall_accuracy']:.2%} | "
        f"Overall win rate: {results['overall_win_rate']:.2%}"
    )
    return results


def run_baseline_comparison(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Fits and evaluates all baseline models, returning a comparison DataFrame."""
    from neuroplay.models.baselines import (
        MajorityBaseline,
        MarkovOrder1Baseline,
        MarkovOrder2Baseline,
        RandomBaseline,
    )

    models = [RandomBaseline(), MajorityBaseline(), MarkovOrder1Baseline(), MarkovOrder2Baseline()]
    all_results = []

    for model in models:
        model.fit(train_df)
        result = evaluate_model(model, test_df)
        all_results.append(
            {
                "model": result["model"],
                "accuracy": result["overall_accuracy"],
                "win_rate": result["overall_win_rate"],
            }
        )

    return pd.DataFrame(all_results)
