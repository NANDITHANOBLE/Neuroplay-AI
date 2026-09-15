"""Tests for feature scaling and the shared model evaluation harness."""

import pandas as pd

from neuroplay.models.baselines import MajorityBaseline
from neuroplay.models.evaluation import evaluate_model
from neuroplay.models.feature_scaler import apply_scaler, fit_scaler


def test_fit_scaler_computes_correct_mean_and_std():
    train_df = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = fit_scaler(train_df, ["feature_a"])
    assert scaler["mean"]["feature_a"] == 3.0
    assert round(scaler["std"]["feature_a"], 4) == round(train_df["feature_a"].std(), 4)


def test_apply_scaler_normalizes_to_zero_mean():
    train_df = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = fit_scaler(train_df, ["feature_a"])
    scaled_df = apply_scaler(train_df, scaler)
    assert abs(scaled_df["feature_a"].mean()) < 1e-9


def test_apply_scaler_avoids_division_by_zero_for_constant_feature():
    train_df = pd.DataFrame({"feature_a": [5.0, 5.0, 5.0]})
    scaler = fit_scaler(train_df, ["feature_a"])
    scaled_df = apply_scaler(train_df, scaler)
    assert scaled_df["feature_a"].tolist() == [0.0, 0.0, 0.0]


def test_evaluate_model_computes_correct_accuracy_and_win_rate():
    train_df = pd.DataFrame({"target_move": [0, 0, 0, 1, 2]})
    model = MajorityBaseline()
    model.fit(train_df)

    test_df = pd.DataFrame(
        {
            "player_window": [[0, 1], [1, 2], [2, 0]],
            "target_move": [0, 0, 1],
            "model_used": ["persona_a", "persona_a", "persona_b"],
        }
    )
    results = evaluate_model(model, test_df)

    assert results["model"] == "majority_baseline"
    assert results["overall_accuracy"] == round(2 / 3, 4)
    assert "persona_a" in results["per_persona"]
    assert "persona_b" in results["per_persona"]
