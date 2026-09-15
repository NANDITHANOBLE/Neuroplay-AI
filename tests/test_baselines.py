"""Tests for baseline models — verifying deterministic Markov prediction logic."""

import pandas as pd

from neuroplay.models.baselines import MajorityBaseline, MarkovOrder1Baseline


def test_majority_baseline_predicts_most_common_move():
    train_df = pd.DataFrame({"target_move": [0, 0, 0, 1, 2]})
    model = MajorityBaseline()
    model.fit(train_df)

    test_df = pd.DataFrame({"player_window": [[0, 1]], "target_move": [0]})
    predictions = model.predict(test_df)
    assert predictions == [0]


def test_markov_order1_learns_deterministic_transition():
    # Every time last move is Rock (0), next move is always Paper (1)
    train_df = pd.DataFrame(
        {
            "player_window": [[0, 0], [1, 0], [2, 0]],
            "target_move": [1, 1, 1],
        }
    )
    model = MarkovOrder1Baseline()
    model.fit(train_df)

    test_df = pd.DataFrame({"player_window": [[2, 0]]})
    predictions = model.predict(test_df)
    assert predictions == [1]
