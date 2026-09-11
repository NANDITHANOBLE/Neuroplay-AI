"""
RPS environment for RL training — wraps a persona bot as the opponent,
using the ANN's predictions as part of the observable state.
"""

import numpy as np

from neuroplay.constants import BEATS, Move
from neuroplay.data_generation.personas import BasePersona
from neuroplay.models.ann_model import ANNBaseline

STATE_DIM = 7


class RPSEnvironment:
    """
    A single-match RPS environment. The opponent is a persona bot; the agent's
    state includes the ANN's move prediction + confidence + behavioral features.
    """

    def __init__(self, persona: BasePersona, ann: ANNBaseline, max_rounds: int = 100):
        self.persona = persona
        self.ann = ann
        self.max_rounds = max_rounds
        self.round_num = 0
        self.streak = 0
        self.recent_results: list[int] = []  # 1=win, 0=draw/loss for rolling win rate

    def reset(self) -> np.ndarray:
        self.round_num = 0
        self.streak = 0
        self.recent_results = []
        self.persona.history = []
        self.persona.opponent_history = []
        return self._get_state(predicted_move=0, confidence=0.34)

    def _get_state(self, predicted_move: int, confidence: float) -> np.ndarray:
        pred_onehot = [0, 0, 0]
        pred_onehot[predicted_move] = 1
        rolling_win_rate_5 = np.mean(self.recent_results[-5:]) if self.recent_results else 0.0
        drift_warning = 0.0  # placeholder; wired to Phase 14 in later integration
        return np.array(
            pred_onehot + [confidence, self.streak, rolling_win_rate_5, drift_warning],
            dtype=np.float32,
        )

    def step(self, agent_action: int) -> tuple[np.ndarray, float, bool]:
        """
        agent_action: the move (0/1/2) the RL agent chooses to play.
        Returns (next_state, reward, done).
        """
        opponent_move = self.persona.next_move()

        if agent_action == opponent_move:
            reward = 0.0
            self.streak = 0
            self.recent_results.append(0)
        elif BEATS[Move(agent_action)] == opponent_move:
            reward = 1.0
            self.streak = self.streak + 1 if self.streak >= 0 else 1
            self.recent_results.append(1)
        else:
            reward = -1.0
            self.streak = self.streak - 1 if self.streak <= 0 else -1
            self.recent_results.append(0)

        self.persona.record(opponent_move, agent_action)
        self.round_num += 1
        done = self.round_num >= self.max_rounds

        # Simplified prediction for next state (real integration in Phase 17)
        predicted_move = opponent_move
        confidence = 0.5
        next_state = self._get_state(predicted_move, confidence)

        return next_state, reward, done
