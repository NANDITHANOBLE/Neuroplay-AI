"""
Persona bot implementations for synthetic Rock-Paper-Scissors data generation.
Each persona models a distinct, research-grounded human decision-making pattern.
"""

import random
from abc import ABC, abstractmethod

from neuroplay.constants import Move


class BasePersona(ABC):
    """Base class for all synthetic persona bots."""

    name: str = "base"

    def __init__(self) -> None:
        self.history: list[Move] = []
        self.opponent_history: list[Move] = []

    @abstractmethod
    def next_move(self) -> Move:
        """Returns this persona's next move given its history so far."""
        raise NotImplementedError

    def record(self, own_move: Move, opponent_move: Move) -> None:
        self.history.append(own_move)
        self.opponent_history.append(opponent_move)


class RandomBot(BasePersona):
    """Plays uniformly at random — the theoretical baseline persona."""

    name = "random"

    def next_move(self) -> Move:
        return random.choice(list(Move))


class WinStayLoseShiftBot(BasePersona):
    """
    Models the most well-documented human heuristic:
    - After a win: repeat the same move.
    - After a loss: switch to the move that just beat them.
    - After a draw: pick randomly.
    """

    name = "win_stay_lose_shift"

    def next_move(self) -> Move:
        if not self.history:
            return random.choice(list(Move))

        last_own = self.history[-1]
        last_opp = self.opponent_history[-1]

        if last_own == last_opp:  # draw
            return random.choice(list(Move))
        elif self._beats(last_own, last_opp):  # won
            return last_own
        else:  # lost -> shift to the move that beat them
            return last_opp

    @staticmethod
    def _beats(a: Move, b: Move) -> bool:
        from neuroplay.constants import BEATS

        return BEATS[a] == b


class CyclicBot(BasePersona):
    """Plays a fixed deterministic cycle: Rock -> Paper -> Scissors -> repeat."""

    name = "cyclic"

    def next_move(self) -> Move:
        idx = len(self.history) % 3
        return list(Move)[idx]


class FrequencyBiasedBot(BasePersona):
    """Favors one move with a fixed probability distribution."""

    name = "frequency_biased"

    def __init__(self, weights: tuple[float, float, float] = (0.5, 0.25, 0.25)) -> None:
        super().__init__()
        self.weights = weights

    def next_move(self) -> Move:
        return random.choices(list(Move), weights=self.weights, k=1)[0]


class MarkovOrder2Bot(BasePersona):
    """
    Chooses moves based on a probabilistic transition matrix conditioned on
    the last 2 moves played. Falls back to random until enough history exists.
    """

    name = "markov_order2"

    def __init__(self) -> None:
        super().__init__()
        # Randomly initialized transition matrix: {(last2, last1): weights}
        self.transition_bias = random.choice(list(Move))

    def next_move(self) -> Move:
        if len(self.history) < 2:
            return random.choice(list(Move))

        # Simple heuristic Markov-2 behavior: biased toward one move
        # after seeing a specific 2-move pattern, else random.
        last_two = (self.history[-2], self.history[-1])
        if last_two == (Move.ROCK, Move.ROCK):
            return self.transition_bias
        return random.choice(list(Move))


class DriftingBot(BasePersona):
    """
    Switches between two underlying personas at a randomly chosen round.
    Provides labeled ground-truth for concept drift detection (Phase 14).
    """

    name = "drifting"

    def __init__(self, persona_a: BasePersona, persona_b: BasePersona, drift_round: int) -> None:
        super().__init__()
        self.persona_a = persona_a
        self.persona_b = persona_b
        self.drift_round = drift_round
        self.current_round = 0

    def next_move(self) -> Move:
        active = self.persona_a if self.current_round < self.drift_round else self.persona_b
        move = active.next_move()
        return move

    def record(self, own_move: Move, opponent_move: Move) -> None:
        super().record(own_move, opponent_move)
        active = self.persona_a if self.current_round < self.drift_round else self.persona_b
        active.record(own_move, opponent_move)
        self.current_round += 1


PERSONA_REGISTRY = {
    "random": RandomBot,
    "win_stay_lose_shift": WinStayLoseShiftBot,
    "cyclic": CyclicBot,
    "frequency_biased": FrequencyBiasedBot,
    "markov_order2": MarkovOrder2Bot,
}
