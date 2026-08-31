"""
Global constants for NeuroPlay-AI.
Move encodings and core Rock-Paper-Scissors game rules used across
data generation, modeling, backend, and frontend modules.
"""

from enum import IntEnum


class Move(IntEnum):
    """Integer-encoded moves — numeric so they can feed directly into ML models."""

    ROCK = 0
    PAPER = 1
    SCISSORS = 2


MOVE_NAMES = {
    Move.ROCK: "Rock",
    Move.PAPER: "Paper",
    Move.SCISSORS: "Scissors",
}

# beats[A] = B means "A beats B"
BEATS = {
    Move.ROCK: Move.SCISSORS,
    Move.PAPER: Move.ROCK,
    Move.SCISSORS: Move.PAPER,
}

# The theoretical win rate of a purely random player against a purely
# random opponent — our baseline that every model must outperform.
RANDOM_BASELINE_WIN_RATE = 1 / 3
