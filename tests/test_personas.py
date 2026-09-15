"""Tests for synthetic persona bots — validates deterministic behavioral logic."""

from neuroplay.constants import Move
from neuroplay.data_generation.personas import CyclicBot, WinStayLoseShiftBot


def test_cyclic_bot_follows_rock_paper_scissors_cycle():
    bot = CyclicBot()
    moves = []
    for _ in range(6):
        move = bot.next_move()
        moves.append(move)
        bot.record(move, Move.ROCK)  # opponent move irrelevant to CyclicBot
    assert moves == [Move.ROCK, Move.PAPER, Move.SCISSORS] * 2


def test_win_stay_lose_shift_repeats_after_win():
    bot = WinStayLoseShiftBot()
    bot.record(Move.ROCK, Move.SCISSORS)  # Rock beats Scissors -> bot won
    next_move = bot.next_move()
    assert next_move == Move.ROCK


def test_win_stay_lose_shift_switches_after_loss():
    bot = WinStayLoseShiftBot()
    bot.record(Move.ROCK, Move.PAPER)  # Paper beats Rock -> bot lost
    next_move = bot.next_move()
    assert next_move == Move.PAPER  # shifts to the move that beat it
