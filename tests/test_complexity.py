"""Tests for Lempel-Ziv complexity — validates known high/low complexity sequences."""

from neuroplay.eda.complexity import lempel_ziv_complexity


def test_lz_complexity_zero_for_empty_or_single_element():
    assert lempel_ziv_complexity([]) == 0.0
    assert lempel_ziv_complexity([0]) == 0.0


def test_lz_complexity_low_for_repetitive_cyclic_sequence():
    cyclic_sequence = [0, 1, 2] * 10  # highly repetitive
    score = lempel_ziv_complexity(cyclic_sequence)
    assert score < 1.0


def test_lz_complexity_higher_for_less_repetitive_sequence():
    repetitive = [0] * 30
    varied = [0, 1, 2, 1, 0, 2, 2, 1, 0, 1, 2, 0, 1, 2, 0] * 2
    assert lempel_ziv_complexity(varied) > lempel_ziv_complexity(repetitive)
