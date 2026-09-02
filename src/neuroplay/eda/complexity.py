"""
Sequence complexity metrics for behavioral analysis.
Implements Lempel-Ziv complexity to quantify how "random" vs "patterned"
a persona's move sequence is — validates persona design and feeds Phase 21
(Psychology Classification).
"""

import numpy as np


def lempel_ziv_complexity(sequence: list[int]) -> float:
    """
    Computes a normalized Lempel-Ziv complexity score for a move sequence.
    Higher score = more random/unpredictable; lower = more patterned.
    """
    s = "".join(str(x) for x in sequence)
    n = len(s)
    if n == 0:
        return 0.0

    i, k, l_ = 0, 1, 1
    c = 1  # complexity count
    k_max = 1

    while True:
        if s[i + k - 1] == s[l_ + k - 1]:
            k += 1
            if l_ + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l_:
                c += 1
                l_ += k_max
                if l_ + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1

    normalized = (c * np.log2(n)) / n if n > 1 else 0.0
    return float(normalized)
