"""
Statistical significance testing — McNemar's test to confirm model performance
differences are not due to random chance.
"""

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

from neuroplay.logger import get_logger

logger = get_logger(__name__)


def mcnemar_test(actuals: list[int], predictions_a: list[int], predictions_b: list[int]) -> dict:
    """
    Compares two models' predictions against ground truth using McNemar's test.
    Returns whether model A significantly outperforms model B (p < 0.05).
    """
    a_correct_b_wrong = sum(
        (a == act) and (b != act) for a, b, act in zip(predictions_a, predictions_b, actuals)
    )
    a_wrong_b_correct = sum(
        (a != act) and (b == act) for a, b, act in zip(predictions_a, predictions_b, actuals)
    )

    table = np.array([[0, a_correct_b_wrong], [a_wrong_b_correct, 0]])
    result = mcnemar(table, exact=True)

    is_significant = result.pvalue < 0.05
    logger.info(
        f"McNemar's test: statistic={result.statistic:.4f}, p-value={result.pvalue:.4f}, "
        f"significant={is_significant}"
    )

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(is_significant),
    }
