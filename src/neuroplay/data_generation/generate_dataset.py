"""
CLI entry point for synthetic dataset generation.
Run: python -m neuroplay.data_generation.generate_dataset
"""

from neuroplay.data_generation.personas import (
    CyclicBot,
    DriftingBot,
    FrequencyBiasedBot,
    MarkovOrder2Bot,
    RandomBot,
    WinStayLoseShiftBot,
)
from neuroplay.data_generation.simulator import ensure_synthetic_user, simulate_match
from neuroplay.logger import get_logger

logger = get_logger(__name__)

MATCHES_PER_PERSONA = 20
ROUNDS_PER_MATCH = 100


def main() -> None:
    user_id = ensure_synthetic_user()

    persona_factories = [
        lambda: RandomBot(),
        lambda: WinStayLoseShiftBot(),
        lambda: CyclicBot(),
        lambda: FrequencyBiasedBot(),
        lambda: MarkovOrder2Bot(),
    ]

    total_matches = 0

    for factory in persona_factories:
        for _ in range(MATCHES_PER_PERSONA):
            persona = factory()
            simulate_match(persona, user_id, num_rounds=ROUNDS_PER_MATCH)
            total_matches += 1

    # Generate DriftingBot matches: switches persona mid-match
    for _ in range(MATCHES_PER_PERSONA):
        persona_a = WinStayLoseShiftBot()
        persona_b = CyclicBot()
        drift_round = ROUNDS_PER_MATCH // 2
        drifting = DriftingBot(persona_a, persona_b, drift_round)
        simulate_match(drifting, user_id, num_rounds=ROUNDS_PER_MATCH, model_used="synthetic_drift")
        total_matches += 1

    logger.info(f"✅ Dataset generation complete. Total matches created: {total_matches}")


if __name__ == "__main__":
    main()
