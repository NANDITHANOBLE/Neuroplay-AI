"""
Training script for the DQN counter-strategy agent.
Trains against a mix of persona bots (Phase 4) to learn a general policy.
Run: python -m neuroplay.rl.train_dqn
"""

from neuroplay.config import settings
from neuroplay.data_generation.personas import (
    CyclicBot,
    FrequencyBiasedBot,
    MarkovOrder2Bot,
    RandomBot,
    WinStayLoseShiftBot,
)
from neuroplay.logger import get_logger
from neuroplay.models.ann_model import ANNBaseline
from neuroplay.rl.dqn_agent import DQNAgent
from neuroplay.rl.environment import RPSEnvironment

logger = get_logger(__name__)

NUM_EPISODES = 500
TARGET_UPDATE_FREQ = 10


def main() -> None:
    device = "cpu"
    ann_dir = settings.models_dir / "ann"
    ann = ANNBaseline(device=device)
    ann.load_weights(str(ann_dir / "ann_best.pt"))
    ann.load_scaler(str(ann_dir / "feature_scaler.json"))

    agent = DQNAgent(device=device)

    persona_factories = [
        lambda: RandomBot(),
        lambda: WinStayLoseShiftBot(),
        lambda: CyclicBot(),
        lambda: FrequencyBiasedBot(),
        lambda: MarkovOrder2Bot(),
    ]

    episode_rewards = []

    for episode in range(1, NUM_EPISODES + 1):
        persona = persona_factories[episode % len(persona_factories)]()
        env = RPSEnvironment(persona, ann)
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        if episode % 50 == 0:
            avg_reward = sum(episode_rewards[-50:]) / 50
            logger.info(
                f"Episode {episode}/{NUM_EPISODES} | Avg Reward (last 50): {avg_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    models_dir = settings.models_dir / "dqn"
    models_dir.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(agent.q_network.state_dict(), models_dir / "dqn_best.pt")
    logger.info(f"✅ DQN training complete. Model saved to {models_dir / 'dqn_best.pt'}")

    final_avg_reward = sum(episode_rewards[-50:]) / 50
    logger.info(f"Final average reward (last 50 episodes): {final_avg_reward:.3f}")
    logger.info(
        "Note: reward range is [-1, 1] per round; positive average indicates "
        "net winning strategy against the mixed persona pool."
    )


if __name__ == "__main__":
    main()
