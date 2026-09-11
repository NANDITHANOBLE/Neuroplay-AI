"""
DQN agent — learns optimal counter-strategy via Q-learning with a small
feed-forward network, experience replay, and epsilon-greedy exploration.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from neuroplay.rl.environment import STATE_DIM


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, num_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN with target network, experience replay, and epsilon-greedy exploration."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        num_actions: int = 3,
        gamma: float = 0.95,
        lr: float = 0.001,
        device: str = "cpu",
    ):
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma

        self.q_network = QNetwork(state_dim, num_actions).to(device)
        self.target_network = QNetwork(state_dim, num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self, batch_size: int = 64) -> float | None:
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)

        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(dim=1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
