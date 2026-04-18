import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim, action_dim, config, device):
        self.action_dim = action_dim
        self.device = device
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]

        self.q_network = QNetwork(
            state_dim, action_dim, hidden_dim=config.get("hidden_dim", 128)
        ).to(device)
        self.target_network = QNetwork(
            state_dim, action_dim, hidden_dim=config.get("hidden_dim", 128)
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=config["learning_rate"]
        )
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(config["buffer_size"])

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        current_q = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        mean_max_q = self.q_network(states).max(dim=1)[0].mean().item()
        return loss.item(), mean_max_q

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


def train_agent(config, device, verbose=True):
    set_seed(config.get("seed", 42))

    env = gym.make(config.get("env_name", "LunarLander-v3"))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, config, device)

    rewards_history = []
    avg_losses = []
    epsilons = []
    mean_q_values = []
    solved_at = None

    epsilon = config["epsilon_start"]

    for episode in range(config["num_episodes"]):
        state, _ = env.reset(seed=config.get("seed", 42) + episode)
        done = False
        episode_reward = 0.0
        episode_losses = []
        episode_qs = []

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)

            loss, mean_q = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
                episode_qs.append(mean_q)

            state = next_state
            episode_reward += reward

        if (episode + 1) % config["target_update_freq"] == 0:
            agent.update_target_network()

        epsilon = max(config["epsilon_end"], epsilon * config["epsilon_decay"])

        rewards_history.append(float(episode_reward))
        epsilons.append(float(epsilon))
        avg_losses.append(float(np.mean(episode_losses)) if episode_losses else float("nan"))
        mean_q_values.append(float(np.mean(episode_qs)) if episode_qs else float("nan"))

        if solved_at is None and len(rewards_history) >= 100:
            if np.mean(rewards_history[-100:]) > 200:
                solved_at = episode + 1

        if verbose and (episode + 1) % 50 == 0:
            print(
                f"Episode {episode + 1}, "
                f"Reward: {episode_reward:.2f}, "
                f"Epsilon: {epsilon:.3f}, "
                f"Avg Loss: {avg_losses[-1]:.4f}, "
                f"Mean Q: {mean_q_values[-1]:.4f}"
            )

    env.close()

    metrics = {
        "episode_rewards": rewards_history,
        "avg_losses": avg_losses,
        "epsilons": epsilons,
        "mean_q_values": mean_q_values,
        "solved_at": solved_at,
    }

    return agent, metrics


def test_agent(agent, env_name="LunarLander-v3", num_episodes=100, seed=42):
    env = gym.make(env_name)

    rewards = []
    lengths = []
    successes = 0

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            steps += 1

        rewards.append(float(episode_reward))
        lengths.append(int(steps))
        if episode_reward > 200:
            successes += 1

    env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(successes / num_episodes),
        "episode_rewards": rewards,
        "episode_lengths": lengths,
    }