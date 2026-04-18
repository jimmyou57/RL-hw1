import os
import json
import random
import numpy as np
import gymnasium as gym

from utils import print_stats, plot_baseline, record_episodes


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def run_random_baseline(num_episodes: int = 100, seed: int = 42) -> dict:
    """
    Run random policy on LunarLander-v3 for num_episodes and collect statistics.
    Returns a stats dict compatible with utils.print_stats() and utils.plot_baseline().
    """
    set_seed(seed)

    env = gym.make("LunarLander-v3")

    episode_rewards = []
    episode_lengths = []
    successes = 0

    for ep in range(num_episodes):
        state, info = env.reset(seed=seed + ep)

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            state = next_state
            done = terminated or truncated

        episode_rewards.append(float(total_reward))
        episode_lengths.append(int(steps))

        # Part A 常見的簡化成功判準：episode reward > 200
        if total_reward > 200:
            successes += 1

        if (ep + 1) % 10 == 0:
            print(f"[{ep + 1:3d}/{num_episodes}] reward = {total_reward:8.2f}, steps = {steps}")

    env.close()

    stats = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": float(successes / num_episodes),
    }
    return stats


def save_stats_files(stats: dict, out_dir: str = "outputs/part_a"):
    """
    Save baseline statistics as txt and json for later report writing.
    """
    os.makedirs(out_dir, exist_ok=True)

    txt_path = os.path.join(out_dir, "baseline_stats.txt")
    json_path = os.path.join(out_dir, "baseline_stats.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Random Policy Baseline Statistics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Mean reward : {stats['mean_reward']:.2f}\n")
        f.write(f"Std reward  : {stats['std_reward']:.2f}\n")
        f.write(f"Min reward  : {stats['min_reward']:.2f}\n")
        f.write(f"Max reward  : {stats['max_reward']:.2f}\n")
        f.write(f"Mean length : {stats['mean_length']:.2f}\n")
        f.write(f"Success rate: {stats['success_rate'] * 100:.2f}%\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved text stats  -> {txt_path}")
    print(f"Saved json stats  -> {json_path}")


def record_random_gifs(num_episodes: int = 5, out_dir: str = "videos/part_a_random", seed: int = 42):
    """
    Record random-policy episodes as GIFs using utils.record_episodes().
    """
    set_seed(seed)

    # 用一個獨立 env 只提供 action_space.sample() 給 policy_fn
    action_env = gym.make("LunarLander-v3")

    try:
        record_episodes(
            num_episodes=num_episodes,
            out_dir=out_dir,
            policy_fn=lambda state: action_env.action_space.sample(),
        )
    finally:
        action_env.close()


def main():
    os.makedirs("outputs/part_a", exist_ok=True)
    os.makedirs("videos/part_a", exist_ok=True)

    # 1. 跑 100 episodes random baseline
    stats = run_random_baseline(num_episodes=100, seed=42)

    # 2. 用 utils.py 印出整理好的統計
    print_stats(stats)

    # 3. 存成 txt/json
    save_stats_files(stats, out_dir="outputs/part_a")

    # 4. 用 utils.py 畫 baseline 圖
    plot_baseline(stats, out_path="outputs/part_a/baseline_stats.png")

    # 5. 用 utils.py 存 5 段 random policy GIF
    record_random_gifs(num_episodes=5, out_dir="videos/part_a_random", seed=42)


if __name__ == "__main__":
    main()