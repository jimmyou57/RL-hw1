import json
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_core import train_agent, test_agent
from utils import plot_training_curves, save_checkpoint


BASE_CONFIG = {
    "env_name": "LunarLander-v3",
    "seed": 56,
    "learning_rate": 5e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "batch_size": 64,
    "buffer_size": 10000,
    "target_update_freq": 10,
    "num_episodes": 1000,
    "hidden_dim": 128,
}

# ===== 改需要實驗的參數 =====
PARAM_NAME = "buffer_size"
BASELINE_VALUE = 10000
NEW_VALUES = [5000, 20000]

BASELINE_BUNDLE_PATH = "outputs/part_c/baseline_bundle.json"


def moving_average(x, window=50):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def safe_label(param_name, value):
    return f"{param_name}_{value}"


def load_baseline_bundle(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def run_one_experiment(config, device, label, out_root):
    print(f"\n===== Running experiment: {label} =====")

    agent, metrics = train_agent(config, device, verbose=True)
    test_stats = test_agent(
        agent,
        env_name=config["env_name"],
        num_episodes=100,
        seed=config["seed"] + 20000,
    )

    curve_dir = os.path.join(out_root, "curves", label)
    os.makedirs(curve_dir, exist_ok=True)
    plot_training_curves(metrics, out_dir=curve_dir)

    ckpt_path = os.path.join(out_root, "checkpoints", f"{label}.pt")
    save_checkpoint(
        agent,
        config["num_episodes"],
        metrics["episode_rewards"],
        filename=ckpt_path,
    )

    print(f"Test mean reward  : {test_stats['mean_reward']:.2f}")
    print(f"Test success rate : {test_stats['success_rate'] * 100:.2f}%")
    print(f"Solved at         : {metrics['solved_at']}")

    return {
        "config": config,
        "metrics": metrics,
        "test_stats": test_stats,
    }


def plot_param_comparison(results, param_name, out_path):
    plt.figure(figsize=(10, 6))

    for label, result in results.items():
        rewards = result["metrics"]["episode_rewards"]
        ma = moving_average(rewards, window=50)
        if len(rewards) >= 50:
            episodes = np.arange(len(ma)) + 50
        else:
            episodes = np.arange(len(ma)) + 1
        plt.plot(episodes, ma, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Reward (50-episode moving average)")
    plt.title(f"{param_name} comparison on LunarLander-v3")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_summary(results, param_name, out_path):
    summary = {}
    for label, result in results.items():
        metrics = result["metrics"]
        test_stats = result["test_stats"]
        config = result["config"]

        summary[label] = {
            param_name: config[param_name],
            "solved_at": metrics["solved_at"],
            "final_100_avg_reward": float(np.mean(metrics["episode_rewards"][-100:])),
            "test_mean_reward": test_stats["mean_reward"],
            "test_std_reward": test_stats["std_reward"],
            "test_success_rate": test_stats["success_rate"],
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_root = os.path.join("outputs", "part_d")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, "curves"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "checkpoints"), exist_ok=True)

    results = {}

    # 1. 先讀 baseline，不重跑
    if not os.path.exists(BASELINE_BUNDLE_PATH):
        raise FileNotFoundError(
            f"找不到 baseline 檔案：{BASELINE_BUNDLE_PATH}\n"
            "請先跑 main.py，產生 outputs/part_c/baseline_bundle.json"
        )

    baseline_bundle = load_baseline_bundle(BASELINE_BUNDLE_PATH)
    baseline_label = f"baseline_{PARAM_NAME}_{BASELINE_VALUE}"
    results[baseline_label] = baseline_bundle

    print(f"Loaded baseline from {BASELINE_BUNDLE_PATH}")
    print(f"Baseline {PARAM_NAME} = {baseline_bundle['config'][PARAM_NAME]}")

    # 2. 只跑新的參數值
    for value in NEW_VALUES:
        label = safe_label(PARAM_NAME, value)

        config = deepcopy(BASE_CONFIG)
        config[PARAM_NAME] = value

        result = run_one_experiment(config, device, label, out_root)
        results[label] = result

    # 3. 畫比較圖
    comparison_plot_path = os.path.join(out_root, f"{PARAM_NAME}_comparison.png")
    plot_param_comparison(results, PARAM_NAME, comparison_plot_path)

    # 4. 存 summary
    summary_path = os.path.join(out_root, f"{PARAM_NAME}_summary.json")
    summary = save_summary(results, PARAM_NAME, summary_path)

    print("\n===== Final Summary =====")
    for label, item in summary.items():
        print(
            f"{label}: "
            f"{PARAM_NAME}={item[PARAM_NAME]}, "
            f"solved_at={item['solved_at']}, "
            f"final_100_avg={item['final_100_avg_reward']:.2f}, "
            f"test_mean={item['test_mean_reward']:.2f}, "
            f"success_rate={item['test_success_rate'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()