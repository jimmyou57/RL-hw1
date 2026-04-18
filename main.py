import json
import os
import torch

from rl_core import train_agent, test_agent
from utils import plot_training_curves, save_checkpoint, record_episodes

CONFIG = {
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs("outputs/part_b", exist_ok=True)

    agent, metrics = train_agent(CONFIG, device, verbose=True)

    plot_training_curves(metrics, out_dir="outputs/part_c")
    save_checkpoint(
        agent,
        CONFIG["num_episodes"],
        metrics["episode_rewards"],
        filename="outputs/part_b/checkpoint_final.pt",
    )

    test_stats = test_agent(
        agent,
        env_name=CONFIG["env_name"],
        num_episodes=100,
        seed=CONFIG["seed"] + 10000,
    )
    
    baseline_bundle = {
        "config": CONFIG,
        "metrics": metrics,
        "test_stats": test_stats,
    }

    with open("outputs/part_c/baseline_bundle.json", "w", encoding="utf-8") as f:
        json.dump(baseline_bundle, f, indent=2)

    print("Saved baseline bundle → outputs/part_c/baseline_bundle.json")

    print(f"Mean reward  : {test_stats['mean_reward']:.2f}")
    print(f"Std reward   : {test_stats['std_reward']:.2f}")
    print(f"Min reward   : {test_stats['min_reward']:.2f}")
    print(f"Max reward   : {test_stats['max_reward']:.2f}")
    print(f"Mean length  : {test_stats['mean_length']:.2f}")
    print(f"Success rate : {test_stats['success_rate'] * 100:.2f}%")

    with open("outputs/part_c/test_stats.txt", "w", encoding="utf-8") as f:
        f.write("=== Test Performance (No Exploration) ===\n")
        f.write(f"Mean reward  : {test_stats['mean_reward']:.2f}\n")
        f.write(f"Std reward   : {test_stats['std_reward']:.2f}\n")
        f.write(f"Min reward   : {test_stats['min_reward']:.2f}\n")
        f.write(f"Max reward   : {test_stats['max_reward']:.2f}\n")
        f.write(f"Mean length  : {test_stats['mean_length']:.2f}\n")
        f.write(f"Success rate : {test_stats['success_rate'] * 100:.2f}%\n")

    record_episodes(
        num_episodes=5,
        out_dir="videos/part_c",
        policy_fn=lambda state: agent.select_action(state, epsilon=0.0),
    )


if __name__ == "__main__":
    main()