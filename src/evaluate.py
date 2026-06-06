"""Evaluation entry point.

Runs the three protocols described in the report:

1. Stochastic performance over N episodes (action sampling).
2. Deterministic stability over M episodes (mean actions).
3. Generalisation across unseen track seeds.

Usage:
    python -m src.evaluate \
        --config configs/model_A.yaml \
        --model-path runs/model_A/models/best_model.zip
"""

import argparse

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

from src.env_utils import make_env


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_single_env(cfg: dict, seed: int):
    env_cfg = cfg["env"]
    env = DummyVecEnv(
        [
            make_env(
                rank=0,
                seed=seed,
                env_id=env_cfg["env_id"],
                skip_zoom_steps=env_cfg["skip_zoom_steps"],
                frame_size=env_cfg["frame_size"],
            )
        ]
    )
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=env_cfg["n_stack"], channels_order="first")
    return env


def run_episodes(model, env, n_episodes: int, deterministic: bool) -> list:
    rewards = []
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, d, _ = env.step(action)
            episode_reward += float(r[0])
            done = bool(d[0])
        rewards.append(episode_reward)
        print(f"  Episode {i + 1}: {episode_reward:.2f}")
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg["evaluation"]

    print(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path)

    eval_env = make_single_env(cfg, seed=cfg["eval_seed"])

    print("\n=== Stochastic test ===")
    stochastic_rewards = run_episodes(
        model,
        eval_env,
        n_episodes=eval_cfg["stochastic_episodes"],
        deterministic=False,
    )
    print(f"Stochastic mean: {np.mean(stochastic_rewards):.2f}")

    print("\n=== Deterministic test ===")
    deterministic_rewards = run_episodes(
        model,
        eval_env,
        n_episodes=eval_cfg["deterministic_episodes"],
        deterministic=True,
    )
    print(f"Deterministic mean: {np.mean(deterministic_rewards):.2f}")

    eval_env.close()

    print("\n=== Generalisation test (unseen seeds) ===")
    unseen_rewards = []
    for s in eval_cfg["unseen_seeds"]:
        temp_env = make_single_env(cfg, seed=s)
        obs = temp_env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, r, d, _ = temp_env.step(action)
            episode_reward += float(r[0])
            done = bool(d[0])
        unseen_rewards.append(episode_reward)
        print(f"  Seed {s}: {episode_reward:.2f}")
        temp_env.close()

    mean_unseen = float(np.mean(unseen_rewards))
    std_unseen = float(np.std(unseen_rewards))
    capacity = mean_unseen / eval_cfg["solved_threshold"] * 100.0

    print("\n=== Summary ===")
    print(f"Stochastic mean   : {np.mean(stochastic_rewards):.2f}")
    print(f"Deterministic mean: {np.mean(deterministic_rewards):.2f}")
    print(f"Unseen tracks     : {mean_unseen:.2f} +/- {std_unseen:.2f}")
    print(f"Generalisation cap: {capacity:.1f}% of {eval_cfg['solved_threshold']}")


if __name__ == "__main__":
    main()
