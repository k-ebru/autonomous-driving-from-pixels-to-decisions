"""Record a GIF of a trained policy driving on CarRacing v3.

Usage:
    python -m src.record_gif \
        --config configs/model_A.yaml \
        --model-path runs/model_A/models/best_model.zip \
        --output results/agent.gif \
        --max-steps 1000 \
        --fps 30

The GIF captures the raw RGB render of the environment, not the
preprocessed agent input, so the result is visually meaningful for a
README.
"""

import argparse

import gymnasium as gym
import imageio
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

from src.env_utils import make_env


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a driving GIF.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="results/agent.gif")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env"]

    # Agent environment for inference (preprocessed observations).
    agent_env = DummyVecEnv(
        [
            make_env(
                rank=0,
                seed=args.seed,
                env_id=env_cfg["env_id"],
                skip_zoom_steps=env_cfg["skip_zoom_steps"],
                frame_size=env_cfg["frame_size"],
            )
        ]
    )
    agent_env = VecMonitor(agent_env)
    agent_env = VecFrameStack(agent_env, n_stack=env_cfg["n_stack"], channels_order="first")

    # Render environment for the GIF (raw RGB frames).
    render_env = gym.make(env_cfg["env_id"], continuous=True, render_mode="rgb_array")
    render_env.reset(seed=args.seed)

    print(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path)

    obs = agent_env.reset()
    # Step the render env through the same zoom skip so the two envs stay aligned.
    no_op = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(env_cfg["skip_zoom_steps"]):
        render_env.step(no_op)

    frames = []
    done = False
    step = 0
    total_reward = 0.0

    while not done and step < args.max_steps:
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, r, d, _ = agent_env.step(action)
        # Apply the same continuous action to the render env to keep frames in sync.
        _, render_r, terminated, truncated, _ = render_env.step(action[0])
        frame = render_env.render()
        frames.append(frame)
        total_reward += float(r[0])
        done = bool(d[0]) or terminated or truncated
        step += 1

    print(f"Captured {len(frames)} frames. Episode reward: {total_reward:.2f}")
    print(f"Writing GIF to {args.output} at {args.fps} fps")
    imageio.mimsave(args.output, frames, fps=args.fps, loop=0)

    agent_env.close()
    render_env.close()
    print("Done.")


if __name__ == "__main__":
    main()
