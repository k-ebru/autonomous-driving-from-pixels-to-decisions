"""Config driven PPO training entry point.

Usage:
    python -m src.train --config configs/model_A.yaml

Outputs are written under runs/<run_name>/ with subfolders for the model
checkpoints, CSV logs, and TensorBoard event files.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
)

from src.callbacks import TimeLimitCallback, linear_schedule
from src.env_utils import make_env


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_train_env(cfg: dict):
    env_cfg = cfg["env"]
    seed = cfg["seed"]
    env_fns = [
        make_env(
            rank=i,
            seed=seed,
            env_id=env_cfg["env_id"],
            skip_zoom_steps=env_cfg["skip_zoom_steps"],
            frame_size=env_cfg["frame_size"],
        )
        for i in range(env_cfg["n_envs"])
    ]
    env = SubprocVecEnv(env_fns, start_method="spawn")
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=env_cfg["n_stack"], channels_order="first")
    return env


def build_eval_env(cfg: dict):
    env_cfg = cfg["env"]
    eval_seed = cfg["eval_seed"]
    env = DummyVecEnv(
        [
            make_env(
                rank=0,
                seed=eval_seed,
                env_id=env_cfg["env_id"],
                skip_zoom_steps=env_cfg["skip_zoom_steps"],
                frame_size=env_cfg["frame_size"],
            )
        ]
    )
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=env_cfg["n_stack"], channels_order="first")
    return env


def train(cfg: dict, run_dir: Path) -> None:
    set_global_seeds(cfg["seed"])

    model_dir = run_dir / "models"
    log_dir = run_dir / "logs"
    tb_dir = run_dir / "tensorboard"
    for d in (model_dir, log_dir, tb_dir):
        d.mkdir(parents=True, exist_ok=True)

    new_logger = configure(str(tb_dir), ["stdout", "csv", "tensorboard"])

    train_env = build_train_env(cfg)
    eval_env = build_eval_env(cfg)

    training_cfg = cfg["training"]
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=training_cfg["eval_freq"],
        n_eval_episodes=training_cfg["n_eval_episodes"],
        deterministic=True,
        verbose=1,
    )
    time_callback = TimeLimitCallback(
        time_limit_secs=training_cfg["time_limit_secs"],
        save_path=str(model_dir),
    )

    ppo_cfg = cfg["ppo"]
    model = PPO(
        ppo_cfg["policy"],
        train_env,
        verbose=1,
        tensorboard_log=str(tb_dir),
        learning_rate=linear_schedule(ppo_cfg["learning_rate"]),
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        ent_coef=ppo_cfg["ent_coef"],
        clip_range=ppo_cfg["clip_range"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        device="auto",
        seed=cfg["seed"],
    )
    model.set_logger(new_logger)
    model.learn(
        total_timesteps=training_cfg["total_timesteps"],
        callback=[eval_callback, time_callback],
    )
    model.save(os.path.join(str(model_dir), "final_model_completed"))

    eval_env.close()
    train_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on CarRacing-v3.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML config file (see configs/).",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Root directory for run outputs.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.runs_dir) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist a copy of the config next to the outputs for reproducibility.
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train(cfg, run_dir)


if __name__ == "__main__":
    main()
