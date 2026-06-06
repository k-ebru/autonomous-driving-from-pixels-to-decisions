"""Environment factory helpers.

A small wrapper stack is applied around CarRacing v3, then vectorised by the
caller using SubprocVecEnv or DummyVecEnv plus VecFrameStack.
"""

import gymnasium as gym

from src.wrappers import ImageProcessWrapper, SafeCarAction, SkipZoomWrapper


def make_env(rank: int, seed: int, env_id: str, skip_zoom_steps: int, frame_size: int):
    """Return a thunk that builds one fully wrapped environment instance."""

    def _init():
        env = gym.make(env_id, continuous=True, render_mode="rgb_array")
        env.reset(seed=seed + rank)
        env = SafeCarAction(env)
        env = SkipZoomWrapper(env, skip_steps=skip_zoom_steps)
        env = ImageProcessWrapper(env, frame_size=frame_size)
        return env

    return _init
