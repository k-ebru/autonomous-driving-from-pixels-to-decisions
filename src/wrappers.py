"""Environment wrappers for CarRacing-v3.

Three wrappers are applied to every environment instance:

1. SafeCarAction normalises action arrays so PPO output is accepted by the
   underlying continuous action space.
2. SkipZoomWrapper advances past the camera zoom intro that CarRacing plays
   at the start of every episode, avoiding wasted gradient signal.
3. ImageProcessWrapper converts RGB frames to grayscale and resizes to
   84 by 84. This matches the input pipeline used by Mnih et al. (2015)
   and keeps the CNN policy compact.
"""

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SafeCarAction(gym.Wrapper):
    def step(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        if action.ndim > 1:
            action = action.reshape(-1)
        return self.env.step(action)


class SkipZoomWrapper(gym.Wrapper):
    def __init__(self, env, skip_steps: int = 50):
        super().__init__(env)
        self.skip_steps = skip_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        no_op = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(self.skip_steps):
            obs, _, done, _, _ = self.env.step(no_op)
            if done:
                break
        return obs, info


class ImageProcessWrapper(gym.ObservationWrapper):
    def __init__(self, env, frame_size: int = 84):
        super().__init__(env)
        self.frame_size = frame_size
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, frame_size, frame_size),
            dtype=np.uint8,
        )

    def observation(self, obs):
        obs = obs.astype(np.uint8)
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(
            gray,
            (self.frame_size, self.frame_size),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized[None, :, :]
