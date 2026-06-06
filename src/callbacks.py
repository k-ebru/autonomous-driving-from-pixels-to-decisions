"""Custom callbacks and learning rate schedule for PPO training."""

import os
import time

from stable_baselines3.common.callbacks import BaseCallback


def linear_schedule(initial_value: float):
    """Linear decay schedule used for the PPO learning rate.

    Returns a callable that maps progress_remaining (1.0 to 0.0) to the
    current learning rate.
    """

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


class TimeLimitCallback(BaseCallback):
    """Stops training after a wall clock time budget and saves a checkpoint.

    Used to make the three sensitivity runs comparable on equal compute.
    """

    def __init__(self, time_limit_secs: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.time_limit_secs = time_limit_secs
        self.save_path = save_path
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if (time.time() - self.start_time) > self.time_limit_secs:
            if self.verbose:
                print("Time limit reached. Saving final timeout checkpoint.")
            self.model.save(os.path.join(self.save_path, "final_model_timeout"))
            return False
        return True
