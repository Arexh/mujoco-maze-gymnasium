"""Common APIs for defining mujoco robot.
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MuJocoPyEnv
from gymnasium.utils import EzPickle


class AgentModel(ABC, MuJocoPyEnv, EzPickle):
    FILE: str
    MANUAL_COLLISION: bool
    ORI_IND: Optional[int] = None
    RADIUS: Optional[float] = None
    OBJBALL_TYPE: Optional[str] = None
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, file_path: str, frame_skip: int, observation_space) -> None:
        MuJocoPyEnv.__init__(self, file_path, frame_skip, observation_space)
        EzPickle.__init__(self)

    def close(self):
        if self.viewer is not None and hasattr(self.viewer, "window"):
            import glfw

            glfw.destroy_window(self.viewer.window)
        super().close()

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Returns the observation from the model."""
        pass

    def get_xy(self) -> np.ndarray:
        """Returns the coordinate of the agent."""
        pass

    def set_xy(self, xy: np.ndarray) -> None:
        """Set the coordinate of the agent."""
        pass
