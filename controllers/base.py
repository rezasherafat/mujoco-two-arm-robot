"""Controller contracts shared by IK, policy, and torque approaches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np


class ActionType(str, Enum):
    JOINT_POSITION = "joint_position"
    JOINT_DELTA = "joint_delta"
    TORQUE = "torque"


@dataclass(frozen=True)
class Observation:
    current_q: np.ndarray
    qvel: np.ndarray
    target_xz: np.ndarray
    q_command: np.ndarray | None = None


@dataclass(frozen=True)
class ControllerAction:
    action_type: ActionType
    values: np.ndarray


class Controller(ABC):
    name: str
    action_type: ActionType

    @abstractmethod
    def predict(self, observation: Observation) -> ControllerAction:
        """Return one control action for an observation."""
