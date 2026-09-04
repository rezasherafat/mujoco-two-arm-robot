"""PyTorch MLP that learns the target-and-current-pose to IK mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .analytic_ik import JOINT_LIMITS
from .base import ActionType, Controller, ControllerAction, Observation

INPUT_DIM = 6
OUTPUT_DIM = 2


def encode_observation(current_q: np.ndarray, target_xz: np.ndarray) -> np.ndarray:
    q1, q2 = current_q
    return np.array(
        [np.sin(q1), np.cos(q1), np.sin(q2), np.cos(q2), *target_xz],
        dtype=np.float32,
    )


class IKMLPNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_DIM, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, OUTPUT_DIM),
        )
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)


class IKMLPController(Controller):
    name = "ik_mlp"
    action_type = ActionType.JOINT_POSITION

    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.network = IKMLPNetwork().to(self.device)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.network.load_state_dict(payload["model_state"])
        self.network.eval()

    def predict(self, observation: Observation) -> ControllerAction:
        encoded = encode_observation(observation.current_q, observation.target_xz)
        with torch.inference_mode():
            prediction = self.network(
                torch.from_numpy(encoded).unsqueeze(0).to(self.device)
            )[0]
        angles = np.clip(prediction.cpu().numpy().astype(float), JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        return ControllerAction(self.action_type, angles)
