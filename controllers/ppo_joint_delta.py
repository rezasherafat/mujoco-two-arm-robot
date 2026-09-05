"""PPO actor/value model and deterministic browser inference controller."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn

from .base import ActionType, Controller, ControllerAction, Observation

OBSERVATION_DIM = 10
ACTION_DIM = 2


def encode_ppo_observation(observation: Observation) -> np.ndarray:
    q1, q2 = observation.current_q
    command = observation.current_q if observation.q_command is None else observation.q_command
    command_error = command - observation.current_q
    l1, l2 = 0.60, 0.50
    end_effector = np.array([
        l1 * np.sin(q1) + l2 * np.sin(q1 + q2),
        l1 * np.cos(q1) + l2 * np.cos(q1 + q2),
    ])
    error = observation.target_xz - end_effector
    return np.array([
        np.sin(q1), np.cos(q1), np.sin(q2), np.cos(q2),
        observation.qvel[0] / 5.0, observation.qvel[1] / 5.0,
        error[0], error[1], command_error[0], command_error[1],
    ], dtype=np.float32)


def make_mlp(output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(OBSERVATION_DIM, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, output_dim),
    )


class PPOActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = make_mlp(ACTION_DIM)
        self.critic = make_mlp(1)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -2.0))
        self.apply(self._initialize)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def distribution(self, observations: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(observations)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations).squeeze(-1)

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        latent = distribution.rsample()
        actions = torch.tanh(latent)
        log_probability = (
            distribution.log_prob(latent) - torch.log(1.0 - actions.square() + 1e-6)
        ).sum(-1)
        return actions, log_probability, self.value(observations)

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions = actions.clamp(-0.999999, 0.999999)
        latent = torch.atanh(actions)
        distribution = self.distribution(observations)
        log_probability = (
            distribution.log_prob(latent) - torch.log(1.0 - actions.square() + 1e-6)
        ).sum(-1)
        entropy = distribution.entropy().sum(-1)
        return log_probability, entropy, self.value(observations)


class PPOJointDeltaController(Controller):
    name = "ppo_joint_delta"
    action_type = ActionType.JOINT_DELTA

    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.network = PPOActorCritic().to(self.device)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.network.load_state_dict(payload["model_state"])
        self.network.eval()

    def predict(self, observation: Observation) -> ControllerAction:
        encoded = torch.from_numpy(encode_ppo_observation(observation)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            action = torch.tanh(self.network.actor(encoded))[0].cpu().numpy()
        return ControllerAction(self.action_type, action.astype(float))
