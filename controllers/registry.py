"""Controller registry used by the browser and future experiments."""

from pathlib import Path

from .analytic_ik import AnalyticIKController
from .base import Controller
from .ik_mlp import IKMLPController
from .ppo_joint_delta import PPOJointDeltaController


def create_controllers(checkpoint: Path) -> dict[str, Controller]:
    controllers: dict[str, Controller] = {"analytic_ik": AnalyticIKController()}
    if checkpoint.exists():
        controllers["ik_mlp"] = IKMLPController(checkpoint)
    ppo_checkpoint = checkpoint.parent / "org_reward" / "ppo_joint_delta.pt"
    if ppo_checkpoint.exists():
        controllers["ppo_joint_delta"] = PPOJointDeltaController(ppo_checkpoint)
    return controllers
