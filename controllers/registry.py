"""Controller registry used by the browser and future experiments."""

from pathlib import Path

from .analytic_ik import AnalyticIKController
from .base import Controller
from .ik_mlp import IKMLPController


def create_controllers(checkpoint: Path) -> dict[str, Controller]:
    controllers: dict[str, Controller] = {"analytic_ik": AnalyticIKController()}
    if checkpoint.exists():
        controllers["ik_mlp"] = IKMLPController(checkpoint)
    return controllers
