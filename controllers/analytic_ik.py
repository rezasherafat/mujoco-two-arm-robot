"""Closed-form two-link IK controller and shared kinematic utilities."""

from __future__ import annotations

import numpy as np

from .base import ActionType, Controller, ControllerAction, Observation

LINK_LENGTHS = (0.60, 0.50)
JOINT_LIMITS = np.array([[-2.8, 2.8], [-2.8, 2.8]], dtype=np.float64)


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    q1, q2 = q
    l1, l2 = LINK_LENGTHS
    return np.array([
        l1 * np.sin(q1) + l2 * np.sin(q1 + q2),
        l1 * np.cos(q1) + l2 * np.cos(q1 + q2),
    ])


def solve_all(target_xz: np.ndarray) -> list[np.ndarray]:
    x, z = target_xz
    l1, l2 = LINK_LENGTHS
    cosine = (x * x + z * z - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if cosine < -1.0 or cosine > 1.0:
        return []
    solutions = []
    for q2 in (np.arccos(cosine), -np.arccos(cosine)):
        q1 = np.arctan2(x, z) - np.arctan2(
            l2 * np.sin(q2), l1 + l2 * np.cos(q2)
        )
        q = np.array([q1, q2], dtype=np.float64)
        if np.all(q >= JOINT_LIMITS[:, 0]) and np.all(q <= JOINT_LIMITS[:, 1]):
            solutions.append(q)
    return solutions


class AnalyticIKController(Controller):
    name = "analytic_ik"
    action_type = ActionType.JOINT_POSITION

    def predict(self, observation: Observation) -> ControllerAction:
        solutions = solve_all(observation.target_xz)
        if not solutions:
            raise ValueError("Target is outside the reachable workspace")
        goal = min(
            solutions,
            key=lambda q: float(np.linalg.norm(q - observation.current_q)),
        )
        return ControllerAction(self.action_type, goal)
