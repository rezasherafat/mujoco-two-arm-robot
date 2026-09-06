"""Regression checks for physics and rollout bugs that blocked PPO reaching."""
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from controllers.analytic_ik import forward_kinematics, solve_all
from controllers.base import ActionType, ControllerAction, Observation
from controllers.ppo_joint_delta import encode_ppo_observation
from training.arm_task import MODEL_PATH
from training.train_ppo_joint_delta import (
    VectorArmEnv, MAX_EPISODE_STEPS, SETTLE_STREAK, HOLD_STEPS, HOLD_BONUS,
    SUCCESS_DISTANCE, proximity_reward,
)


class ArmTaskTests(unittest.TestCase):
    def test_zero_action_holds_under_gravity(self):
        env = VectorArmEnv(16, 123)
        initial = np.array([d.qpos.copy() for d in env.data])
        for _ in range(100):
            env.evaluation_step(np.zeros((16, 2)))
        np.testing.assert_allclose([d.qpos for d in env.data], initial, atol=1e-9)

    def test_compensation_matches_browser(self):
        env = VectorArmEnv(1, 77)
        browser_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        browser = mujoco.MjData(browser_model)
        gravity = mujoco.MjData(browser_model)
        browser.qpos[:] = env.data[0].qpos
        browser.ctrl[:] = env.data[0].ctrl
        for _ in range(20):
            action = np.array([0.2, -0.3])
            browser.ctrl[:] = (browser.qpos + 0.4 * action).clip(-2.8, 2.8)
            env.evaluation_step(action[None])
            for _ in range(10):
                gravity.qpos[:] = browser.qpos
                mujoco.mj_forward(browser_model, gravity)
                browser.qfrc_applied[:] = gravity.qfrc_bias
                mujoco.mj_step(browser_model, browser)
        np.testing.assert_allclose(browser.qpos, env.data[0].qpos, atol=1e-8)

    def test_observation_encoder_matches_inference(self):
        env = VectorArmEnv(8, 88)
        for i, data in enumerate(env.data):
            expected = encode_ppo_observation(Observation(
                data.qpos, data.qvel, env.targets[i], data.ctrl,
            ))
            np.testing.assert_array_equal(env.observations()[i], expected)

    def test_optional_teacher_matches_analytic_reference(self):
        from training.ik_teacher import teacher_actions
        env = VectorArmEnv(128, 918)
        labels = teacher_actions(torch.from_numpy(env.observations())).numpy()
        for i, data in enumerate(env.data):
            goal = min(solve_all(env.targets[i]), key=lambda q: np.linalg.norm(q - data.qpos))
            np.testing.assert_allclose(labels[i], ((goal-data.qpos)/.4).clip(-1, 1), atol=3e-4)

    def test_browser_policy_runs_at_training_frequency(self):
        from web_controller import Simulation

        class CountingPolicy:
            calls = 0

            def predict(self, observation):
                self.calls += 1
                return ControllerAction(ActionType.JOINT_DELTA, np.zeros(2))

        # No renderer/window needed: exercise the actual browser scheduling method.
        sim = Simulation.__new__(Simulation)
        sim.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        sim.data = mujoco.MjData(sim.model)
        sim.gravity_data = mujoco.MjData(sim.model)
        policy = CountingPolicy()
        sim.controllers = {"test": policy}
        sim.active_controller = "test"
        sim.policy_target_xz = np.array([0.1, 0.9])
        sim.next_policy_time = 0.0
        sim.motion_speed = 1.0
        for _ in range(500):  # one second of 2 ms physics steps
            sim.apply_policy_controls()
            sim.apply_gravity_compensation()
            mujoco.mj_step(sim.model, sim.data)
        self.assertEqual(policy.calls, 50)

    def test_checkpoint_reload_preserves_active_controller(self):
        from web_controller import Simulation

        old_controller = object()
        new_controller = object()
        sim = Simulation.__new__(Simulation)
        sim.controllers = {"ppo_joint_delta": old_controller}
        sim.active_controller = "ppo_joint_delta"
        sim.joint_goal = None
        sim.policy_target_xz = np.array([0.1, 0.2])
        sim.training_metrics = None
        sim.ppo_metrics = None
        sim.message = ""
        with patch("web_controller.create_controllers",
                   return_value={"ppo_joint_delta": new_controller}), \
             patch.object(Simulation, "checkpoint_info", return_value=[]):
            result = sim.reload_checkpoints()
        self.assertTrue(result["ok"])
        self.assertIs(sim.controllers["ppo_joint_delta"], new_controller)
        self.assertEqual(sim.active_controller, "ppo_joint_delta")
        np.testing.assert_array_equal(sim.policy_target_xz, [0.1, 0.2])

    def test_failed_checkpoint_reload_keeps_live_controller(self):
        from web_controller import Simulation

        old_controller = object()
        sim = Simulation.__new__(Simulation)
        sim.controllers = {"ppo_joint_delta": old_controller}
        sim.active_controller = "ppo_joint_delta"
        sim.message = ""
        with patch("web_controller.create_controllers", side_effect=ValueError("bad checkpoint")):
            result = sim.reload_checkpoints()
        self.assertFalse(result["ok"])
        self.assertIs(sim.controllers["ppo_joint_delta"], old_controller)
        self.assertIn("bad checkpoint", result["message"])

    def test_timeout_returns_final_observation_before_reset(self):
        env = VectorArmEnv(1, 99)
        env.targets[0] = forward_kinematics(env.data[0].qpos) + [0.1, 0]
        expected = env.observations()[0].copy()
        env.episode_steps[0] = MAX_EPISODE_STEPS - 1
        _, _, done, info = env.step(np.zeros((1, 2)))
        self.assertTrue(done[0])
        self.assertTrue(info["truncated"][0])
        np.testing.assert_allclose(info["terminal_observations"][0], expected, atol=1e-8)

    def test_precision_well_is_monotone_and_never_pays_outside(self):
        """The well is steep near the goal only because it is offset to zero at the
        success radius. Both properties are what keep it from being a hovering bonus."""
        distances = np.linspace(0.0, 1.2, 4001)
        values = np.array([proximity_reward(d) for d in distances])
        self.assertTrue(np.all(np.diff(values) < 0), "closing distance must always pay")
        outside = distances >= SUCCESS_DISTANCE
        self.assertLess(values[outside].max(), 0.0)
        self.assertAlmostEqual(proximity_reward(SUCCESS_DISTANCE),
                               -np.sqrt(SUCCESS_DISTANCE + 1e-6), places=12)
        # Steep where the measured failures stop, flat where the traverse happens.
        near = proximity_reward(0.02) - proximity_reward(0.03)
        far = proximity_reward(0.50) - proximity_reward(0.51)
        self.assertGreater(near, 20 * far)

    def test_no_positive_reward_outside_the_success_region(self):
        """Hovering still cannot pay: the only positive term is gated on success."""
        env = VectorArmEnv(1, 123)
        env.targets[0] = forward_kinematics(env.data[0].qpos) + [0.045, 0]
        env.previous_distance[0] = 0.045
        _, reward, done, _ = env.step(np.zeros((1, 2)))
        self.assertLess(reward[0], 0)
        self.assertFalse(done[0])

    def test_settling_streak_does_not_end_the_episode(self):
        """The benchmark streak used to terminate training episodes, so the policy
        was never scored on staying. Only the time limit ends an episode now."""
        env = VectorArmEnv(1, 123)
        env.targets[0] = forward_kinematics(env.data[0].qpos)
        env.previous_distance[0] = 0.0
        for _ in range(SETTLE_STREAK + 1):
            _, reward, done, _ = env.step(np.zeros((1, 2)))
            self.assertFalse(done[0])
            self.assertGreater(reward[0], 0)
        self.assertGreaterEqual(env.success_streak[0], SETTLE_STREAK)

    def test_holding_the_goal_beats_dithering_through_it(self):
        """At training noise a five-step streak is reachable by passing through the
        goal. Requiring HOLD_STEPS of continuous settling makes that unprofitable."""
        steps = 2 * HOLD_STEPS

        held = VectorArmEnv(1, 321)
        held.targets[0] = forward_kinematics(held.data[0].qpos)
        held.previous_distance[0] = 0.0
        held_reward = 0.0
        for _ in range(steps):
            _, reward, _, info = held.step(np.zeros((1, 2)))
            held.targets[0] = forward_kinematics(held.data[0].qpos)
            held_reward += float(reward[0])

        dithered = VectorArmEnv(1, 321)
        dithered_reward = 0.0
        for step in range(steps):
            offset = [0.0, 0.0] if step % 2 else [0.03, 0.0]
            dithered.targets[0] = forward_kinematics(dithered.data[0].qpos) + offset
            _, reward, _, _ = dithered.step(np.zeros((1, 2)))
            dithered_reward += float(reward[0])

        self.assertEqual(held.settled_steps, steps)
        self.assertLess(dithered.settled_steps, steps)
        self.assertGreater(held_reward, dithered_reward)
        self.assertGreaterEqual(held.completed_reaches, 1)
        self.assertEqual(dithered.completed_reaches, 0)

    def test_held_goal_is_retargeted_without_resetting_the_arm(self):
        env = VectorArmEnv(1, 555)
        env.targets[0] = forward_kinematics(env.data[0].qpos)
        env.previous_distance[0] = 0.0
        original_target = env.targets[0].copy()
        pose = env.data[0].qpos.copy()
        for _ in range(HOLD_STEPS):
            env.step(np.zeros((1, 2)))
            env.targets[0] = forward_kinematics(env.data[0].qpos) \
                if env.success_streak[0] else env.targets[0]
        self.assertEqual(env.completed_reaches, 1)
        self.assertFalse(np.allclose(env.targets[0], original_target))
        # A chained reach continues from the arm's current state, not a fresh reset.
        np.testing.assert_allclose(env.data[0].qpos, pose, atol=1e-6)
        self.assertGreater(env.episode_steps[0], 0)
        self.assertEqual(env.success_streak[0], 0)

    def test_stochastic_and_deterministic_evaluation_are_both_reported(self):
        """The gap between them is the objective mismatch and must stay visible."""
        from training.train_ppo_joint_delta import evaluate_policy
        from controllers.ppo_joint_delta import PPOActorCritic
        network = PPOActorCritic()
        device = torch.device("cpu")
        deterministic = evaluate_policy(network, device, 5, episodes=4)
        repeated = evaluate_policy(network, device, 5, episodes=4)
        noisy = evaluate_policy(network, device, 5, episodes=4, stochastic=True)
        self.assertEqual(deterministic, repeated)
        self.assertNotEqual(noisy["final_mean_distance_m"],
                            deterministic["final_mean_distance_m"])


if __name__ == "__main__":
    unittest.main()
