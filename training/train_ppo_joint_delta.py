"""Train a continuous-action PPO joint-delta controller in batched MuJoCo."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import mujoco
import numpy as np
import torch

from controllers.analytic_ik import forward_kinematics, solve_all
from controllers.base import Observation
from controllers.ppo_joint_delta import PPOActorCritic, encode_ppo_observation
from training.arm_task import load_training_model

ROOT = Path(__file__).resolve().parents[1]
JOINT_LIMITS = np.array([[-2.8, 2.8], [-2.8, 2.8]])
MAX_DELTA = 0.40
CONTROL_STEPS = 10
MAX_EPISODE_STEPS = 400
SUCCESS_DISTANCE = 0.02
EVAL_SUCCESS_DISTANCE = 0.02
SUCCESS_SPEED = 0.25
# Streak that defines success for the benchmark; unchanged so published numbers stay
# comparable. Training deliberately requires a much longer hold: at training-time
# exploration noise a five-step streak is reachable by dithering through the goal,
# which inflates the reward without producing a policy that stays there.
SETTLE_STREAK = 5
HOLD_STEPS = 25
HOLD_BONUS = 1.0
# Precision well. The failures this targets stop dead a few millimetres outside
# tolerance, where the plain sqrt cost has a slope of 0.03 per cm and says almost
# nothing. Anchoring the well's scale to SUCCESS_DISTANCE keeps the shaping and the
# criterion from drifting apart.
NEAR_WEIGHT = 2.0
NEAR_SCALE = SUCCESS_DISTANCE
# Speed is priced only near the goal, so the traverse is unaffected.
APPROACH_SCALE = 0.05
APPROACH_VELOCITY_WEIGHT = 0.2
# A joint limit must cost something to sit at, not just to push against.
LIMIT_MARGIN = 2.3
LIMIT_WEIGHT = 0.1
EVAL_EPISODES = 256
EVAL_INTERVAL = 10


def proximity_reward(distance: float) -> float:
    """Coarse time cost plus a precision well anchored to the success radius.

    The well is offset to be exactly zero at SUCCESS_DISTANCE, which is what lets it
    be steep without ever making the reward positive outside the terminal region.
    Strictly decreasing in distance, so closing the last centimetre always pays and
    holding station short of the goal never does.
    """
    well = NEAR_WEIGHT * (np.exp(-(distance / NEAR_SCALE) ** 2) - np.exp(-1.0))
    return float(well - np.sqrt(distance + 1e-6))


class VectorArmEnv:
    def __init__(self, count: int, seed: int, ik_reward_weight: float = 0.0) -> None:
        self.count = count
        self.rng = np.random.default_rng(seed)
        self.model = load_training_model()
        self.data = [mujoco.MjData(self.model) for _ in range(count)]
        self.targets = np.zeros((count, 2))
        self.previous_distance = np.zeros(count)
        self.episode_steps = np.zeros(count, dtype=np.int32)
        self.success_streak = np.zeros(count, dtype=np.int32)
        self.episode_returns = np.zeros(count)
        self.max_goal_distance = 2.20
        self.ik_reward_weight = ik_reward_weight
        self.goal_solutions: list[np.ndarray] = [np.empty((0, 2)) for _ in range(count)]
        self.previous_joint_distance = np.zeros(count)
        self.completed_returns: list[float] = []
        self.completed_successes: list[float] = []
        self.completed_reaches = 0
        self.settled_steps = 0
        self.total_steps = 0
        self.episode_reaches = np.zeros(count, dtype=np.int32)
        for index in range(count):
            self.reset_one(index)

    @staticmethod
    def safe(q: np.ndarray) -> bool:
        elbow_z = 0.60 + 0.60 * np.cos(q[0])
        end_z = 0.60 + forward_kinematics(q)[1]
        return min(elbow_z, end_z) > 0.10

    def random_safe_q(self) -> np.ndarray:
        while True:
            q = self.rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            if self.safe(q):
                return q

    def reset_one(self, index: int) -> None:
        data = self.data[index]
        mujoco.mj_resetData(self.model, data)
        q = self.random_safe_q()
        goal_q = self.random_safe_q()
        while np.linalg.norm(forward_kinematics(goal_q) - forward_kinematics(q)) > self.max_goal_distance:
            goal_q = self.random_safe_q()
        data.qpos[:] = q
        data.ctrl[:] = q
        data.qvel[:] = 0.0
        self.targets[index] = forward_kinematics(goal_q)
        self.goal_solutions[index] = np.asarray(solve_all(self.targets[index]))
        self.previous_joint_distance[index] = self.joint_distance(index)
        mujoco.mj_forward(self.model, data)
        self.previous_distance[index] = np.linalg.norm(
            self.targets[index] - forward_kinematics(data.qpos)
        )
        self.episode_steps[index] = 0
        self.success_streak[index] = 0
        self.episode_returns[index] = 0.0
        self.episode_reaches[index] = 0

    def retarget_one(self, index: int) -> None:
        """Give a settled arm a new goal without disturbing its state.

        Chaining reaches inside one episode keeps target throughput high while
        making the settled state something the policy has to hold rather than an
        exit that ends the episode before holding is ever scored.
        """
        data = self.data[index]
        goal_q = self.random_safe_q()
        while np.linalg.norm(
            forward_kinematics(goal_q) - forward_kinematics(data.qpos)
        ) > self.max_goal_distance:
            goal_q = self.random_safe_q()
        self.targets[index] = forward_kinematics(goal_q)
        self.goal_solutions[index] = np.asarray(solve_all(self.targets[index]))
        self.previous_joint_distance[index] = self.joint_distance(index)
        self.previous_distance[index] = np.linalg.norm(
            self.targets[index] - forward_kinematics(data.qpos)
        )
        self.success_streak[index] = 0

    def joint_distance(self, index: int) -> float:
        return float(np.linalg.norm(
            self.goal_solutions[index] - self.data[index].qpos, axis=1,
        ).min())

    def observations(self) -> np.ndarray:
        rows = np.empty((self.count, 10), dtype=np.float32)
        for index, data in enumerate(self.data):
            q1, q2 = data.qpos
            error = self.targets[index] - forward_kinematics(data.qpos)
            command_error = data.ctrl - data.qpos
            rows[index] = (
                np.sin(q1), np.cos(q1), np.sin(q2), np.cos(q2),
                data.qvel[0] / 5.0, data.qvel[1] / 5.0,
                error[0], error[1], command_error[0], command_error[1],
            )
        return rows

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        rewards = np.empty(self.count, dtype=np.float32)
        dones = np.zeros(self.count, dtype=np.float32)
        truncated = np.zeros(self.count, dtype=bool)
        terminal_observations = np.zeros((self.count, 10), dtype=np.float32)
        distances = np.empty(self.count)
        for index, data in enumerate(self.data):
            data.ctrl[:] = np.clip(
                data.qpos + MAX_DELTA * actions[index],
                JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1],
            )
            for _ in range(CONTROL_STEPS):
                mujoco.mj_step(self.model, data)
            distance = np.linalg.norm(self.targets[index] - forward_kinematics(data.qpos))
            speed = np.linalg.norm(data.qvel)
            progress = self.previous_distance[index] - distance
            limit_fraction = np.maximum((np.abs(data.qpos) - 2.5) / 0.3, 0.0)
            outward = np.maximum(actions[index] * np.sign(data.qpos), 0.0) * limit_fraction
            # Crossing the goal at speed is what exploration noise does; make it cost.
            approach = APPROACH_VELOCITY_WEIGHT * np.dot(data.qvel, data.qvel) * np.exp(
                -(distance / APPROACH_SCALE) ** 2
            )
            # The old term priced only the action pushing outward, so an arm already
            # pinned at a stop with zero action paid nothing to stay there.
            dwell = LIMIT_WEIGHT * np.square(np.maximum(
                (np.abs(data.qpos) - LIMIT_MARGIN) / (2.8 - LIMIT_MARGIN), 0.0
            )).sum()
            reward = (
                10.0 * progress + proximity_reward(distance)
                - approach - dwell
                - 0.10 * np.dot(outward, outward)
            )
            if self.ik_reward_weight:
                # Optional privileged reward, not an input or an inference-time fallback.
                # Distance to either valid IK branch discourages Cartesian local minima
                # where a bent arm pushes farther into a joint limit.
                joint_distance = self.joint_distance(index)
                joint_progress = self.previous_joint_distance[index] - joint_distance
                reward += self.ik_reward_weight * (10.0 * joint_progress - 0.25 * joint_distance)
                self.previous_joint_distance[index] = joint_distance
            settled = distance < SUCCESS_DISTANCE and speed < SUCCESS_SPEED
            self.success_streak[index] = self.success_streak[index] + 1 if settled else 0
            if settled:
                # The only positive term, and it pays strictly inside the region that
                # defines success, so hovering outside still cannot earn it. What it
                # rewards is time spent settled: an arm dithered through the goal by
                # exploration noise collects it only on the steps it is actually inside.
                reward += HOLD_BONUS
                self.settled_steps += 1
            self.episode_steps[index] += 1
            self.total_steps += 1
            timeout = self.episode_steps[index] >= MAX_EPISODE_STEPS
            self.episode_returns[index] += reward
            rewards[index] = reward
            dones[index] = float(timeout)
            distances[index] = distance
            self.previous_distance[index] = distance
            if self.success_streak[index] >= HOLD_STEPS:
                # A reach counts only once the arm has held the goal for HOLD_STEPS.
                self.completed_reaches += 1
                self.episode_reaches[index] += 1
                if not timeout:
                    self.retarget_one(index)
            if timeout:
                truncated[index] = True
                terminal_observations[index] = encode_ppo_observation(Observation(
                    data.qpos, data.qvel, self.targets[index], data.ctrl,
                ))
                self.completed_returns.append(float(self.episode_returns[index]))
                self.completed_successes.append(float(self.episode_reaches[index] > 0))
                self.reset_one(index)
        info = {
            "truncated": truncated,
            "terminal_observations": terminal_observations,
            "completed_episodes": len(self.completed_successes),
            "mean_distance_m": float(distances.mean()),
            "settled_fraction": self.settled_steps / max(self.total_steps, 1),
            "reaches_completed": self.completed_reaches,
            "success_rate": float(np.mean(self.completed_successes[-100:])) if self.completed_successes else 0.0,
            "mean_episode_return": float(np.mean(self.completed_returns[-100:])) if self.completed_returns else 0.0,
        }
        return self.observations(), rewards, dones, info


    def evaluation_step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances = np.empty(self.count)
        speeds = np.empty(self.count)
        for index, data in enumerate(self.data):
            data.ctrl[:] = np.clip(
                data.qpos + MAX_DELTA * actions[index],
                JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1],
            )
            for _ in range(CONTROL_STEPS):
                mujoco.mj_step(self.model, data)
            distances[index] = np.linalg.norm(
                self.targets[index] - forward_kinematics(data.qpos)
            )
            speeds[index] = np.linalg.norm(data.qvel)
        return distances, speeds


@torch.no_grad()
def evaluate_policy(
    network: PPOActorCritic, device: torch.device, seed: int,
    episodes: int = EVAL_EPISODES, stochastic: bool = False,
) -> dict[str, float]:
    """Roll out the benchmark criterion. `stochastic` samples from the same
    distribution PPO optimizes; the default mean action is what gets deployed."""
    env = VectorArmEnv(episodes, seed)
    env.max_goal_distance = 2.20
    for index in range(episodes):
        env.reset_one(index)
    initial = env.previous_distance.copy()
    best = initial.copy()
    distances = initial
    speeds = np.zeros(episodes)
    streak = np.zeros(episodes, dtype=int)
    succeeded = np.zeros(episodes, dtype=bool)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(MAX_EPISODE_STEPS):
        observations = torch.from_numpy(env.observations()).to(device)
        mean = network.actor(observations)
        if stochastic:
            noise = torch.randn(mean.shape, generator=generator, device=device)
            mean = mean + network.log_std.exp() * noise
        actions = torch.tanh(mean).cpu().numpy()
        distances, speeds = env.evaluation_step(actions)
        best = np.minimum(best, distances)
        settled = (distances < SUCCESS_DISTANCE) & (speeds < SUCCESS_SPEED)
        streak = np.where(settled, streak + 1, 0)
        succeeded |= streak >= SETTLE_STREAK
    return {
        "initial_mean_distance_m": float(initial.mean()),
        "final_mean_distance_m": float(distances.mean()),
        "final_median_distance_m": float(np.median(distances)),
        "success_rate": float(succeeded.mean()),
        "final_settled_rate": float(np.mean(streak >= SETTLE_STREAK)),
        "episodes": episodes,
        "horizon_seconds": MAX_EPISODE_STEPS * CONTROL_STEPS * env.model.opt.timestep,
        "worst_final_distance_m": float(distances.max()),
        "within_5cm_rate": float(np.mean(best < 0.05)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--initial-log-std", type=float, default=-1.0)
    parser.add_argument("--final-log-std", type=float, default=-4.0,
                        help="Anneal a ceiling on log_std down to this value so the "
                             "stochastic objective PPO optimizes converges to the "
                             "deterministic mean action that is actually deployed")
    parser.add_argument("--log-std-anneal-fraction", type=float, default=0.8,
                        help="Fraction of training over which the ceiling reaches its floor")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Initialize model weights only; optimizer and exploration are reset")
    parser.add_argument("--ik-reward-weight", type=float, default=0.0,
                        help="Optional IK-guided joint-progress reward; no IK at inference")
    parser.add_argument("--teacher-pretrain-steps", type=int, default=0,
                        help="Optional supervised actor warm-start steps; not pure PPO")
    parser.add_argument("--teacher-weight", type=float, default=0.0,
                        help="Optional analytic action imitation loss during PPO; not pure PPO")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-watch", action="store_true")
    args = parser.parse_args()
    if min(args.envs, args.rollout_steps, args.updates, args.epochs, args.minibatch,
           args.eval_episodes, args.eval_interval) <= 0:
        parser.error("Training counts and intervals must be positive")
    if args.teacher_pretrain_steps < 0 or args.teacher_weight < 0 or args.ik_reward_weight < 0:
        parser.error("Optional teaching steps and weights must be nonnegative")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive")
    if not 0.0 < args.log_std_anneal_fraction <= 1.0:
        parser.error("Exploration anneal fraction must be in (0, 1]")
    if args.final_log_std > args.initial_log_std:
        parser.error("Final log_std must not exceed the initial value")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "ppo_joint_delta.pt"
    metrics_path = args.output_dir / "ppo_joint_delta_metrics.json"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = VectorArmEnv(args.envs, args.seed, args.ik_reward_weight)
    network = PPOActorCritic().to(device)
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        if payload.get("physics_version") != "compile-time-gravcomp":
            parser.error("Initialize only from a checkpoint trained with corrected gravity compensation")
        network.load_state_dict(payload["model_state"])
    with torch.no_grad():
        network.log_std.fill_(args.initial_log_std)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    if args.teacher_pretrain_steps or args.teacher_weight:
        from training.ik_teacher import teacher_actions, teaching_batch
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                "algorithm": "PPO",
                "controller": "joint_delta",
                "seed": args.seed,
                "device": str(device),
                "environments": args.envs,
                "rollout_steps": args.rollout_steps,
                "updates": args.updates,
                "ppo_epochs": args.epochs,
                "minibatch_size": args.minibatch,
                "learning_rate": args.learning_rate,
                "physics_version": "compile-time-gravcomp",
                "reward_version": "precision-well-hold-to-complete",
                "near_weight": NEAR_WEIGHT, "near_scale": NEAR_SCALE,
                "approach_velocity_weight": APPROACH_VELOCITY_WEIGHT,
                "approach_scale": APPROACH_SCALE,
                "limit_weight": LIMIT_WEIGHT, "limit_margin": LIMIT_MARGIN,
                "training_hold_steps": HOLD_STEPS,
                "hold_bonus": HOLD_BONUS,
                "final_log_std": args.final_log_std,
                "log_std_anneal_fraction": args.log_std_anneal_fraction,
                "ik_reward_weight": args.ik_reward_weight,
                "teacher_pretrain_steps": args.teacher_pretrain_steps,
                "teacher_weight": args.teacher_weight,
                "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                "success_hold_steps": SETTLE_STREAK,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_coefficient": 0.2,
                "max_joint_offset_rad": MAX_DELTA,
                "max_episode_steps": MAX_EPISODE_STEPS,
                "training_success_distance_m": SUCCESS_DISTANCE,
                "evaluation_success_distance_m": EVAL_SUCCESS_DISTANCE,
            },
        )
        if args.wandb_watch:
            wandb_run.watch(network, log="gradients", log_freq=100)
    pretraining_history = []
    if args.teacher_pretrain_steps:
        teacher_optimizer = torch.optim.Adam(network.actor.parameters(), lr=1e-3)
        for step in range(args.teacher_pretrain_steps):
            inputs, labels = teaching_batch(4096, device)
            teaching_loss = (torch.tanh(network.actor(inputs)) - labels).square().mean()
            teacher_optimizer.zero_grad(set_to_none=True)
            teaching_loss.backward()
            teacher_optimizer.step()
            if (step + 1) % 500 == 0 or step == 0 or step + 1 == args.teacher_pretrain_steps:
                pretraining_record = {"teacher/step": step+1, "teacher/action_mse": teaching_loss.item()}
                pretraining_history.append(pretraining_record)
                if wandb_run is not None:
                    wandb_run.log(pretraining_record)
                print(f"teacher_step={step+1} action_mse={teaching_loss.item():.6f}", flush=True)
    observation = env.observations()
    history = []
    best_eval_success = -1.0
    best_eval_final_settled = -1.0
    best_eval_distance = float("inf")
    gamma, gae_lambda, clip = 0.99, 0.95, 0.2
    started = time.monotonic()

    for update in range(1, args.updates + 1):
        observations = torch.empty((args.rollout_steps, args.envs, 10), device=device)
        actions = torch.empty((args.rollout_steps, args.envs, 2), device=device)
        log_probs = torch.empty((args.rollout_steps, args.envs), device=device)
        values = torch.empty((args.rollout_steps, args.envs), device=device)
        rewards = torch.empty((args.rollout_steps, args.envs), device=device)
        dones = torch.empty((args.rollout_steps, args.envs), device=device)
        distance_sum = 0.0
        for step in range(args.rollout_steps):
            obs_tensor = torch.from_numpy(observation).to(device)
            with torch.no_grad():
                action, log_prob, value = network.sample(obs_tensor)
            next_observation, reward, done, info = env.step(action.cpu().numpy())
            # Bootstrap time limits from the last state before auto-reset.
            # GAE still stops at the reset, so distinct episodes are not mixed.
            with torch.no_grad():
                ids = info["truncated"]
                if ids.any():
                    terminal_values = network.value(torch.from_numpy(
                        info["terminal_observations"][ids]
                    ).to(device)).cpu().numpy()
                    reward[ids] += gamma * terminal_values
            observations[step], actions[step] = obs_tensor, action
            log_probs[step], values[step] = log_prob, value
            rewards[step] = torch.from_numpy(reward).to(device)
            dones[step] = torch.from_numpy(done).to(device)
            observation = next_observation
            distance_sum += info["mean_distance_m"]

        with torch.no_grad():
            next_value = network.value(torch.from_numpy(observation).to(device))
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros(args.envs, device=device)
        for step in reversed(range(args.rollout_steps)):
            next_values = next_value if step == args.rollout_steps - 1 else values[step + 1]
            nonterminal = 1.0 - dones[step]
            delta = rewards[step] + gamma * next_values * nonterminal - values[step]
            last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
            advantages[step] = last_advantage
        returns = advantages + values
        flat_obs = observations.flatten(0, 1)
        flat_actions = actions.flatten(0, 1)
        flat_old_log_probs = log_probs.flatten()
        flat_advantages = advantages.flatten()
        flat_returns = returns.flatten()
        flat_old_values = values.flatten()
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        actor_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approximate_kls: list[float] = []
        clip_fractions: list[float] = []
        imitation_losses: list[float] = []
        labels = teacher_actions(flat_obs) if args.teacher_weight else None
        sample_count = flat_obs.shape[0]
        for _ in range(args.epochs):
            permutation = torch.randperm(sample_count, device=device)
            for start in range(0, sample_count, args.minibatch):
                indices = permutation[start:start + args.minibatch]
                new_log_prob, entropy, new_value = network.evaluate_actions(
                    flat_obs[indices], flat_actions[indices]
                )
                log_ratio = new_log_prob - flat_old_log_probs[indices]
                ratio = log_ratio.exp()
                advantage_batch = flat_advantages[indices]
                actor_loss = -torch.min(
                    ratio * advantage_batch,
                    ratio.clamp(1.0 - clip, 1.0 + clip) * advantage_batch,
                ).mean()
                value_loss = 0.5 * (new_value - flat_returns[indices]).square().mean()
                entropy_loss = entropy.mean()
                loss = actor_loss + 0.5 * value_loss - 0.0005 * entropy_loss
                if labels is not None:
                    imitation_loss = (torch.tanh(network.actor(flat_obs[indices])) - labels[indices]).square().mean()
                    loss = loss + args.teacher_weight * imitation_loss
                    imitation_losses.append(float(imitation_loss.item()))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Clip separate networks separately; critic gradients should not shrink actor steps.
                torch.nn.utils.clip_grad_norm_(list(network.actor.parameters()) + [network.log_std], 0.5)
                torch.nn.utils.clip_grad_norm_(network.critic.parameters(), 0.5)
                optimizer.step()
                actor_losses.append(float(actor_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_loss.item()))
                approximate_kls.append(float(((ratio - 1.0) - log_ratio).mean().item()))
                clip_fractions.append(float(((ratio - 1.0).abs() > clip).float().mean().item()))

        # Force exploration down on a schedule. Left free, log_std settles wherever
        # it maximizes the stochastic return, which is not where the deployed mean
        # action performs best; annealing makes the two objectives converge.
        progress_fraction = min(
            1.0, (update - 1) / max(args.log_std_anneal_fraction * args.updates - 1.0, 1.0)
        )
        log_std_ceiling = args.initial_log_std + progress_fraction * (
            args.final_log_std - args.initial_log_std
        )
        with torch.no_grad():
            network.log_std.clamp_(max=log_std_ceiling)

        mean_distance = distance_sum / args.rollout_steps
        explained_variance = 1.0 - (flat_returns - flat_old_values).var() / (
            flat_returns.var() + 1e-8
        )
        record = {
            "update": update,
            "environment_steps": update * args.rollout_steps * args.envs,
            "mean_distance_m": mean_distance,
            "settled_fraction": info["settled_fraction"],
            "reaches_completed": info["reaches_completed"],
            "log_std_ceiling": log_std_ceiling,
            "success_rate": info["success_rate"],
            "mean_episode_return": info["mean_episode_return"],
            "actor_loss": float(np.mean(actor_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approximate_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "explained_variance": float(explained_variance.item()),
            "policy_std": network.log_std.exp().detach().cpu().tolist(),
            "max_goal_distance_m": env.max_goal_distance,
            "elapsed_seconds": time.monotonic() - started,
            "completed_episodes": info["completed_episodes"],
            "imitation_loss": float(np.mean(imitation_losses)) if imitation_losses else 0.0,
        }
        should_evaluate = update == 1 or update % args.eval_interval == 0 or update == args.updates
        evaluation = None
        if should_evaluate:
            evaluation = evaluate_policy(network, device, args.seed + 10000, args.eval_episodes)
            noisy = evaluate_policy(
                network, device, args.seed + 10000, args.eval_episodes, stochastic=True
            )
            # The gap between these is the objective mismatch: PPO maximizes the
            # stochastic return while the benchmark and the browser run the mean.
            # A large positive gap means exploration noise, not skill, is scoring.
            evaluation["stochastic_success_rate"] = noisy["success_rate"]
            evaluation["stochastic_final_settled_rate"] = noisy["final_settled_rate"]
            evaluation["exploration_gap"] = noisy["success_rate"] - evaluation["success_rate"]
        if evaluation is not None:
            record["evaluation"] = evaluation
            # Prefer staying at the target, not merely passing through it earlier.
            better = (evaluation["final_settled_rate"], evaluation["success_rate"],
                      -evaluation["final_mean_distance_m"]) > (
                          best_eval_final_settled, best_eval_success, -best_eval_distance)
            if better:
                best_eval_success = evaluation["success_rate"]
                best_eval_final_settled = evaluation["final_settled_rate"]
                best_eval_distance = evaluation["final_mean_distance_m"]
                temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
                torch.save({
                    "model_state": network.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "update": update,
                    "evaluation": evaluation,
                    "observation": "sin/cos(q), qvel/5, Cartesian error, q_command-q",
                    "action": "continuous normalized q-relative joint offset in [-1,1]^2",
                    "max_training_delta_rad": MAX_DELTA,
                    "physics_version": "compile-time-gravcomp",
                    "reward_version": "precision-well-hold-to-complete",
                    "training_hold_steps": HOLD_STEPS,
                    "final_log_std": args.final_log_std,
                    "control_steps": CONTROL_STEPS,
                    "ik_reward_weight": args.ik_reward_weight,
                    "training_method": "teacher_assisted_ppo" if args.teacher_weight or args.teacher_pretrain_steps else "ppo",
                    "checkpoint_selection": "final_settled_rate_then_success_then_distance",
                }, temporary_checkpoint)
                temporary_checkpoint.replace(checkpoint_path)
        history.append(record)
        if wandb_run is not None:
            wandb_metrics = {
                "train/update": update,
                "train/environment_steps": record["environment_steps"],
                "rollout/mean_distance_m": record["mean_distance_m"],
                "rollout/settled_fraction": record["settled_fraction"],
                "rollout/reaches_completed": record["reaches_completed"],
                "policy/log_std_ceiling": record["log_std_ceiling"],
                "rollout/success_rate": record["success_rate"],
                "rollout/mean_episode_return": record["mean_episode_return"],
                "rollout/completed_episodes": record["completed_episodes"],
                "loss/policy": record["actor_loss"],
                "loss/value": record["value_loss"],
                "loss/imitation": record["imitation_loss"],
                "policy/entropy": record["entropy"],
                "policy/approx_kl": record["approx_kl"],
                "policy/clip_fraction": record["clip_fraction"],
                "critic/explained_variance": record["explained_variance"],
                "policy/std_shoulder": record["policy_std"][0],
                "policy/std_elbow": record["policy_std"][1],
                "curriculum/max_goal_distance_m": record["max_goal_distance_m"],
            }
            if evaluation is not None:
                wandb_metrics.update({
                    "evaluation/initial_mean_distance_m": evaluation["initial_mean_distance_m"],
                    "evaluation/final_mean_distance_m": evaluation["final_mean_distance_m"],
                    "evaluation/final_median_distance_m": evaluation["final_median_distance_m"],
                    "evaluation/success_rate": evaluation["success_rate"],
                    "evaluation/final_settled_rate": evaluation["final_settled_rate"],
                    "evaluation/stochastic_success_rate": evaluation["stochastic_success_rate"],
                    "evaluation/stochastic_final_settled_rate": evaluation["stochastic_final_settled_rate"],
                    "evaluation/exploration_gap": evaluation["exploration_gap"],
                    "evaluation/worst_final_distance_m": evaluation["worst_final_distance_m"],
                    "evaluation/within_5cm_rate": evaluation["within_5cm_rate"],
                    "evaluation/best_success_rate": best_eval_success,
                    "evaluation/best_final_settled_rate": best_eval_final_settled,
                    "evaluation/best_mean_distance_m": best_eval_distance,
                })
            wandb_run.log(wandb_metrics, step=record["environment_steps"])
        report = {
            "controller": "ppo_joint_delta", "device": str(device), "seed": args.seed,
            "updates_completed": update, "total_environment_steps": record["environment_steps"],
            "best_evaluation_success_rate": best_eval_success,
            "best_evaluation_final_settled_rate": best_eval_final_settled,
            "best_evaluation_mean_distance_m": best_eval_distance,
            "success_distance_m": SUCCESS_DISTANCE, "success_speed_rad_s": SUCCESS_SPEED,
            "success_hold_steps": SETTLE_STREAK, "episode_steps": MAX_EPISODE_STEPS,
            "architecture": "separate 10-128-128-2 actor and 10-128-128-1 critic",
            "reward": "10*progress - sqrt(distance+1e-6) + 2*(exp(-(d/.02)^2)-exp(-1)) "
                      "- .2*velocity^2*exp(-(d/.05)^2) - .1*limit_dwell^2 "
                      "- .1*outward_limit^2 + 1.0 per step while settled",
            "near_weight": NEAR_WEIGHT, "near_scale": NEAR_SCALE,
            "approach_velocity_weight": APPROACH_VELOCITY_WEIGHT,
            "approach_scale": APPROACH_SCALE,
            "limit_weight": LIMIT_WEIGHT, "limit_margin": LIMIT_MARGIN,
            "episode_termination": "time limit only; a held goal is re-targeted in place",
            "training_hold_steps": HOLD_STEPS, "hold_bonus": HOLD_BONUS,
            "final_log_std": args.final_log_std,
            "log_std_anneal_fraction": args.log_std_anneal_fraction,
            "physics_version": "compile-time-gravcomp",
            "gamma": gamma, "gae_lambda": gae_lambda, "clip": clip,
            "environments": args.envs, "updates": args.updates,
            "ik_reward_weight": args.ik_reward_weight,
            "teacher_pretrain_steps": args.teacher_pretrain_steps,
            "teacher_weight": args.teacher_weight,
            "ik_reward": "weight * (10*joint_progress - .25*distance_to_nearest_valid_IK_branch)",
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "history": history,
            "pretraining_history": pretraining_history,
            "checkpoint_selection": "final_settled_rate_then_success_then_distance",
        }
        # Save on every update so an interruption does not erase experiment history.
        temporary_metrics = metrics_path.with_suffix(".json.tmp")
        temporary_metrics.write_text(json.dumps(report, indent=2))
        temporary_metrics.replace(metrics_path)
        if evaluation is not None:
            print(
                f"update={update:03d} steps={record["environment_steps"]} "
                f"train_dist={mean_distance:.3f}m train_settled={info["settled_fraction"]:.1%} "
                f"reaches={info["reaches_completed"]} "
                f"eval_dist={evaluation["final_mean_distance_m"]:.3f}m "
                f"eval_success={evaluation["success_rate"]:.1%} "
                f"final_settled={evaluation["final_settled_rate"]:.1%} "
                f"noisy_success={evaluation["stochastic_success_rate"]:.1%} "
                f"gap={evaluation["exploration_gap"]:+.1%} "
                f"within5cm={evaluation["within_5cm_rate"]:.1%} "
                f"policy_loss={record["actor_loss"]:.4f} "
                f"value_loss={record["value_loss"]:.4f} "
                f"entropy={record["entropy"]:.3f} kl={record["approx_kl"]:.5f} "
                f"std={record["policy_std"]}",
                flush=True,
            )

    if wandb_run is not None:
        wandb_run.summary["best_evaluation_success_rate"] = best_eval_success
        wandb_run.summary["best_evaluation_final_settled_rate"] = best_eval_final_settled
        wandb_run.summary["best_evaluation_mean_distance_m"] = best_eval_distance
        wandb_run.summary["checkpoint_path"] = str(checkpoint_path)
        wandb_run.finish()
    print(f"checkpoint={checkpoint_path}\nmetrics={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
