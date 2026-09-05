"""Reproducible reaching benchmark; never trains or overwrites a checkpoint.

The optional analytic reference is explicitly reported as a different controller.
PPO inference has no IK fallback.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from controllers.analytic_ik import solve_all
from controllers.ppo_joint_delta import PPOActorCritic
from training.train_ppo_joint_delta import VectorArmEnv, MAX_DELTA, CONTROL_STEPS


@torch.no_grad()
def benchmark(checkpoint: Path | None, episodes: int, seed: int, steps: int):
    torch.set_num_threads(1)
    env = VectorArmEnv(episodes, seed)
    initial_q = np.array([d.qpos.copy() for d in env.data])
    initial_targets = env.targets.copy()
    initial_distance = env.previous_distance.copy()
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        policy = PPOActorCritic()
        policy.load_state_dict(payload["model_state"])
        policy.eval()
        offset = payload.get("max_training_delta_rad", MAX_DELTA)
        if not np.isclose(offset, MAX_DELTA):
            raise ValueError("Checkpoint and benchmark action scales differ")
    else:
        goals = np.array([min(solve_all(target), key=lambda q: np.linalg.norm(q-current))
                          for current, target in zip(initial_q, env.targets)])
    streak = np.zeros(episodes, int)
    success = np.zeros(episodes, bool)
    time_to_success = np.full(episodes, np.nan)
    best_distance = initial_distance.copy()
    for step in range(steps):
        if checkpoint:
            action = torch.tanh(policy.actor(torch.from_numpy(env.observations()))).numpy()
        else:
            q = np.array([d.qpos.copy() for d in env.data])
            action = np.clip((goals - q) / MAX_DELTA, -1, 1)
        distance, speed = env.evaluation_step(action)
        best_distance = np.minimum(best_distance, distance)
        streak = np.where((distance < 0.02) & (speed < 0.25), streak + 1, 0)
        newly_succeeded = (streak >= 5) & ~success
        time_to_success[newly_succeeded] = (step + 1) * CONTROL_STEPS * env.model.opt.timestep
        success |= streak >= 5
    failures = [
        {"index": int(i), "start_q": initial_q[i].tolist(),
         "target_xz": initial_targets[i].tolist(), "final_q": env.data[i].qpos.tolist(),
         "final_error_m": float(distance[i]), "best_error_m": float(best_distance[i])}
        for i in np.flatnonzero(~success)
    ]
    return {
        "controller": str(checkpoint) if checkpoint else "analytic_ik_reference",
        "seed": seed, "episodes": episodes, "horizon_seconds": steps * 0.02,
        "success_distance_m": 0.02, "success_speed_rad_s": 0.25,
        "consecutive_settled_steps": 5, "successes": int(success.sum()),
        "success_rate": float(success.mean()), "final_settled_rate": float((streak >= 5).mean()),
        "initial_mean_error_m": float(initial_distance.mean()),
        "final_mean_error_m": float(distance.mean()), "final_max_error_m": float(distance.max()),
        "within_5cm_rate": float((best_distance < 0.05).mean()),
        "median_time_to_success_s": float(np.nanmedian(time_to_success)) if success.any() else None,
        "failures": failures,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--analytic-reference", action="store_true")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--seed", type=int, default=10719)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if min(args.episodes, args.steps) <= 0:
        p.error("episodes and steps must be positive")
    report = benchmark(args.checkpoint, args.episodes, args.seed, args.steps)
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
