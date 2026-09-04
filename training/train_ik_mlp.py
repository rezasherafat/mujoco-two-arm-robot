"""Generate supervised IK data and train the pluggable IK MLP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from controllers.analytic_ik import JOINT_LIMITS, forward_kinematics, solve_all
from controllers.ik_mlp import IKMLPNetwork, encode_observation

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATASET_PATH = ARTIFACTS / "ik_training_data.npz"
CHECKPOINT_PATH = ARTIFACTS / "ik_mlp.pt"
METRICS_PATH = ARTIFACTS / "ik_mlp_metrics.json"
L1, L2 = 0.60, 0.50
CARTESIAN_LOSS_WEIGHT = 10.0


def generate_dataset(sample_count: int, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    current_rows, target_rows, label_rows, observation_rows = [], [], [], []
    while len(current_rows) < sample_count:
        current_q = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        sampled_goal = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        target_xz = forward_kinematics(sampled_goal)
        solutions = solve_all(target_xz)
        if not solutions:
            continue
        label_q = min(solutions, key=lambda q: (q[1] >= 0) != (current_q[1] >= 0))
        current_rows.append(current_q)
        target_rows.append(target_xz)
        label_rows.append(label_q)
        observation_rows.append(encode_observation(current_q, target_xz))
    return tuple(np.asarray(rows, dtype=np.float32) for rows in (
        observation_rows, current_rows, target_rows, label_rows
    ))


def torch_forward_kinematics(q: torch.Tensor) -> torch.Tensor:
    q1, q2 = q[:, 0], q[:, 1]
    return torch.stack((
        L1 * torch.sin(q1) + L2 * torch.sin(q1 + q2),
        L1 * torch.cos(q1) + L2 * torch.cos(q1 + q2),
    ), dim=1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"joint_squared": 0.0, "cart_squared": 0.0, "cart_distance": 0.0}
    count = 0
    with torch.inference_mode():
        for observations, targets_q, targets_xz in loader:
            observations, targets_q, targets_xz = (
                observations.to(device), targets_q.to(device), targets_xz.to(device)
            )
            predictions = model(observations)
            predicted_xz = torch_forward_kinematics(predictions)
            batch = observations.shape[0]
            totals["joint_squared"] += ((predictions - targets_q) ** 2).sum().item()
            totals["cart_squared"] += ((predicted_xz - targets_xz) ** 2).sum().item()
            totals["cart_distance"] += torch.linalg.vector_norm(
                predicted_xz - targets_xz, dim=1
            ).sum().item()
            count += batch
    return {
        "joint_rmse_rad": (totals["joint_squared"] / (count * 2)) ** 0.5,
        "cartesian_rmse_m": (totals["cart_squared"] / (count * 2)) ** 0.5,
        "mean_cartesian_error_m": totals["cart_distance"] / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=60_000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    observations, current_q, target_xz, label_q = generate_dataset(args.samples, args.seed)
    permutation = np.random.default_rng(args.seed).permutation(args.samples)
    split = int(args.samples * 0.8)
    train_indices, validation_indices = permutation[:split], permutation[split:]
    np.savez_compressed(
        DATASET_PATH,
        observations=observations,
        current_q=current_q,
        target_xz=target_xz,
        label_q=label_q,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )

    train_dataset = TensorDataset(
        torch.from_numpy(observations[train_indices]),
        torch.from_numpy(label_q[train_indices]),
        torch.from_numpy(target_xz[train_indices]),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(observations[validation_indices]),
        torch.from_numpy(label_q[validation_indices]),
        torch.from_numpy(target_xz[validation_indices]),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IKMLPNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    history = []
    best_error = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_observations, batch_q, batch_xz in train_loader:
            batch_observations = batch_observations.to(device)
            batch_q = batch_q.to(device)
            batch_xz = batch_xz.to(device)
            prediction = model(batch_observations)
            joint_loss = nn.functional.mse_loss(prediction, batch_q)
            cartesian_loss = nn.functional.mse_loss(
                torch_forward_kinematics(prediction), batch_xz
            )
            loss = joint_loss + CARTESIAN_LOSS_WEIGHT * cartesian_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_observations.shape[0]
            seen += batch_observations.shape[0]
        scheduler.step()

        metrics = evaluate(model, validation_loader, device)
        metrics.update({
            "epoch": epoch,
            "training_objective": running_loss / seen,
            "learning_rate": scheduler.get_last_lr()[0],
        })
        history.append(metrics)
        if metrics["mean_cartesian_error_m"] < best_error:
            best_error = metrics["mean_cartesian_error_m"]
            torch.save({
                "model_state": model.state_dict(),
                "architecture": "6-128-128-128-2-silu-linear",
                "input_encoding": "sin_q1,cos_q1,sin_q2,cos_q2,target_x,target_z",
                "samples": args.samples,
                "epoch": epoch,
            }, CHECKPOINT_PATH)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} objective={metrics['training_objective']:.6f} "
                f"joint_rmse={metrics['joint_rmse_rad']:.4f}rad "
                f"mean_cart_error={metrics['mean_cartesian_error_m']:.4f}m"
            )

    report = {
        "controller": "ik_mlp",
        "device": str(device),
        "architecture": "6 → 128 → 128 → 128 → 2 (SiLU, linear output)",
        "input": "sin/cos of current joint angles plus target x/z relative to shoulder",
        "output": "desired shoulder and elbow angles in radians",
        "objective": "joint MSE + 10 × Cartesian FK MSE",
        "samples": args.samples,
        "training_samples": split,
        "validation_samples": args.samples - split,
        "best_mean_cartesian_error_m": best_error,
        "history": history,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    print(f"checkpoint={CHECKPOINT_PATH}")
    print(f"dataset={DATASET_PATH}")
    print(f"metrics={METRICS_PATH}")


if __name__ == "__main__":
    main()
