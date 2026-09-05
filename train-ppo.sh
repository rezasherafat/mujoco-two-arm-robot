#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e WANDB_API_KEY \
  -e WANDB_ENTITY \
  -e WANDB_PROJECT \
  -e WANDB_MODE \
  -v "${project_dir}:/workspace" \
  -w /workspace \
  mujoco-thor:26.05 \
  python -u -m training.train_ppo_joint_delta "$@"
