#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e MUJOCO_GL=egl \
  -v "${project_dir}:/workspace" \
  -w /workspace \
  mujoco-thor:26.05 \
  python animate.py
