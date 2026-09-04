#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --rm \
  --name mujoco-arm-web \
  --runtime=nvidia \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e MUJOCO_GL=egl \
  -p 127.0.0.1:8000:8000 \
  -v "${project_dir}:/workspace" \
  -w /workspace \
  mujoco-thor:26.05 \
  uvicorn web_controller:app --host 0.0.0.0 --port 8000
