#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="/home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz"

cd "${REPO_ROOT}"
uv run gt-play \
  --task GeneralTracking-Flat-Unitree-G1 \
  --checkpoint  \
  --motion-lib-path "${DATA_DIR}/motion_manifest.yaml" \
  --num-envs 4 \
  "$@"
