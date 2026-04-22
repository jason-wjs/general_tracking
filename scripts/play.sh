#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="/home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz"
CHECKPOINT_DIR="/home/humanoid/Projects/Junsong_WU/learning/locomotion/controller/general_tracking/logs/rsl_rl/"

cd "${REPO_ROOT}"
uv run gt-play \
  --task GeneralTracking-Flat-Unitree-G1 \
  --checkpoint "${CHECKPOINT_DIR}/g1_general_tracking/2026-04-21_22-36-12/model_4500.pt" \
  --motion-lib-path "${DATA_DIR}/motion_manifest.yaml" \
  --num-envs 4 \
  --viewer viser \
  "$@"

