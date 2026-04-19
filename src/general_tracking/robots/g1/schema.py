"""Frozen G1 schema for general tracking."""

from __future__ import annotations

import torch

ANCHOR_BODY_NAME = "torso_link"

BODY_NAMES: tuple[str, ...] = (
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "waist_roll_link",
  "torso_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
)

JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

PARENT_INDICES: tuple[int, ...] = (
  -1,
  0,
  1,
  2,
  3,
  4,
  5,
  0,
  7,
  8,
  9,
  10,
  11,
  0,
  13,
  14,
  15,
  16,
  17,
  18,
  19,
  20,
  21,
  15,
  23,
  24,
  25,
  26,
  27,
  28,
)

LOCAL_POS: tuple[tuple[float, float, float], ...] = (
  (0.0, 0.0, 0.793),
  (0.0, 0.064452, -0.1027),
  (0.0, 0.052, -0.030465),
  (0.025001, 0.0, -0.12412),
  (-0.078273, 0.0021489, -0.17734),
  (0.0, -9.4445e-05, -0.30001),
  (0.0, 0.0, -0.017558),
  (0.0, -0.064452, -0.1027),
  (0.0, -0.052, -0.030465),
  (0.025001, 0.0, -0.12412),
  (-0.078273, -0.0021489, -0.17734),
  (0.0, 9.4445e-05, -0.30001),
  (0.0, 0.0, -0.017558),
  (0.0, 0.0, 0.0),
  (-0.0039635, 0.0, 0.044),
  (0.0, 0.0, 0.0),
  (0.0039563, 0.10022, 0.24778),
  (0.0, 0.038, -0.013831),
  (0.0, 0.00624, -0.1032),
  (0.015783, 0.0, -0.080518),
  (0.1, 0.00188791, -0.01),
  (0.038, 0.0, 0.0),
  (0.046, 0.0, 0.0),
  (0.0039563, -0.10021, 0.24778),
  (0.0, -0.038, -0.013831),
  (0.0, -0.00624, -0.1032),
  (0.015783, 0.0, -0.080518),
  (0.1, -0.00188791, -0.01),
  (0.038, 0.0, 0.0),
  (0.046, 0.0, 0.0),
)

NUM_BODIES = len(BODY_NAMES)
NUM_DOFS = len(JOINT_NAMES)
ANCHOR_BODY_INDEX = BODY_NAMES.index(ANCHOR_BODY_NAME)
CONTROL_FPS = 50.0


def compute_body_density_weights(
  parent_indices: torch.Tensor,
  local_pos: torch.Tensor,
  discount: float = 0.9,
) -> torch.Tensor:
  """Compute body weights from kinematic chain density."""
  num_bodies = int(parent_indices.numel())
  bone_lengths = local_pos.norm(dim=-1).cpu()

  paths_to_root: list[list[tuple[int, float]]] = []
  for i in range(num_bodies):
    path: list[tuple[int, float]] = []
    current = i
    cumulative_dist = 0.0
    while current != -1:
      path.append((current, cumulative_dist))
      parent = int(parent_indices[current].item())
      if parent != -1:
        cumulative_dist += float(bone_lengths[current].item())
      current = parent
    paths_to_root.append(path)

  ancestor_dists = [{body_idx: dist for body_idx, dist in path} for path in paths_to_root]
  chain_distances = torch.zeros(num_bodies, num_bodies, dtype=torch.float32)
  for i in range(num_bodies):
    for j in range(i + 1, num_bodies):
      j_ancestors = ancestor_dists[j]
      lca = -1
      lca_dist_i = 0.0
      for ancestor, dist_i in paths_to_root[i]:
        if ancestor in j_ancestors:
          lca = ancestor
          lca_dist_i = dist_i
          break
      if lca != -1:
        chain_dist = lca_dist_i + j_ancestors[lca]
      else:
        chain_dist = float("inf")
      chain_distances[i, j] = chain_dist
      chain_distances[j, i] = chain_dist

  discounted = torch.pow(
    torch.tensor(discount, dtype=torch.float32), chain_distances
  )
  discounted.fill_diagonal_(0.0)
  densities = discounted.sum(dim=1)
  weights = 1.0 / densities
  return weights / weights.sum() * num_bodies


PARENT_INDICES_TENSOR = torch.tensor(PARENT_INDICES, dtype=torch.long)
LOCAL_POS_TENSOR = torch.tensor(LOCAL_POS, dtype=torch.float32)
DENSITY_WEIGHTS = compute_body_density_weights(
  parent_indices=PARENT_INDICES_TENSOR,
  local_pos=LOCAL_POS_TENSOR,
  discount=0.9,
)

