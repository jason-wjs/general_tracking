from pathlib import Path

import numpy as np
import torch

from general_tracking.robots.g1 import schema

DATA_ROOT = Path("/home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz")


def _reference_density_weights(
  parent_indices: torch.Tensor,
  local_pos: torch.Tensor,
  discount: float = 0.9,
) -> torch.Tensor:
  num_bodies = int(parent_indices.numel())
  bone_lengths = local_pos.norm(dim=-1).cpu()
  paths_to_root = []
  for i in range(num_bodies):
    path = []
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

  discounted = torch.pow(torch.tensor(discount, dtype=torch.float32), chain_distances)
  discounted.fill_diagonal_(0.0)
  densities = discounted.sum(dim=1)
  weights = 1.0 / densities
  return weights / weights.sum() * num_bodies


def test_schema_matches_reference_npz_shapes():
  sample = next(DATA_ROOT.glob("*.npz"))
  with np.load(sample) as data:
    assert data["joint_pos"].shape[1] == schema.NUM_DOFS
    assert data["body_pos_w"].shape[1] == schema.NUM_BODIES


def test_anchor_and_counts_are_frozen():
  assert schema.ANCHOR_BODY_NAME == "torso_link"
  assert schema.BODY_NAMES[schema.ANCHOR_BODY_INDEX] == "torso_link"
  assert schema.NUM_DOFS == 29
  assert schema.NUM_BODIES == 30


def test_density_weights_match_reference_impl():
  expected = _reference_density_weights(
    schema.PARENT_INDICES_TENSOR,
    schema.LOCAL_POS_TENSOR,
    discount=0.9,
  )
  assert torch.allclose(schema.DENSITY_WEIGHTS, expected, atol=1e-6)
  assert torch.isclose(
    schema.DENSITY_WEIGHTS.sum(),
    torch.tensor(float(schema.NUM_BODIES)),
    atol=1e-5,
  )
  assert torch.all(schema.DENSITY_WEIGHTS > 0)

