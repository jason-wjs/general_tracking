"""L2C2 regularization helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def compute_l2c2_loss(
  *,
  mu_noisy: torch.Tensor,
  mu_clean: torch.Tensor,
  obs_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
  lambda_coef: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  """Compute ProtoMotions-style L2C2 penalty.

  The input-distance term is detached to match ProtoMotions and avoid pushing
  gradients back into the observation perturbation magnitude itself.
  """
  device = mu_noisy.device
  zero = torch.zeros((), device=device, dtype=mu_noisy.dtype)
  if lambda_coef == 0.0:
    return zero, {
      "input_dist": zero,
      "output_dist": zero,
      "unweighted_loss": zero,
      "weighted_loss": zero,
    }

  input_ss = zero
  input_n = 0
  for noisy_obs, clean_obs in obs_pairs:
    diff = noisy_obs - clean_obs
    input_ss = input_ss + diff.pow(2).sum()
    input_n += diff.numel()

  if input_n == 0:
    raise ValueError("obs_pairs must contain at least one non-empty pair")

  input_dist = (input_ss / float(input_n)).detach()
  output_dist = (mu_noisy - mu_clean).pow(2).mean()
  unweighted_loss = output_dist / (input_dist + 1e-8)
  weighted_loss = unweighted_loss * lambda_coef

  return weighted_loss, {
    "input_dist": input_dist,
    "output_dist": output_dist,
    "unweighted_loss": unweighted_loss.detach(),
    "weighted_loss": weighted_loss.detach(),
  }
