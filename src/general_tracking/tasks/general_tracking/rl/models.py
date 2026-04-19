"""Custom actor model for L2C2 training."""

from __future__ import annotations

import torch
from rsl_rl.models import MLPModel


class GeneralTrackingActorModel(MLPModel):
  """Actor model that can re-run the same trunk on clean observations."""

  def __init__(
    self,
    obs,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    *,
    clean_obs_set: str = "actor_clean",
    **kwargs,
  ) -> None:
    self.clean_obs_set_name = clean_obs_set
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    self.clean_obs_groups, self.clean_obs_dim = self._get_obs_dim(
      obs,
      obs_groups,
      clean_obs_set,
    )
    if self.clean_obs_dim != self.obs_dim:
      raise ValueError(
        f"clean obs dim must match actor obs dim: {self.clean_obs_dim} != {self.obs_dim}"
      )

  def forward_clean(
    self,
    obs,
    masks: torch.Tensor | None = None,
    hidden_state=None,
  ) -> torch.Tensor:
    del masks, hidden_state
    obs_list = [obs[group] for group in self.clean_obs_groups]
    latent = torch.cat(obs_list, dim=-1)
    latent = self.obs_normalizer(latent)
    mlp_output = self.mlp(latent)
    if self.distribution is not None:
      return self.distribution.deterministic_output(mlp_output)
    return mlp_output
