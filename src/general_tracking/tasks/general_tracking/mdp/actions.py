"""Custom action terms for general tracking."""

from __future__ import annotations

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


class BMPositionActionCfg(JointPositionActionCfg):
  """BeyondMimic-style joint position action config."""

  def build(self, env):
    return BMPositionAction(self, env)


class BMPositionAction(JointPositionAction):
  """Joint position action that exposes a 1-slot processed-action history."""

  def __init__(self, cfg: BMPositionActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    self.history = torch.zeros_like(self._processed_actions)

  def process_actions(self, actions: torch.Tensor):
    super().process_actions(actions)
    self.history[:] = self._processed_actions

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    super().reset(env_ids=env_ids)
    self._processed_actions[env_ids] = 0.0
    self.history[env_ids] = 0.0
