"""Termination kernels for general tracking."""

from __future__ import annotations

from typing import Protocol, cast

import torch


class _CommandLike(Protocol):
  anchor_pos_w: torch.Tensor
  robot_anchor_pos_w: torch.Tensor


class _CommandManagerLike(Protocol):
  def get_term(self, name: str) -> _CommandLike: ...


class _EnvLike(Protocol):
  command_manager: _CommandManagerLike


def motion_anchor_height_error(
  env: _EnvLike,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  command = cast(_CommandLike, env.command_manager.get_term(command_name))
  return torch.abs(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]) > threshold
