"""Reward kernels for general tracking."""

from __future__ import annotations

from typing import Any, Protocol, cast

import torch
from mjlab.utils.lab_api.math import quat_error_magnitude


class _CommandLike(Protocol):
  anchor_quat_w: torch.Tensor
  robot_anchor_quat_w: torch.Tensor
  body_pos_relative_w: torch.Tensor
  robot_body_pos_w: torch.Tensor
  body_quat_relative_w: torch.Tensor
  robot_body_quat_w: torch.Tensor
  body_lin_vel_w: torch.Tensor
  robot_body_lin_vel_w: torch.Tensor
  body_ang_vel_w: torch.Tensor
  robot_body_ang_vel_w: torch.Tensor


def motion_global_anchor_orientation_error_exp(
  env: Any,
  command_name: str,
  std: float,
) -> torch.Tensor:
  command_manager = cast(Any, env.command_manager)
  command = cast(_CommandLike, command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def region_weighted_body_position_error_exp(
  env: Any,
  command_name: str,
  std: float,
  region_weights: torch.Tensor | None = None,
) -> torch.Tensor:
  command_manager = cast(Any, env.command_manager)
  command = cast(_CommandLike, command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.body_pos_relative_w - command.robot_body_pos_w),
    dim=-1,
  )
  return _exp_from_per_body_error(error, std=std, region_weights=region_weights)


def region_weighted_body_orientation_error_exp(
  env: Any,
  command_name: str,
  std: float,
  region_weights: torch.Tensor | None = None,
) -> torch.Tensor:
  command_manager = cast(Any, env.command_manager)
  command = cast(_CommandLike, command_manager.get_term(command_name))
  error = (
    quat_error_magnitude(command.body_quat_relative_w, command.robot_body_quat_w) ** 2
  )
  return _exp_from_per_body_error(error, std=std, region_weights=region_weights)


def region_weighted_body_linear_velocity_error_exp(
  env: Any,
  command_name: str,
  std: float,
  region_weights: torch.Tensor | None = None,
) -> torch.Tensor:
  command_manager = cast(Any, env.command_manager)
  command = cast(_CommandLike, command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.body_lin_vel_w - command.robot_body_lin_vel_w),
    dim=-1,
  )
  return _exp_from_per_body_error(error, std=std, region_weights=region_weights)


def region_weighted_body_angular_velocity_error_exp(
  env: Any,
  command_name: str,
  std: float,
  region_weights: torch.Tensor | None = None,
) -> torch.Tensor:
  command_manager = cast(Any, env.command_manager)
  command = cast(_CommandLike, command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.body_ang_vel_w - command.robot_body_ang_vel_w),
    dim=-1,
  )
  return _exp_from_per_body_error(error, std=std, region_weights=region_weights)


def _exp_from_per_body_error(
  error: torch.Tensor,
  *,
  std: float,
  region_weights: torch.Tensor | None,
) -> torch.Tensor:
  if error.ndim != 2:
    raise ValueError(f"expected per-body error with shape [num_envs, num_bodies], got {error.shape}")
  if region_weights is not None:
    weights = region_weights.to(device=error.device, dtype=error.dtype).view(1, -1)
    if weights.shape[1] != error.shape[1]:
      raise ValueError(
        "region_weights length does not match number of bodies: "
        f"{weights.shape[1]} != {error.shape[1]}"
      )
    error = error * weights
  return torch.exp(-error.mean(dim=-1) / std**2)
