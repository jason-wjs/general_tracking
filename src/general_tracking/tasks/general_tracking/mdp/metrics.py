"""Evaluation metrics for general tracking."""

from __future__ import annotations

from typing import Protocol, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_error_magnitude


class _CommandLike(Protocol):
  body_pos_w: torch.Tensor
  robot_body_pos_w: torch.Tensor
  body_quat_w: torch.Tensor
  robot_body_quat_w: torch.Tensor
  anchor_pos_w: torch.Tensor
  robot_anchor_pos_w: torch.Tensor
  anchor_quat_w: torch.Tensor
  robot_anchor_quat_w: torch.Tensor
  body_pos_relative_w: torch.Tensor


class _CommandManagerLike(Protocol):
  def get_term(self, name: str) -> _CommandLike: ...


class _EnvLike(Protocol):
  command_manager: _CommandManagerLike


def gt_error(env: _EnvLike, command_name: str) -> torch.Tensor:
  command = cast(_CommandLike, env.command_manager.get_term(command_name))
  return torch.norm(command.body_pos_w - command.robot_body_pos_w, dim=-1).mean(dim=-1)


def max_joint_error(env: _EnvLike, command_name: str) -> torch.Tensor:
  command = cast(_CommandLike, env.command_manager.get_term(command_name))
  return torch.norm(command.body_pos_w - command.robot_body_pos_w, dim=-1).max(dim=-1)[0]


def gr_error(env: _EnvLike, command_name: str) -> torch.Tensor:
  command = cast(_CommandLike, env.command_manager.get_term(command_name))
  return quat_error_magnitude(command.body_quat_w, command.robot_body_quat_w).mean(dim=-1)


def anchor_ori_metric(env: _EnvLike, command_name: str) -> torch.Tensor:
  command = cast(_CommandLike, env.command_manager.get_term(command_name))
  gravity = torch.zeros_like(command.anchor_pos_w)
  gravity[:, 2] = -1.0
  current_proj = quat_apply_inverse(command.robot_anchor_quat_w, gravity)
  ref_proj = quat_apply_inverse(command.anchor_quat_w, gravity)
  return torch.abs(current_proj[:, 2] - ref_proj[:, 2])


def relative_body_pos_metric(env: _EnvLike, command_name: str) -> torch.Tensor:
  command = cast(_CommandLike, env.command_manager.get_term(command_name))
  return torch.norm(
    command.body_pos_relative_w - command.robot_body_pos_w,
    dim=-1,
  ).max(dim=-1)[0]


def anchor_height_error(env: _EnvLike, command_name: str) -> torch.Tensor:
  command = cast(_CommandLike, env.command_manager.get_term(command_name))
  return torch.abs(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
