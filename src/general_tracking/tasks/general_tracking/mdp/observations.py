"""Observation helpers for general tracking."""

from __future__ import annotations

from typing import Protocol, cast

import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
  yaw_quat,
)


class _CommandLike(Protocol):
  robot_joint_pos: torch.Tensor
  robot_joint_vel: torch.Tensor
  robot_anchor_quat_w: torch.Tensor
  robot_anchor_ang_vel_w: torch.Tensor
  robot_body_pos_w: torch.Tensor
  robot_body_quat_w: torch.Tensor
  robot_body_lin_vel_w: torch.Tensor
  robot_body_ang_vel_w: torch.Tensor
  motion_anchor_body_index: int
  cfg: object

  def get_future_states(self, offsets: list[int] | tuple[int, ...]) -> dict[str, torch.Tensor]: ...


class _CommandManagerLike(Protocol):
  def get_term(self, name: str) -> _CommandLike: ...


class _EnvLike(Protocol):
  command_manager: _CommandManagerLike


def build_reduced_coords_obs(
  *,
  dof_pos: torch.Tensor,
  dof_vel: torch.Tensor,
  anchor_quat_w: torch.Tensor,
  root_local_ang_vel: torch.Tensor,
) -> torch.Tensor:
  """Build reduced-coordinate proprioception used by the actor."""
  gravity_vec_w = torch.zeros(
    dof_pos.shape[0],
    3,
    device=dof_pos.device,
    dtype=dof_pos.dtype,
  )
  gravity_vec_w[:, 2] = -1.0
  projected_gravity = quat_apply_inverse(anchor_quat_w, gravity_vec_w)
  return torch.cat(
    [dof_pos, dof_vel, root_local_ang_vel, projected_gravity],
    dim=-1,
  )


def build_reduced_coords_target_poses(
  *,
  current_anchor_quat_w: torch.Tensor,
  future_anchor_quat_w: torch.Tensor,
  future_dof_vel: torch.Tensor,
  future_dof_pos: torch.Tensor,
  future_steps: list[int] | tuple[int, ...] | None = None,
  include_dof_vel: bool = True,
) -> torch.Tensor:
  """Build reduced-coordinate future targets."""
  if future_steps is not None:
    future_anchor_quat_w = future_anchor_quat_w[:, future_steps]
    future_dof_vel = future_dof_vel[:, future_steps]
    future_dof_pos = future_dof_pos[:, future_steps]

  num_envs, num_steps = future_anchor_quat_w.shape[:2]
  current_quat = current_anchor_quat_w[:, None, :].expand(-1, num_steps, -1)
  rel_quat = quat_mul(quat_inv(current_quat.reshape(-1, 4)), future_anchor_quat_w.reshape(-1, 4))
  rel_rot_6d = matrix_from_quat(rel_quat)[..., :2].reshape(num_envs, num_steps, -1)

  obs_parts = [rel_rot_6d]
  if include_dof_vel:
    obs_parts.append(future_dof_vel)
  obs_parts.append(future_dof_pos)
  return torch.cat(obs_parts, dim=-1).reshape(num_envs, -1)


def build_max_coords_obs(
  *,
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  body_lin_vel_w: torch.Tensor,
  body_ang_vel_w: torch.Tensor,
  root_height: bool = True,
  local_obs: bool = True,
) -> torch.Tensor:
  """Build full-body critic observations in max coordinates."""
  root_pos_w = body_pos_w[:, 0]
  root_quat_w = body_quat_w[:, 0]
  root_height_obs = root_pos_w[:, 2:3] if root_height else torch.zeros_like(root_pos_w[:, 2:3])

  if local_obs:
    heading_inv = quat_inv(yaw_quat(root_quat_w))
    heading_inv_expand = heading_inv[:, None, :].expand(-1, body_pos_w.shape[1], -1)

    body_pos_local = quat_apply(
      heading_inv_expand.reshape(-1, 4),
      (body_pos_w - root_pos_w[:, None, :]).reshape(-1, 3),
    ).reshape(body_pos_w.shape)
    body_quat_local = quat_mul(
      heading_inv_expand.reshape(-1, 4),
      body_quat_w.reshape(-1, 4),
    ).reshape(body_quat_w.shape)
    body_lin_vel_local = quat_apply(
      heading_inv_expand.reshape(-1, 4),
      body_lin_vel_w.reshape(-1, 3),
    ).reshape(body_lin_vel_w.shape)
    body_ang_vel_local = quat_apply(
      heading_inv_expand.reshape(-1, 4),
      body_ang_vel_w.reshape(-1, 3),
    ).reshape(body_ang_vel_w.shape)
  else:
    body_pos_local = body_pos_w
    body_quat_local = body_quat_w
    body_lin_vel_local = body_lin_vel_w
    body_ang_vel_local = body_ang_vel_w

  body_pos_obs = body_pos_local[:, 1:, :].reshape(body_pos_w.shape[0], -1)
  body_rot_obs = matrix_from_quat(body_quat_local)[..., :2].reshape(body_quat_w.shape[0], -1)
  body_vel_obs = body_lin_vel_local.reshape(body_lin_vel_w.shape[0], -1)
  body_ang_vel_obs = body_ang_vel_local.reshape(body_ang_vel_w.shape[0], -1)
  return torch.cat(
    [root_height_obs, body_pos_obs, body_rot_obs, body_vel_obs, body_ang_vel_obs],
    dim=-1,
  )


def build_max_coords_target_poses(
  *,
  current_body_pos_w: torch.Tensor,
  current_body_quat_w: torch.Tensor,
  current_body_lin_vel_w: torch.Tensor,
  current_body_ang_vel_w: torch.Tensor,
  future_body_pos_w: torch.Tensor,
  future_body_quat_w: torch.Tensor,
  future_body_lin_vel_w: torch.Tensor,
  future_body_ang_vel_w: torch.Tensor,
  future_steps: list[int] | tuple[int, ...] | None = None,
  with_velocities: bool = True,
  with_relative: bool = True,
) -> torch.Tensor:
  """Build full-body future targets for the critic."""
  if future_steps is not None:
    future_body_pos_w = future_body_pos_w[:, future_steps]
    future_body_quat_w = future_body_quat_w[:, future_steps]
    future_body_lin_vel_w = future_body_lin_vel_w[:, future_steps]
    future_body_ang_vel_w = future_body_ang_vel_w[:, future_steps]

  num_envs, num_steps, num_bodies = future_body_pos_w.shape[:3]
  current_root_pos = current_body_pos_w[:, 0]
  current_root_quat = current_body_quat_w[:, 0]
  heading_inv = quat_inv(yaw_quat(current_root_quat))
  heading_inv_expand = heading_inv[:, None, None, :].expand(-1, num_steps, num_bodies, -1)

  current_body_pos_expand = current_body_pos_w[:, None].expand(-1, num_steps, -1, -1)
  current_body_quat_expand = current_body_quat_w[:, None].expand(-1, num_steps, -1, -1)
  current_body_lin_vel_expand = current_body_lin_vel_w[:, None].expand(-1, num_steps, -1, -1)
  current_body_ang_vel_expand = current_body_ang_vel_w[:, None].expand(-1, num_steps, -1, -1)

  current_root_pos_expand = current_root_pos[:, None, None, :].expand(-1, num_steps, num_bodies, -1)

  body_pos_abs = quat_apply(
    heading_inv_expand.reshape(-1, 4),
    (future_body_pos_w - current_root_pos_expand).reshape(-1, 3),
  ).reshape(num_envs, num_steps, num_bodies, 3)
  body_quat_abs = quat_mul(
    heading_inv_expand.reshape(-1, 4),
    future_body_quat_w.reshape(-1, 4),
  ).reshape(num_envs, num_steps, num_bodies, 4)

  obs_parts = [
    body_pos_abs.reshape(num_envs, num_steps, -1),
    matrix_from_quat(body_quat_abs)[..., :2].reshape(num_envs, num_steps, -1),
  ]

  if with_relative:
    body_pos_rel = quat_apply(
      heading_inv_expand.reshape(-1, 4),
      (future_body_pos_w - current_body_pos_expand).reshape(-1, 3),
    ).reshape(num_envs, num_steps, num_bodies, 3)
    body_quat_rel = quat_mul(
      quat_inv(current_body_quat_expand.reshape(-1, 4)),
      future_body_quat_w.reshape(-1, 4),
    ).reshape(num_envs, num_steps, num_bodies, 4)
    obs_parts.extend(
      [
        body_pos_rel.reshape(num_envs, num_steps, -1),
        matrix_from_quat(body_quat_rel)[..., :2].reshape(num_envs, num_steps, -1),
      ]
    )

  if with_velocities:
    body_lin_vel = quat_apply(
      heading_inv_expand.reshape(-1, 4),
      (future_body_lin_vel_w - current_body_lin_vel_expand).reshape(-1, 3),
    ).reshape(num_envs, num_steps, num_bodies, 3)
    body_ang_vel = quat_apply(
      heading_inv_expand.reshape(-1, 4),
      (future_body_ang_vel_w - current_body_ang_vel_expand).reshape(-1, 3),
    ).reshape(num_envs, num_steps, num_bodies, 3)
    obs_parts.extend(
      [
        body_lin_vel.reshape(num_envs, num_steps, -1),
        body_ang_vel.reshape(num_envs, num_steps, -1),
      ]
    )

  return torch.cat(obs_parts, dim=-1).reshape(num_envs, -1)


class _ActionHistoryTerm(Protocol):
  history: torch.Tensor


def processed_action_history(env, action_name: str = "joint_pos") -> torch.Tensor:
  """Read the 1-slot processed-action history buffer from a custom action term."""
  term = env.action_manager.get_term(action_name)
  if not isinstance(term.history, torch.Tensor):
    raise TypeError("action term history must be a torch.Tensor")
  return term.history


def reduced_coords_obs(env: _EnvLike, command_name: str = "motion") -> torch.Tensor:
  command = cast(_CommandLike, env.command_manager.get_term(command_name))
  root_local_ang_vel = quat_apply_inverse(
    command.robot_anchor_quat_w,
    command.robot_anchor_ang_vel_w,
  )
  return build_reduced_coords_obs(
    dof_pos=command.robot_joint_pos,
    dof_vel=command.robot_joint_vel,
    anchor_quat_w=command.robot_anchor_quat_w,
    root_local_ang_vel=root_local_ang_vel,
  )


def reduced_coords_target_poses(
  env: _EnvLike,
  command_name: str = "motion",
  future_steps: list[int] | tuple[int, ...] = (1, 2, 4, 8),
) -> torch.Tensor:
  command = cast(_CommandLike, env.command_manager.get_term(command_name))
  future = command.get_future_states(future_steps)
  return build_reduced_coords_target_poses(
    current_anchor_quat_w=command.robot_anchor_quat_w,
    future_anchor_quat_w=future["body_quat_w"][:, :, command.motion_anchor_body_index],
    future_dof_vel=future["joint_vel"],
    future_dof_pos=future["joint_pos"],
    include_dof_vel=True,
  )


def max_coords_obs(env: _EnvLike, command_name: str = "motion") -> torch.Tensor:
  command = cast(_CommandLike, env.command_manager.get_term(command_name))
  return build_max_coords_obs(
    body_pos_w=command.robot_body_pos_w,
    body_quat_w=command.robot_body_quat_w,
    body_lin_vel_w=command.robot_body_lin_vel_w,
    body_ang_vel_w=command.robot_body_ang_vel_w,
    root_height=True,
    local_obs=True,
  )


def max_coords_target_poses(
  env: _EnvLike,
  command_name: str = "motion",
  future_steps: list[int] | tuple[int, ...] = (1, 2, 4, 8),
) -> torch.Tensor:
  command = cast(_CommandLike, env.command_manager.get_term(command_name))
  future = command.get_future_states(future_steps)
  return build_max_coords_target_poses(
    current_body_pos_w=command.robot_body_pos_w,
    current_body_quat_w=command.robot_body_quat_w,
    current_body_lin_vel_w=command.robot_body_lin_vel_w,
    current_body_ang_vel_w=command.robot_body_ang_vel_w,
    future_body_pos_w=future["body_pos_w"],
    future_body_quat_w=future["body_quat_w"],
    future_body_lin_vel_w=future["body_lin_vel_w"],
    future_body_ang_vel_w=future["body_ang_vel_w"],
    with_velocities=True,
    with_relative=True,
  )
