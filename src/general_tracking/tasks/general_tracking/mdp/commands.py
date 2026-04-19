"""Multi-clip motion command for general tracking."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from general_tracking.data.motion_library import MotionLibrary

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  import viser
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class MultiClipMotionCommand(CommandTerm):
  cfg: "MultiClipMotionCommandCfg"
  _env: "ManagerBasedRlEnv"

  def __init__(self, cfg: "MultiClipMotionCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    env_control_fps = 1.0 / env.step_dt
    self.motion = MotionLibrary(
      cfg.motion_library_path,
      env_control_fps=env_control_fps,
      device=self.device,
    )
    self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs,
      len(cfg.body_names),
      3,
      device=self.device,
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs,
      len(cfg.body_names),
      4,
      device=self.device,
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def current_state(self) -> dict[str, torch.Tensor]:
    return self.motion.get_state_at(self.clip_ids, self.time_steps)

  def get_future_states(self, offsets: list[int] | tuple[int, ...]) -> dict[str, torch.Tensor]:
    return self.motion.get_future_states(self.clip_ids, self.time_steps, offsets)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.current_state["joint_pos"]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.current_state["joint_vel"]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self.current_state["body_pos_w"] + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.current_state["body_quat_w"]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.current_state["body_lin_vel_w"]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.current_state["body_ang_vel_w"]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.motion_anchor_body_index]

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w[:, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    from mjlab.utils.lab_api.math import quat_error_magnitude

    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w,
      dim=-1,
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w,
      self.robot_anchor_quat_w,
    )
    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w,
      dim=-1,
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w,
      self.robot_body_quat_w,
    ).mean(dim=-1)
    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w,
      dim=-1,
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w,
      dim=-1,
    ).mean(dim=-1)
    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos,
      dim=-1,
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel,
      dim=-1,
    )

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    if len(env_ids) == 0:
      return

    self.clip_ids[env_ids] = self.motion.sample_clip_ids(len(env_ids))
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    else:
      self.time_steps[env_ids] = self.motion.sample_init_time(
        self.clip_ids[env_ids],
        init_start_prob=self.cfg.init_start_prob,
      )

    states = self.motion.get_state_at(self.clip_ids[env_ids], self.time_steps[env_ids])
    root_pos = states["body_pos_w"][:, 0].clone() + self._env.scene.env_origins[env_ids]
    root_ori = states["body_quat_w"][:, 0].clone()
    root_lin_vel = states["body_lin_vel_w"][:, 0].clone()
    root_ang_vel = states["body_ang_vel_w"][:, 0].clone()

    pose_ranges = torch.tensor(
      [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
      device=self.device,
      dtype=torch.float32,
    )
    pose_noise = sample_uniform(
      pose_ranges[:, 0],
      pose_ranges[:, 1],
      (len(env_ids), 6),
      device=self.device,
    )
    root_pos += pose_noise[:, :3]
    root_ori = quat_mul(
      quat_from_euler_xyz(pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5]),
      root_ori,
    )

    vel_ranges = torch.tensor(
      [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
      device=self.device,
      dtype=torch.float32,
    )
    vel_noise = sample_uniform(
      vel_ranges[:, 0],
      vel_ranges[:, 1],
      (len(env_ids), 6),
      device=self.device,
    )
    root_lin_vel += vel_noise[:, :3]
    root_ang_vel += vel_noise[:, 3:]

    joint_pos = states["joint_pos"].clone()
    joint_pos += sample_uniform(
      self.cfg.joint_position_range[0],
      self.cfg.joint_position_range[1],
      joint_pos.shape,
      device=self.device,
    )

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      states["joint_vel"],
    )
    self.update_relative_body_poses()

  def update_relative_body_poses(self) -> None:
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

    delta_pos_w = robot_anchor_pos_w_repeat.clone()
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w,
      self.body_pos_w - anchor_pos_w_repeat,
    )

  def _update_command(self):
    self.time_steps += 1
    clip_lengths = self.motion.clip_lengths[self.clip_ids]
    env_ids = torch.where(self.time_steps >= clip_lengths)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)
    self.update_relative_body_poses()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        for gi in range(self._ghost_model.ngeom):
          if self._ghost_model.geom_contype[gi] != 0 or self._ghost_model.geom_conaffinity[gi] != 0:
            self._ghost_model.geom_rgba[gi, 3] = 0
          else:
            self._ghost_model.geom_rgba[gi] = self._ghost_color

      entity = self.robot
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()
      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()
        visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")
      return

    for batch in env_indices:
      desired_body_pos = self.body_pos_w[batch].cpu().numpy()
      desired_body_quat = self.body_quat_w[batch]
      desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()
      current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
      current_body_quat = self.robot_body_quat_w[batch]
      current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()
      for i, body_name in enumerate(self.cfg.body_names):
        visualizer.add_frame(
          position=desired_body_pos[i],
          rotation_matrix=desired_body_rotm[i],
          scale=0.08,
          label=f"desired_{body_name}_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )
        visualizer.add_frame(
          position=current_body_pos[i],
          rotation_matrix=current_body_rotm[i],
          scale=0.12,
          label=f"current_{body_name}_{batch}",
        )

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: "Callable[[], int]",
    on_change: "Callable[[], None] | None" = None,
    request_action: "Callable[[str, Any], None] | None" = None,
  ) -> None:
    max_frame = int(self.motion.clip_lengths.max().item()) - 1
    with server.gui.add_folder(name.capitalize()):
      scrubber = server.gui.add_slider("Frame", min=0, max=max_frame, step=1, initial_value=0)

      @scrubber.on_update
      def _(_) -> None:
        idx = get_env_idx()
        self.reset_to_frame(torch.tensor([idx], device=self.device), int(scrubber.value))
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action("CUSTOM", {"type": "gui_reset", "all_envs": all_envs_cb.value})

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn)
    self._set_scrubber_disabled(True)

  def _set_scrubber_disabled(self, disabled: bool) -> None:
    for handle in self._scrubber_handles:
      handle.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    if hasattr(self, "_scrubber_handles"):
      self._set_scrubber_disabled(not paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    if not hasattr(self, "_scrubber_handles"):
      return False
    frame = int(self._scrubber_handles[0].value)
    self.reset_to_frame(env_ids, frame)
    self.update_relative_body_poses()
    return True

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    self.time_steps[env_ids] = frame
    states = self.motion.get_state_at(self.clip_ids[env_ids], self.time_steps[env_ids])
    self._write_reference_state_to_sim(
      env_ids,
      states["body_pos_w"][:, 0] + self._env.scene.env_origins[env_ids],
      states["body_quat_w"][:, 0],
      states["body_lin_vel_w"][:, 0],
      states["body_ang_vel_w"][:, 0],
      states["joint_pos"],
      states["joint_vel"],
    )

  def reset_to_clip_frame(
    self,
    env_ids: torch.Tensor,
    clip_ids: torch.Tensor,
    frame: int = 0,
  ) -> None:
    self.clip_ids[env_ids] = clip_ids.to(device=self.device, dtype=torch.long)
    self.reset_to_frame(env_ids, frame)


@dataclass(kw_only=True)
class MultiClipMotionCommandCfg(CommandTermCfg):
  motion_library_path: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  init_start_prob: float = 0.2
  future_steps: tuple[int, ...] = (1, 2, 4, 8)
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  sampling_mode: Literal["uniform", "start"] = "uniform"

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: "ManagerBasedRlEnv") -> MultiClipMotionCommand:
    return MultiClipMotionCommand(self, env)
