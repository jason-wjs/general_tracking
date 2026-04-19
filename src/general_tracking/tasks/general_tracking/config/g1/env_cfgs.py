"""G1 general tracking environment configuration."""

from __future__ import annotations

import os
from pathlib import Path

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.rewards import action_rate_l2, joint_pos_limits
from mjlab.envs.mdp.terminations import time_out
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.tracking_env_cfg import VELOCITY_RANGE, make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from general_tracking.robots.g1.action_scale import build_g1_bm_action_scale
from general_tracking.robots.g1.schema import (
  ANCHOR_BODY_NAME,
  BODY_NAMES,
  DENSITY_WEIGHTS,
)
from general_tracking.tasks.general_tracking.mdp.actions import BMPositionActionCfg
from general_tracking.tasks.general_tracking.mdp.commands import (
  MultiClipMotionCommandCfg,
)
from general_tracking.tasks.general_tracking.mdp.observations import (
  max_coords_obs,
  max_coords_target_poses,
  processed_action_history,
  reduced_coords_obs,
  reduced_coords_target_poses,
)
from general_tracking.tasks.general_tracking.mdp.rewards import (
  motion_global_anchor_orientation_error_exp,
  region_weighted_body_angular_velocity_error_exp,
  region_weighted_body_linear_velocity_error_exp,
  region_weighted_body_orientation_error_exp,
  region_weighted_body_position_error_exp,
)
from general_tracking.tasks.general_tracking.mdp.terminations import (
  motion_anchor_height_error,
)


def _default_motion_library_path() -> str:
  return os.environ.get(
    "MOTION_LIB_PATH",
    str(Path.home() / "Downloads/Data/G1_retargeted/lafan1_npz/motion_manifest.yaml"),
  )


def unitree_g1_general_tracking_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()
  cfg.scene.entities = {"robot": get_g1_robot_cfg()}
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
  )

  cfg.actions = {
    "joint_pos": BMPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=build_g1_bm_action_scale(),
      use_default_offset=True,
    ),
  }

  cfg.commands = {
    "motion": MultiClipMotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      motion_library_path=_default_motion_library_path(),
      anchor_body_name=ANCHOR_BODY_NAME,
      body_names=BODY_NAMES,
      init_start_prob=0.2,
      future_steps=(1, 2, 4, 8),
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      sampling_mode="start" if play else "uniform",
    ),
  }

  cfg.observations = {
    "noisy_reduced_coords_obs": ObservationGroupCfg(
      terms={
        "value": ObservationTermCfg(
          func=reduced_coords_obs,
          params={"command_name": "motion"},
          noise=Unoise(n_min=-0.01, n_max=0.01),
        )
      },
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "noisy_reduced_coords_target_poses": ObservationGroupCfg(
      terms={
        "value": ObservationTermCfg(
          func=reduced_coords_target_poses,
          params={"command_name": "motion", "future_steps": [1, 2, 4, 8]},
          noise=Unoise(n_min=-0.01, n_max=0.01),
        )
      },
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "clean_reduced_coords_obs": ObservationGroupCfg(
      terms={"value": ObservationTermCfg(func=reduced_coords_obs, params={"command_name": "motion"})},
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "clean_reduced_coords_target_poses": ObservationGroupCfg(
      terms={
        "value": ObservationTermCfg(
          func=reduced_coords_target_poses,
          params={"command_name": "motion", "future_steps": [1, 2, 4, 8]},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "max_coords_obs": ObservationGroupCfg(
      terms={"value": ObservationTermCfg(func=max_coords_obs, params={"command_name": "motion"})},
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "mimic_max_coords_target_poses": ObservationGroupCfg(
      terms={
        "value": ObservationTermCfg(
          func=max_coords_target_poses,
          params={"command_name": "motion", "future_steps": [1, 2, 4, 8]},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "historical_previous_processed_actions": ObservationGroupCfg(
      terms={
        "value": ObservationTermCfg(
          func=processed_action_history,
          params={"action_name": "joint_pos"},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  cfg.events = {
    "push_robot": EventTermCfg(
      func=env_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": VELOCITY_RANGE},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=r"^(left|right)_foot[1-7]_collision$"),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
  }

  cfg.rewards = {
    "motion_global_anchor_ori": RewardTermCfg(
      func=motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_rel_body_pos": RewardTermCfg(
      func=region_weighted_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3, "region_weights": DENSITY_WEIGHTS},
    ),
    "motion_rel_body_ori": RewardTermCfg(
      func=region_weighted_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4, "region_weights": DENSITY_WEIGHTS},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=region_weighted_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0, "region_weights": DENSITY_WEIGHTS},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=region_weighted_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14, "region_weights": DENSITY_WEIGHTS},
    ),
    "action_rate_l2": RewardTermCfg(func=action_rate_l2, weight=-0.1),
    "joint_limit": RewardTermCfg(
      func=joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
  }

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "fall": TerminationTermCfg(
      func=motion_anchor_height_error,
      params={"command_name": "motion", "threshold": 0.25},
    ),
  }

  cfg.viewer.body_name = ANCHOR_BODY_NAME
  cfg.episode_length_s = 20.0
  cfg.scene.num_envs = 64 if play else 4096

  if play:
    cfg.observations["noisy_reduced_coords_obs"].enable_corruption = False
    cfg.observations["noisy_reduced_coords_target_poses"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg
