import math
from dataclasses import dataclass

import pytest
import torch

from general_tracking.tasks.general_tracking.mdp import rewards


@dataclass
class _FakeCommand:
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


class _FakeCommandManager:
  def __init__(self, command: _FakeCommand):
    self._command = command

  def get_term(self, name: str) -> _FakeCommand:
    assert name == "motion"
    return self._command


@dataclass
class _FakeEnv:
  command_manager: _FakeCommandManager


def _make_env() -> _FakeEnv:
  quarter_turn = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]])
  identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  command = _FakeCommand(
    anchor_quat_w=quarter_turn,
    robot_anchor_quat_w=identity,
    body_pos_relative_w=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
    robot_body_pos_w=torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
    body_quat_relative_w=torch.stack([identity, quarter_turn], dim=1),
    robot_body_quat_w=torch.stack([identity, identity], dim=1),
    body_lin_vel_w=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
    robot_body_lin_vel_w=torch.zeros(1, 2, 3),
    body_ang_vel_w=torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]),
    robot_body_ang_vel_w=torch.zeros(1, 2, 3),
  )
  return _FakeEnv(command_manager=_FakeCommandManager(command))


def test_global_anchor_orientation_error_exp_matches_expected_kernel():
  env = _make_env()
  reward = rewards.motion_global_anchor_orientation_error_exp(
    env,
    command_name="motion",
    std=0.4,
  )
  expected = math.exp(-(math.pi / 2) ** 2 / (0.4**2))
  assert reward.item() == pytest.approx(expected)


def test_region_weighted_body_position_error_exp_applies_per_body_weights_before_mean():
  env = _make_env()
  region_weights = torch.tensor([0.5, 1.5])
  reward = rewards.region_weighted_body_position_error_exp(
    env,
    command_name="motion",
    std=1.0,
    region_weights=region_weights,
  )
  weighted_mean_error = (0.0 * 0.5 + 1.0 * 1.5) / 2.0
  assert reward.item() == pytest.approx(math.exp(-weighted_mean_error))


def test_region_weighted_body_orientation_error_exp_uses_quat_error_squared():
  env = _make_env()
  region_weights = torch.tensor([1.0, 3.0])
  reward = rewards.region_weighted_body_orientation_error_exp(
    env,
    command_name="motion",
    std=1.0,
    region_weights=region_weights,
  )
  weighted_mean_error = ((0.0**2) * 1.0 + ((math.pi / 2) ** 2) * 3.0) / 2.0
  assert reward.item() == pytest.approx(math.exp(-weighted_mean_error))
