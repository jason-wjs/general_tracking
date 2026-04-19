from dataclasses import dataclass

import pytest
import torch

from general_tracking.tasks.general_tracking.mdp import observations


def test_build_reduced_coords_obs_concatenates_expected_terms():
  obs = observations.build_reduced_coords_obs(
    dof_pos=torch.tensor([[0.1, 0.2]]),
    dof_vel=torch.tensor([[1.0, 2.0]]),
    anchor_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    root_local_ang_vel=torch.tensor([[0.3, 0.4, 0.5]]),
  )

  assert obs.shape == (1, 10)
  assert torch.allclose(
    obs,
    torch.tensor([[0.1, 0.2, 1.0, 2.0, 0.3, 0.4, 0.5, 0.0, 0.0, -1.0]]),
  )


def test_build_reduced_coords_target_poses_uses_future_steps_and_identity_6d():
  target = observations.build_reduced_coords_target_poses(
    current_anchor_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    future_anchor_quat_w=torch.tensor(
      [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
    ),
    future_dof_vel=torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
    future_dof_pos=torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]]),
    future_steps=[1, 2],
  )

  assert target.shape == (1, 20)
  first_step = target[0, :10]
  assert torch.allclose(
    first_step,
    torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 4.0, 0.3, 0.4]),
  )


def test_build_max_coords_obs_heading_normalizes_positions_and_keeps_root_height():
  obs = observations.build_max_coords_obs(
    body_pos_w=torch.tensor([[[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]]]),
    body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
    body_lin_vel_w=torch.zeros(1, 2, 3),
    body_ang_vel_w=torch.zeros(1, 2, 3),
    root_height=True,
    local_obs=True,
  )

  assert obs.shape == (1, 1 + 3 + 12 + 6 + 6)
  assert obs[0, 0].item() == pytest.approx(3.0)
  assert torch.allclose(obs[0, 1:4], torch.tensor([1.0, 0.0, 0.0]))
  assert torch.allclose(obs[0, 4:16], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0] * 2))


@dataclass
class _FakeActionTerm:
  history: torch.Tensor


class _FakeActionManager:
  def __init__(self, term: _FakeActionTerm):
    self._term = term

  def get_term(self, name: str) -> _FakeActionTerm:
    assert name == "joint_pos"
    return self._term


@dataclass
class _FakeEnv:
  action_manager: _FakeActionManager


def test_processed_action_history_reads_one_slot_buffer():
  env = _FakeEnv(
    action_manager=_FakeActionManager(
      _FakeActionTerm(history=torch.tensor([[0.1, 0.2, 0.3]]))
    )
  )
  obs = observations.processed_action_history(env, action_name="joint_pos")
  assert torch.allclose(obs, torch.tensor([[0.1, 0.2, 0.3]]))
