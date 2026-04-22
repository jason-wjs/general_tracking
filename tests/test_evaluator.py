import pytest
import torch
from torch import nn
from typing import cast

from general_tracking.tasks.general_tracking.rl.evaluator import (
  MotionSuccessEvaluator,
  MotionSuccessEvaluatorCfg,
  apply_motion_weight_update,
  compute_failed_mask,
)


def test_compute_failed_mask_only_uses_components_with_threshold():
  values = {
    "anchor_height_error": torch.tensor([0.10, 0.35, 0.05]),
    "relative_body_pos": torch.tensor([0.40, 0.60, 0.80]),
  }
  component_cfg = {
    "anchor_height_error": {"threshold": 0.25},
    "relative_body_pos": {},
  }

  failed, component_failures = compute_failed_mask(values, component_cfg)

  assert torch.equal(failed, torch.tensor([False, True, False]))
  assert "anchor_height_error" in component_failures
  assert "relative_body_pos" not in component_failures


def test_apply_motion_weight_update_matches_proto_reweight_rule():
  weights = torch.tensor([0.25, 0.5, 0.75])
  failed = torch.tensor([False, True, False])

  updated = apply_motion_weight_update(
    clip_weights=weights,
    failed_mask=failed,
    success_discount=0.999,
    eval_interval=200,
    failure_weight=1.0,
  )

  expected_success = 0.999**200
  assert updated[0].item() == pytest.approx(0.25 * expected_success)
  assert updated[1].item() == pytest.approx(1.0)
  assert updated[2].item() == pytest.approx(0.75 * expected_success)


class _FakeMotion:
  def __init__(self, clip_lengths: list[int], clip_weights: list[float]):
    self.num_clips = len(clip_lengths)
    self.clip_lengths = torch.tensor(clip_lengths, dtype=torch.long)
    self.clip_weights = torch.tensor(clip_weights, dtype=torch.float32)

  def update_weights(self, new_weights: torch.Tensor) -> None:
    self.clip_weights = new_weights.clone()


class _FakeMotionCmd:
  def __init__(self, motion: _FakeMotion, num_envs: int):
    self.motion = motion
    self.time_steps = torch.zeros(num_envs, dtype=torch.long)
    self.current_clip_ids = torch.zeros(num_envs, dtype=torch.long)

  def reset_to_clip_frame(
    self,
    env_ids: torch.Tensor,
    clip_ids: torch.Tensor,
    frame: int,
  ) -> None:
    self.current_clip_ids[env_ids] = clip_ids
    self.time_steps[env_ids] = frame


class _FakeObservationManager:
  def __init__(self, num_envs: int):
    self.num_envs = num_envs

  def compute(self, update_history: bool = False):
    del update_history
    return {"actor": torch.zeros((self.num_envs, 3), dtype=torch.float32)}


class _FakeCommandManager:
  def __init__(self, motion_cmd: _FakeMotionCmd):
    self._motion_cmd = motion_cmd

  def get_term(self, name: str) -> _FakeMotionCmd:
    assert name == "motion"
    return self._motion_cmd

  def compute(self, dt: float) -> None:
    del dt


class _FakeScene:
  def write_data_to_sim(self) -> None:
    return None


class _FakeSim:
  def forward(self) -> None:
    return None

  def sense(self) -> None:
    return None


class _FakeCfg:
  def __init__(self):
    self.auto_reset = True


class _FakeEnv:
  def __init__(
    self,
    *,
    num_envs: int,
    clip_lengths: list[int],
    clip_weights: list[float],
    allow_runtime_ops: bool,
    step_done_env_ids: tuple[int, ...] = (),
  ):
    self.device = torch.device("cpu")
    self.num_envs = num_envs
    self.cfg = _FakeCfg()
    self.scene = _FakeScene()
    self.sim = _FakeSim()
    self.motion_cmd = _FakeMotionCmd(
      _FakeMotion(clip_lengths=clip_lengths, clip_weights=clip_weights),
      num_envs=num_envs,
    )
    self.command_manager = _FakeCommandManager(self.motion_cmd)
    self.observation_manager = _FakeObservationManager(num_envs)
    self.reset_modes: list[bool] = []
    self.reset_calls = 0
    self.step_calls = 0
    self.allow_runtime_ops = allow_runtime_ops
    self.step_done_env_ids = set(step_done_env_ids)
    self._manual_reset_pending = torch.zeros(num_envs, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if not self.allow_runtime_ops:
      raise AssertionError("training env must not be reset during evaluation")
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, dtype=torch.long)
    self.motion_cmd.time_steps[env_ids] = 0
    self._manual_reset_pending[env_ids] = False
    self.reset_modes.append(torch.is_inference_mode_enabled())
    self.reset_calls += 1

  def step(self, actions: torch.Tensor):
    if not self.allow_runtime_ops:
      raise AssertionError("training env must not be stepped during evaluation")
    pending_ids = self._manual_reset_pending.nonzero(as_tuple=False).flatten().tolist()
    if pending_ids:
      raise RuntimeError(
        f"Environments {pending_ids} must be reset via reset(env_ids=...) before calling step() again when auto_reset=False."
      )
    del actions
    self.step_calls += 1
    self.motion_cmd.time_steps += 1
    dones = torch.zeros(self.num_envs, dtype=torch.long)
    if self.step_done_env_ids:
      done_ids = torch.tensor(sorted(self.step_done_env_ids), dtype=torch.long)
      dones[done_ids] = 1
      self._manual_reset_pending[done_ids] = True
    return (
      torch.zeros((self.num_envs, 3), dtype=torch.float32),
      torch.zeros(self.num_envs, dtype=torch.float32),
      dones,
      {},
    )


class _FakeVecEnv:
  def __init__(self, env: _FakeEnv):
    self.unwrapped = env
    self.device = torch.device("cpu")
    self.clip_actions = None

  def get_observations(self) -> torch.Tensor:
    return torch.zeros((self.unwrapped.num_envs, 3), dtype=torch.float32)

  def step(self, actions: torch.Tensor):
    return self.unwrapped.step(actions)


def _make_metric(env: _FakeEnv, command_name: str) -> torch.Tensor:
  del command_name
  return env.motion_cmd.current_clip_ids.to(dtype=torch.float32)


def test_motion_success_evaluator_uses_parameter_device_and_resets_in_inference_mode(
  tmp_path,
):
  train_vec_env = _FakeVecEnv(
    _FakeEnv(
      num_envs=1,
      clip_lengths=[2],
      clip_weights=[1.0],
      allow_runtime_ops=False,
    )
  )
  eval_vec_env = _FakeVecEnv(
    _FakeEnv(
      num_envs=1,
      clip_lengths=[2],
      clip_weights=[1.0],
      allow_runtime_ops=True,
    )
  )
  evaluator = MotionSuccessEvaluator(
    MotionSuccessEvaluatorCfg(eval_metrics_every=200),
    train_vec_env,
    str(tmp_path),
    eval_vec_env_factory=lambda num_envs: eval_vec_env,
  )
  evaluator.metric_fns = cast(dict, {"anchor_height_error": _make_metric})
  policy = nn.Linear(3, 2)

  log_dict = evaluator.run_eval(policy=policy, iteration=200)

  assert "eval/failure_rate" in log_dict
  assert eval_vec_env.unwrapped.reset_modes == [True, True]
  assert train_vec_env.unwrapped.reset_calls == 0
  assert train_vec_env.unwrapped.step_calls == 0


def test_motion_success_evaluator_uses_dedicated_eval_env_and_updates_train_weights(
  tmp_path,
):
  train_vec_env = _FakeVecEnv(
    _FakeEnv(
      num_envs=4,
      clip_lengths=[2, 2, 2],
      clip_weights=[0.25, 0.5, 0.75],
      allow_runtime_ops=False,
    )
  )
  eval_vec_env = _FakeVecEnv(
    _FakeEnv(
      num_envs=2,
      clip_lengths=[2, 2, 2],
      clip_weights=[99.0, 99.0, 99.0],
      allow_runtime_ops=True,
    )
  )
  evaluator = MotionSuccessEvaluator(
    MotionSuccessEvaluatorCfg(
      eval_metrics_every=200,
      failure_weight=1.0,
      evaluation_components={"anchor_height_error": {"threshold": 1.5}},
    ),
    train_vec_env,
    str(tmp_path),
    eval_vec_env_factory=lambda num_envs: eval_vec_env,
  )
  evaluator.metric_fns = cast(dict, {"anchor_height_error": _make_metric})
  policy = nn.Linear(3, 2)

  log_dict = evaluator.run_eval(policy=policy, iteration=200)

  expected_success = 0.999**200
  updated_weights = train_vec_env.unwrapped.motion_cmd.motion.clip_weights
  assert updated_weights.tolist() == pytest.approx(
    [0.25 * expected_success, 0.5 * expected_success, 1.0]
  )
  assert eval_vec_env.unwrapped.motion_cmd.motion.clip_weights.tolist() == pytest.approx(
    updated_weights.tolist()
  )
  assert log_dict["eval/failure_rate"] == pytest.approx(1.0 / 3.0)
  assert train_vec_env.unwrapped.reset_calls == 0
  assert train_vec_env.unwrapped.step_calls == 0
  assert train_vec_env.unwrapped.cfg.auto_reset is True


def test_motion_success_evaluator_resets_inactive_done_envs_before_next_step(
  tmp_path,
):
  train_vec_env = _FakeVecEnv(
    _FakeEnv(
      num_envs=2,
      clip_lengths=[1, 3],
      clip_weights=[1.0, 1.0],
      allow_runtime_ops=False,
    )
  )
  eval_vec_env = _FakeVecEnv(
    _FakeEnv(
      num_envs=2,
      clip_lengths=[1, 3],
      clip_weights=[1.0, 1.0],
      allow_runtime_ops=True,
      step_done_env_ids=(0,),
    )
  )
  evaluator = MotionSuccessEvaluator(
    MotionSuccessEvaluatorCfg(eval_metrics_every=200),
    train_vec_env,
    str(tmp_path),
    eval_vec_env_factory=lambda num_envs: eval_vec_env,
  )
  evaluator.metric_fns = cast(dict, {"anchor_height_error": lambda env, command_name: torch.zeros(env.num_envs)})
  policy = nn.Linear(3, 2)

  log_dict = evaluator.run_eval(policy=policy, iteration=200)

  assert "eval/failure_rate" in log_dict
  assert eval_vec_env.unwrapped.reset_calls >= 3
