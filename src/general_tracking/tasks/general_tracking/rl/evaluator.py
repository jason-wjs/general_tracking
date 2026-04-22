"""Evaluation helpers for motion reweighting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from general_tracking.tasks.general_tracking.mdp import metrics


@dataclass(slots=True)
class MotionSuccessEvaluatorCfg:
  eval_metrics_every: int = 200
  eval_num_envs: int = 64
  max_eval_steps: int = 600
  success_discount: float = 0.999
  failure_weight: float = 1.0
  evaluation_components: dict[str, dict[str, Any]] = field(default_factory=dict)


def compute_failed_mask(
  component_values: dict[str, torch.Tensor],
  component_cfg: dict[str, dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  """Compute per-clip failure mask using only components with thresholds."""
  first_value = next(iter(component_values.values()), None)
  if first_value is None:
    raise ValueError("component_values must not be empty")

  failed = torch.zeros_like(first_value, dtype=torch.bool)
  component_failures: dict[str, torch.Tensor] = {}
  for name, value in component_values.items():
    threshold = component_cfg.get(name, {}).get("threshold")
    if threshold is None:
      continue
    fail_above = component_cfg.get(name, {}).get("fail_above", True)
    failure = value > threshold if fail_above else value < threshold
    component_failures[name] = failure
    failed = failed | failure
  return failed, component_failures


def apply_motion_weight_update(
  *,
  clip_weights: torch.Tensor,
  failed_mask: torch.Tensor,
  success_discount: float,
  eval_interval: int,
  failure_weight: float = 1.0,
) -> torch.Tensor:
  """Apply ProtoMotions-style success/failure reweighting."""
  updated = clip_weights.clone()
  success_scale = success_discount**eval_interval
  updated[~failed_mask] = updated[~failed_mask] * success_scale
  updated[failed_mask] = failure_weight
  return updated


class MotionSuccessEvaluator:
  """Periodic rollout evaluator with ProtoMotions-style motion reweighting."""

  def __init__(
    self,
    cfg: MotionSuccessEvaluatorCfg,
    vec_env,
    log_dir: str | None,
    eval_vec_env_factory: Callable[[int], Any] | None = None,
  ):
    self.cfg = cfg
    self.vec_env = vec_env
    self.log_dir = Path(log_dir) if log_dir is not None else None
    self.eval_vec_env_factory = eval_vec_env_factory
    self._eval_vec_env = None
    self.metric_fns = {
      "anchor_ori": metrics.anchor_ori_metric,
      "relative_body_pos": metrics.relative_body_pos_metric,
      "anchor_height_error": metrics.anchor_height_error,
      "gt_error": metrics.gt_error,
      "gr_error": metrics.gr_error,
      "max_joint_error": metrics.max_joint_error,
    }

  def _policy_device(self, policy) -> torch.device:
    device = getattr(policy, "device", None)
    if device is not None:
      return torch.device(device)
    try:
      return next(policy.parameters()).device
    except StopIteration:
      return self.vec_env.device

  def _build_eval_vec_env(self, num_clips: int):
    if self._eval_vec_env is not None:
      return self._eval_vec_env

    eval_num_envs = min(self.cfg.eval_num_envs, num_clips)
    if self.eval_vec_env_factory is None:
      self._eval_vec_env = self.vec_env
    else:
      self._eval_vec_env = self.eval_vec_env_factory(eval_num_envs)
    return self._eval_vec_env

  @staticmethod
  def _get_motion_cmd(vec_env):
    return vec_env.unwrapped.command_manager.get_term("motion")

  @staticmethod
  def _reset_env(env, env_ids: torch.Tensor | None = None) -> None:
    with torch.inference_mode():
      env.reset(env_ids=env_ids)

  def _sync_weights_from_train_env(self, eval_motion_cmd) -> None:
    train_weights = self._get_motion_cmd(self.vec_env).motion.clip_weights
    eval_motion_cmd.motion.update_weights(train_weights.clone())

  def _sync_weights_to_train_env(self, new_weights: torch.Tensor) -> None:
    self._get_motion_cmd(self.vec_env).motion.update_weights(new_weights.clone())

  @staticmethod
  def _build_eval_batch(
    clip_ids_all: torch.Tensor,
    start: int,
    batch_capacity: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    batch_clip_ids = clip_ids_all[start : start + batch_capacity]
    tracked_mask = torch.zeros(batch_capacity, dtype=torch.bool, device=clip_ids_all.device)
    tracked_mask[: batch_clip_ids.numel()] = True
    if batch_clip_ids.numel() == batch_capacity:
      return batch_clip_ids, tracked_mask

    padded_clip_ids = torch.empty(batch_capacity, dtype=torch.long, device=clip_ids_all.device)
    padded_clip_ids[: batch_clip_ids.numel()] = batch_clip_ids
    padded_clip_ids[batch_clip_ids.numel() :] = clip_ids_all[0]
    return padded_clip_ids, tracked_mask

  def run_eval(self, *, policy, iteration: int) -> dict[str, float]:
    train_motion_cmd = self._get_motion_cmd(self.vec_env)
    num_clips = train_motion_cmd.motion.num_clips
    eval_vec_env = self._build_eval_vec_env(num_clips)
    env = eval_vec_env.unwrapped
    motion_cmd = self._get_motion_cmd(eval_vec_env)
    num_envs = env.num_envs
    clip_ids_all = torch.arange(num_clips, device=env.device, dtype=torch.long)

    policy_was_training = policy.training
    policy_device = self._policy_device(policy)
    original_auto_reset = env.cfg.auto_reset
    env.cfg.auto_reset = False
    self._sync_weights_from_train_env(motion_cmd)
    policy.eval()

    failed_all = torch.zeros(num_clips, dtype=torch.bool, device=env.device)
    metric_sums = {
      name: torch.zeros(num_clips, dtype=torch.float32, device=env.device)
      for name in self.metric_fns
    }
    metric_counts = torch.zeros(num_clips, dtype=torch.float32, device=env.device)

    try:
      for start in range(0, num_clips, num_envs):
        batch_clip_ids, tracked_mask = self._build_eval_batch(clip_ids_all, start, num_envs)
        env_ids = torch.arange(num_envs, device=env.device, dtype=torch.long)
        clip_lengths = motion_cmd.motion.clip_lengths[batch_clip_ids]
        rollout_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        active = torch.ones(num_envs, dtype=torch.bool, device=env.device)

        motion_cmd.reset_to_clip_frame(env_ids, batch_clip_ids, frame=0)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.command_manager.compute(dt=0.0)
        env.sim.sense()
        env.observation_manager.compute(update_history=True)
        obs = eval_vec_env.get_observations().to(policy_device)

        while torch.any(active):
          reached_end = motion_cmd.time_steps[env_ids] >= (clip_lengths - 1)
          reached_budget = rollout_steps >= self.cfg.max_eval_steps
          completed_success = active & (reached_end | reached_budget)
          if torch.any(completed_success):
            completed_env_ids = env_ids[completed_success]
            self._reset_env(env, env_ids=completed_env_ids)
            active[completed_success] = False
            obs = eval_vec_env.get_observations().to(policy_device)
            if not torch.any(active):
              break

          inactive_env_ids = env_ids[~active]
          if inactive_env_ids.numel() > 0:
            self._reset_env(env, env_ids=inactive_env_ids)
            obs = eval_vec_env.get_observations().to(policy_device)

          with torch.inference_mode():
            actions = policy(obs)
          obs, _, dones, _ = eval_vec_env.step(actions.to(eval_vec_env.device))
          obs = obs.to(policy_device)

          component_values = {name: fn(env, "motion") for name, fn in self.metric_fns.items()}
          current_active = active & tracked_mask
          for name, value in component_values.items():
            metric_sums[name][batch_clip_ids[current_active]] += value[current_active]
          metric_counts[batch_clip_ids[current_active]] += 1.0

          failed_step, _ = compute_failed_mask(component_values, self.cfg.evaluation_components)
          done_step = dones.to(dtype=torch.bool)
          completed = active & (failed_step | done_step)
          tracked_completed = completed & tracked_mask
          if torch.any(tracked_completed):
            failed_all[batch_clip_ids[tracked_completed]] = True
          if torch.any(completed):
            completed_env_ids = env_ids[completed]
            self._reset_env(env, env_ids=completed_env_ids)
            active[completed] = False
            obs = eval_vec_env.get_observations().to(policy_device)

          rollout_steps[active] += 1

      new_weights = apply_motion_weight_update(
        clip_weights=train_motion_cmd.motion.clip_weights,
        failed_mask=failed_all,
        success_discount=self.cfg.success_discount,
        eval_interval=self.cfg.eval_metrics_every,
        failure_weight=self.cfg.failure_weight,
      )
      motion_cmd.motion.update_weights(new_weights)
      self._sync_weights_to_train_env(new_weights)
      self._write_failed_motion_report(failed_all, iteration)

      log_dict: dict[str, float] = {}
      denom = torch.clamp(metric_counts, min=1.0)
      for name, sums in metric_sums.items():
        log_dict[f"eval/{name}"] = float((sums / denom).mean().item())
      log_dict["eval/failure_rate"] = float(failed_all.float().mean().item())
      log_dict["eval/mean_motion_weight"] = float(motion_cmd.motion.clip_weights.mean().item())
      return log_dict
    finally:
      env.cfg.auto_reset = original_auto_reset
      self._reset_env(env)
      if policy_was_training:
        policy.train()

  def close(self) -> None:
    if self._eval_vec_env is None or self._eval_vec_env is self.vec_env:
      return
    self._eval_vec_env.close()
    self._eval_vec_env = None

  def _write_failed_motion_report(self, failed_mask: torch.Tensor, iteration: int) -> None:
    if self.log_dir is None:
      return
    failed_ids = failed_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    out_dir = self.log_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"failed_motions_epoch_{iteration}.txt").write_text(
      "\n".join(str(idx) for idx in failed_ids)
    )
