"""Evaluation helpers for motion reweighting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from general_tracking.tasks.general_tracking.mdp import metrics


@dataclass(slots=True)
class MotionSuccessEvaluatorCfg:
  eval_metrics_every: int = 200
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

  def __init__(self, cfg: MotionSuccessEvaluatorCfg, vec_env, log_dir: str | None):
    self.cfg = cfg
    self.vec_env = vec_env
    self.log_dir = Path(log_dir) if log_dir is not None else None
    self.metric_fns = {
      "anchor_ori": metrics.anchor_ori_metric,
      "relative_body_pos": metrics.relative_body_pos_metric,
      "anchor_height_error": metrics.anchor_height_error,
      "gt_error": metrics.gt_error,
      "gr_error": metrics.gr_error,
      "max_joint_error": metrics.max_joint_error,
    }

  def run_eval(self, *, policy, iteration: int) -> dict[str, float]:
    env = self.vec_env.unwrapped
    motion_cmd = env.command_manager.get_term("motion")
    num_envs = env.num_envs
    num_clips = motion_cmd.motion.num_clips
    clip_ids_all = torch.arange(num_clips, device=env.device, dtype=torch.long)

    original_auto_reset = env.cfg.auto_reset
    policy_was_training = policy.training
    env.cfg.auto_reset = False
    policy.eval()

    failed_all = torch.zeros(num_clips, dtype=torch.bool, device=env.device)
    metric_sums = {
      name: torch.zeros(num_clips, dtype=torch.float32, device=env.device)
      for name in self.metric_fns
    }
    metric_counts = torch.zeros(num_clips, dtype=torch.float32, device=env.device)

    try:
      for start in range(0, num_clips, num_envs):
        batch_clip_ids = clip_ids_all[start : start + num_envs]
        batch_size = int(batch_clip_ids.numel())
        env_ids = torch.arange(batch_size, device=env.device, dtype=torch.long)
        clip_lengths = motion_cmd.motion.clip_lengths[batch_clip_ids]
        rollout_steps = torch.zeros(batch_size, dtype=torch.long, device=env.device)
        active = torch.ones(batch_size, dtype=torch.bool, device=env.device)

        motion_cmd.reset_to_clip_frame(env_ids, batch_clip_ids, frame=0)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.command_manager.compute(dt=0.0)
        env.sim.sense()
        env.observation_manager.compute(update_history=True)
        obs = self.vec_env.get_observations().to(policy.device)

        while torch.any(active):
          reached_end = motion_cmd.time_steps[env_ids] >= (clip_lengths - 1)
          reached_budget = rollout_steps >= self.cfg.max_eval_steps
          completed_success = active & (reached_end | reached_budget)
          if torch.any(completed_success):
            completed_env_ids = env_ids[completed_success]
            env.reset(env_ids=completed_env_ids)
            active[completed_success] = False
            obs = self.vec_env.get_observations().to(policy.device)
            if not torch.any(active):
              break

          with torch.inference_mode():
            actions = policy(obs)
          obs, _, dones, _ = self.vec_env.step(actions.to(self.vec_env.device))
          obs = obs.to(policy.device)

          component_values = {
            name: fn(env, "motion")[:batch_size]
            for name, fn in self.metric_fns.items()
          }
          current_active = active.clone()
          for name, value in component_values.items():
            metric_sums[name][batch_clip_ids[current_active]] += value[current_active]
          metric_counts[batch_clip_ids[current_active]] += 1.0

          failed_step, _ = compute_failed_mask(component_values, self.cfg.evaluation_components)
          done_step = dones[:batch_size].to(dtype=torch.bool)
          completed = active & (failed_step | done_step)
          if torch.any(completed):
            failed_all[batch_clip_ids[completed]] = True
            completed_env_ids = env_ids[completed]
            env.reset(env_ids=completed_env_ids)
            active[completed] = False
            obs = self.vec_env.get_observations().to(policy.device)

          rollout_steps[active] += 1

      new_weights = apply_motion_weight_update(
        clip_weights=motion_cmd.motion.clip_weights,
        failed_mask=failed_all,
        success_discount=self.cfg.success_discount,
        eval_interval=self.cfg.eval_metrics_every,
        failure_weight=self.cfg.failure_weight,
      )
      motion_cmd.motion.update_weights(new_weights)
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
      env.reset()
      if policy_was_training:
        policy.train()

  def _write_failed_motion_report(self, failed_mask: torch.Tensor, iteration: int) -> None:
    if self.log_dir is None:
      return
    failed_ids = failed_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    out_dir = self.log_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"failed_motions_epoch_{iteration}.txt").write_text(
      "\n".join(str(idx) for idx in failed_ids)
    )
