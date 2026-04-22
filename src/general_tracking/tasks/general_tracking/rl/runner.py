"""Runner with evaluator hook for general tracking."""

from __future__ import annotations

from copy import deepcopy
import os
import time

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from rsl_rl.runners.on_policy_runner import check_nan

from general_tracking.tasks.general_tracking.rl.evaluator import (
  MotionSuccessEvaluator,
  MotionSuccessEvaluatorCfg,
)


class GeneralTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    evaluator_cfg = train_cfg.get("evaluator")
    super().__init__(env, train_cfg, log_dir, device)
    if evaluator_cfg is not None:
      self.evaluator = MotionSuccessEvaluator(
        MotionSuccessEvaluatorCfg(**evaluator_cfg),
        self.env,
        log_dir,
        eval_vec_env_factory=self._build_eval_vec_env,
      )
    else:
      self.evaluator = None

  def _build_eval_vec_env(self, num_envs: int):
    env_cfg = deepcopy(self.env.unwrapped.cfg)
    env_cfg.scene.num_envs = num_envs
    env_cfg.auto_reset = False
    eval_env = ManagerBasedRlEnv(cfg=env_cfg, device=str(self.env.device))
    return RslRlVecEnvWrapper(eval_env, clip_actions=self.env.clip_actions)

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf,
        high=int(self.env.max_episode_length),
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()

    if self.is_distributed:
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    try:
      for it in range(start_it, total_it):
        start = time.time()
        with torch.inference_mode():
          for _ in range(self.cfg["num_steps_per_env"]):
            actions = self.alg.act(obs)
            obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
            if self.cfg.get("check_for_nan", True):
              check_nan(obs, rewards, dones)
            obs = obs.to(self.device)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)
            self.alg.process_env_step(obs, rewards, dones, extras)
            self.logger.process_env_step(rewards, dones, extras, None)

          stop = time.time()
          collect_time = stop - start
          start = stop
          self.alg.compute_returns(obs)

        loss_dict = self.alg.update()
        if self.evaluator is not None and it > 0 and it % self.evaluator.cfg.eval_metrics_every == 0:
          loss_dict.update(self.evaluator.run_eval(policy=self.alg.get_policy(), iteration=it))
          obs = self.env.get_observations().to(self.device)
          self.alg.train_mode()

        stop = time.time()
        learn_time = stop - start
        self.current_learning_iteration = it

        self.logger.log(
          it=it,
          start_it=start_it,
          total_it=total_it,
          collect_time=collect_time,
          learn_time=learn_time,
          loss_dict=loss_dict,
          learning_rate=self.alg.learning_rate,
          action_std=self.alg.get_policy().output_std,
          rnd_weight=None,
        )

        if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
          self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]
    finally:
      if self.evaluator is not None:
        self.evaluator.close()

    if self.logger.writer is not None:
      self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore[arg-type]
      self.logger.stop_logging_writer()
