"""RL configuration for G1 general tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mjlab.rl.config import (
  RslRlBaseRunnerCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from general_tracking.tasks.general_tracking.rl.evaluator import (
  MotionSuccessEvaluatorCfg,
)


@dataclass
class GeneralTrackingActorCfg(RslRlModelCfg):
  class_name: str = "general_tracking.tasks.general_tracking.rl.models:GeneralTrackingActorModel"
  clean_obs_set: str = "actor_clean"


@dataclass
class GeneralTrackingPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "general_tracking.tasks.general_tracking.rl.ppo:GeneralTrackingPPO"
  actor_learning_rate: float = 2e-5
  critic_learning_rate: float = 1e-4
  actor_betas: tuple[float, float] = (0.95, 0.99)
  critic_betas: tuple[float, float] = (0.95, 0.99)
  lambda_l2c2: float = 1.0
  l2c2_obs_pairs: dict[str, str] = field(
    default_factory=lambda: {
      "noisy_reduced_coords_obs": "clean_reduced_coords_obs",
      "noisy_reduced_coords_target_poses": "clean_reduced_coords_target_poses",
    }
  )


@dataclass
class GeneralTrackingRunnerCfg(RslRlOnPolicyRunnerCfg):
  actor: RslRlModelCfg = field(default_factory=GeneralTrackingActorCfg)
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=GeneralTrackingPPOAlgorithmCfg)
  evaluator: MotionSuccessEvaluatorCfg | None = None


def unitree_g1_general_tracking_runner_cfg() -> RslRlBaseRunnerCfg:
  return GeneralTrackingRunnerCfg(
    obs_groups={
      "actor": (
        "noisy_reduced_coords_obs",
        "noisy_reduced_coords_target_poses",
        "historical_previous_processed_actions",
      ),
      "actor_clean": (
        "clean_reduced_coords_obs",
        "clean_reduced_coords_target_poses",
        "historical_previous_processed_actions",
      ),
      "critic": (
        "max_coords_obs",
        "mimic_max_coords_target_poses",
        "historical_previous_processed_actions",
      ),
    },
    actor=GeneralTrackingActorCfg(
      hidden_dims=(1024, 1024, 1024, 1024, 1024, 1024),
      activation="relu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": math.exp(-2.9),
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(1024, 1024, 1024, 1024),
      activation="relu",
      obs_normalization=True,
    ),
    algorithm=GeneralTrackingPPOAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0,
      num_learning_epochs=2,
      num_mini_batches=4,
      learning_rate=2e-5,
      schedule="fixed",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.0,
      max_grad_norm=50.0,
      normalize_advantage_per_mini_batch=False,
      actor_learning_rate=2e-5,
      critic_learning_rate=1e-4,
      lambda_l2c2=1.0,
      l2c2_obs_pairs={
        "noisy_reduced_coords_obs": "clean_reduced_coords_obs",
        "noisy_reduced_coords_target_poses": "clean_reduced_coords_target_poses",
      },
    ),
    evaluator=MotionSuccessEvaluatorCfg(
      eval_metrics_every=200,
      max_eval_steps=600,
      success_discount=0.999,
      failure_weight=1.0,
      evaluation_components={"anchor_height_error": {"threshold": 0.25}},
    ),
    experiment_name="g1_general_tracking",
    wandb_project="whole_body_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
