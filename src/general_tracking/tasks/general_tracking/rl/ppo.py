"""Custom PPO with dual optimizers and L2C2 regularization."""

from __future__ import annotations

from typing import cast

import torch
from rsl_rl.algorithms import PPO
from torch import nn, optim

from general_tracking.learning.ppo import compute_l2c2_loss
from general_tracking.tasks.general_tracking.rl.models import GeneralTrackingActorModel


class GeneralTrackingPPO(PPO):
  def __init__(
    self,
    actor,
    critic,
    storage,
    *,
    actor_learning_rate: float = 2e-5,
    critic_learning_rate: float = 1e-4,
    actor_betas: tuple[float, float] = (0.95, 0.99),
    critic_betas: tuple[float, float] = (0.95, 0.99),
    lambda_l2c2: float = 1.0,
    l2c2_obs_pairs: dict[str, str] | None = None,
    **kwargs,
  ) -> None:
    kwargs.pop("learning_rate", None)
    super().__init__(
      actor,
      critic,
      storage,
      learning_rate=actor_learning_rate,
      **kwargs,
    )
    self.actor_optimizer = optim.Adam(
      self.actor.parameters(),
      lr=actor_learning_rate,
      betas=actor_betas,
    )
    self.critic_optimizer = optim.Adam(
      self.critic.parameters(),
      lr=critic_learning_rate,
      betas=critic_betas,
    )
    self.optimizer = self.actor_optimizer
    self.learning_rate = actor_learning_rate
    self.actor_learning_rate = actor_learning_rate
    self.critic_learning_rate = critic_learning_rate
    self.lambda_l2c2 = lambda_l2c2
    self.l2c2_obs_pairs = l2c2_obs_pairs or {}

  def update(self) -> dict[str, float]:
    if self.rnd or self.symmetry:
      raise NotImplementedError("GeneralTrackingPPO does not support RND or symmetry")

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_l2c2_loss = 0.0

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches,
        self.num_learning_epochs,
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches,
        self.num_learning_epochs,
      )

    for batch in generator:
      assert batch.observations is not None
      assert batch.actions is not None
      assert batch.old_actions_log_prob is not None
      assert batch.advantages is not None
      assert batch.values is not None
      assert batch.returns is not None
      actor_model = cast(GeneralTrackingActorModel, self.actor)

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore[operator]

      actor_model(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = actor_model.get_output_log_prob(batch.actions)
      values = self.critic(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[1],
      )
      entropy = actor_model.output_entropy
      mu_noisy = actor_model.output_mean
      mu_clean = actor_model.forward_clean(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
      )

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore[arg-type]
      surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore[arg-type]
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore[arg-type]
        ratio,
        1.0 - self.clip_param,
        1.0 + self.clip_param,
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)  # type: ignore[operator]
        value_losses = (values - batch.returns).pow(2)  # type: ignore[operator]
        value_losses_clipped = (value_clipped - batch.returns).pow(2)  # type: ignore[operator]
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()  # type: ignore[operator]

      obs_pairs = [
        (batch.observations[noisy_key], batch.observations[clean_key])
        for noisy_key, clean_key in self.l2c2_obs_pairs.items()
      ]
      l2c2_loss, _ = compute_l2c2_loss(
        mu_noisy=mu_noisy,
        mu_clean=mu_clean,
        obs_pairs=obs_pairs,
        lambda_coef=self.lambda_l2c2,
      )

      actor_loss = surrogate_loss - self.entropy_coef * entropy.mean() + l2c2_loss
      critic_loss = self.value_loss_coef * value_loss

      self.actor_optimizer.zero_grad()
      self.critic_optimizer.zero_grad()
      actor_loss.backward()
      critic_loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(actor_model.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.actor_optimizer.step()
      self.critic_optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_l2c2_loss += l2c2_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    return {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      "l2c2": mean_l2c2_loss / num_updates,
    }

  def save(self) -> dict:
    return {
      "actor_state_dict": self.actor.state_dict(),
      "critic_state_dict": self.critic.state_dict(),
      "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
      "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
    }

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    if load_cfg is None:
      load_cfg = {
        "actor": True,
        "critic": True,
        "optimizer": True,
        "iteration": True,
      }
    if load_cfg.get("actor"):
      self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
    if load_cfg.get("critic"):
      self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
    if load_cfg.get("optimizer"):
      actor_opt_state = loaded_dict.get("actor_optimizer_state_dict")
      critic_opt_state = loaded_dict.get("critic_optimizer_state_dict")
      if actor_opt_state is not None:
        self.actor_optimizer.load_state_dict(actor_opt_state)
      if critic_opt_state is not None:
        self.critic_optimizer.load_state_dict(critic_opt_state)
      if actor_opt_state is None and critic_opt_state is None and "optimizer_state_dict" in loaded_dict:
        self.actor_optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
    return load_cfg.get("iteration", False)
