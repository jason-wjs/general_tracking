import pytest
from rsl_rl.algorithms import PPO
from torch import nn

from general_tracking.tasks.general_tracking.rl.ppo import GeneralTrackingPPO


class _DummyStorage:
  pass


def test_general_tracking_ppo_ignores_legacy_learning_rate_kwarg(monkeypatch):
  captured: dict[str, float] = {}

  def fake_init(self, actor, critic, storage, **kwargs):
    captured["learning_rate"] = kwargs["learning_rate"]
    self.actor = actor
    self.critic = critic
    self.storage = storage
    self.rnd = None
    self.symmetry = None
    self.num_mini_batches = kwargs["num_mini_batches"]
    self.num_learning_epochs = kwargs["num_learning_epochs"]
    self.clip_param = kwargs["clip_param"]
    self.value_loss_coef = kwargs["value_loss_coef"]
    self.entropy_coef = kwargs["entropy_coef"]
    self.max_grad_norm = kwargs["max_grad_norm"]
    self.use_clipped_value_loss = kwargs["use_clipped_value_loss"]
    self.normalize_advantage_per_mini_batch = kwargs["normalize_advantage_per_mini_batch"]
    self.is_multi_gpu = False
    self.learning_rate = kwargs["learning_rate"]

  monkeypatch.setattr(PPO, "__init__", fake_init)

  actor = nn.Linear(3, 2)
  actor.is_recurrent = False  # type: ignore[attr-defined]
  critic = nn.Linear(3, 1)
  critic.is_recurrent = False  # type: ignore[attr-defined]

  algo = GeneralTrackingPPO(
    actor,
    critic,
    _DummyStorage(),
    num_learning_epochs=2,
    num_mini_batches=4,
    clip_param=0.2,
    value_loss_coef=1.0,
    entropy_coef=0.0,
    max_grad_norm=50.0,
    use_clipped_value_loss=True,
    normalize_advantage_per_mini_batch=False,
    learning_rate=1e-3,
    actor_learning_rate=2e-5,
    critic_learning_rate=1e-4,
  )

  assert captured["learning_rate"] == pytest.approx(2e-5)
  assert algo.actor_optimizer.param_groups[0]["lr"] == pytest.approx(2e-5)
  assert algo.critic_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
