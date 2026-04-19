import pytest
import torch

from general_tracking.learning.ppo.l2c2 import compute_l2c2_loss


def test_compute_l2c2_loss_matches_proto_formula():
  mu_noisy = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
  mu_clean = torch.tensor([[0.0, 1.0], [1.0, 2.0]])
  obs_pairs = [
    (
      torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
      torch.tensor([[0.0, 2.0], [2.0, 2.0]]),
    ),
    (
      torch.tensor([[0.0], [2.0]]),
      torch.tensor([[0.0], [1.0]]),
    ),
  ]

  loss, stats = compute_l2c2_loss(
    mu_noisy=mu_noisy,
    mu_clean=mu_clean,
    obs_pairs=obs_pairs,
    lambda_coef=1.0,
  )

  input_ss = (
    (obs_pairs[0][0] - obs_pairs[0][1]).pow(2).sum()
    + (obs_pairs[1][0] - obs_pairs[1][1]).pow(2).sum()
  )
  input_n = obs_pairs[0][0].numel() + obs_pairs[1][0].numel()
  input_dist = input_ss / input_n
  output_dist = (mu_noisy - mu_clean).pow(2).mean()
  expected = output_dist / (input_dist + 1e-8)

  assert loss.item() == pytest.approx(expected.item())
  assert stats["input_dist"].item() == pytest.approx(input_dist.item())
  assert stats["output_dist"].item() == pytest.approx(output_dist.item())


def test_compute_l2c2_loss_zero_lambda_matches_baseline():
  mu_noisy = torch.randn(4, 3)
  mu_clean = torch.randn(4, 3)
  obs_pairs = [(torch.randn(4, 5), torch.randn(4, 5))]

  loss, stats = compute_l2c2_loss(
    mu_noisy=mu_noisy,
    mu_clean=mu_clean,
    obs_pairs=obs_pairs,
    lambda_coef=0.0,
  )

  assert loss.item() == pytest.approx(0.0)
  assert stats["weighted_loss"].item() == pytest.approx(0.0)


def test_compute_l2c2_loss_detaches_input_distance_from_backward():
  mu_noisy = torch.tensor([[1.0, 0.0]], requires_grad=True)
  mu_clean = torch.tensor([[0.0, 0.0]], requires_grad=True)
  noisy_obs = torch.tensor([[1.0, 2.0]], requires_grad=True)
  clean_obs = torch.tensor([[0.5, 1.5]], requires_grad=True)

  loss, _ = compute_l2c2_loss(
    mu_noisy=mu_noisy,
    mu_clean=mu_clean,
    obs_pairs=[(noisy_obs, clean_obs)],
    lambda_coef=1.0,
  )
  loss.backward()

  assert mu_noisy.grad is not None
  assert mu_clean.grad is not None
  assert noisy_obs.grad is None
  assert clean_obs.grad is None
