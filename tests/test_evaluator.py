import pytest
import torch

from general_tracking.tasks.general_tracking.rl.evaluator import (
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
