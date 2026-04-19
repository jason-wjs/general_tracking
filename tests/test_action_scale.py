import pytest
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  G1_ARTICULATION,
)

from general_tracking.robots.g1.action_scale import build_g1_bm_action_scale


def test_build_g1_bm_action_scale_uses_effort_over_stiffness():
  actual = build_g1_bm_action_scale()

  expected: dict[str, float] = {}
  for actuator in G1_ARTICULATION.actuators:
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    assert actuator.effort_limit is not None
    for pattern in actuator.target_names_expr:
      expected[pattern] = actuator.effort_limit / actuator.stiffness

  assert actual == pytest.approx(expected)


def test_build_g1_bm_action_scale_is_four_times_mjlab_tracking_scale():
  actual = build_g1_bm_action_scale()
  for pattern, value in actual.items():
    assert value == pytest.approx(G1_ACTION_SCALE[pattern] * 4.0)
