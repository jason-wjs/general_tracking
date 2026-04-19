"""BeyondMimic action scaling for G1."""

from __future__ import annotations

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ARTICULATION


def build_g1_bm_action_scale() -> dict[str, float]:
  """Build BeyondMimic-style joint scale from mjlab's G1 actuator definitions."""
  action_scale: dict[str, float] = {}
  for actuator in G1_ARTICULATION.actuators:
    if not isinstance(actuator, BuiltinPositionActuatorCfg):
      raise TypeError(f"Unsupported actuator cfg type: {type(actuator)!r}")
    effort_limit = actuator.effort_limit
    if effort_limit is None:
      raise ValueError("G1 actuator is missing effort_limit")
    for pattern in actuator.target_names_expr:
      action_scale[pattern] = float(effort_limit / actuator.stiffness)
  return action_scale
