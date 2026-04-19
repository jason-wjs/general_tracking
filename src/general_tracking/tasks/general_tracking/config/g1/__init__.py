"""Register G1 general tracking tasks."""

from mjlab.tasks.registry import register_mjlab_task

from general_tracking.tasks.general_tracking.config.g1.env_cfgs import (
  unitree_g1_general_tracking_env_cfg,
)
from general_tracking.tasks.general_tracking.config.g1.rl_cfg import (
  unitree_g1_general_tracking_runner_cfg,
)
from general_tracking.tasks.general_tracking.rl.runner import (
  GeneralTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="GeneralTracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_general_tracking_env_cfg(play=False),
  play_env_cfg=unitree_g1_general_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_general_tracking_runner_cfg(),
  runner_cls=GeneralTrackingOnPolicyRunner,
)
