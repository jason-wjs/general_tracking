"""Training entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _set_motion_library_path(path: str | None) -> None:
  if path:
    os.environ["MOTION_LIB_PATH"] = path


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="GeneralTracking-Flat-Unitree-G1")
  parser.add_argument("--num-envs", type=int, default=None)
  parser.add_argument("--max-iterations", type=int, default=None)
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--motion-lib-path", default=os.environ.get("MOTION_LIB_PATH"))
  return parser


def main(argv: list[str] | None = None) -> None:
  args = _build_parser().parse_args(argv)
  _set_motion_library_path(args.motion_lib_path)

  import mjlab  # noqa: F401
  from mjlab.scripts.train import TrainConfig, launch_training

  cfg = TrainConfig.from_task(args.task)
  if args.num_envs is not None:
    cfg.env.scene.num_envs = args.num_envs
  if args.max_iterations is not None:
    cfg.agent.max_iterations = args.max_iterations
  if args.checkpoint is not None:
    ckpt = Path(args.checkpoint)
    cfg.agent.resume = True
    cfg.agent.load_run = ckpt.parent.name
    cfg.agent.load_checkpoint = ckpt.name

  launch_training(task_id=args.task, args=cfg)
