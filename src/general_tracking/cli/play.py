"""Play/evaluation entry point."""

from __future__ import annotations

import argparse
import os


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="GeneralTracking-Flat-Unitree-G1")
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--motion-lib-path", default=os.environ.get("MOTION_LIB_PATH"))
  parser.add_argument("--num-envs", type=int, default=None)
  return parser


def main(argv: list[str] | None = None) -> None:
  args = _build_parser().parse_args(argv)
  if args.motion_lib_path:
    os.environ["MOTION_LIB_PATH"] = args.motion_lib_path

  import mjlab  # noqa: F401
  from mjlab.scripts.play import PlayConfig, run_play

  cfg = PlayConfig(
    checkpoint_file=args.checkpoint,
    num_envs=args.num_envs,
  )
  run_play(args.task, cfg)
