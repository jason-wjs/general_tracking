"""Build a motion manifest from a directory of NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from general_tracking.data.manifest import (
  MotionClipEntry,
  MotionManifest,
  save_manifest,
)


def build_manifest(
  input_dir: Path,
  *,
  output: Path,
  control_fps: float,
) -> MotionManifest:
  clips: list[MotionClipEntry] = []
  for npz_path in sorted(input_dir.glob("*.npz")):
    with np.load(npz_path) as data:
      clips.append(
        MotionClipEntry(
          path=npz_path.name,
          weight=1.0,
          num_frames=int(data["joint_pos"].shape[0]),
        )
      )
  manifest = MotionManifest(version=1, control_fps=control_fps, clips=clips)
  save_manifest(output, manifest)
  return manifest


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-dir", type=Path, required=True)
  parser.add_argument("--output", type=Path, default=None)
  parser.add_argument("--control-fps", type=float, default=50.0)
  return parser


def main(argv: list[str] | None = None) -> None:
  args = _build_parser().parse_args(argv)
  output = args.output or (args.input_dir / "motion_manifest.yaml")
  build_manifest(args.input_dir, output=output, control_fps=args.control_fps)
