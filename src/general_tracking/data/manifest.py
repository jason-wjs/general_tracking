"""Motion manifest helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class MotionClipEntry:
  path: str
  weight: float
  num_frames: int


@dataclass(slots=True)
class MotionManifest:
  version: int
  control_fps: float
  clips: list[MotionClipEntry]


def load_manifest(path: str | Path) -> MotionManifest:
  raw = yaml.safe_load(Path(path).read_text())
  clips = [MotionClipEntry(**item) for item in raw["clips"]]
  return MotionManifest(
    version=int(raw["version"]),
    control_fps=float(raw["control_fps"]),
    clips=clips,
  )


def save_manifest(path: str | Path, manifest: MotionManifest) -> None:
  payload = {
    "version": manifest.version,
    "control_fps": manifest.control_fps,
    "clips": [asdict(clip) for clip in manifest.clips],
  }
  Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))

