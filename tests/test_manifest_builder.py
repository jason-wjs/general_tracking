from pathlib import Path

import numpy as np

from general_tracking.data.cli import build_manifest
from general_tracking.data.manifest import load_manifest


def _write_clip(path: Path, frames: int) -> None:
  joint = np.zeros((frames, 29), dtype=np.float32)
  body = np.zeros((frames, 30, 3), dtype=np.float32)
  quat = np.zeros((frames, 30, 4), dtype=np.float32)
  quat[..., 0] = 1.0
  np.savez(
    path,
    fps=np.array([50.0], dtype=np.float64),
    joint_pos=joint,
    joint_vel=joint,
    body_pos_w=body,
    body_quat_w=quat,
    body_lin_vel_w=body,
    body_ang_vel_w=body,
  )


def test_build_manifest_writes_relative_paths_and_frame_counts(tmp_path: Path):
  _write_clip(tmp_path / "a.npz", 3)
  _write_clip(tmp_path / "b.npz", 5)
  out_path = tmp_path / "motion_manifest.yaml"

  build_manifest.main(
    argv=[
      "--input-dir",
      str(tmp_path),
      "--output",
      str(out_path),
      "--control-fps",
      "50.0",
    ]
  )

  manifest = load_manifest(out_path)
  assert manifest.version == 1
  assert manifest.control_fps == 50.0
  assert [clip.path for clip in manifest.clips] == ["a.npz", "b.npz"]
  assert [clip.num_frames for clip in manifest.clips] == [3, 5]
  assert [clip.weight for clip in manifest.clips] == [1.0, 1.0]

