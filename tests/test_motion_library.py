from pathlib import Path

import numpy as np
import pytest
import torch

from general_tracking.data.manifest import (
  MotionClipEntry,
  MotionManifest,
  save_manifest,
)
from general_tracking.data.motion_library import MotionLibrary


def _write_clip(path: Path, length: int, base: float) -> None:
  joint = np.arange(length * 29, dtype=np.float32).reshape(length, 29) + base
  body = np.arange(length * 30 * 3, dtype=np.float32).reshape(length, 30, 3) + base
  quat = np.zeros((length, 30, 4), dtype=np.float32)
  quat[..., 0] = 1.0
  np.savez(
    path,
    fps=np.array([50.0], dtype=np.float64),
    joint_pos=joint,
    joint_vel=joint + 1000.0,
    body_pos_w=body,
    body_quat_w=quat,
    body_lin_vel_w=body + 2000.0,
    body_ang_vel_w=body + 3000.0,
  )


def _write_manifest(tmp_path: Path) -> Path:
  clip_a = tmp_path / "clip_a.npz"
  clip_b = tmp_path / "clip_b.npz"
  _write_clip(clip_a, length=3, base=0.0)
  _write_clip(clip_b, length=5, base=10000.0)
  manifest = MotionManifest(
    version=1,
    control_fps=50.0,
    clips=[
      MotionClipEntry(path=clip_a.name, weight=0.0, num_frames=3),
      MotionClipEntry(path=clip_b.name, weight=1.0, num_frames=5),
    ],
  )
  manifest_path = tmp_path / "motion_manifest.yaml"
  save_manifest(manifest_path, manifest)
  return manifest_path


def test_motion_library_rejects_control_fps_mismatch(tmp_path: Path):
  manifest_path = _write_manifest(tmp_path)
  with pytest.raises(ValueError, match="manifest.control_fps"):
    MotionLibrary(manifest_path, env_control_fps=60.0)


def test_sample_clip_ids_respects_zero_weight(tmp_path: Path):
  manifest_path = _write_manifest(tmp_path)
  library = MotionLibrary(manifest_path, env_control_fps=50.0)
  samples = library.sample_clip_ids(32)
  assert torch.equal(samples, torch.ones_like(samples))


def test_sample_init_time_and_future_clamp(tmp_path: Path):
  manifest_path = _write_manifest(tmp_path)
  library = MotionLibrary(manifest_path, env_control_fps=50.0)
  clip_ids = torch.tensor([0, 1], dtype=torch.long)
  zeros = library.sample_init_time(clip_ids, init_start_prob=1.0)
  assert torch.equal(zeros, torch.zeros_like(clip_ids))

  states = library.get_state_at(
    clip_ids=torch.tensor([0, 1], dtype=torch.long),
    time_steps=torch.tensor([2, 100], dtype=torch.long),
  )
  assert states["joint_pos"].shape == (2, 29)
  assert states["joint_pos"][0, 0].item() == pytest.approx(58.0)
  assert states["joint_pos"][1, 0].item() == pytest.approx(10116.0)

  future = library.get_future_states(
    clip_ids=torch.tensor([0], dtype=torch.long),
    time_steps=torch.tensor([1], dtype=torch.long),
    offsets=[1, 2, 4, 8],
  )
  assert future["joint_pos"].shape == (1, 4, 29)
  assert future["joint_pos"][0, 0, 0].item() == pytest.approx(58.0)
  assert future["joint_pos"][0, 1, 0].item() == pytest.approx(58.0)
  assert future["joint_pos"][0, 2, 0].item() == pytest.approx(58.0)
  assert future["joint_pos"][0, 3, 0].item() == pytest.approx(58.0)

