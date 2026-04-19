"""Multi-clip motion library for NPZ tracking data."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from general_tracking.data.manifest import MotionManifest, load_manifest

_FIELDS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


class MotionLibrary:
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  body_pos_w: torch.Tensor
  body_quat_w: torch.Tensor
  body_lin_vel_w: torch.Tensor
  body_ang_vel_w: torch.Tensor
  clip_starts: torch.Tensor
  clip_lengths: torch.Tensor
  clip_weights: torch.Tensor

  def __init__(
    self,
    manifest: MotionManifest | str | Path,
    *,
    env_control_fps: float,
    device: str | torch.device = "cpu",
  ) -> None:
    if isinstance(manifest, MotionManifest):
      self.manifest = manifest
      self._manifest_path: Path | None = None
    else:
      self._manifest_path = Path(manifest)
      self.manifest = load_manifest(self._manifest_path)

    if abs(self.manifest.control_fps - env_control_fps) > 1e-6:
      raise ValueError(
        "manifest.control_fps does not match env_control_fps: "
        f"{self.manifest.control_fps} != {env_control_fps}"
      )

    self.device = torch.device(device)
    self.control_fps = self.manifest.control_fps

    clip_starts: list[int] = []
    clip_lengths: list[int] = []
    clip_weights: list[float] = []
    field_buffers: dict[str, list[torch.Tensor]] = {name: [] for name in _FIELDS}

    total_frames = 0
    base_dir = self._manifest_path.parent if self._manifest_path is not None else None
    for clip in self.manifest.clips:
      clip_path = Path(clip.path)
      if base_dir is not None and not clip_path.is_absolute():
        clip_path = base_dir / clip_path
      with np.load(clip_path) as data:
        clip_starts.append(total_frames)
        clip_lengths.append(int(data["joint_pos"].shape[0]))
        clip_weights.append(float(clip.weight))
        total_frames += clip_lengths[-1]
        for field in _FIELDS:
          field_buffers[field].append(
            torch.tensor(data[field], dtype=torch.float32, device=self.device)
          )

    self.clip_starts = torch.tensor(clip_starts, dtype=torch.long, device=self.device)
    self.clip_lengths = torch.tensor(
      clip_lengths, dtype=torch.long, device=self.device
    )
    self.clip_weights = torch.tensor(
      clip_weights, dtype=torch.float32, device=self.device
    )

    for field, chunks in field_buffers.items():
      setattr(self, field, torch.cat(chunks, dim=0))

  @property
  def num_clips(self) -> int:
    return int(self.clip_starts.numel())

  @property
  def num_frames(self) -> int:
    return int(self.joint_pos.shape[0])

  def sample_clip_ids(self, n: int) -> torch.Tensor:
    return torch.multinomial(self.clip_weights, n, replacement=True)

  def sample_init_time(
    self,
    clip_ids: torch.Tensor,
    init_start_prob: float = 0.2,
  ) -> torch.Tensor:
    clip_ids = clip_ids.to(device=self.device, dtype=torch.long)
    lengths = self.clip_lengths[clip_ids]
    max_time = (lengths - 1).clamp_min(1)
    uniform_t = (torch.rand_like(max_time, dtype=torch.float32) * max_time).long()
    start_mask = torch.rand_like(max_time, dtype=torch.float32) < init_start_prob
    return torch.where(start_mask, torch.zeros_like(uniform_t), uniform_t)

  def get_state_at(
    self,
    clip_ids: torch.Tensor,
    time_steps: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    clip_ids = clip_ids.to(device=self.device, dtype=torch.long)
    time_steps = time_steps.to(device=self.device, dtype=torch.long)
    flat_indices = self.clip_starts[clip_ids] + self._clamp_time(clip_ids, time_steps)
    return {field: getattr(self, field)[flat_indices] for field in _FIELDS}

  def get_future_states(
    self,
    clip_ids: torch.Tensor,
    time_steps: torch.Tensor,
    offsets: Sequence[int],
  ) -> dict[str, torch.Tensor]:
    clip_ids = clip_ids.to(device=self.device, dtype=torch.long)
    time_steps = time_steps.to(device=self.device, dtype=torch.long)
    offset_tensor = torch.tensor(offsets, dtype=torch.long, device=self.device)
    future_steps = time_steps[:, None] + offset_tensor[None, :]
    clamped = self._clamp_time(clip_ids[:, None], future_steps)
    flat_indices = self.clip_starts[clip_ids][:, None] + clamped
    return {field: getattr(self, field)[flat_indices] for field in _FIELDS}

  def update_weights(self, new_weights: torch.Tensor) -> None:
    if new_weights.shape != self.clip_weights.shape:
      raise ValueError("new_weights shape mismatch")
    self.clip_weights[:] = new_weights.to(
      device=self.device, dtype=self.clip_weights.dtype
    )

  def _clamp_time(
    self,
    clip_ids: torch.Tensor,
    time_steps: torch.Tensor,
  ) -> torch.Tensor:
    max_steps = self.clip_lengths[clip_ids] - 1
    zeros = torch.zeros_like(time_steps)
    return torch.minimum(torch.maximum(time_steps, zeros), max_steps)
