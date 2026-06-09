"""TorchSource: batched torch-tensor representation for use within the ML pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from tcfuse.data.sources.source import SourceKind


@dataclass
class TorchSource:
    """A batched, torch-tensor observation source for use inside the ML pipeline.

    Carries B stacked samples (one per window in the batch). Time is stored
    separately as a normalised 2-vector so that coordinates remain purely
    spatial. There is no metadata or instrument descriptor on this class;
    those are consumed upstream in preprocessing/evaluation.

    Args:
        kind: Dimensionality class of this source.
        values: Observed measurements, always batched.
            - SCALAR:  (B, C)
            - PROFILE: (B, L, C)  — L levels, C channels
            - FIELD:   (B, H, W, C)
        coords: Spatial coordinates paired with each measurement (no time channel).
            - SCALAR:  (B, 2)             — [lat, lon]
            - PROFILE: (B, L, 3)          — [lat, lon, alt] per level
            - FIELD:   (B, H, W, 2)       — [lat, lon] per pixel
        source_name: Human-readable source identifier, e.g. "pmw_amsr2".
        channels: Names of each channel in the last axis of ``values``.
            Length must equal ``values.shape[-1]``.
        mask: Per-value availability mask, True = finite/available.
            Shape matches ``values``:
            - SCALAR:  (B, C)
            - PROFILE: (B, L, C)
            - FIELD:   (B, H, W, C)
        time: Normalised temporal encoding for each sample.
            Shape (B, 2) — ``[day_of_year / 366.0, minute_of_day / 1440.0]``.
            NaN-filled for missing slots (samples that had no snapshot for this key).
    """

    kind: SourceKind
    values: Tensor
    coords: Tensor
    source_name: str
    channels: list[str]
    mask: Tensor
    time: Tensor  # (B, 2)

    def __post_init__(self) -> None:
        """Cast mask to bool and validate shapes."""
        self.mask = self.mask.to(dtype=torch.bool)
        self._validate()

    def _validate(self) -> None:
        """Check shape consistency between values, coords, mask, and time."""
        v, c = self.values, self.coords
        B = v.shape[0]

        if self.kind is SourceKind.SCALAR:
            # SCALAR values: (B, C); coords: (B, 2)
            if v.ndim != 2:
                raise ValueError(f"SCALAR values must be 2-D (B, C), got {v.shape}")
            expected_coords = (B, 2)
            if c.shape != expected_coords:
                raise ValueError(f"SCALAR coords must be {expected_coords}, got {c.shape}")

        elif self.kind is SourceKind.PROFILE:
            # PROFILE values: (B, L, C); coords: (B, L, 3)
            if v.ndim != 3:
                raise ValueError(f"PROFILE values must be 3-D (B, L, C), got {v.shape}")
            expected_coords = (B, v.shape[1], 3)
            if c.shape != expected_coords:
                raise ValueError(f"PROFILE coords must be {expected_coords}, got {c.shape}")

        elif self.kind is SourceKind.FIELD:
            # FIELD values: (B, H, W, C); coords: (B, H, W, 2)
            if v.ndim != 4:
                raise ValueError(f"FIELD values must be 4-D (B, H, W, C), got {v.shape}")
            expected_coords = (B, v.shape[1], v.shape[2], 2)
            if c.shape != expected_coords:
                raise ValueError(f"FIELD coords must be {expected_coords}, got {c.shape}")

        if self.mask.shape != v.shape:
            raise ValueError(f"mask shape {self.mask.shape} must match values shape {v.shape}")

        # time must be (B, 2) — one normalised [day/366, minute/1440] per sample.
        expected_time = (B, 2)
        if self.time.shape != expected_time:
            raise ValueError(f"time must be {expected_time}, got {self.time.shape}")

    @property
    def batch_size(self) -> int:
        """Number of samples in this batched source."""
        return int(self.values.shape[0])

    @property
    def shape(self) -> tuple[int, ...]:
        """Spatial shape of this source (excluding batch and channel dims).

        Returns:
            - SCALAR:  ``()``
            - PROFILE: ``(L,)``
            - FIELD:   ``(H, W)``
        """
        if self.kind is SourceKind.SCALAR:
            return ()
        elif self.kind is SourceKind.PROFILE:
            # values: (B, L, C) → spatial shape is (L,)
            return (self.values.shape[1],)
        else:  # FIELD
            # values: (B, H, W, C) → spatial shape is (H, W)
            return (self.values.shape[1], self.values.shape[2])

    @property
    def n_tokens(self) -> int:
        """Number of (value, coord) pairs per sample (flattened spatial dims)."""
        return max(1, math.prod(self.shape))

    def to(self, device: torch.device | str) -> TorchSource:
        """Move all tensors to ``device``, returning a new TorchSource."""
        return TorchSource(
            kind=self.kind,
            values=self.values.to(device),
            coords=self.coords.to(device),
            source_name=self.source_name,
            channels=self.channels,
            mask=self.mask.to(device),
            time=self.time.to(device),
        )
