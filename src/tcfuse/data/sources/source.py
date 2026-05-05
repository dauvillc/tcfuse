"""Source abstraction: a single (value, coordinate) collection from one instrument."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, auto

import h5py
import numpy as np
import torch
from torch import Tensor


class SourceKind(Enum):
    """Dimensionality class of a source."""

    SCALAR = auto()  # 0D: single point measurement
    PROFILE = auto()  # 1D: vertical profile (L levels)
    FIELD = auto()  # 2D: image or gridded field (H x W)


# Map SourceKind to top-level HDF5 group name.
_KIND_TO_GROUP: dict[SourceKind, str] = {
    SourceKind.SCALAR: "scalar",
    SourceKind.PROFILE: "profile",
    SourceKind.FIELD: "field",
}
_GROUP_TO_KIND: dict[str, SourceKind] = {v: k for k, v in _KIND_TO_GROUP.items()}

_FLOAT_COMPRESSION = {"compression": "gzip", "compression_opts": 4}


@dataclass
class Source:
    """A single observation source: values paired with explicit spatio-temporal coordinates.

    All coordinates are continuous and physical — no learned bin embeddings.
    Missing values are represented as NaN and must propagate correctly through
    any embedding or loss computation.

    Args:
        kind: Dimensionality class of this source.
        values: Observed measurements.
            - SCALAR:  (C,)
            - PROFILE: (L, C)   — L levels, C channels
            - FIELD:   (H, W, C)
        coords: Spatio-temporal coordinates paired with each measurement.
            - SCALAR:  (3,)         — [time, lat, lon]
            - PROFILE: (L, 4)       — [time, lat, lon, alt] per level
            - FIELD:   (H, W, 3)    — [time (scalar broadcast), lat, lon] per pixel
        source_name: Human-readable source identifier, e.g. "pmw_amsr2" or "era5_surface".
        channels: Names of each channel in the last axis of ``values``, in order.
            Length must equal ``values.shape[-1]``.
        mask: Boolean mask of valid (non-missing) entries; True = valid.
            Same leading shape as values. If None, all entries are assumed valid.
    """

    kind: SourceKind
    values: Tensor
    coords: Tensor
    source_name: str
    channels: list[str]
    mask: Tensor | None = None

    def __post_init__(self) -> None:
        """Validate shape consistency between values, coords, and mask."""
        self._validate()

    def _validate(self) -> None:
        """Check shape consistency between values, coords, and mask."""
        v, c = self.values, self.coords

        if self.kind is SourceKind.SCALAR:
            if v.ndim != 1:
                raise ValueError(f"SCALAR values must be 1-D, got shape {v.shape}")
            if c.shape != (3,):
                raise ValueError(f"SCALAR coords must be (3,), got {c.shape}")

        elif self.kind is SourceKind.PROFILE:
            if v.ndim != 2:
                raise ValueError(f"PROFILE values must be 2-D (L, C), got {v.shape}")
            if c.ndim != 2 or c.shape[0] != v.shape[0] or c.shape[1] != 4:
                raise ValueError(
                    f"PROFILE coords must be (L, 4), got {c.shape} for values {v.shape}"
                )

        elif self.kind is SourceKind.FIELD:
            if v.ndim != 3:
                raise ValueError(f"FIELD values must be 3-D (H, W, C), got {v.shape}")
            if c.shape != (*v.shape[:2], 3):
                raise ValueError(
                    f"FIELD coords must be (H, W, 3), got {c.shape} for values {v.shape}"
                )

        if self.mask is not None and self.mask.shape != v.shape[: self.mask.ndim]:
            raise ValueError(
                f"mask leading shape {self.mask.shape} incompatible with values {v.shape}"
            )

    @property
    def n_tokens(self) -> int:
        """Number of (value, coord) pairs in this source (flattened spatial dims)."""
        if self.kind is SourceKind.SCALAR:
            return 1
        elif self.kind is SourceKind.PROFILE:
            return self.values.shape[0]  # L
        else:  # FIELD
            h, w = self.values.shape[:2]
            return h * w

    def to(self, device: torch.device | str) -> Source:
        """Move tensors to device, returning a new Source."""
        return Source(
            kind=self.kind,
            values=self.values.to(device),
            coords=self.coords.to(device),
            source_name=self.source_name,
            channels=self.channels,
            mask=self.mask.to(device) if self.mask is not None else None,
        )

    # ------------------------------------------------------------------
    # HDF5 per-source I/O
    # ------------------------------------------------------------------

    def to_hdf5_group(self, group: h5py.Group) -> None:
        """Write this source into an open, writable HDF5 group.

        The group should already be a sub-group named by ``source_name``
        (created by the caller).  Datasets ``values`` and ``coords`` are
        always written; ``mask`` is written only when present.

        Args:
            group: Open, writable h5py Group for this source.
        """
        group.create_dataset(
            "values",
            data=self.values.detach().cpu().numpy().astype(np.float32),
            **_FLOAT_COMPRESSION,
        )
        # FIELD coords stored as float32 (lat/lon precision sufficient); others float64.
        coord_dtype = np.float32 if self.kind is SourceKind.FIELD else np.float64
        group.create_dataset(
            "coords",
            data=self.coords.detach().cpu().numpy().astype(coord_dtype),
            **_FLOAT_COMPRESSION,
        )
        if self.mask is not None:
            group.create_dataset(
                "mask",
                data=self.mask.detach().cpu().numpy().astype(bool),
            )
        group.attrs["source_name"] = self.source_name
        group.attrs["channels"] = json.dumps(self.channels)

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group, kind: SourceKind) -> Source:
        """Read a Source from an HDF5 group previously written by :meth:`to_hdf5_group`.

        Args:
            group: Open h5py Group for this source.
            kind: SourceKind of this source (determined by parent group name).

        Returns:
            Reconstructed Source with tensors on CPU.
        """
        values = torch.from_numpy(np.array(group["values"], dtype=np.float32))
        coord_dtype = np.float32 if kind is SourceKind.FIELD else np.float64
        coords = torch.from_numpy(np.array(group["coords"], dtype=coord_dtype))
        mask: torch.Tensor | None = None
        if "mask" in group:
            mask = torch.from_numpy(np.array(group["mask"], dtype=bool))
        source_name = str(group.attrs["source_name"])
        channels: list[str] = json.loads(str(group.attrs["channels"]))
        return cls(
            kind=kind,
            values=values,
            coords=coords,
            source_name=source_name,
            channels=channels,
            mask=mask,
        )
