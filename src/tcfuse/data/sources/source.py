"""Source abstraction: a single (value, coordinate) collection from one instrument."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

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


def _as_numpy_dtype(tensor: Tensor, dtype: np.dtype[Any]) -> np.ndarray:
    """Return a CPU NumPy array, casting only when the dtype changes."""
    array = tensor.detach().cpu().numpy()
    return array if array.dtype == dtype else array.astype(dtype)


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
        mask: Per-value availability mask; True = finite/available, False = missing.
            Must have the same shape as ``values``:
            - SCALAR:  (C,)
            - PROFILE: (L, C)
            - FIELD:   (H, W, C)
        char_vars: Instrument-level descriptor variables that are constant across all
            snapshots of this source (e.g. ``{"ifov": {"tb_89.0h": [7.2, 4.4, 7.2, 4.4]}}``).
            Values must be JSON-serialisable (lists, dicts, scalars).
    """

    kind: SourceKind
    values: Tensor
    coords: Tensor
    source_name: str
    channels: list[str]
    mask: Tensor
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)
    char_vars: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shape consistency between values, coords, and mask."""
        self.mask = self.mask.to(dtype=torch.bool)
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

        if self.mask.shape != v.shape:
            raise ValueError(f"mask shape {self.mask.shape} must match values shape {v.shape}")

    @property
    def shape(self) -> tuple[int, ...]:
        """Spatial shape of this source (excluding channels).

        Returns:
            - SCALAR:  ``()`` — a single measurement point.
            - PROFILE: ``(L,)`` — L vertical levels.
            - FIELD:   ``(H, W)`` — spatial grid dimensions.
        """
        if self.kind is SourceKind.SCALAR:
            return ()
        elif self.kind is SourceKind.PROFILE:
            return (self.values.shape[0],)
        else:  # FIELD
            return (self.values.shape[0], self.values.shape[1])

    @property
    def n_tokens(self) -> int:
        """Number of (value, coord) pairs in this source (flattened spatial dims)."""
        # SCALAR has an empty shape, so math.prod(()) == 1.
        return max(1, math.prod(self.shape))

    def to(self, device: torch.device | str) -> Source:
        """Move tensors to device, returning a new Source."""
        return Source(
            kind=self.kind,
            values=self.values.to(device),
            coords=self.coords.to(device),
            source_name=self.source_name,
            channels=self.channels,
            mask=self.mask.to(device),
            meta=self.meta,
            char_vars=self.char_vars,
        )

    # ------------------------------------------------------------------
    # HDF5 per-source I/O
    # ------------------------------------------------------------------

    def to_hdf5_group(self, group: h5py.Group) -> None:
        """Write this source into an open, writable HDF5 group.

        The group should already be a sub-group named by ``source_name``
        (created by the caller).  Datasets ``values`` and ``coords`` are
        always written; ``mask`` is the per-value availability mask.

        Args:
            group: Open, writable h5py Group for this source.
        """
        group.create_dataset(
            "values",
            data=_as_numpy_dtype(self.values, np.dtype(np.float32)),
            **_FLOAT_COMPRESSION,
        )
        # FIELD coords stored as float32 (lat/lon precision sufficient); others float64.
        coord_dtype = np.dtype(np.float32 if self.kind is SourceKind.FIELD else np.float64)
        group.create_dataset(
            "coords",
            data=_as_numpy_dtype(self.coords, coord_dtype),
            **_FLOAT_COMPRESSION,
        )
        group.create_dataset(
            "mask",
            data=_as_numpy_dtype(self.mask, np.dtype(bool)),
        )
        group.attrs["source_name"] = self.source_name
        group.attrs["channels"] = json.dumps(self.channels)
        # Instrument-level descriptors (same for every snapshot of this source).
        group.attrs["char_vars"] = json.dumps(self.char_vars)

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
        if "mask" not in group:
            raise ValueError("Source HDF5 group is missing mandatory 'mask' dataset.")
        mask = torch.from_numpy(np.array(group["mask"], dtype=bool))
        source_name = str(group.attrs["source_name"])
        channels: list[str] = json.loads(str(group.attrs["channels"]))
        char_vars: dict[str, Any] = json.loads(str(group.attrs["char_vars"]))
        return cls(
            kind=kind,
            values=values,
            coords=coords,
            source_name=source_name,
            channels=channels,
            mask=mask,
            char_vars=char_vars,
        )

    # ------------------------------------------------------------------
    # File-level HDF5 I/O (single-source files)
    # ------------------------------------------------------------------

    def write(self, path: Path) -> None:
        """Write this source to a self-contained HDF5 file.

        Root-level attributes hold ``self.meta`` (storm / observation metadata).
        Tensor data lives under ``/{kind}/{source_name}/``, matching the layout
        used by :meth:`to_hdf5_group`.  Parent directories are created as needed.

        Args:
            path: Destination ``.h5`` file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            # Write storm / observation metadata as root attributes.
            for key, value in self.meta.items():
                f.attrs[key] = value
            # Write tensors into /{kind}/{source_name}/.
            kind_group_name = _KIND_TO_GROUP[self.kind]
            src_group = f.create_group(f"{kind_group_name}/{self.source_name}")
            self.to_hdf5_group(src_group)

    @classmethod
    def from_disk(cls, path: Path) -> Source:
        """Load a Source from an HDF5 file written by :meth:`write`.

        Root-level attributes are loaded into ``meta``.  The single
        ``/{kind}/{source_name}/`` group provides the tensor data.

        Args:
            path: Path to the ``.h5`` file.

        Returns:
            Reconstructed :class:`Source` with tensors on CPU and root
            attributes in ``meta``.
        """
        with h5py.File(path, "r") as f:
            meta = dict(f.attrs)
            # Find the one source group (there is exactly one per file).
            for kind_group_name, kind in _GROUP_TO_KIND.items():
                if kind_group_name not in f:
                    continue
                for _name, group in f[kind_group_name].items():  # type: ignore[union-attr]
                    source = cls.from_hdf5_group(group, kind)
                    source.meta = meta
                    return source
        raise ValueError(f"No source group found in {path}")

    @staticmethod
    def read_meta(path: Path) -> dict[str, Any]:
        """Read only root-level metadata attributes without loading tensors.

        Useful for building or refreshing an index without touching source data.

        Args:
            path: Path to the ``.h5`` file.

        Returns:
            Dict of root-level HDF5 attributes.
        """
        with h5py.File(path, "r") as f:
            return dict(f.attrs)

    @staticmethod
    def path(
        sources_root: Path,
        source_name: str,
        storm_id: str,
        snapshot_time_utc: str,
    ) -> Path:
        """Return the canonical path for a single-source HDF5 file.

        Args:
            sources_root: Root directory for preprocessed sources
                (``cfg.paths.preprocessed_sources``).
            source_name: Source identifier, e.g. ``"pmw_amsr2_gcomw1"``.
            storm_id: Storm identifier, e.g. ``"2016AL10"``.
            snapshot_time_utc: Compact UTC timestamp string,
                e.g. ``"20160912T010942Z"``.

        Returns:
            Absolute path:
            ``{sources_root}/{source_name}/snapshots/{storm_id}_{snapshot_time_utc}.h5``
        """
        return sources_root / source_name / "snapshots" / f"{storm_id}_{snapshot_time_utc}.h5"
