"""SamplePrediction: one window's model predictions paired with their targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py

from tcfuse.data.sources.source import Source, SourceKind

# Top-level HDF5 group names separating model output from ground truth.
_PREDICTED_GROUP = "predicted"
_TARGET_GROUP = "target"


@dataclass
class SamplePrediction:
    """Model predictions for a single window, alongside their ground-truth targets.

    A :class:`SamplePrediction` is the prediction-side companion of one
    :class:`~tcfuse.data.dataset.WindowSample`.  It only covers the sources the
    model actually reconstructed (the target subset of the window), and stores
    both the predicted and the ground-truth :class:`~tcfuse.data.sources.source.Source`
    for each so that evaluation metrics can be computed from this object alone —
    no need to re-open the dataset.

    Predicted and target values are stored in **physical units** (already
    de-normalized), matching the output of ``predict_step``.

    Args:
        sample_id: Window identifier (``window_id``), shared with the source
            :class:`~tcfuse.data.dataset.WindowSample`.
        sid: IBTrACS storm identifier.
        season: TC season year.
        basin: Ocean basin code.
        subbasin: IBTrACS sub-basin code.
        window_ref_time_utc: Assimilation anchor time ``t0`` (ISO 8601 string).
        predicted: Maps ``(source_name, source_index)`` to the predicted
            :class:`Source` (model output values, with coords/mask/channels
            copied from the ground truth).
        target: Maps the same keys to the ground-truth :class:`Source`.
            ``predicted`` and ``target`` always share the exact same key set.
    """

    sample_id: str
    sid: str
    season: int
    basin: str
    subbasin: str
    window_ref_time_utc: str
    predicted: dict[tuple[str, int], Source]
    target: dict[tuple[str, int], Source]

    def write(self, path: Path) -> None:
        """Write this sample's predictions and targets to a self-contained HDF5 file.

        Layout::

            /
            ├── attrs: {sample_id, sid, season, basin, subbasin, window_ref_time_utc}
            ├── predicted/{source_name}/{source_index}/   # Source group (+ kind attr)
            └── target/{source_name}/{source_index}/      # Source group (+ kind attr)

        Args:
            path: Destination ``.h5`` file path.  Parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            # Sample-level metadata as root attributes (used to build the run index).
            f.attrs["sample_id"] = self.sample_id
            f.attrs["sid"] = self.sid
            f.attrs["season"] = self.season
            f.attrs["basin"] = self.basin
            f.attrs["subbasin"] = self.subbasin
            f.attrs["window_ref_time_utc"] = self.window_ref_time_utc

            # Write predicted and target Sources under their respective groups.
            self._write_sources(f.create_group(_PREDICTED_GROUP), self.predicted)
            self._write_sources(f.create_group(_TARGET_GROUP), self.target)

    @staticmethod
    def _write_sources(
        parent: h5py.Group,
        sources: dict[tuple[str, int], Source],
    ) -> None:
        """Write a dict of keyed Sources under ``{source_name}/{source_index}/``."""
        for (source_name, source_index), source in sources.items():
            # source_index is the chronological rank; stringified for the group name.
            snap_group = parent.require_group(f"{source_name}/{source_index}")
            source.to_hdf5_group(snap_group)
            # Persist the kind name so from_hdf5_group can reconstruct without guessing.
            snap_group.attrs["kind"] = source.kind.name

    @classmethod
    def from_disk(cls, path: Path) -> SamplePrediction:
        """Load a :class:`SamplePrediction` from a file written by :meth:`write`.

        Args:
            path: Path to the ``.h5`` file.

        Returns:
            Reconstructed :class:`SamplePrediction` with numpy-backed Sources.
        """
        with h5py.File(path, "r") as f:
            # Read sample-level metadata from root attributes.
            sample_id = str(f.attrs["sample_id"])
            sid = str(f.attrs["sid"])
            season = int(str(f.attrs["season"]))
            basin = str(f.attrs["basin"])
            subbasin = str(f.attrs["subbasin"])
            window_ref_time_utc = str(f.attrs["window_ref_time_utc"])

            # Rebuild predicted and target Source dicts from their groups.
            predicted = cls._read_sources(cast(h5py.Group, f[_PREDICTED_GROUP]))
            target = cls._read_sources(cast(h5py.Group, f[_TARGET_GROUP]))

        return cls(
            sample_id=sample_id,
            sid=sid,
            season=season,
            basin=basin,
            subbasin=subbasin,
            window_ref_time_utc=window_ref_time_utc,
            predicted=predicted,
            target=target,
        )

    @staticmethod
    def _read_sources(parent: h5py.Group) -> dict[tuple[str, int], Source]:
        """Read keyed Sources from a ``{source_name}/{source_index}/`` hierarchy."""
        sources: dict[tuple[str, int], Source] = {}
        # Outer level: one group per source_name.
        for source_name, source_group in parent.items():
            if not isinstance(source_group, h5py.Group):
                continue
            # Inner level: one group per stringified source_index.
            for source_index, snap_group in source_group.items():
                if not isinstance(snap_group, h5py.Group):
                    continue
                # kind drives the coord dtype and validation inside from_hdf5_group.
                kind = SourceKind[str(snap_group.attrs["kind"])]
                source = Source.from_hdf5_group(snap_group, kind)
                sources[(source_name, int(source_index))] = source
        return sources
