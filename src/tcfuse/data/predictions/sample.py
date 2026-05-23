"""SamplePrediction: per-window container for predicted (and target) Source objects.

A :class:`SamplePrediction` is the per-window analogue of
:class:`~tcfuse.data.sources.storm_data.StormData`. It holds the model's predicted
sources for one ``(storm_id, init_time)`` forecast window, plus the matching
ground-truth sources, both keyed by ``(source_name, snapshot_time_utc)`` so the
key matches the input :class:`StormData`.

Tensor serialisation is delegated verbatim to
:meth:`~tcfuse.data.sources.source.Source.to_hdf5_group` /
:meth:`~tcfuse.data.sources.source.Source.from_hdf5_group`; the container only adds
the ``pred/`` and ``target/`` subtrees and a small set of root attributes.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py

from tcfuse.data.sources.source import Source, SourceKind
from tcfuse.utils.time import lead_hours_rounded, to_compact_time

SAMPLES_SUBDIR = "samples"
_PRED_GROUP = "pred"
_TARGET_GROUP = "target"
_ROOT_ATTRS = (
    "sample_id",
    "storm_id",
    "init_time_utc",
    "basin",
    "season",
    "atcf_id",
    "run_id",
)


@dataclass
class SamplePrediction:
    """Predicted and (optionally) target Sources for a single forecast window.

    Args:
        sample_id: Window identifier ``f"{storm_id}_{anchor_time:%Y%m%dT%H%M%SZ}"``,
            matching ``build_splits.py``.
        storm_id: Storm identifier, e.g. ``"2016AL10"``.
        init_time_utc: Window anchor time as a repository-style ISO timestamp.
        basin: Ocean basin code, e.g. ``"AL"``.
        season: TC season year, e.g. 2016.
        atcf_id: Optional ATCF identifier carried over from the split parquet.
        run_id: Optional run identifier; written to the file root attrs to make
            each per-window file self-identifying.
        pred_sources: Predicted sources keyed by ``(source_name, snapshot_time_utc)``.
        target_sources: Ground-truth sources keyed by ``(source_name, snapshot_time_utc)``.
            Same keys as ``pred_sources`` are expected when both are present, but the
            container does not enforce equality so partial coverage is allowed.
    """

    sample_id: str
    storm_id: str
    init_time_utc: str
    basin: str
    season: int
    atcf_id: str | None = None
    run_id: str | None = None
    pred_sources: dict[tuple[str, str], Source] = dataclasses.field(default_factory=dict)
    target_sources: dict[tuple[str, str], Source] = dataclasses.field(default_factory=dict)

    @property
    def predicted_source_names(self) -> list[str]:
        """Sorted unique source names present in ``pred_sources``."""
        return sorted({source_name for source_name, _ in self.pred_sources})

    # ------------------------------------------------------------------
    # Canonical path helper
    # ------------------------------------------------------------------

    @staticmethod
    def path(run_root: Path, sample_id: str) -> Path:
        """Return the canonical path for a sample's per-window HDF5 file.

        Args:
            run_root: Root directory of the prediction run
                (``{cfg.paths.predictions}/{run_id}``).
            sample_id: Window identifier.

        Returns:
            ``{run_root}/samples/{sample_id}.h5``.
        """
        return run_root / SAMPLES_SUBDIR / f"{sample_id}.h5"

    # ------------------------------------------------------------------
    # HDF5 I/O
    # ------------------------------------------------------------------

    def write(self, run_root: Path) -> Path:
        """Write the predicted (and target) sources to a per-window HDF5 file.

        Layout::

            /
            ├── attrs: {sample_id, storm_id, init_time_utc, basin, season,
            │           atcf_id?, run_id?}
            ├── pred/
            │   └── {source_name}/{compact_valid_time}/  (values, coords, mask, attrs)
            └── target/
                └── {source_name}/{compact_valid_time}/  (values, coords, mask, attrs)

        Args:
            run_root: Root directory of the prediction run.

        Returns:
            Absolute path of the written file.
        """
        dest = SamplePrediction.path(run_root, self.sample_id)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(dest, "w") as f:
            # Window-level constants as root attributes.
            f.attrs["sample_id"] = self.sample_id
            f.attrs["storm_id"] = self.storm_id
            f.attrs["init_time_utc"] = self.init_time_utc
            f.attrs["basin"] = self.basin
            f.attrs["season"] = int(self.season)
            if self.atcf_id is not None:
                f.attrs["atcf_id"] = self.atcf_id
            if self.run_id is not None:
                f.attrs["run_id"] = self.run_id

            # Predicted sources go under /pred/, targets under /target/.
            self._write_sources(f.require_group(_PRED_GROUP), self.pred_sources)
            self._write_sources(f.require_group(_TARGET_GROUP), self.target_sources)

        return dest

    def _write_sources(
        self,
        parent: h5py.Group,
        sources: dict[tuple[str, str], Source],
    ) -> None:
        """Write a {(source_name, snapshot_time_utc): Source} mapping under ``parent``."""
        for (source_name, snapshot_time_utc), source in sources.items():
            compact_time = to_compact_time(snapshot_time_utc)
            snap_group = parent.require_group(f"{source_name}/{compact_time}")

            # Reuse the existing Source -> HDF5 group serializer verbatim.
            source.to_hdf5_group(snap_group)

            # Round-trip support: kind + the original isoformat timestamp.
            snap_group.attrs["kind"] = source.kind.name
            snap_group.attrs["snapshot_time_utc"] = snapshot_time_utc

            # Lead hour is a convenience attr; it is derivable from the timestamps
            # but cheap to store and useful for ad-hoc inspection of HDF5 files.
            try:
                lead_hour = lead_hours_rounded(self.init_time_utc, snapshot_time_utc)
                snap_group.attrs["lead_hour"] = int(lead_hour)
            except (TypeError, ValueError):
                pass

            # Forward Source.meta (lat, lon, vmax_kt, ...) as snapshot-level attrs.
            skip_keys = {"source_name", "channels", "kind", "snapshot_time_utc", "lead_hour"}
            for key, value in source.meta.items():
                if key in skip_keys:
                    continue
                try:
                    snap_group.attrs[key] = value
                except TypeError:
                    warnings.warn(
                        f"Could not write meta key '{key}' as HDF5 attr: {type(value)}",
                        stacklevel=2,
                    )

    @classmethod
    def from_disk(cls, run_root: Path, sample_id: str) -> SamplePrediction:
        """Load the predicted and target sources for a window from its HDF5 file.

        Args:
            run_root: Root directory of the prediction run.
            sample_id: Window identifier.

        Returns:
            Reconstructed :class:`SamplePrediction` with tensors on CPU.
        """
        path = SamplePrediction.path(run_root, sample_id)
        with h5py.File(path, "r") as f:
            # h5py exposes attrs as a typed-dict-ish object; reify into a plain dict
            # so the per-key coercions below have a stable Any-valued lookup.
            attrs = cast(dict[str, Any], dict(f.attrs))
            loaded_sample_id = str(attrs["sample_id"])
            storm_id = str(attrs["storm_id"])
            init_time_utc = str(attrs["init_time_utc"])
            basin = str(attrs["basin"])
            season = int(attrs["season"])
            atcf_id = str(attrs["atcf_id"]) if "atcf_id" in attrs else None
            run_id = str(attrs["run_id"]) if "run_id" in attrs else None

            pred_sources = cls._read_sources(f, _PRED_GROUP)
            target_sources = cls._read_sources(f, _TARGET_GROUP)

        return cls(
            sample_id=loaded_sample_id,
            storm_id=storm_id,
            init_time_utc=init_time_utc,
            basin=basin,
            season=season,
            atcf_id=atcf_id,
            run_id=run_id,
            pred_sources=pred_sources,
            target_sources=target_sources,
        )

    @staticmethod
    def _read_sources(
        file: h5py.File,
        group_name: str,
    ) -> dict[tuple[str, str], Source]:
        """Read a ``pred/`` or ``target/`` subtree into a {(name, time): Source} dict."""
        if group_name not in file:
            return {}

        sources: dict[tuple[str, str], Source] = {}
        parent = file[group_name]
        if not isinstance(parent, h5py.Group):
            return {}

        # Iterate source_name groups, then compact_time sub-groups.
        for source_name, source_group in parent.items():
            if not isinstance(source_group, h5py.Group):
                continue
            for _compact_time, snap_group in source_group.items():
                if not isinstance(snap_group, h5py.Group):
                    continue

                kind = SourceKind[str(snap_group.attrs["kind"])]
                source = Source.from_hdf5_group(snap_group, kind)
                snapshot_time_utc = str(snap_group.attrs["snapshot_time_utc"])

                # Forward snapshot-level attrs into Source.meta.
                meta: dict[str, Any] = {
                    "snapshot_time_utc": snapshot_time_utc,
                }
                skip_keys = {"source_name", "channels", "kind", "snapshot_time_utc"}
                for key in snap_group.attrs:
                    if key not in skip_keys:
                        meta[key] = snap_group.attrs[key]
                source.meta = meta

                sources[(source_name, snapshot_time_utc)] = source
        return sources

    @staticmethod
    def read_meta(run_root: Path, sample_id: str) -> dict[str, Any]:
        """Read only root-level attributes without loading any tensors.

        Useful for quickly listing what is available in a per-window file before
        deciding whether to read the full tensor payload.
        """
        path = SamplePrediction.path(run_root, sample_id)
        with h5py.File(path, "r") as f:
            attrs = cast(dict[str, Any], dict(f.attrs))

        # Coerce the few attrs we know carry well-defined types so callers do not
        # have to deal with numpy/h5py wrapper scalars.
        result: dict[str, Any] = {}
        if "season" in attrs:
            result["season"] = int(attrs["season"])
        for key in ("sample_id", "storm_id", "init_time_utc", "basin", "atcf_id", "run_id"):
            if key in attrs:
                result[key] = str(attrs[key])
        return {key: value for key, value in result.items() if key in _ROOT_ATTRS}
