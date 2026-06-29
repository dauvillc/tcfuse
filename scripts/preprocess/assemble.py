#!/usr/bin/env python3
"""Stage 2 — assemble per-source HDF5 snapshots into one HDF5 file per storm.

Reads:
- Stage 1 outputs under ``${paths.preprocessed_sources}/<source>/``.
- Stage 0 IBTrACS artifacts under ``${paths.preprocessed_sources}/ibtracs/``.

Writes:
- ``${paths.preprocessed_data}/storm_data/{sid}.h5`` — one assembled HDF5 file
  per IBTrACS SID. Contains every available Stage 1 source plus an injected
  ``ibtracs_best_track`` SCALAR Source (16 channels).
- ``${paths.preprocessed_data}/index.parquet`` — concatenated index of
  satellite-source snapshot rows and full IBTrACS rows. Satellite rows leave
  IBTrACS-specific columns NaN; IBTrACS rows leave nothing extra.
- ``${paths.preprocessed_data}/sources_metadata.yaml`` — merged source
  descriptors (channels, shape, kind) for downstream ML pipeline use.

Storms not present in IBTrACS are simply not assembled — the SID set comes
straight from the IBTrACS parquet, after the ``TRACK_TYPE == "MAIN"`` filter
already applied at Stage 0.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, cast

import h5py
import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from scripts.preprocess.utils.assembler import assemble_storm
from scripts.preprocess.utils.runner import INDEX_COLUMNS, resolve_preproc_cfg
from tcfuse.data.ibtracs import (
    IBTRACS_SOURCE_NAME,
    group_ibtracs_by_sid,
    load_atcf_to_sid,
    load_ibtracs_snapshots,
)
from tcfuse.data.sources import MultisourceMetadata, StormData
from tcfuse.utils.archive import submit_archive_job

# Stage 0 outputs live here; not a Stage 1 measurement source.
_IBTRACS_DIR_NAME = "ibtracs"


def _is_stage1_source_dir(entry: Path) -> bool:
    """Return True when a directory holds both Stage-1 metadata and index files."""
    if not entry.is_dir() or entry.name == _IBTRACS_DIR_NAME:
        return False
    return (entry / "metadata.yaml").is_file() and (entry / "index.parquet").is_file()


def _discover_stage1_metadata_yaml_paths(sources_root: Path) -> list[Path]:
    """Collect per-source ``metadata.yaml`` paths under ``sources_root``."""
    return [
        entry / "metadata.yaml"
        for entry in sorted(sources_root.iterdir())
        if _is_stage1_source_dir(entry)
    ]


def _load_stage1_snapshot_index(sources_root: Path) -> pd.DataFrame:
    """Concatenate every Stage-1 ``index.parquet`` under ``sources_root``."""
    frames: list[pd.DataFrame] = []
    for entry in sorted(sources_root.iterdir()):
        if not _is_stage1_source_dir(entry):
            continue
        frames.append(pd.read_parquet(entry / "index.parquet"))
    if not frames:
        return pd.DataFrame(columns=list(INDEX_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def _assemble_storms_batch(
    sids: list[str],
    index: pd.DataFrame,
    sources_root: Path,
    assembled_root: Path,
    skip_existing: bool,
    num_workers: int,
    ibtracs_by_sid: dict[str, pd.DataFrame],
    sid_attrs: dict[str, dict[str, Any]],
    atcf_for_sid: dict[str, str],
) -> list[str | None]:
    """Assemble a batch of storms, optionally in parallel."""
    sid_set = set(sids)
    grouped = {sid: grp for sid, grp in index.groupby("sid") if sid in sid_set}
    empty = pd.DataFrame(columns=index.columns)
    rows_per_sid = [grouped.get(sid, empty) for sid in sids]

    if num_workers <= 1:
        return [
            assemble_storm(
                sid,
                rows,
                sources_root,
                assembled_root,
                skip_existing,
                ibtracs_by_sid,
                sid_attrs,
                atcf_for_sid,
            )
            for sid, rows in zip(tqdm(sids, desc="assemble"), rows_per_sid, strict=True)
        ]

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        return list(
            tqdm(
                pool.map(
                    assemble_storm,
                    sids,
                    rows_per_sid,
                    repeat(sources_root),
                    repeat(assembled_root),
                    repeat(skip_existing),
                    repeat(ibtracs_by_sid),
                    repeat(sid_attrs),
                    repeat(atcf_for_sid),
                    chunksize=max(1, len(sids) // (num_workers * 4)),
                ),
                total=len(sids),
                desc="assemble",
            )
        )


def _scan_storm_satellite_index(
    assembled_root: Path,
    sids: list[str],
    sid_attrs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Browse every assembled storm file and collect one row per non-IBTrACS snapshot."""
    rows: list[dict[str, Any]] = []
    for sid in sids:
        path = StormData.path(assembled_root, sid)
        if not path.exists():
            continue
        info = sid_attrs.get(sid)
        if info is None:
            continue

        with h5py.File(path, "r") as storm_file:
            for source_name, source_group in storm_file.items():
                if source_name == IBTRACS_SOURCE_NAME:
                    continue
                if not isinstance(source_group, h5py.Group):
                    continue
                for snap_group in source_group.values():
                    if not isinstance(snap_group, h5py.Group):
                        continue
                    time_utc = str(snap_group.attrs["time_utc"])
                    rows.append(
                        {
                            "sid": sid,
                            "source_name": source_name,
                            "time_utc": time_utc,
                            "season": info["season"],
                            "basin": info["basin"],
                            "subbasin": info["subbasin"],
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    return pd.DataFrame(rows, columns=INDEX_COLUMNS)


def build_assembled_index(
    ibtracs_snapshots: pd.DataFrame,
    assembled_root: Path,
    assembled_sids: list[str],
    sid_attrs: dict[str, dict[str, Any]],
    atcf_for_sid: dict[str, str],
) -> pd.DataFrame:
    """Build the concatenated index of satellite rows + IBTrACS rows.

    Every row — satellite or IBTrACS — carries the same uniform schema:
    ``INDEX_COLUMNS`` plus ``usa_atcf_id``.  IBTrACS numeric channel data
    (wind, pressure, radii, …) is stored in the assembled HDF5 files and is
    deliberately excluded from the index so that IBTrACS is treated like any
    other source.
    """
    sat_index = _scan_storm_satellite_index(assembled_root, assembled_sids, sid_attrs)

    ibt_rows = cast(
        pd.DataFrame,
        ibtracs_snapshots[ibtracs_snapshots["sid"].isin(assembled_sids)].copy(),
    )
    ibt_rows = cast(pd.DataFrame, ibt_rows.rename(columns={"iso_time": "time_utc"}))
    ibt_rows["source_name"] = IBTRACS_SOURCE_NAME

    # Retain only the uniform schema columns from both sets of rows.
    output_columns = [*INDEX_COLUMNS, "usa_atcf_id"]
    for df in (sat_index, ibt_rows):
        for col in output_columns:
            if col not in df.columns:
                df[col] = None

    combined = cast(
        pd.DataFrame,
        pd.concat(
            [sat_index[output_columns], ibt_rows[output_columns]],
            ignore_index=True,
        ),
    )

    # Populate usa_atcf_id for satellite rows from the ATCF translation table;
    # IBTrACS rows already carry it from the parquet.
    missing_atcf = combined["usa_atcf_id"].isna()
    combined.loc[missing_atcf, "usa_atcf_id"] = combined.loc[missing_atcf, "sid"].map(atcf_for_sid)

    return cast(
        pd.DataFrame,
        combined.sort_values(["sid", "time_utc"]).reset_index(drop=True),
    )


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Assemble all available preprocessed sources into per-storm HDF5 files."""
    cfg = resolve_preproc_cfg(raw_cfg)
    sources_root = Path(cfg["paths"]["preprocessed_sources"])
    assembled_root = Path(cfg["paths"]["preprocessed_data"])
    assembled_root.mkdir(parents=True, exist_ok=True)

    num_workers = int(cfg.get("num_workers", 4))
    skip_existing = bool(cfg.get("skip_existing", False))

    print(f"Loading IBTrACS Stage 0 artifacts from {sources_root / 'ibtracs'} …")
    ibtracs_snapshots = load_ibtracs_snapshots(sources_root)
    ibtracs_by_sid = group_ibtracs_by_sid(ibtracs_snapshots)
    translation = load_atcf_to_sid(sources_root)
    print(f"Loaded {len(ibtracs_by_sid)} IBTrACS storms; {len(translation)} ATCF↔SID pairings.")

    # Storm-level constants keyed by SID, derived from the translation table.
    subset_df = cast(pd.DataFrame, translation[["sid", "season", "basin", "subbasin"]])
    keep = cast(pd.DataFrame, subset_df.drop_duplicates(subset=["sid"]))

    # Guard against a missing basin/subbasin sneaking in as a float NaN: never
    # stringify it to "nan" — a truly-missing code is the empty string instead.
    def _clean_code(value: Any) -> str:
        return "" if pd.isna(value) else str(value)

    sid_attrs: dict[str, dict[str, Any]] = {
        str(rec["sid"]): {
            "season": int(rec["season"]),
            "basin": _clean_code(rec["basin"]),
            "subbasin": _clean_code(rec["subbasin"]),
        }
        for rec in cast(list[dict[str, Any]], keep.to_dict(orient="records"))
    }
    atcf_for_sid: dict[str, str] = {
        str(rec["sid"]): str(rec["usa_atcf_id"])
        for rec in cast(list[dict[str, Any]], translation.to_dict(orient="records"))
        if str(rec["usa_atcf_id"]).strip() != ""
    }

    print(f"Loading per-source metadata from {sources_root} …")
    yaml_paths = _discover_stage1_metadata_yaml_paths(sources_root)
    multi_meta = MultisourceMetadata.from_multiple_yaml(yaml_paths)
    index = _load_stage1_snapshot_index(sources_root)
    if multi_meta.sources:
        print(
            f"Found {len(multi_meta)} source(s), {len(index)} total snapshots, "
            f"{index['sid'].nunique() if not index.empty else 0} unique SIDs."
        )
    else:
        print("No Stage 1 source indices found; will still write IBTrACS-only storm files.")

    sids = sorted(ibtracs_by_sid.keys())
    if not sids:
        print("No IBTrACS storms to assemble. Nothing to do.")
        return

    if cfg.get("submitit", False):
        from tcfuse.utils.submitit_utils import make_executor

        chunk_size = int(cfg.get("chunk_size", 200))
        chunks = [sids[i : i + chunk_size] for i in range(0, len(sids), chunk_size)]
        slurm_executor = make_executor(cfg, "assemble")
        print(
            f"Submitting {len(chunks)} SLURM jobs ({len(sids)} storms, chunk_size={chunk_size}) …"
        )
        results: list[str | None] = []
        for job in tqdm(
            [
                slurm_executor.submit(
                    _assemble_storms_batch,
                    chunk,
                    index,
                    sources_root,
                    assembled_root,
                    skip_existing,
                    num_workers,
                    ibtracs_by_sid,
                    sid_attrs,
                    atcf_for_sid,
                )
                for chunk in chunks
            ],
            desc="collecting",
        ):
            results.extend(job.result())
    else:
        results = _assemble_storms_batch(
            sids,
            index,
            sources_root,
            assembled_root,
            skip_existing,
            num_workers,
            ibtracs_by_sid,
            sid_attrs,
            atcf_for_sid,
        )

    written = [r for r in results if r is not None]
    skipped = len(results) - len(written)
    print(f"Assembled {len(written)}/{len(sids)} storms → {assembled_root}")
    if skipped:
        print(f"Skipped / empty: {skipped}")

    print("Building assembled index …")
    index_df = build_assembled_index(
        ibtracs_snapshots, assembled_root, written, sid_attrs, atcf_for_sid
    )
    index_path = assembled_root / "index.parquet"
    index_df.to_parquet(index_path, index=False)
    print(f"Wrote assembled index: {len(index_df)} rows → {index_path}")

    if multi_meta.sources:
        metadata_path = assembled_root / "sources_metadata.yaml"
        multi_meta.to_yaml(metadata_path)
        print(f"Wrote sources metadata: {len(multi_meta)} source(s) → {metadata_path}")

    submit_archive_job(
        assembled_root,
        Path(cfg["paths"]["archives"]["preprocessed_data"]),
        cfg,
        job_name="archive_assembled",
    )


if __name__ == "__main__":
    main()
