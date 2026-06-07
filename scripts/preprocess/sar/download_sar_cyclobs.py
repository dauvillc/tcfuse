#!/usr/bin/env python3
"""
download_sar_cyclobs.py
-----------------------

Download SAR acquisitions from the CyclObs API
(https://cyclobs.ifremer.fr).
Destination is read from paths.raw_datasets.cyclobs in the Hydra config.

Examples
========
# 1)  Run locally (no SLURM submission)
python scripts/preprocess/sar/download_sar_cyclobs.py submitit=false paths=local

# 2)  Override the instrument filter
python scripts/preprocess/sar/download_sar_cyclobs.py submitit=false paths=local +instrument=C-Band_SAR

# 3)  Download only selected seasons and basins
python scripts/preprocess/sar/download_sar_cyclobs.py \
    submitit=false paths=local \
    '+include_seasons=[2020,2021]' '+include_basins=[AL,EP]'

# 4)  On Jean-Zay — submits to the prepost partition (has internet access)
python scripts/preprocess/sar/download_sar_cyclobs.py setup=jz_prepost
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import hydra
import pandas as pd
import requests
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from tcfuse.utils.archive import submit_archive_job
from tcfuse.utils.submitit_utils import make_executor

# Constants
API_URL = "https://cyclobs.ifremer.fr/app/api/getData"
DEFAULT_INSTRUMENT = "C-Band_SAR"
DEFAULT_WORKERS = 4
MAX_RETRIES = 3
BACKOFF_FACTOR = 2  # Seconds between retries, multiplied by attempt number


def _normalize_filter_values(value: object, *, uppercase: bool = False) -> tuple[str, ...] | None:
    """Normalize optional Hydra scalar/list filters to comparable strings."""
    if value is None:
        return None

    # Hydra overrides may arrive as a scalar or a list after OmegaConf conversion.
    if isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        raw_values = (value,)

    normalized: list[str] = []
    for raw_value in raw_values:
        text = str(raw_value).strip()
        if not text:
            continue
        normalized.append(text.upper() if uppercase else text)

    return tuple(normalized) or None


def filter_sar_acquisitions_metadata(
    df_metadata: pd.DataFrame,
    include_seasons: object = None,
    include_basins: object = None,
) -> pd.DataFrame:
    """Filter CyclObs acquisition metadata by seasons and basins derived from ``sid``.

    CyclObs ``sid`` values use ``bbNNYYYY`` formatting, for example ``al022024``.
    Seasons are derived from the final four characters and basins from the first
    two characters, matching ``prepare_sar.py``.
    """
    seasons = _normalize_filter_values(include_seasons)
    basins = _normalize_filter_values(include_basins, uppercase=True)

    if seasons is None and basins is None:
        return df_metadata

    if "sid" not in df_metadata.columns:
        raise ValueError("'sid' column is required when filtering by season or basin.")

    initial_count = len(df_metadata)
    # Derive temporary filter keys from CyclObs storm identifiers.
    sid = df_metadata["sid"].astype(str)
    derived_seasons = sid.str[-4:]
    derived_basins = sid.str[:2].str.upper()
    keep_rows = pd.Series(True, index=df_metadata.index)

    if seasons is not None:
        keep_rows &= derived_seasons.isin(seasons)

    if basins is not None:
        keep_rows &= derived_basins.isin(basins)

    filtered = df_metadata[keep_rows].reset_index(drop=True)
    filtered = cast(pd.DataFrame, filtered)
    print(
        "Filtered SAR acquisitions: "
        f"{initial_count:,} -> {len(filtered):,} "
        f"(seasons={seasons or 'all'}, basins={basins or 'all'})."
    )
    return filtered


def get_sar_acquisitions_metadata(
    instrument: str = DEFAULT_INSTRUMENT, include_cols: str = "all"
) -> pd.DataFrame:
    """Query the CyclObs API for available SAR acquisitions.

    Args:
        instrument: Instrument filter passed to the API.
        include_cols: Columns to include; "all" returns comprehensive metadata.

    Returns:
        DataFrame of acquisition metadata, or an empty DataFrame on error.
    """
    params = {"instrument": instrument, "include_cols": include_cols}
    print(f"Querying CyclObs API at {API_URL} for instrument='{instrument}'...")

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        df = pd.read_csv(response.url)
        print(f"Found {len(df)} acquisitions.")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error querying API: {e}")
        return pd.DataFrame()


def download_file(url: str, output_dir: Path) -> str:
    """Download a single file with retries and atomic rename.

    Steps:
    1. Skip if the final file already exists (idempotent re-runs).
    2. Download to ``{filename}.tmp``.
    3. Rename to ``{filename}`` only on success.

    Args:
        url: Direct download URL.
        output_dir: Target directory.

    Returns:
        One of ``"skipped"``, ``"downloaded"``, or ``"failed: <reason>"``.
    """
    if not isinstance(url, str) or not url.startswith("http"):
        return "failed: invalid URL"

    # Derive local paths
    filename = url.split("/")[-1]
    final_path = output_dir / filename
    temp_path = output_dir / f"{filename}.tmp"

    # Skip completed downloads
    if final_path.exists():
        return "skipped"

    # Retry loop with exponential backoff
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with temp_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            # Atomic rename on success
            temp_path.rename(final_path)
            return "downloaded"

        except (OSError, requests.exceptions.RequestException) as e:
            # Remove partial temp file before retrying
            if temp_path.exists():
                temp_path.unlink()

            if attempt == MAX_RETRIES:
                return f"failed: {e}"

            time.sleep(BACKOFF_FACTOR**attempt)

    return "failed: exceeded max retries"


class DownloadJob:
    """Submitit-compatible callable that downloads CyclObs SAR data."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def __call__(self) -> None:
        """Run the download and archive steps."""
        cfg = self.cfg

        # Destination root and optional overrides from config
        cyclobs_root = Path(cfg["paths"]["raw_datasets"]["cyclobs"])
        instrument: str = cfg.get("instrument", DEFAULT_INSTRUMENT)
        workers: int = int(cfg.get("workers", DEFAULT_WORKERS))
        include_seasons = cfg.get("include_seasons")
        include_basins = cfg.get("include_basins")

        cyclobs_root.mkdir(parents=True, exist_ok=True)
        print(f"Saving data to: {cyclobs_root.resolve()}")
        print(f"Instrument    : {instrument}")
        print(f"Workers       : {workers}")
        print(f"Seasons       : {_normalize_filter_values(include_seasons) or 'all'}")
        print(
            f"Basins        : {_normalize_filter_values(include_basins, uppercase=True) or 'all'}"
        )

        # Query the API for acquisition metadata
        df_metadata = get_sar_acquisitions_metadata(instrument)
        if df_metadata.empty:
            print("No data found or API error.")
            return

        # Apply optional season/basin filters before saving metadata or downloading.
        df_metadata = filter_sar_acquisitions_metadata(
            df_metadata,
            include_seasons=include_seasons,
            include_basins=include_basins,
        )

        # Persist metadata alongside the raw files
        metadata_path = cyclobs_root / "sar_acquisitions_metadata.csv"
        df_metadata.to_csv(metadata_path, index=False)
        print(f"Metadata saved to {metadata_path}")

        if df_metadata.empty:
            print("No acquisitions match the requested filters.")
            return

        # Validate that download URLs are present
        if "data_url" not in df_metadata.columns:
            print("Error: 'data_url' column missing from API response.")
            return

        # Filter out rows with missing URLs
        urls = [u for u in df_metadata["data_url"] if pd.notna(u)]
        total_files = len(urls)
        print(f"Starting parallel download of {total_files} files with {workers} threads...")

        # Counters for the final summary
        counters: dict[str, int] = {"downloaded": 0, "skipped": 0, "failed": 0}
        failed_urls: list[str] = []

        # Parallel downloads with a tqdm progress bar
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_url = {executor.submit(download_file, url, cyclobs_root): url for url in urls}

            with tqdm(total=total_files, unit="file", desc="Downloading") as pbar:
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    result = future.result()
                    pbar.update(1)

                    # Tally result by category
                    if result == "downloaded":
                        counters["downloaded"] += 1
                    elif result == "skipped":
                        counters["skipped"] += 1
                    else:
                        counters["failed"] += 1
                        failed_urls.append(f"{url}: {result}")
                        tqdm.write(f"Failed: {url} — {result}")

        print(
            f"\nFinished — "
            f"{counters['downloaded']:,} downloaded, "
            f"{counters['skipped']:,} skipped (already present), "
            f"{counters['failed']:,} failed."
        )

        if failed_urls:
            print("Failed URLs:")
            for msg in failed_urls:
                print(f"  {msg}")

        # Archive the raw CyclObs directory to STORE
        submit_archive_job(
            cyclobs_root,
            Path(cfg["paths"]["archives"]["raw_cyclobs"]),
            cfg,
            job_name="archive_raw_cyclobs",
        )


@hydra.main(config_path="../../../conf/", config_name="preproc", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    """Download CyclObs SAR data to the path configured in paths.raw_datasets.cyclobs."""
    cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    cfg = cast(dict[str, Any], cfg)

    job = DownloadJob(cfg)

    # submitit=false → run directly (local debug); otherwise submit to SLURM
    launch_local = not bool(cfg.get("submitit", False))
    if launch_local:
        job()
    else:
        executor = make_executor(cfg, "download_sar")
        submitted = executor.submit(job)
        print(f"Submitted job {submitted.job_id}")


if __name__ == "__main__":
    main()
