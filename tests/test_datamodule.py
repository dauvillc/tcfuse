"""Unit tests for TCWindowDataModule device transfer."""

import pytest
import torch

from tcfuse.data.collate import WindowBatch
from tcfuse.lightning.datamodule import TCWindowDataModule
from tests.test_sources import make_batched_scalar_source


def _make_window_batch() -> WindowBatch:
    """Minimal WindowBatch with one batched scalar source."""
    src = make_batched_scalar_source(B=2)
    key = ("buoy", 0)
    return WindowBatch(
        sources={key: src},
        is_target={key: torch.tensor([False, False], dtype=torch.bool)},
        sample_ids=["sample_a", "sample_b"],
        window_ref_times_utc=["2020-08-01T12:00:00", "2020-08-01T18:00:00"],
        window_start_times_utc=["2020-08-01T06:00:00", "2020-08-01T12:00:00"],
        window_end_times_utc=["2020-08-01T18:00:00", "2020-08-02T00:00:00"],
        sids=["2020123N12345", "2020456N67890"],
        seasons=[2020, 2020],
        basins=["NA", "NA"],
        subbasins=["GM", "GM"],
        usa_atcf_ids=["AL012020", "AL022020"],
    )


class TestTransferBatchToDevice:
    def test_moves_sources_to_cpu(self) -> None:
        dm = TCWindowDataModule("/tmp", "test_windows", dataloader_kwargs={"batch_size": 2})
        batch = _make_window_batch()
        moved = dm.transfer_batch_to_device(batch, torch.device("cpu"), 0)
        assert isinstance(moved, WindowBatch)
        assert moved.sample_ids == batch.sample_ids
        src = moved.sources[("buoy", 0)]
        assert src.values.device.type == "cpu"
        assert src.coords.device.type == "cpu"
        assert src.mask.device.type == "cpu"
        assert src.time.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_moves_sources_to_cuda(self) -> None:
        dm = TCWindowDataModule("/tmp", "test_windows", dataloader_kwargs={"batch_size": 2})
        batch = _make_window_batch()
        moved = dm.transfer_batch_to_device(batch, torch.device("cuda"), 0)
        assert isinstance(moved, WindowBatch)
        assert moved.sample_ids == batch.sample_ids
        src = moved.sources[("buoy", 0)]
        assert src.values.is_cuda
        assert src.coords.is_cuda
        assert src.mask.is_cuda
        assert src.time.is_cuda
