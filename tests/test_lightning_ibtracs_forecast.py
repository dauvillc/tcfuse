"""Tests for IBTrACSForecastLightningModule skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from torch import nn
from torch.optim import AdamW

from tcfuse.data.sources.metadata import MultisourceMetadata, SourceMetadata
from tcfuse.data.sources.source import SourceKind
from tcfuse.lightning.ibtracs_forecast import IBTrACSForecastLightningModule
from tcfuse.lightning.lr_scheduler import CosineAnnealingWarmupRestarts


def _sample_multisource_metadata() -> MultisourceMetadata:
    """Minimal metadata for Lightning module construction tests."""
    return MultisourceMetadata(
        sources={
            "pmw_amsr2_gcomw1": SourceMetadata(
                name="pmw_amsr2_gcomw1",
                type="microwave",
                kind=SourceKind.FIELD,
                channels=["tb_36.5h"],
                shape=(10, 10),
            )
        }
    )


def test_configure_optimizers_returns_adamw_and_scheduler(tmp_path: Path) -> None:
    """configure_optimizers wires AdamW and CosineAnnealingWarmupRestarts."""
    model = nn.Linear(2, 1)
    metadata = _sample_multisource_metadata()
    module = IBTrACSForecastLightningModule(
        model=model,
        sources_metadata=metadata,
        adamw_kwargs={"lr": 1.0e-4, "weight_decay": 0.01},
        lr_scheduler_kwargs={
            "first_cycle_steps": 10,
            "warmup_steps": 2,
            "max_lr": 1.0e-4,
            "min_lr": 1.0e-6,
        },
        validation_dir=tmp_path / "val_figures",
    )

    assert module.model is model

    raw_config = module.configure_optimizers()
    assert isinstance(raw_config, dict)
    config = cast(dict[str, Any], raw_config)
    optimizer = config["optimizer"]
    assert isinstance(optimizer, AdamW)

    lr_block = config["lr_scheduler"]
    assert isinstance(lr_block, dict)
    scheduler = lr_block["scheduler"]
    assert isinstance(scheduler, CosineAnnealingWarmupRestarts)
    assert lr_block.get("interval") == "step"

    # Linear weight is 2D (decay); bias is 1D (no decay).
    decay_group = next(g for g in optimizer.param_groups if g["weight_decay"] == 0.01)
    no_decay_group = next(g for g in optimizer.param_groups if g["weight_decay"] == 0.0)
    weight_param = next(p for n, p in module.named_parameters() if n.endswith("weight"))
    bias_param = next(p for n, p in module.named_parameters() if n.endswith("bias"))
    assert weight_param in decay_group["params"]
    assert bias_param in no_decay_group["params"]


def test_sources_metadata_property_returns_snapshot(tmp_path: Path) -> None:
    """sources_metadata property returns a defensive copy."""
    metadata = _sample_multisource_metadata()
    module = IBTrACSForecastLightningModule(
        model=nn.Linear(1, 1),
        sources_metadata=metadata,
        adamw_kwargs={"lr": 1.0e-4},
        lr_scheduler_kwargs={"first_cycle_steps": 10, "warmup_steps": 0, "max_lr": 1.0e-4},
        validation_dir=tmp_path,
    )
    restored = module.sources_metadata
    assert restored.to_dict() == metadata.to_dict()
    restored.sources["pmw_amsr2_gcomw1"] = metadata.sources["pmw_amsr2_gcomw1"]
    assert module.sources_metadata.to_dict() == metadata.to_dict()
