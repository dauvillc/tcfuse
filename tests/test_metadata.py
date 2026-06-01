"""Unit tests for SourceMetadata and MultisourceMetadata YAML I/O."""

from __future__ import annotations

from pathlib import Path

import yaml

from tcfuse.data.sources.metadata import MultisourceMetadata, SourceMetadata
from tcfuse.data.sources.source import SourceKind


def _sample_source_meta(name: str = "pmw_amsr2_gcomw1") -> SourceMetadata:
    """Build a minimal FIELD source metadata object for tests."""
    return SourceMetadata(
        name=name,
        type="microwave",
        kind=SourceKind.FIELD,
        channels=["tb_36.5h", "tb_36.5v"],
        shape=(400, 400),
        char_vars={"ifov": {"tb_36.5h": [7.2, 4.4]}},
    )


def test_source_metadata_to_dict_from_dict_roundtrip() -> None:
    """to_dict / from_dict preserve all descriptor fields."""
    original = _sample_source_meta()
    restored = SourceMetadata.from_dict(original.to_dict())

    assert restored.name == original.name
    assert restored.type == original.type
    assert restored.kind == original.kind
    assert restored.channels == original.channels
    assert restored.shape == original.shape
    assert restored.char_vars == original.char_vars
    assert restored.num_channels == 2
    assert restored.num_tokens == 400 * 400


def test_source_metadata_yaml_roundtrip(tmp_path: Path) -> None:
    """to_yaml / from_yaml roundtrip through a file on disk."""
    yaml_path = tmp_path / "metadata.yaml"
    original = _sample_source_meta()
    original.to_yaml(yaml_path)

    restored = SourceMetadata.from_yaml(yaml_path)
    assert restored.to_dict() == original.to_dict()


def test_multisource_metadata_from_multiple_yaml(tmp_path: Path) -> None:
    """from_multiple_yaml unions per-source YAML files."""
    source_a = _sample_source_meta("pmw_amsr2_gcomw1")
    source_b = SourceMetadata(
        name="ir_tcirar",
        type="infrared",
        kind=SourceKind.FIELD,
        channels=["tb"],
        shape=(401, 401),
    )

    path_a = tmp_path / "pmw_amsr2_gcomw1" / "metadata.yaml"
    path_b = tmp_path / "ir_tcirar" / "metadata.yaml"
    source_a.to_yaml(path_a)
    source_b.to_yaml(path_b)

    multi = MultisourceMetadata.from_multiple_yaml([path_a, path_b])
    assert len(multi) == 2
    assert multi.names == ["pmw_amsr2_gcomw1", "ir_tcirar"]
    assert multi["ir_tcirar"].type == "infrared"


def test_multisource_metadata_yaml_roundtrip(tmp_path: Path) -> None:
    """to_yaml / from_yaml roundtrip for merged multi-source metadata."""
    multi = MultisourceMetadata(
        sources={
            "pmw_amsr2_gcomw1": _sample_source_meta(),
            "ir_tcirar": SourceMetadata(
                name="ir_tcirar",
                type="infrared",
                kind=SourceKind.FIELD,
                channels=["tb"],
                shape=(401, 401),
            ),
        }
    )

    yaml_path = tmp_path / "sources_metadata.yaml"
    multi.to_yaml(yaml_path)

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    assert set(raw.keys()) == {"pmw_amsr2_gcomw1", "ir_tcirar"}

    restored = MultisourceMetadata.from_yaml(yaml_path)
    assert restored.to_dict() == multi.to_dict()


def test_multisource_metadata_has_no_index_property() -> None:
    """MultisourceMetadata does not expose a merged snapshot index."""
    multi = MultisourceMetadata(sources={"pmw_amsr2_gcomw1": _sample_source_meta()})
    assert not hasattr(multi, "index")
