"""Generate the Jean-Zay pip requirements overlay from Pixi metadata."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIXI_TOML = PROJECT_ROOT / "pixi.toml"
DEFAULT_OUTPUT = PROJECT_ROOT / "requirements-jz.txt"

MODULE_PROVIDED_PACKAGES = {
    "python",
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "matplotlib",
    "pillow",
}

LOCAL_OR_DEV_ONLY_PACKAGES = {
    "tcfuse",
    "jupyterlab",
    "ipykernel",
    "basedpyright",
    "pre-commit",
    "pytest",
    "ruff",
}

EXCLUDED_PACKAGES = MODULE_PROVIDED_PACKAGES | LOCAL_OR_DEV_ONLY_PACKAGES

SECTION_PACKAGES = {
    "ML / training": ("lightning", "timm", "einops"),
    "Config / experiment management": (
        "hydra-core",
        "hydra-submitit-launcher",
        "submitit",
    ),
    "Logging": ("wandb",),
    "Data / geospatial": (
        "xarray",
        "zarr",
        "h5py",
        "netcdf4",
        "pandas",
        "pyresample",
        "cartopy",
        "cmocean",
        "boto3",
        "dask",
        "polars",
    ),
}


def _requirement_from_spec(name: str, spec: Any) -> str | None:
    """Convert one Pixi dependency entry to a pip requirement line."""
    if name in EXCLUDED_PACKAGES:
        return None

    # Pixi uses either plain version strings or tables for PyPI dependencies.
    if isinstance(spec, str):
        version = spec
    elif isinstance(spec, dict):
        version = spec.get("version")
    else:
        msg = f"Unsupported dependency spec for {name!r}: {spec!r}"
        raise TypeError(msg)

    if version in (None, "*"):
        return name

    return f"{name}{str(version).replace(' ', '')}"


def _load_requirements(pixi_toml: Path) -> dict[str, str]:
    """Load requirements generated from Pixi conda and PyPI dependency tables."""
    with pixi_toml.open("rb") as file:
        pixi_config = tomllib.load(file)

    requirements: dict[str, str] = {}

    # Merge conda and PyPI declarations into one pip-compatible overlay.
    for table_name in ("dependencies", "pypi-dependencies"):
        dependencies = pixi_config.get(table_name, {})
        for name, spec in dependencies.items():
            requirement = _requirement_from_spec(name, spec)
            if requirement is not None:
                requirements[name] = requirement

    return requirements


def render_requirements(pixi_toml: Path) -> str:
    """Render the Jean-Zay requirements file content from a Pixi manifest."""
    requirements = _load_requirements(pixi_toml)
    emitted_packages: set[str] = set()

    lines = [
        "# This file is generated from pixi.toml by scripts/export_jz_requirements.py.",
        "# Do not edit it by hand; run: pixi run export-jz-requirements",
        "# Jean-Zay-specific packages to pip install --user after loading modules.",
        "# The pytorch-gpu/py3/2.8.0 module already provides: torch, torchvision,",
        "# numpy, scipy, matplotlib, Pillow, and CUDA libraries.",
        "# Run: pip install --user -r requirements-jz.txt",
        "",
    ]

    # Emit stable, readable sections matching the project dependency groups.
    for section_name, package_names in SECTION_PACKAGES.items():
        section_requirements = [
            requirements[package_name]
            for package_name in package_names
            if package_name in requirements
        ]
        if not section_requirements:
            continue

        lines.append(f"# {section_name}")
        lines.extend(section_requirements)
        lines.append("")
        emitted_packages.update(package_names)

    remaining_requirements = [
        requirement
        for package_name, requirement in sorted(requirements.items())
        if package_name not in emitted_packages
    ]
    if remaining_requirements:
        lines.append("# Other")
        lines.extend(remaining_requirements)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate the Jean-Zay pip requirements overlay from pixi.toml.",
    )
    parser.add_argument(
        "--pixi-toml",
        type=Path,
        default=DEFAULT_PIXI_TOML,
        help="Path to the Pixi manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the generated requirements file.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the output file is stale instead of rewriting it.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate or validate the Jean-Zay requirements file."""
    args = parse_args()
    rendered = render_requirements(args.pixi_toml)

    if args.check:
        current = args.output.read_text() if args.output.exists() else ""
        if current != rendered:
            print(
                f"{args.output} is stale. Run: pixi run export-jz-requirements",
                file=sys.stderr,
            )
            return 1
        return 0

    args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
