"""Spatio-temporal Fourier positional encoding for source tokens.

Encodes each token's physical coordinates (lat, lon, optional altitude/depth)
and time with deterministic axial sinusoidal Fourier features, then projects
the concatenated features to the embedding dimension so they can be **added**
to the value-only token embedding (standard additive positional encoding).

Frequencies are log-spaced (geometric) within an interpretable wavelength range
per axis group, so low frequencies capture long ranges and high frequencies
capture short differences. Wavelengths are expressed in each axis' physical
units, so no separate coordinate normalization is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

# Axis-group identifiers. Each group has its own wavelength range because the
# axes carry different physical units (degrees vs meters vs normalized time).
ANGULAR = "angular"  # latitude / longitude, in degrees
VERTICAL = "vertical"  # altitude / depth, in meters
TEMPORAL = "temporal"  # day-of-year and minute-of-day fractions, in [0, 1]

# Time is always two axes: [day_of_year / 366, minute_of_day / 1440].
_TEMPORAL_AXIS_GROUPS = (TEMPORAL, TEMPORAL)


@dataclass(frozen=True)
class CoordEncodingConfig:
    """Hyperparameters for :class:`SpatioTemporalEncoding`.

    Bundled into one object so backbones can thread a single config through to
    the embedding layer instead of many loose scalars.

    Args:
        enabled: Master switch. When ``False`` the embedding layer skips
            coordinate encoding entirely (value-only tokens).
        num_frequencies: Number of log-spaced frequency bands per axis. Each band
            contributes a sin and a cos feature, so an axis yields
            ``2 * num_frequencies`` features.
        angular_min_wavelength: Shortest wavelength for lat/lon axes (degrees).
        angular_max_wavelength: Longest wavelength for lat/lon axes (degrees).
        vertical_min_wavelength: Shortest wavelength for altitude/depth (meters).
        vertical_max_wavelength: Longest wavelength for altitude/depth (meters).
        temporal_min_wavelength: Shortest wavelength for time axes (normalized).
        temporal_max_wavelength: Longest wavelength for time axes (normalized).
    """

    enabled: bool = True
    num_frequencies: int = 16
    angular_min_wavelength: float = 0.1
    angular_max_wavelength: float = 360.0
    vertical_min_wavelength: float = 50.0
    vertical_max_wavelength: float = 20000.0
    temporal_min_wavelength: float = 0.01
    temporal_max_wavelength: float = 1.0

    def wavelength_range(self, group: str) -> tuple[float, float]:
        """Return the ``(min_wavelength, max_wavelength)`` for an axis group."""
        if group == ANGULAR:
            return self.angular_min_wavelength, self.angular_max_wavelength
        elif group == VERTICAL:
            return self.vertical_min_wavelength, self.vertical_max_wavelength
        else:  # TEMPORAL
            return self.temporal_min_wavelength, self.temporal_max_wavelength


def _log_spaced_frequencies(num_frequencies: int, min_wl: float, max_wl: float) -> Tensor:
    """Log-spaced angular frequencies for wavelengths in ``[min_wl, max_wl]``.

    Returns:
        Float tensor ``(num_frequencies,)`` of frequencies ``1 / wavelength``,
        ordered from highest frequency (shortest wavelength) to lowest.
    """
    # Geometric (log-spaced) wavelengths between the configured bounds.
    wavelengths = torch.logspace(math.log10(min_wl), math.log10(max_wl), num_frequencies)
    # Frequency is the reciprocal of wavelength.
    return 1.0 / wavelengths


class SpatioTemporalEncoding(nn.Module):
    """Additive Fourier positional encoding over spatial + temporal coordinates.

    Builds, per coordinate axis, a fixed bank of sin/cos features from log-spaced
    frequencies, concatenates them across axes, and projects to ``embed_dim`` with
    a single learnable linear layer. The output is meant to be summed onto the
    value-only token features.

    Args:
        embed_dim: Output embedding dimension D (matches the token features).
        spatial_axis_groups: Group identifier for each spatial coordinate axis,
            in coordinate order. SCALAR / FIELD use ``[ANGULAR, ANGULAR]`` (lat,
            lon); PROFILE uses ``[ANGULAR, ANGULAR, VERTICAL]`` (lat, lon, alt).
            The two temporal axes are appended automatically.
        config: Frequency / wavelength hyperparameters.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        spatial_axis_groups: list[str],
        config: CoordEncodingConfig,
    ) -> None:
        super().__init__()
        self.num_spatial_axes = len(spatial_axis_groups)
        self.num_frequencies = config.num_frequencies
        # Full axis order: spatial coordinate axes first, then the two time axes.
        axis_groups = list(spatial_axis_groups) + list(_TEMPORAL_AXIS_GROUPS)
        # Stack one log-spaced frequency bank per axis into a (num_axes, num_freq) buffer.
        freqs_per_axis = [
            _log_spaced_frequencies(config.num_frequencies, *config.wavelength_range(group))
            for group in axis_groups
        ]
        # Registered as a non-trainable buffer: moves with the module, no gradient.
        # Annotated so the type checker knows attribute access yields a Tensor.
        self.frequencies: Tensor
        self.register_buffer("frequencies", torch.stack(freqs_per_axis, dim=0))
        # Each axis yields 2 * num_frequencies features (sin and cos).
        num_features = len(axis_groups) * 2 * config.num_frequencies
        # Project the concatenated Fourier features down to the embedding dim.
        self.proj = nn.Linear(num_features, embed_dim)

    def forward(self, coords: Tensor, time: Tensor) -> Tensor:
        """Encode per-token coordinates and per-sample time into ``(..., D)``.

        Args:
            coords: Per-token spatial coordinates, shape ``(B, *spatial, S)`` where
                S == ``num_spatial_axes``. May carry NaN at missing/masked slots.
            time: Per-sample time encoding, shape ``(B, 2)``. May carry NaN.

        Returns:
            Positional embedding of shape ``(B, *spatial, D)``, ready to be added
            to the token features.
        """
        # Match the buffer/projection dtype (coords may arrive as float64 from numpy,
        # or the module may be in half precision under AMP).
        coords = coords.to(self.frequencies.dtype)
        time = time.to(self.frequencies.dtype)
        # Zero NaN-fill at missing/masked slots so the encoding stays finite; those
        # tokens are masked out downstream anyway.
        coords = torch.nan_to_num(coords, nan=0.0)
        time = torch.nan_to_num(time, nan=0.0)
        # Broadcast the per-sample time (B, 2) across every token's spatial dims.
        # coords is (B, *spatial, S); time must become (B, *spatial, 2).
        num_spatial_dims = coords.ndim - 2
        # Insert singleton axes between batch and the time channel, then expand.
        time_shape = (time.shape[0],) + (1,) * num_spatial_dims + (time.shape[1],)
        time_expanded = time.view(time_shape).expand(*coords.shape[:-1], time.shape[1])
        # Concatenate spatial coords and time into a single (..., num_axes) tensor.
        full_coords = torch.cat([coords, time_expanded], dim=-1)
        # Angles: (..., num_axes, num_freq) = coord * frequency * 2*pi.
        angles = full_coords.unsqueeze(-1) * self.frequencies * (2.0 * math.pi)
        # Stack sin and cos along the frequency axis -> (..., num_axes, 2*num_freq).
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        # Flatten the (num_axes, 2*num_freq) feature block -> (..., num_features).
        fourier = fourier.flatten(start_dim=-2)
        # Project to the embedding dimension for additive combination.
        return self.proj(fourier)
