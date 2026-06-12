"""Channelwise affine backbone for testing the masked-reconstruction pipeline."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from tcfuse.data.collate import WindowBatch
from tcfuse.data.sources.metadata import MultisourceMetadata
from tcfuse.data.sources.torch_source import TorchSource


class ChannelwiseAffineBackbone(nn.Module):
    """Learn a per-channel scale and shift for every source in the batch.

    A single ``(weight, bias)`` pair of shape ``(C,)`` is registered per source
    name.  The transform ``y = weight * x + bias`` is applied independently at
    each ``(batch, spatial)`` position; the channel axis is the only one with
    learned parameters.  Sources absent from ``sources_metadata`` pass through
    unchanged.

    Weights are initialised to 1 and biases to 0, so the model starts as an
    identity map.  This makes the module useful as a lightweight sanity-check
    backbone: a perfectly-trained model on MSE reconstruction loss should
    converge weights to 1 and biases to 0 in normalised space.

    Args:
        sources_metadata: Static descriptors for all sources in the dataset.
            Used to determine the number of channels per source at construction
            time.
    """

    def __init__(self, sources_metadata: MultisourceMetadata) -> None:
        super().__init__()
        weights: dict[str, nn.Parameter] = {}
        biases: dict[str, nn.Parameter] = {}
        # Map original source name -> sanitized ParameterDict key.
        # ParameterDict does not allow '.' or '-' in keys.
        self._key_map: dict[str, str] = {}
        for name in sources_metadata.names:
            key = name.replace(".", "_").replace("-", "_")
            C = sources_metadata[name].num_channels
            weights[key] = nn.Parameter(torch.ones(C))
            biases[key] = nn.Parameter(torch.zeros(C))
            self._key_map[name] = key
        self._weights = nn.ParameterDict(weights)
        self._biases = nn.ParameterDict(biases)

    def forward(self, batch: WindowBatch) -> WindowBatch:
        """Apply per-channel affine transform to every source in the batch.

        Args:
            batch: Collated window batch (values expected in normalised space).

        Returns:
            A new :class:`~tcfuse.data.collate.WindowBatch` with transformed
            source values; masks, coords, and all other fields are unchanged.
        """
        # Shallow-copy so untouched sources share tensors with the original batch.
        new_sources: dict[tuple[str, int], TorchSource] = {}
        for key, source in batch.sources.items():
            source_name, _idx = key
            if source_name not in self._key_map:
                # Source was not registered at construction — pass through.
                new_sources[key] = source
                continue
            param_key = self._key_map[source_name]
            weight = self._weights[param_key]  # (C,)
            bias = self._biases[param_key]  # (C,)
            # values is (B, *spatial, C); (C,) broadcasts from the right.
            new_values = source.values * weight + bias
            new_sources[key] = dataclasses.replace(source, values=new_values)
        return dataclasses.replace(batch, sources=new_sources)
