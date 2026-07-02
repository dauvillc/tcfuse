---
name: tcfuse-motif
description: >-
  TC-Fuse MoTiF backbone — the diffusion-transformer-style architecture where
  each source is embedded into separate value and coordinate tokens
  (encoders_motif) and the coordinate tokens condition every layer. Covers the
  block design (cross-source attention + spatial self-attention + MLP), what is
  implemented (dual embedding, windowed cross-source attention with
  relative positional bias, value-only decoders) versus missing (self-attention,
  MLP, block, backbone, config, PROFILE/SCALAR cross-source), the
  value/coord dict layer interface, and MoTiF conventions (RMSNorm QK-norm,
  RPB not RoPE, fully self-contained packages). Use when working in
  `src/tcfuse/models/motif/`, `encoders_motif/`, or `decoders_motif/`, or adding
  any MoTiF layer, block, backbone, or config.
---

> **Content lives in [`.agents/motif.md`](../../.agents/motif.md).**
> Read that file for the MoTiF architecture overview, the implemented vs. missing components, the
> data structures (`MotifEmbeddedSource`/`MotifDecodedSource`), the cross-source attention mechanism,
> the extension points, and the MoTiF-specific design decisions and caveats.
>
> Project-wide, architecture-agnostic contracts (embedding/encoder/decoder philosophy, encoder
> interface, pre-training task) are in [`.agents/architecture.md`](../../.agents/architecture.md).
