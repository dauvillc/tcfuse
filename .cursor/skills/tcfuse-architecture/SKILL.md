---
name: tcfuse-architecture
description: >-
  TC-Fuse model backbone design — the architecture-agnostic embedding layer
  (TorchSource → EmbeddedSource via MultiSourceEncoder) and un-embedding
  layer (EmbeddedBatch → DecodedBatch via MultiSourceDecoder), the swappable
  encoder interface, candidate backbones to benchmark (Perceiver,
  cross-attention Transformer, hierarchical windowed attention), and the
  masked-source-reconstruction self-supervised pre-training task. Use when
  working in `src/tcfuse/models/`, adding or benchmarking a backbone, or
  changing the encoder/decoder interface.
---

> **Content has moved.** The full skill documentation is in [`.agents/architecture.md`](../../.agents/architecture.md).
> Read that file for the embedding/un-embedding layer contracts, the encoder interface, candidate architectures, and the self-supervised pre-training task.
