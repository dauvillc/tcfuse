# /architecture — TC-Fuse Model Architecture Agent

Source of truth: [`.agents/architecture.md`](../../.agents/architecture.md).

This command activates the TC-Fuse architecture skill. **Before working in `src/tcfuse/models/`, adding or benchmarking a backbone, or changing the embedding/encoder/decoder interface**, read the skill file. The embedding/un-embedding layer contracts, encoder interface, candidate architectures, and pre-training task are defined there.

Preprocessing pipeline: [`/preprocess`](preprocess.md). Jean-Zay submission: [`/jz`](jz.md).

Keep docs in sync: when the embedding/encoder/decoder interface or candidate architecture list changes, update `.agents/architecture.md` and this file together; update the architecture pointer in `.agents/context.md` if the section moves again.

---

## Quick pointer

| Need | Start here (in architecture.md) |
|---|---|
| Overall design philosophy + embedding/encoder/decoder diagram | "Architecture philosophy" |
| `EmbeddedSource`/`EmbeddedBatch`, `MultiSourceEncoder` contract | "The embedding layer (implemented)" |
| `DecodedBatch`, `ScalarDecoder`/`ProfileDecoder`/`FieldDecoder`, `MultiSourceDecoder` contract | "The un-embedding layer (implemented)" |
| Encoder input/output shapes, Hydra `_partial_` instantiation | "The encoder interface" |
| Backbones to benchmark (Perceiver, cross-attention Transformer, windowed attention) | "Candidate architectures to benchmark" |
| Masked-source reconstruction pre-training objective | "Self-supervised pre-training task" |
