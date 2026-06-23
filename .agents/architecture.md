# TC-Fuse model architecture

Claude Code: invoke `/architecture` (reads this skill).

Read this before touching `src/tcfuse/models/`, adding or benchmarking a backbone, or changing the embedding/encoder/decoder interface.

## Architecture philosophy

The framework is **architecture-agnostic at the backbone level**. The embedding layer (value + coordinate → token) and the task heads (decoder) are fixed interfaces; the encoder between them is swappable.

```
[Source 1: values + coords] ──┐
[Source 2: values + coords] ──┼──► [Source Embeddings] ──► [Encoder (swappable)] ──► [Task Head]
[Source N: values + coords] ──┘
```

**The embedding layer (implemented):** `tcfuse.models.encoders` patch-embeds each `TorchSource` into an `EmbeddedSource` carrying a single `features` tensor — `(B, D)` SCALAR, `(B, El, D)` PROFILE, `(B, Eh, Ew, D)` FIELD, with `El/Eh/Ew = L/H/W // patch_size`. `MultiSourceEncoder` (Hydra `_partial_: true`, allocates one per-source encoder from `sources_metadata`, mirrors `ChannelwiseAffineBackbone`) maps a `WindowBatch` → `EmbeddedBatch` (`sources` + `is_target` only; coords/mask/time dropped). Current encoders are **value-only**; coordinate/time encoding and a token-validity mask are deferred follow-ups.

**The un-embedding layer (implemented):** `tcfuse.models.decoders` is the symmetric inverse, mapping an `EmbeddedBatch` → `DecodedBatch` (`DecodedSource.values` matching the original `TorchSource.values` layout; `is_target` passed through). `ScalarDecoder`/`ProfileDecoder` are exact shape inverses (`nn.Linear` / `nn.ConvTranspose1d`) of their encoders; `FieldDecoder` instead uses sub-pixel convolution (`Conv2d` + `PixelShuffle`) with ICNR-initialized weights to avoid checkerboard artifacts on image-like FIELD data. `MultiSourceDecoder` mirrors `MultiSourceEncoder`'s Hydra-partial/per-source-allocation pattern. Not yet wired into `BaseLightningModule` or any task head's cross-source querying mechanism.

## The encoder interface

- Input: a list of token sequences, one per source, each of shape `(B, N_i, D)` where `N_i` is the number of tokens for source `i` and `D` is the embedding dimension.
- Output: a representation that the task head can query — exact form depends on architecture (latent array for Perceiver, CLS token for ViT-style, etc.).
- The encoder must be instantiable from a Hydra config node, using `_partial_: true` so that `BaseLightningModule` can pass `sources_metadata` to the constructor at runtime (allowing backbones to allocate per-source parameters from channel counts). The model config lives in `conf/model/` and is imported into the lightning module config via a defaults package-override entry: `- /model@model: <name>`.

## Candidate architectures to benchmark

Extend as needed:

- Perceiver / Perceiver IO.
- Cross-attention Transformer (queries from anchor points or task positions).
- Hierarchical windowed attention (Swin-style, per source + cross-source).

## Self-supervised pre-training task

Randomly mask one source at training time; reconstruct its values from all remaining sources, using only its coordinates and instrument metadata as queries. This is the default pre-training objective. Supervised fine-tuning follows for specific tasks.
