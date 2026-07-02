# TC-Fuse MoTiF backbone

Claude Code: invoke `/motif` (reads this skill).

Read this **before touching `src/tcfuse/models/motif/`, `encoders_motif/`, or `decoders_motif/`**, or
before adding any MoTiF layer, block, backbone, or config. It is the single source of truth for the
MoTiF architecture: what it is, what is implemented, what is missing, and where to extend. For the
project-wide, architecture-agnostic contracts (embedding/encoder/decoder philosophy, the swappable
encoder interface, the pre-training task) see [`.agents/architecture.md`](architecture.md); this skill
covers only the MoTiF-specific stack.

## What MoTiF is

MoTiF is a **diffusion-transformer-style** backbone for multi-source geospatial (tropical-cyclone)
data. Its defining choice: each source is embedded into **two** token tensors — a *value* embedding and
a **standalone** *coordinate* embedding — and the coordinate tokens are injected as positional
conditioning at **every** layer, rather than added once to the values (as the baseline backbones do).
This keeps positional information available as an explicit, per-layer signal (relative positional bias,
shift/scale conditioning) throughout the network.

The backbone chains **blocks**; each block chains **three layers**:

1. **Cross-source attention** — every source reads from the *other* sources at a coarse (windowed)
   spatial resolution while keeping its own full resolution on the value side.
2. **Spatial self-attention** — within-source (Swin-style windowed) attention. *(not yet implemented)*
3. **MLP** — position-wise feed-forward. *(not yet implemented)*

Coordinate tokens condition each layer (relative positional bias today; shift/scale conditioning is the
intended block-level mechanism).

## MoTiF-specific packages (deliberate redundancy)

MoTiF lives in three **self-contained** packages that mirror the baseline `encoders/` /
`decoders/` but never import from them, so MoTiF can evolve without touching the baselines:

| Package | Role | Status |
|---|---|---|
| `src/tcfuse/models/encoders_motif/` | `TorchSource` → dual value/coord tokens (`MotifEmbeddedSource`) | **implemented** |
| `src/tcfuse/models/motif/` | the backbone: layers, block, and top-level model | **partial** (cross-source attention only) |
| `src/tcfuse/models/decoders_motif/` | value tokens → raw values (`MotifDecodedSource`), coords ignored | **implemented** |

## Data structures

- **`MotifEmbeddedSource`** (`encoders_motif/embedded.py`): `.values`, `.coords` (same token layout,
  independent last dims `Dv` / `Dc`), `.kind`, `.source_name`, `.input_shape`. FIELD values are
  `(B, Eh, Ew, Dv)`; PROFILE `(B, El, Dv)`; SCALAR `(B, Dv)`. Validates the shared-spatial-dims
  contract (values/coords agree on all dims but the last).
- **`MotifEmbeddedBatch`**: `.sources: dict[(name, index) → MotifEmbeddedSource]`, `.is_target`.
- **`MotifDecodedSource` / `MotifDecodedBatch`** (`decoders_motif/decoded.py`): coord-free value
  containers; the un-embedding counterpart. Structurally identical to the baseline `DecodedBatch`
  (a downstream task head consuming the backbone output must accept it).
- **MoTiF layer interface** (important): the in-backbone layers do **not** pass `MotifEmbeddedSource`
  around. They take two plain dicts — `values: dict[key → Tensor]` and `coords: dict[key → Tensor]` —
  so a block can feed LN/shift/scale-conditioned tokens (not the raw dataclass) and the layer stays
  decoupled and directly testable. Layers return updated **values** only; the **residual is the
  block's responsibility**.

## Implemented components

### Embedding — `encoders_motif/` (all three kinds)
`MotifMultiSourceEncoder` (Hydra-partial, per-source allocation from `sources_metadata`) dispatches to
`MotifScalarEncoder` / `MotifProfileEncoder` / `MotifFieldEncoder`. Each patch-embeds values into a
**value-only** tensor (no positional term) and separately builds a standalone Fourier coordinate tensor
via `CoordEmbedding`. Per-pixel/level coords are average-pooled to patch centers first. NaN-fill in
values/coords/time is zeroed inside the encoder. Knobs live in `MotifCoordEncodingConfig`
(`num_frequencies=16`, per-group log-spaced wavelength ranges for angular/vertical/temporal axes; **no**
`enabled` switch — the separate coord tensor is the architecture's premise).

### Cross-source attention — `motif/cross_source_attention.py` : `MultiSourceCrossAttention`
The **first** of the three block layers. **FIELD (2-D) sources only** in this version. Mechanism:

- Per source, tile value & coord tokens into `window_size × window_size` windows (0-padding the grid up
  to a whole number of windows via `F.pad`).
- **Pool** (mean over window pixels) the value & coord tokens → one token per window; project and flatten
  to pooled query/key sequences. Token count is divided by `window_size²`.
- Concatenate pooled Q/K of **all** sources into one cross-source sequence → a single attention matrix
  mixes information across sources.
- The **value** tokens are **not** averaged (too lossy): each window's pixels are **stacked** along the
  embedding dim (`Dv → window_size² · value_inner_dim`), so `V' = A·V` keeps full resolution. `V'` is
  un-stacked and cropped back to each source's original `(Eh, Ew)` grid.
- Optional block-diagonal **`mask_self_attention`** (default `True`): mask same-source pairs so a source
  attends only *across* sources in this layer.

Constructor: `MultiSourceCrossAttention(*, dim, inner_dim, window_size, num_heads, coord_dim,
coord_inner_dim, value_inner_dim=None, mask_self_attention=True, dropout=0.0)`. QK-norm uses
`torch.nn.RMSNorm` directly. **Divisibility:** `num_heads` must divide `inner_dim`, `coord_inner_dim`,
and `window_size² · value_inner_dim` (the stacked value width).

### Low-level attention — `motif/attention.py` : `SpatiotemporalAttention`
Fuses feature scores with an α-weighted coordinate-score bias in **one** fused
`F.scaled_dot_product_attention` call, implementing

```
V_out = softmax( Qf·Kfᵀ/√d_f  +  α · Qc·Kcᵀ/√d_c ) · V
```

The coordinate term is computed here (scaled by learnable scalar `alpha`, **init 0** so the layer starts
as pure feature attention), and passed as SDPA's additive float `attn_mask`; the structural keep-mask is
folded in as `-inf`. This is a **relative positional bias (RPB)** — **no RoPE** (explicitly out of scope).
No output projection (the cross-source layer un-stacks and projects values itself). Only learnable
parameter: `alpha`.

### Un-embedding — `decoders_motif/` (all three kinds)
`MotifMultiSourceDecoder` (Hydra-partial) dispatches to `MotifScalarDecoder` (`nn.Linear`),
`MotifProfileDecoder` (`ConvTranspose1d`, exact inverse), `MotifFieldDecoder` (`Conv2d` +
`PixelShuffle` + ICNR init, to avoid checkerboard artifacts). Each reads only `embedded.values`
(**coords ignored**) and crops back to `input_shape`. Identical decode math to the baseline decoders.

## What's missing (extension points, roughly in build order)

1. **Spatial self-attention layer** (block layer 2) — within-source windowed (Swin-style) attention.
2. **MLP layer** (block layer 3) — position-wise feed-forward.
3. **`MotifBlock`** — wire the three layers with pre-LN and, per the diffusion-transformer premise,
   **coordinate-conditioned shift/scale (and gating)** around each sub-layer, plus the residuals.
4. **`MotifBackbone`** — own a `MotifMultiSourceEncoder` + `MotifMultiSourceDecoder`, stack blocks, and
   satisfy `WindowBatch → WindowBatch` (like the other backbones rebuild the batch, replacing values).
5. **`conf/model/motif.yaml`** — Hydra config (`_partial_: true`), plus an experiment config.
6. **PROFILE / SCALAR support in cross-source attention** — currently FIELD-only; PROFILE needs 1-D
   windowing, SCALAR is a single token (no windowing).
7. **Wiring into `BaseLightningModule`** and a task head that accepts `MotifDecodedBatch` (same
   deferred-integration status as the baseline decoders).
8. **Token-validity mask** — genuinely-absent sources are zeroed, not masked out of attention (the
   project-wide deferred follow-up; see `.agents/architecture.md`).

## Design decisions & conventions specific to MoTiF

- **Separate coord tensor at every layer** is the premise — never fold coords into values.
- **RPB, not RoPE.** Coordinate conditioning enters as the α-weighted additive score bias (init α = 0).
- **`torch.nn.RMSNorm`** for QK-norm (not a local implementation).
- **Dict-based layer interface** (`values`/`coords` dicts), residual added by the block, so layers are
  conditioning-agnostic and unit-testable.
- **Value tokens are stacked, never pooled**, on the value path — pooling is only for Q/K.
- **Fully self-contained packages** — `motif/`, `encoders_motif/`, `decoders_motif/` import from
  `tcfuse.data.*` and each other, **never** from the baseline `encoders/` / `decoders/`.
- Follow `.agents/context.md` § Human-readable code: a `#` comment before each logical step, one-line
  docstrings + Args/Returns, inline tensor-shape comments, **no** defensive `raise`/`assert` beyond
  documented invariants, no magic numbers in `src/` (all widths flow from constructor args).

## Known caveats & potential improvements

- **Single-source + `mask_self_attention=True` → NaN.** Cross-source attention needs ≥ 2 sources; with
  one source every key is masked and the softmax rows are all `-inf`. The block/backbone must guarantee
  ≥ 2 sources (or add a guard there); the layer deliberately has no defensive check.
- **Uniform stacked value width** is why cross-source attention is FIELD-only today: all windows stack to
  `window_size² · value_inner_dim`. Mixed kinds (differing per-window pixel counts) need a reconciliation
  strategy (e.g. zero-pad slots to a max) before SCALAR/PROFILE can join the same attention matrix.
- **Windowed pooling loses fine cross-source detail** for Q/K by construction; the stacked-value path
  compensates on the value side. Revisit if targets need finer cross-source routing.
- **`MotifDecodedBatch` duplicates `DecodedBatch`.** Whatever eventually consumes backbone output must
  accept the MoTiF type; unify only if a shared task-head interface is introduced.
- **Shift/scale conditioning** (the diffusion-transformer mechanism) is not yet built — only the RPB
  coord path exists. The block is where per-layer coordinate conditioning should be realized.

## Tests

Synthetic-tensor pytest only (no real data), mirroring `tests/test_embeddings_motif.py` conventions:

- `tests/test_embeddings_motif.py` — dual-tensor encoders (shapes, value/coord independence, NaN, dispatch).
- `tests/test_cross_source_attention_motif.py` — windowing round-trip, pad/crop, self-attention mask,
  determinism, finite backward (incl. `alpha`).
- `tests/test_decoders_motif.py` — per-kind shape inverse + crop-back, **coords-ignored** invariance,
  dispatcher round-trip, finite backward.

Verify with `pixi run pytest <file> -q`, then `pixi run typecheck` and `pixi run lint`.

## Keeping this skill in sync

Per `.agents/context.md` § Workflow rules item 6: whenever a MoTiF layer/block/backbone/config is added
or the MoTiF data structures change, update **this file** and the § "The MoTiF embedding layer" /
candidate-architecture notes in [`.agents/architecture.md`](architecture.md). The `/motif` slash-command
redirect (`.claude/commands/motif.md`) and the Cursor pointer (`.cursor/skills/tcfuse-motif/SKILL.md`)
are thin and rarely need edits; the skills table lives in `context.md`, `CLAUDE.md`, and `AGENTS.md`.
