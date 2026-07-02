# /motif — TC-Fuse MoTiF Backbone Agent

Source of truth: [`.agents/motif.md`](../../.agents/motif.md).

This command activates the TC-Fuse MoTiF backbone skill. **Before working in `src/tcfuse/models/motif/`, `encoders_motif/`, or `decoders_motif/`, or adding any MoTiF layer, block, backbone, or config**, read the skill file. The MoTiF block design, the dual value/coordinate token structures, what is implemented vs. missing, the cross-source attention mechanism, and the MoTiF-specific conventions are defined there.

Architecture-agnostic contracts (embedding/encoder/decoder philosophy, encoder interface, pre-training task): [`/architecture`](architecture.md). Hyperparameter search: [`/sweep`](sweep.md). Cluster submission: [`/jz`](jz.md), [`/cleps`](cleps.md).

Keep docs in sync: when a MoTiF layer/block/backbone/config is added or the MoTiF data structures change, update `.agents/motif.md` and (if the interface shifts) the MoTiF notes in `.agents/architecture.md`; the skills table lives in `.agents/context.md`, `CLAUDE.md`, and `AGENTS.md`.

---

## Quick pointer

| Need | Start here (in motif.md) |
|---|---|
| What MoTiF is (diffusion-transformer, coords at every layer) | "What MoTiF is" |
| The three MoTiF packages and their status | "MoTiF-specific packages" |
| `MotifEmbeddedSource`/`MotifDecodedSource`, the value/coord dict layer interface | "Data structures" |
| Windowed cross-source attention + RPB coordinate bias | "Implemented components" |
| What to build next (self-attn, MLP, block, backbone, config) | "What's missing (extension points…)" |
| MoTiF conventions (RMSNorm QK-norm, RPB not RoPE, self-contained) | "Design decisions & conventions" |
| Known caveats (single-source NaN, FIELD-only, stacked value width) | "Known caveats & potential improvements" |
