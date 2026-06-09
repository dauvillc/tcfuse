---
name: tcfuse-preprocess
description: >-
  TC-Fuse dataset preprocessing pipeline — Stage 0 IBTrACS prep + ATCF→SID
  translation table, Stage 1 per-source HDF5 snapshots, Stage 2 assembled
  per-storm HDF5 + concatenated index, Stage 3A season splits,
  Stage 3B window index building, normalization statistics, the I/O API in
  `src/tcfuse/data/sources/`. Use when preparing any dataset (TC-PRIMED,
  CyclObs, dropsondes, Argo), running `prepare_ibtracs.py` / `assemble.py` /
  `build_splits.py` / `build_windows.py` / `compute_normalization.py`,
  working with `Source` or `StormData`, or extending the per-source HDF5 /
  assembled formats.
---

> **Content has moved.** The full skill documentation is in [`.agents/preprocess.md`](../../.agents/preprocess.md).
> Read that file for pipeline stages, dataset inventory, HDF5 schemas, running instructions, and the I/O API reference.
