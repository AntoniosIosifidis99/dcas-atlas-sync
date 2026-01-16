OLD INTEGRATION STYLE 


# Integration Guide (Overlay into a separate HierFL checkout)

This repository provides an **overlay** (original code + clean-room modules) for experiments on
**Decision-Critical Atlas Synchronization (DCAS)** and decision-centric metrics (DMR / wDMR / TTR / VoidAUC).

**Important**:
- This repo is **NOT** a fork of HierFL.
- This repo **does not redistribute** any code from HierFL.
- You must obtain HierFL separately from its upstream source.

> Upstream baseline (clone separately):
> ```text
> https://github.com/LuminLiu/HierFL
> ```

---

## 1. What you will add to HierFL

This overlay provides:

- `control_plane.py`  
  Control-plane emulator (delay, budget) and update selection policies used in the paper.

- `src/dcas/atlas.py`  
  Clean-room implementation of semantic atlas extraction + aggregation utilities.

- `src/dcas/curriculum.py`  
  Clean-room implementation of curriculum gating + decision mismatch helpers.

- `scripts/run_grid_dB.sh`  
  Example experiment runner used on a server.

Optionally:
- `collect_paper_artifacts.py` (if you use it to generate paper plots/tables from CSV outputs)

---

## 2. Directory layout (recommended)

Clone both repositories side-by-side:

```text
workspace/
  HierFL/                 # cloned from upstream, not redistributed by this repo
  dcas-atlas-sync/        # this repository
