# dcas-atlas-sync


---

# `README.md` (full)

```markdown
# Decision-Critical Atlas Synchronization (DCAS) — Public Research Artifact (Overlay)

This repository contains **original code** for:
- a control-plane emulator for **delay/budget-constrained** atlas dissemination,
- decision-centric metrics (e.g., DMR / wDMR / TTR / VoidAUC),
- and clean-room utilities for semantic atlas extraction + curriculum gating.

It is designed to be used as an **overlay** on top of a separately obtained Hierarchical Federated Learning baseline implementation.

---

## Paper

**Title**: Atlas Synchronization in the Hierarchical Federated Learning Continuum  
**Authors**: [Your Name(s)]  
**Venue/Year**: [Conference/Workshop], [Year]  
**PDF/DOI**: [Add link here]

If you use this artifact, please cite the paper (see **Citation** below).

---

## What this repo provides

### Core contributions implemented
- **Control-plane formulation** of atlas synchronization with:
  - dissemination delay `d` (in rounds)
  - dissemination budget `B` (classes per update / per edge)
- **Decision-centric mismatch metrics**
  - DMR (Decision Mismatch Rate)
  - wDMR (gap-weighted DMR), if enabled in your pipeline
- **Policies**
  - Random selection
  - ACE-greedy
  - DCAS risk-greedy
- **Clean-room modules**
  - `src/dcas/atlas.py`: semantic atlas extraction + aggregation
  - `src/dcas/curriculum.py`: curriculum gating + mismatch helpers
- **Experiment scripts**
  - `scripts/run_grid_dB.sh`: server runner for constrained dissemination experiments

---

## What this repo does NOT include (important)

This repository **does not** redistribute any code from the upstream HierFL baseline and is **not** a fork.

You must obtain the baseline separately from its upstream source:

```text
https://github.com/LuminLiu/HierFL
