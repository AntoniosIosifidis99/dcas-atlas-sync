# REPRODUCIBILITY.md — Paper artifact (v1.0.0)

This document describes how to reproduce the results/figures for:
*Atlas Synchronization in the Hierarchical Federated Learning Continuum*.

This repo provides DCAS control-plane logic, metrics, and plotting. The baseline HFL training framework must be obtained separately (see THIRD_PARTY.md).

---

## A) Reproduce figures from provided CSV artifacts (no training)

If `paper_results/` is included, you can regenerate plots directly:

```bash
python make_paper_plots.py --in_dir paper_results --out_dir figures.
```
---

## B) Reproduce training runs (requires baseline integration)

### Paper configuration (core)

Use the experimental setting reported in the evaluation section of the paper (ICFEC_Research_Paper_47): :contentReference[oaicite:0]{index=0}
- **Dataset:** CIFAR-10  
- **Model:** ResNet-18  
- **Client data distribution:** pathological label skew (**2 classes per client**)  
- **Rounds:** `T = 50`  
- **Control-plane constraints:** delay `d = 5`, budget `B = 3`  
- **Topology (paper):** **5 edges**, **50 clients**

**Seeds:** `11 12 13` (as used in the runner-script pattern).

### Runner script

After integrating the overlay into your baseline (see `INTEGRATION.md`), run:

```bash
bash scripts/run_grid_dB.sh

