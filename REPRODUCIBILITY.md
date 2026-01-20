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
Paper configuration (core)
From the paper evaluation section: 
ICFEC_Research_Paper_47
Dataset: CIFAR-10
Model: ResNet-18
Pathological label skew: 2 classes per client
Rounds: T = 50
Constraints: delay d = 5, budget B = 3
Topology (paper): 5 edges, 50 clients
Use seeds: 11 12 13 (as in the provided runner script pattern).
*Runner script*
After integrating DCAS into your baseline (see INTEGRATION.md), run:
bash scripts/run_grid_dB.sh
*The script runs constrained policies and evaluates them against an existing oracle run.*
*Expected outputs*
Each run directory should include (at minimum):
training_metrics.csv
cp_metrics.csv
*And your aggregation pipeline should produce the curated CSV artifacts*:
eval_table.csv
train_curve_long.csv
cp_curve_long.csv
args_table.csv
