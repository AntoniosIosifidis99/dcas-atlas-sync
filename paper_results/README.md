# paper_results (curated CSV outputs)

This folder contains curated, lightweight CSV artifacts used to regenerate the main paper plots/tables without rerunning training.

These files were produced by our experiment pipeline (oracle + constrained control-plane runs) and are intended for:
- quick inspection of results,
- plot regeneration,


> Note: We intentionally do not include raw run directories (`runs/`), logs, model checkpoints, or per-round atlas snapshots.

---

## Files

### `args_table.csv`
One row per run with the main experiment configuration (CLI args flattened).
Used to:
- identify experiment settings,
- determine the curriculum phase window (phase1/phase2).

### `eval_table.csv`
One row per run with summary metrics such as:
- void accuracy AUC (`void_auc`),
- time-to-recovery (`ttr`),
- final seen/void accuracy,
- mean gap,
- control-plane aggregates (e.g., delivered bytes, mismatch durations).

This is typically the source for paper table numbers.

### `train_curve_long.csv`
Long-format per-round training curves, one row per (run, round), including:
- seen accuracy (`tr_seen_acc`),
- void accuracy (`tr_void_acc`),
- gap (`tr_gap`),
- global accuracy (`tr_global_acc`).

Used to regenerate learning dynamics figures (void recovery and gap curves).

### `cp_curve_long.csv`
Long-format per-round control-plane curves, one row per (run, round), including:
- decision mismatch rate (`cp_DMR_avg`),
- ACE drift proxy (`cp_ACE_avg`),
- delivered and sent bytes (`cp_bytes_delivered`, `cp_bytes_sent`).

Used to regenerate DMR and bandwidth figures.


