

# INTEGRATION — Using DCAS with an external HFL codebase (e.g., HierFL)

This repository is **overlay-only**: it contains the **Decision-Critical Atlas Synchronization (DCAS)** control-plane logic and plotting utilities, but it does **not** redistribute any third-party HFL training code.

Why? The upstream baseline codebase we used during development (HierFL) does not advertise an explicit open-source license on its repository page. Therefore, **we do not copy or publish any of their source files here.** Instead, we provide clean-room DCAS modules and step-by-step integration instructions.

> You must obtain the external HFL codebase yourself and ensure you have the rights to use it.

---

## What you get from this repo

- `src/dcas/atlas.py` — data structures and utilities for semantic atlas state (clean-room implementation)
- `src/dcas/curriculum.py` — curriculum gating + decision-centric metrics helpers (clean-room implementation)
- `src/dcas/control_plane.py` — DCAS / baselines scheduling logic (policy layer)
- `paper_results/` + `make_paper_plots.py` — scripts + CSVs to reproduce the plots from the paper (if included)

This overlay assumes the external HFL codebase provides:
- a training loop over rounds (communication rounds),
- an “atlas” concept (global prototypes at cloud; cached prototypes at edge),
- curriculum mode (unlocking void classes progressively),
- the ability to log per-round metrics.

If your external codebase uses different names, adapt the glue points accordingly.

---

## Integration overview (high level)

You will:
1. Copy `src/dcas/` into your external codebase.
2. Add a minimal “glue layer” so the training loop calls the control-plane scheduler each round.
3. Route the scheduler’s decisions to the code that sends atlas updates to edges (respecting **delay d** and **budget B**).
4. Log metrics into CSVs compatible with `make_paper_plots.py`.

---

## Step 1 — Put DCAS modules inside the external repo

From the root of the external repo, create:

