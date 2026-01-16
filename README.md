# DCAS Atlas Synchronization — Public Research Artifact (Overlay)

Decision-Critical Atlas Synchronization (**DCAS**) is a lightweight **overlay** that supports experiments on **delay- and bandwidth-constrained semantic atlas dissemination** in **hierarchical federated learning** (Client → Edge → Cloud).

This repo contains **original code** for:
- a **control-plane emulator** for atlas update delivery under constraints,
- **decision-centric synchronization metrics** (e.g., DMR, TTR, VoidAUC;  wDMR),
- clean-room utilities for **semantic atlas** handling and **curriculum gating**,
- scripts and curated CSV artifacts to help **reproduce plots and tables**.



---

##  Paper

**Title:** *Atlas Synchronization in the Hierarchical Federated Learning Continuum*  
**Authors:** Antonios Iosifidis, Vasileios Karagiannis, Stefan Schulte  
**Venue/Year:** *(to be updated)*  
**PDF/DOI:** *(to be updated)*  


## ✨ What this repository provides

### ✅ Core contributions implemented

**Control-plane formulation of atlas synchronization with explicit constraints**
- dissemination delay **d** (in rounds)
- dissemination budget **B** (classes per update / per edge per round)

**Decision-centric metrics**
- **DMR** — Decision Mismatch Rate (oracle vs edge cached view)
- **TTR** — Time-to-Recovery (void accuracy recovery threshold-based)
- **VoidAUC** — area under the void-accuracy curve
- *(Optional)* **wDMR** — gap-weighted DMR (if enabled in your evaluation pipeline)

**Update selection policies**
- Random selection  
- **ACE-greedy** (accuracy-centric / geometry proxy)  
- **DCAS risk-greedy** (decision-critical prioritization)

**Clean-room modules (framework-agnostic)**
- `src/dcas/atlas.py` — semantic atlas statistics + merging utilities
- `src/dcas/curriculum.py` — curriculum gating + mismatch helpers

**Experiment runner**
- `scripts/run_grid_dB.sh` — server runner for constrained dissemination experiments (oracle + constrained policies)

**(Optional) Paper artifacts**
- `paper_results/` — curated CSVs (small, reviewer-friendly)
- `make_paper_plots.py` — generates figures from CSV artifacts

---

## 🚫 What this repository does NOT include (important)

This repository **does not redistribute any source code** from the baseline HFL implementation used during development (e.g., HierFL), and it is **not a fork**.

You must obtain the baseline separately from its upstream source:  
https://github.com/LuminLiu/HierFL

Then integrate this overlay by following:  
- `INTEGRATION.md` (step-by-step wiring guide)

See also `THIRD_PARTY.md` for third-party notices.

---

## Structure

.
├── src/dcas/            # Clean-room DCAS overlay modules
├── scripts/             # Experiment runner scripts
├── paper_results/       # (Optional) curated CSV artifacts
├── make_paper_plots.py  # Plot generator from CSV artifacts
├── INTEGRATION.md       # How to wire DCAS into an external HFL codebase
├── THIRD_PARTY.md       # Third-party notices and licensing constraints
└── README.md

