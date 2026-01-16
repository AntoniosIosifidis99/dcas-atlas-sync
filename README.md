# DCAS Atlas Synchronization — Public Research Artifact (Overlay)

Decision-Critical Atlas Synchronization (**DCAS**) is a lightweight **overlay** that supports experiments on **delay- and bandwidth-constrained semantic atlas dissemination** in **hierarchical federated learning** (Client → Edge → Cloud).

This repo contains **original code** for:
- a **control-plane emulator** for atlas update delivery under constraints,
- **decision-centric synchronization metrics** (e.g., DMR, TTR, VoidAUC; optional wDMR),
- clean-room utilities for **semantic atlas** handling and **curriculum gating**,
- scripts and curated CSV artifacts to help **reproduce plots and tables**.

> ✅ **Overlay-only release (Option A):** we do **not** redistribute third-party training code.  
> You plug this overlay into a separately obtained HFL baseline (see below).

---

## 📝 Paper

**Title:** *Atlas Synchronization in the Hierarchical Federated Learning Continuum*  
**Authors:** Antonios Iosifidis, Vasileios Karagiannis, Stefan Schulte  
**Venue/Year:** *(to be updated)*  
**PDF/DOI:** *(to be updated)*  

**Suggested citation (temporary):**
```bibtex
@misc{iosifidis_dcas_atlas_sync,
  title   = {Atlas Synchronization in the Hierarchical Federated Learning Continuum},
  author  = {Iosifidis, Antonios and Karagiannis, Vasileios and Schulte, Stefan},
  note    = {Preprint / Research Artifact},
  year    = {202X}
}
