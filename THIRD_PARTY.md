Third-Party Software Notice

This repository contains original code for the DCAS control-plane emulator, decision-centric metrics, and a clean-room reimplementation of atlas/curriculum utilities.

Baseline Framework Reference:
- HierFL (https://github.com/LuminLiu/HierFL)

This repository is a **public research artifact (overlay)** for DCAS from:
*Atlas Synchronization in the Hierarchical Federated Learning Continuum*. :contentReference[oaicite:11]{index=11}

It contains **original** overlay code (control-plane logic, metrics, plotting, and clean-room utilities) and does **not** redistribute any third-party HFL training framework.



---

## 3) `THIRD_PARTY.md` (full, copy-paste)

```markdown
# THIRD_PARTY.md — Third-party code, data, and licensing notes

This repository is a **public research artifact (overlay)** for DCAS from:
*Atlas Synchronization in the Hierarchical Federated Learning Continuum*. :contentReference[oaicite:11]{index=11}

It contains **original** overlay code (control-plane logic, metrics, plotting, and clean-room utilities) and does **not** redistribute any third-party HFL training framework.

---

## Baseline HFL framework (external)

To run full training experiments, you must obtain an external baseline HFL framework separately.

Example baseline referenced during development:
- https://github.com/LuminLiu/HierFL

Important:
- This repository is **not** a fork of that project.
- We do **not** include or redistribute any of its source files.
- You are responsible for complying with the baseline repository’s terms and any applicable licensing.

---

## Datasets

The paper evaluates on CIFAR-10. :contentReference[oaicite:12]{index=12}  
You must follow the dataset’s original terms when downloading/using it.

This repository does not ship CIFAR-10 data.

---

## Python dependencies

This repository uses standard scientific Python packages (e.g., numpy, pandas, matplotlib) for artifact processing and plotting.

When running training, additional dependencies are determined by the external baseline framework you use.

---

## Citation

If you use this repository, please cite the paper:
*Atlas Synchronization in the Hierarchical Federated Learning Continuum*. :contentReference[oaicite:13]{index=13}
