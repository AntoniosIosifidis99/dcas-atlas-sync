"""dcas.atlas

Original implementation of Semantic Atlas utilities used in Decision-Critical Atlas
Synchronization (DCAS).

This module is intentionally *standalone* (no dependency on HierFL internals).
You can drop it into an existing HierFL checkout and import it from your driver
script / patched training loop.

Concepts
--------
A *Semantic Atlas* maps each class id to compact feature-space statistics:
  - mean vector (prototype)  mu_c in R^d
  - diagonal variance vector var_c in R^d
  - sample count n_c

The atlas is built from intermediate feature representations z(x) (e.g., the
input to the classifier head).

The code below:
  - extracts per-class moments from a model + dataloader using a forward hook
  - merges moments across clients/edges using pooled second moments
  - supports simple EMA smoothing at the cloud
  - provides payload sizing helpers (useful for bandwidth/budget experiments)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Any, List

import numpy as np


# -------------------------
# Data structures
# -------------------------

@dataclass
class PrototypeStats:
    """Per-class prototype statistics."""

    mean: np.ndarray          # shape [d]
    var: np.ndarray           # shape [d], diagonal variance
    n: float                  # number of feature vectors used

    def as_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "n": float(self.n)}


Atlas = Dict[int, PrototypeStats]


# -------------------------
# Feature extraction helpers
# -------------------------

class FeatureTap:
    """Captures the *input* to a target module during forward passes."""

    def __init__(self, module):
        self._buf = []
        self._handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, _module, inputs):
        # inputs is a tuple; inputs[0] expected to be a tensor of shape [B, d]
        self._buf.append(inputs[0].detach())

    def pop(self):
        if not self._buf:
            return None
        x = self._buf[-1]
        self._buf.clear()
        return x

    def close(self):
        try:
            self._handle.remove()
        except Exception:
            pass


def _find_last_linear_module(model) -> Tuple[Optional[str], Optional[Any]]:
    """Heuristic: returns the name+module of the last torch.nn.Linear found."""
    try:
        import torch.nn as nn
    except Exception:
        return None, None

    last_name = None
    last_mod = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last_name = name
            last_mod = mod
    return last_name, last_mod


def _resolve_tap_module(model, feature_layer: Optional[str]) -> Tuple[str, Any]:
    """Pick which module to tap for features.

    If feature_layer is provided, we expect it to be a valid attribute path
    within model (e.g., "classifier" or "shared_layers.fc").

    Otherwise, we default to the last Linear layer.
    """
    if feature_layer:
        cur = model
        for part in feature_layer.split("."):
            cur = getattr(cur, part)
        return feature_layer, cur

    name, mod = _find_last_linear_module(model)
    if mod is None:
        raise ValueError(
            "Could not infer a classifier layer to tap. "
            "Pass feature_layer explicitly (e.g., --feature_layer shared_layers.fc)."
        )
    return name, mod


# -------------------------
# Moment estimation
# -------------------------

@dataclass
class _RunningMoments:
    """Streaming diagonal-moment estimator (Welford-like)."""

    n: float
    mean: np.ndarray
    m2: np.ndarray  # sum of squared deviations

    @classmethod
    def create(cls, d: int) -> "_RunningMoments":
        return cls(n=0.0, mean=np.zeros((d,), dtype=np.float64), m2=np.zeros((d,), dtype=np.float64))

    def update_batch(self, x: np.ndarray) -> None:
        # x: [B, d]
        if x.size == 0:
            return
        x = x.astype(np.float64, copy=False)
        for i in range(x.shape[0]):
            self.n += 1.0
            delta = x[i] - self.mean
            self.mean += delta / self.n
            delta2 = x[i] - self.mean
            self.m2 += delta * delta2

    def finalize(self) -> PrototypeStats:
        if self.n <= 1.0:
            var = np.ones_like(self.mean, dtype=np.float64) * 1e-6
        else:
            var = self.m2 / (self.n - 1.0)
            var = np.maximum(var, 1e-6)
        return PrototypeStats(mean=self.mean.astype(np.float32), var=var.astype(np.float32), n=float(self.n))


def compute_class_prototypes(
    model,
    dataloader,
    num_classes: int,
    device: str,
    *,
    max_batches: Optional[int] = None,
    feature_layer: Optional[str] = None,
) -> Atlas:
    """Compute per-class (mean, var, n) of feature vectors produced by `model`.

    Parameters
    ----------
    model:
      Any PyTorch model.
    dataloader:
      Yields (x, y) with y in [0, num_classes).
    num_classes:
      Total number of classes.
    device:
      "cpu" / "cuda" / "mps".
    max_batches:
      Optional cap for speed.
    feature_layer:
      Optional module path to tap. If None, last nn.Linear is used.

    Returns
    -------
    Atlas: dict[class_id -> PrototypeStats]
    """
    import torch

    model.eval()
    model.to(device)

    layer_name, layer_mod = _resolve_tap_module(model, feature_layer)
    tap = FeatureTap(layer_mod)

    moments: Dict[int, _RunningMoments] = {}
    dim: Optional[int] = None

    try:
        with torch.no_grad():
            for b, (x, y) in enumerate(dataloader):
                if max_batches is not None and b >= int(max_batches):
                    break

                x = x.to(device)
                y = y.to(device)

                _ = model(x)
                z = tap.pop()
                if z is None:
                    continue

                # z is a tensor [B, d]
                z = z.detach().float().cpu()
                if dim is None:
                    dim = int(z.shape[1])

                for cls in torch.unique(y):
                    c = int(cls.item())
                    if c < 0 or c >= int(num_classes):
                        continue
                    mask = (y == cls)
                    if not bool(mask.any()):
                        continue

                    x_c = z[mask].numpy()
                    if c not in moments:
                        moments[c] = _RunningMoments.create(dim)
                    moments[c].update_batch(x_c)

    finally:
        tap.close()

    atlas: Atlas = {}
    for c, rm in moments.items():
        atlas[int(c)] = rm.finalize()

    return atlas


# -------------------------
# Atlas merging + smoothing
# -------------------------

def merge_atlases(atlases: Iterable[Atlas]) -> Atlas:
    """Merge multiple atlases with pooled second moments.

    Each input atlas must provide (mean, var, n). We merge using:
      E[x]  = sum(n_i * mean_i) / sum(n_i)
      E[x^2]= sum(n_i * (var_i + mean_i^2)) / sum(n_i)
      var   = E[x^2] - E[x]^2

    This is order-invariant and matches merging feature samples.
    """
    accum: Dict[int, Dict[str, Any]] = {}

    for atlas in atlases:
        if not atlas:
            continue
        for c, st in atlas.items():
            n = float(st.n)
            if n <= 0.0:
                continue
            mu = np.asarray(st.mean, dtype=np.float64)
            var = np.asarray(st.var, dtype=np.float64)
            ex2 = var + mu * mu

            if int(c) not in accum:
                accum[int(c)] = {
                    "n": 0.0,
                    "sum_mu": np.zeros_like(mu, dtype=np.float64),
                    "sum_ex2": np.zeros_like(mu, dtype=np.float64),
                }
            accum[int(c)]["n"] += n
            accum[int(c)]["sum_mu"] += n * mu
            accum[int(c)]["sum_ex2"] += n * ex2

    out: Atlas = {}
    for c, a in accum.items():
        n = float(a["n"])
        if n <= 0.0:
            continue
        mu = a["sum_mu"] / n
        ex2 = a["sum_ex2"] / n
        var = ex2 - mu * mu
        var = np.maximum(var, 1e-6)
        out[int(c)] = PrototypeStats(mean=mu.astype(np.float32), var=var.astype(np.float32), n=n)

    return out


def ema_update(previous: Atlas, incoming: Atlas, alpha: float) -> Atlas:
    """Exponential moving average update of means/vars.

    alpha in [0, 1]. Higher alpha -> more inertia (more weight on previous).
    Only classes present in `incoming` are updated/added.
    """
    a = float(max(0.0, min(1.0, alpha)))
    out: Atlas = {int(k): v for k, v in (previous or {}).items()}

    for c, st in (incoming or {}).items():
        c = int(c)
        if c in out and a > 0.0:
            mu = a * np.asarray(out[c].mean, dtype=np.float64) + (1.0 - a) * np.asarray(st.mean, dtype=np.float64)
            var = a * np.asarray(out[c].var, dtype=np.float64) + (1.0 - a) * np.asarray(st.var, dtype=np.float64)
            var = np.maximum(var, 1e-6)
            n = float(st.n)
            out[c] = PrototypeStats(mean=mu.astype(np.float32), var=var.astype(np.float32), n=n)
        else:
            out[c] = PrototypeStats(mean=np.asarray(st.mean, dtype=np.float32), var=np.asarray(st.var, dtype=np.float32), n=float(st.n))

    return out


# -------------------------
# Payload / subset utilities
# -------------------------

def atlas_subset(atlas: Atlas, class_ids: Iterable[int]) -> Atlas:
    out: Atlas = {}
    if not atlas:
        return out
    for c in class_ids:
        if int(c) in atlas:
            st = atlas[int(c)]
            out[int(c)] = PrototypeStats(
                mean=np.asarray(st.mean).copy(),
                var=np.asarray(st.var).copy(),
                n=float(st.n),
            )
    return out


def atlas_nbytes(atlas: Atlas) -> int:
    if not atlas:
        return 0
    total = 0
    for st in atlas.values():
        total += np.asarray(st.mean).nbytes
        total += np.asarray(st.var).nbytes
        total += 8
    return int(total)
