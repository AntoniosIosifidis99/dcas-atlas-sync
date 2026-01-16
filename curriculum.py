"""dcas.curriculum

Original implementation of curriculum-gating logic for Semantic Atlases.

A client k has a set of *seen* classes S_k. The *void* classes are V_k = C \ S_k.
Given an atlas A containing class prototype means mu_c, we define the semantic
proximity of a void class v to the seen set as:

  d_k(v) = min_{s in S_k} || mu_v - mu_s ||_2

We bucket void classes by percentiles of d_k(v):
  easy   : d <= tau1
  medium : tau1 < d <= tau2
  hard   : d > tau2

A staged curriculum unlocks void classes progressively as training advances.

No code is copied from the upstream HierFL repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .atlas import Atlas, PrototypeStats


@dataclass(frozen=True)
class CurriculumConfig:
    tau1_percentile: float = 33.0
    tau2_percentile: float = 66.0
    phase1: float = 0.30
    phase2: float = 0.70


def _mu(atlas: Atlas, c: int) -> np.ndarray:
    return np.asarray(atlas[int(c)].mean, dtype=np.float64)


def void_distances(
    atlas: Atlas,
    seen_classes: Set[int],
) -> List[Tuple[int, float, Optional[int]]]:
    """Compute (void_class, distance, nearest_seen_class)."""
    if not atlas:
        return []

    seen_present = [int(c) for c in seen_classes if int(c) in atlas]
    if not seen_present:
        return []

    voids = [int(c) for c in atlas.keys() if int(c) not in seen_classes]
    if not voids:
        return []

    out: List[Tuple[int, float, Optional[int]]] = []
    for v in voids:
        mu_v = _mu(atlas, v)
        best_d = float("inf")
        best_s: Optional[int] = None
        for s in seen_present:
            d = float(np.linalg.norm(mu_v - _mu(atlas, s)))
            if d < best_d:
                best_d = d
                best_s = int(s)
        if np.isfinite(best_d):
            out.append((int(v), float(best_d), best_s))

    out.sort(key=lambda t: t[1])
    return out


def _thresholds(distances: Sequence[float], cfg: CurriculumConfig) -> Tuple[float, float]:
    if len(distances) == 0:
        return 0.0, 0.0
    d = np.asarray(list(distances), dtype=np.float64)
    tau1 = float(np.percentile(d, float(cfg.tau1_percentile)))
    tau2 = float(np.percentile(d, float(cfg.tau2_percentile)))
    return tau1, tau2


def unlock_void_classes(
    atlas: Atlas,
    seen_classes: Set[int],
    round_idx: int,
    total_rounds: int,
    cfg: CurriculumConfig = CurriculumConfig(),
) -> Set[int]:
    """Return the set of *void* classes that are unlocked at this round."""
    trips = void_distances(atlas, seen_classes)
    if not trips:
        return set()

    dists = [d for _, d, _ in trips]
    tau1, tau2 = _thresholds(dists, cfg)

    easy = [v for (v, d, _) in trips if d <= tau1]
    medium = [v for (v, d, _) in trips if (d > tau1 and d <= tau2)]
    hard = [v for (v, d, _) in trips if d > tau2]

    progress = float(round_idx) / float(max(1, total_rounds))

    if progress < float(cfg.phase1):
        return set(easy)
    if progress < float(cfg.phase2):
        return set(easy + medium)
    return set(easy + medium + hard)


def curate_atlas_for_client(
    atlas: Atlas,
    seen_classes: Set[int],
    round_idx: int,
    total_rounds: int,
    *,
    cfg: CurriculumConfig = CurriculumConfig(),
    include_seen: bool = True,
) -> Atlas:
    """Filter an atlas to what a client should receive under curriculum gating."""
    if not atlas:
        return {}

    unlocked_void = unlock_void_classes(atlas, seen_classes, round_idx, total_rounds, cfg)
    keep = set(unlocked_void)
    if include_seen:
        keep |= set(int(c) for c in seen_classes)

    out: Atlas = {}
    for c in keep:
        if int(c) in atlas:
            st = atlas[int(c)]
            out[int(c)] = PrototypeStats(
                mean=np.asarray(st.mean).copy(),
                var=np.asarray(st.var).copy(),
                n=float(st.n),
            )
    return out


def jaccard_distance(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 0.0
    u = a | b
    if not u:
        return 0.0
    return 1.0 - (len(a & b) / float(len(u)))


def decision_mismatch_rate(
    edge_atlas: Atlas,
    oracle_atlas: Atlas,
    clients_seen: Dict[int, Set[int]],
    round_idx: int,
    total_rounds: int,
    cfg: CurriculumConfig = CurriculumConfig(),
) -> float:
    """Average Jaccard distance between unlocked-void sets under two atlases."""
    if not clients_seen:
        return 0.0

    vals: List[float] = []
    for _, seen in clients_seen.items():
        a = unlock_void_classes(edge_atlas, seen, round_idx, total_rounds, cfg)
        b = unlock_void_classes(oracle_atlas, seen, round_idx, total_rounds, cfg)
        vals.append(jaccard_distance(a, b))

    return float(np.mean(vals)) if vals else 0.0


def boundary_sensitivity(
    atlas: Atlas,
    clients_seen: Dict[int, Set[int]],
    round_idx: int,
    total_rounds: int,
    *,
    cfg: CurriculumConfig = CurriculumConfig(),
    credit_anchor: float = 0.30,
    eps: float = 1e-6,
) -> Dict[int, float]:
    """Heuristic sensitivity score per class (for DCAS-style selection).

    Intuition: decision flips are likeliest for classes near the *active* curriculum
    boundary (tau1 or tau2, depending on phase). We score each void class by the
    inverse margin to that boundary, and also credit the nearest-seen anchor class.

    Returns
    -------
    scores: {class_id -> sensitivity_score}
    """
    if not atlas:
        return {}

    progress = float(round_idx) / float(max(1, total_rounds))
    if progress < float(cfg.phase1):
        boundary_mode = "tau1"
    elif progress < float(cfg.phase2):
        boundary_mode = "tau2"
    else:
        boundary_mode = None

    if boundary_mode is None:
        return {}

    scores: Dict[int, float] = {}
    counts: Dict[int, int] = {}

    for _, seen in (clients_seen or {}).items():
        trips = void_distances(atlas, seen)
        if not trips:
            continue

        dists = [d for _, d, _ in trips]
        tau1, tau2 = _thresholds(dists, cfg)
        boundary = tau1 if boundary_mode == "tau1" else tau2

        for v, d, s in trips:
            margin = abs(float(d) - float(boundary))
            inv = 1.0 / (margin + float(eps))

            scores[int(v)] = float(scores.get(int(v), 0.0) + inv)
            counts[int(v)] = int(counts.get(int(v), 0) + 1)

            if s is not None:
                scores[int(s)] = float(scores.get(int(s), 0.0) + float(credit_anchor) * inv)
                counts[int(s)] = int(counts.get(int(s), 0) + 1)

    # normalize by client count contribution
    out: Dict[int, float] = {}
    for c, val in scores.items():
        out[int(c)] = float(val) / float(max(1, counts.get(int(c), 1)))
    return out
