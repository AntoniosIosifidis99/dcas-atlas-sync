"""control_plane.py

Compute-continuum control-plane emulation for the Semantic Atlas.

Purpose
-------
In hierarchical FL (clients -> edges -> cloud), the Semantic Atlas (class prototypes)
is effectively *control-plane state*. In compute-continuum settings, atlas updates
may be delayed (staleness) and/or budget-limited (partial dissemination).

This module provides:
  - A minimal ControlPlaneEmulator that injects (delay, budget, policy) effects.
  - Decision extraction for curriculum (pure function, no side effects).
  - Metrics: ACE (atlas disagreement) and DMR (decision mismatch).

Designed to be used from hierfavg.py without modifying Edge/Cloud classes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Any, List, Set, Tuple, Optional

import numpy as np


Prototypes = Dict[int, Dict[str, Any]]  # {class_id: {'mean': np.ndarray, 'var': np.ndarray, ...}}


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def protos_nbytes(protos: Optional[Prototypes]) -> int:
    if not protos:
        return 0
    total = 0
    for _, st in protos.items():
        if st is None:
            continue
        total += _to_np(st.get("mean", [])).nbytes
        total += _to_np(st.get("var", [])).nbytes
        total += 8  # count/metadata
    return int(total)


def copy_subset(oracle: Prototypes, class_ids: List[int]) -> Prototypes:
    out: Prototypes = {}
    for c in class_ids:
        if c in oracle:
            # keep numpy arrays (pickleable) and copy to avoid aliasing
            st = oracle[c]
            out[int(c)] = {
                "mean": _to_np(st.get("mean", [])).copy(),
                "var": _to_np(st.get("var", [])).copy(),
            }
            if "n" in st:
                out[int(c)]["n"] = float(st["n"])
    return out


def ace_avg(edge_atlas: Prototypes, oracle_atlas: Prototypes) -> float:
    """Average L2 distance between class means over common classes."""
    if not edge_atlas or not oracle_atlas:
        return 0.0
    common = set(edge_atlas.keys()).intersection(set(oracle_atlas.keys()))
    if not common:
        return 0.0
    vals = []
    for c in common:
        mu_e = _to_np(edge_atlas[c]["mean"]).astype(np.float64)
        mu_o = _to_np(oracle_atlas[c]["mean"]).astype(np.float64)
        vals.append(float(np.linalg.norm(mu_e - mu_o)))
    return float(np.mean(vals)) if vals else 0.0


def decision_void_set_curriculum(
    atlas: Prototypes,
    seen_classes: Set[int],
    current_round: int,
    total_rounds: int,
    tau1p: float = 33.0,
    tau2p: float = 66.0,
    phase1: float = 0.30,
    phase2: float = 0.70,
) -> Set[int]:
    """Pure decision function: return which void classes would be "unlocked".

    Matches the spirit of Edge.get_curriculum_prototypes but returns only the
    decision set (void classes), with no side effects.

    Important semantics (aligned with your replay):
      - If atlas is empty OR we cannot compute distances, decision set is empty.
        (client receives no synthetic help).
    """
    if not atlas:
        return set()

    # Only use classes present in the atlas.
    seen_present = [c for c in seen_classes if c in atlas]
    if not seen_present:
        return set()

    void_classes = [c for c in atlas.keys() if c not in seen_classes]
    if not void_classes:
        return set()

    # Compute d(k,v) = min_{s in S_k} ||mu_v - mu_s||
    distances: List[Tuple[int, float]] = []
    for v in void_classes:
        mu_v = _to_np(atlas[v]["mean"]).astype(np.float64)
        best = float("inf")
        for s in seen_present:
            mu_s = _to_np(atlas[s]["mean"]).astype(np.float64)
            d = float(np.linalg.norm(mu_v - mu_s))
            if d < best:
                best = d
        if best < float("inf"):
            distances.append((int(v), best))

    if not distances:
        return set()

    distances.sort(key=lambda x: x[1])
    dvals = np.asarray([d for _, d in distances], dtype=np.float64)

    tau1 = float(np.percentile(dvals, tau1p))
    tau2 = float(np.percentile(dvals, tau2p))

    easy = [v for (v, d) in distances if d <= tau1]
    medium = [v for (v, d) in distances if (d > tau1 and d <= tau2)]
    hard = [v for (v, d) in distances if d > tau2]

    progress = float(current_round) / float(max(1, total_rounds))
    if progress < phase1:
        allowed = set(easy)
    elif progress < phase2:
        allowed = set(easy + medium)
    else:
        allowed = set(easy + medium + hard)

    return allowed


def jaccard_distance(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 0.0
    u = a.union(b)
    if not u:
        return 0.0
    return 1.0 - (len(a.intersection(b)) / float(len(u)))


def dmr_avg(
    edge_atlas: Prototypes,
    oracle_atlas: Prototypes,
    clients_seen: Dict[int, Set[int]],
    current_round: int,
    total_rounds: int,
    tau1p: float,
    tau2p: float,
    phase1: float,
    phase2: float,
) -> float:
    if not clients_seen:
        return 0.0
    vals: List[float] = []
    for _, seen in clients_seen.items():
        a = decision_void_set_curriculum(
            edge_atlas, seen, current_round, total_rounds, tau1p, tau2p, phase1, phase2
        )
        b = decision_void_set_curriculum(
            oracle_atlas, seen, current_round, total_rounds, tau1p, tau2p, phase1, phase2
        )
        vals.append(jaccard_distance(a, b))
    return float(np.mean(vals)) if vals else 0.0


def void_criticality_counts(clients_seen: Dict[int, Set[int]], num_classes: int) -> Dict[int, int]:
    """For each class c, count how many clients do NOT have c in seen."""
    counts = {c: 0 for c in range(num_classes)}
    for _, seen in clients_seen.items():
        for c in range(num_classes):
            if c not in seen:
                counts[c] += 1
    return counts


@dataclass
class ControlPlaneConfig:
    # Continuum effects
    delay_rounds: int = 0
    class_budget: int = -1

    # DCAS-G knobs (policy-only, does not affect other policies)
    dcas_g_decision_slots: int = 1

    # Dissemination strategy
    #  - full: send first B classes (deterministic fallback)
    #  - random: random subset of classes
    #  - ace_greedy: pick classes with highest prototype drift (ACE)
    #  - sens_greedy: ACE weighted by void criticality
    #  - dcas_risk_greedy: decision-critical (proxy) selection (our novelty baseline)
    #  - dcas_g: guarded, delay-aware DCAS (utility + horizon decision slot)
    #  - dmr_oracle_greedy: oracle upper bound (uses true DMR) — evaluation only
    policy: str = "full"

    # Event-driven trigger
    #  - always: send whenever cloud produces a new atlas snapshot
    #  - ace: send if ACE_avg(edge, oracle) > ace_threshold
    #  - dmr_oracle: send if DMR_avg(edge, oracle) > dmr_threshold (oracle, evaluation only)
    #  - dcas_risk: send if decision-risk proxy exceeds risk_threshold (our novelty trigger)
    trigger: str = "always"
    ace_threshold: float = 0.0
    dmr_threshold: float = 0.0
    risk_threshold: float = 0.0
    cooldown_rounds: int = 0

    seed: int = 1


class ControlPlaneEmulator:
    """Inject staleness (delay) + budgets into atlas dissemination.

    Usage:
      - register edge_ids and their client seen sets.
      - each time cloud produces a new oracle atlas at round t, call
          cp.maybe_send_updates(round=t, oracle_atlas=A_t, ...)
        which schedules deliveries at round t+delay.
      - at the start of each training round r, call cp.deliver_due(r)
        then set edge.global_prototypes = cp.edge_state[eid].
    """

    def __init__(
        self,
        config: ControlPlaneConfig,
        edge_clients_seen: Dict[int, Dict[int, Set[int]]],
        num_classes: int,
        tau1p: float,
        tau2p: float,
        phase1: float,
        phase2: float,
    ):
        self.cfg = config
        self.edge_clients_seen = edge_clients_seen
        self.num_classes = int(num_classes)
        self.rng = np.random.RandomState(int(config.seed))

        self.tau1p = float(tau1p)
        self.tau2p = float(tau2p)
        self.phase1 = float(phase1)
        self.phase2 = float(phase2)

        # per-edge atlas state
        self.edge_state: Dict[int, Prototypes] = {eid: {} for eid in edge_clients_seen.keys()}

        # pending messages: (deliver_round, edge_id, payload_prototypes)
        self.pending: List[Tuple[int, int, Prototypes]] = []

        # instrumentation
        self.sent_bytes: Dict[int, int] = {}      # round -> bytes sent
        self.delivered_bytes: Dict[int, int] = {} # round -> bytes delivered

        # per-edge delivery bookkeeping (for decision-critical risk)
        self.last_delivered_round: Dict[int, Dict[int, int]] = {eid: {} for eid in edge_clients_seen.keys()}
        self.last_trigger_round: Dict[int, int] = {eid: -10**9 for eid in edge_clients_seen.keys()}

        # optional risk instrumentation (round -> per-edge)
        self.risk_sum_topB_log: Dict[int, Dict[int, float]] = {}
        self.risk_max_log: Dict[int, Dict[int, float]] = {}
        self.triggered_edges: Dict[int, int] = {}

        self.edge_num_clients: Dict[int, int] = {eid: len(seen_map) for eid, seen_map in edge_clients_seen.items()}

        # precompute void criticality per edge
        self.edge_void_counts: Dict[int, Dict[int, int]] = {
            eid: void_criticality_counts(seen_map, self.num_classes)
            for eid, seen_map in edge_clients_seen.items()
        }

    # --------------------
    # Message scheduling
    # --------------------
    def _select_classes_for_edge(
        self,
        eid: int,
        oracle_atlas: Prototypes,
        current_round: int,
        total_rounds: int,
    ) -> Prototypes:
        """Select subset of oracle_atlas for an edge under the class_budget."""
        budget = int(self.cfg.class_budget)
        if budget < 0:
            return copy.deepcopy(oracle_atlas)

        classes = sorted([int(c) for c in oracle_atlas.keys()])
        if len(classes) <= budget:
            return copy.deepcopy(oracle_atlas)

        policy = str(self.cfg.policy)
        cur_state = self.edge_state.get(eid, {})

        if policy == "full":
            chosen = classes[:budget]
            return copy_subset(oracle_atlas, chosen)

        if policy == "random":
            chosen = self.rng.choice(classes, size=budget, replace=False).tolist()
            return copy_subset(oracle_atlas, [int(x) for x in chosen])

        # per-class ACE score
        def class_ace(c: int) -> float:
            if c not in cur_state:
                return float("inf")
            mu_e = _to_np(cur_state[c]["mean"]).astype(np.float64)
            mu_o = _to_np(oracle_atlas[c]["mean"]).astype(np.float64)
            return float(np.linalg.norm(mu_e - mu_o))

        if policy == "ace_greedy":
            scored = [(c, class_ace(c)) for c in classes]
            scored.sort(key=lambda x: x[1], reverse=True)
            chosen = [c for c, _ in scored[:budget]]
            return copy_subset(oracle_atlas, chosen)

        if policy == "sens_greedy":
            void_counts = self.edge_void_counts.get(eid, {})
            scored = []
            for c in classes:
                ace = class_ace(c)
                crit = float(void_counts.get(c, 0) + 1)
                # If ace is inf (missing), keep it very large.
                score = (1e9 if np.isinf(ace) else ace) * crit
                scored.append((c, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            chosen = [c for c, _ in scored[:budget]]
            return copy_subset(oracle_atlas, chosen)

        if policy == "dcas_risk_greedy":
            # Decision-Critical Atlas Synchronization (DCAS-lite):
            # pick classes that are both stale/drifting AND likely to flip curriculum decisions.
            risk = self._risk_scores_for_edge(eid, oracle_atlas, current_round, total_rounds)
            scored = [(c, float(risk.get(c, 0.0))) for c in classes]
            scored.sort(key=lambda x: x[1], reverse=True)
            chosen = [c for c, _ in scored[:budget]]
            return copy_subset(oracle_atlas, chosen)


        if policy == "dcas_g":
            # DCAS-G (Guarded, Delay-aware):
            #  - reserve a small number of "decision" slots scored at the *horizon* (t + delay)
            #  - fill the remaining slots with "utility" updates to avoid harming learning dynamics
            # This remains policy-only: Edge/clients/gating are unchanged.
            decision_slots = int(getattr(self.cfg, "dcas_g_decision_slots", 1))
            decision_slots = max(0, min(int(budget), int(decision_slots)))
            utility_slots = max(0, int(budget) - int(decision_slots))

            # Actuation-delay awareness: choose decision-critical classes for when the update arrives.
            horizon_round = int(current_round) + int(getattr(self.cfg, "delay_rounds", 0))
            horizon_round = max(0, min(int(total_rounds) - 1, int(horizon_round)))

            decision_scores = self._risk_scores_for_edge(int(eid), oracle_atlas, int(horizon_round), int(total_rounds))
            utility_scores = self._utility_scores_for_edge(int(eid), oracle_atlas, int(current_round), int(total_rounds))

            chosen: List[int] = []

            if decision_slots > 0 and decision_scores:
                scored = [(int(c), float(s)) for c, s in decision_scores.items()]
                scored.sort(key=lambda x: (-x[1], x[0]))  # deterministic ties
                chosen.extend([c for c, _ in scored[:decision_slots]])

            if utility_slots > 0 and utility_scores:
                scored = [(int(c), float(s)) for c, s in utility_scores.items()]
                scored.sort(key=lambda x: (-x[1], x[0]))  # deterministic ties
                for c, _ in scored:
                    if c not in chosen:
                        chosen.append(c)
                    if len(chosen) >= int(budget):
                        break

            # Fallback: fill any remaining slots deterministically.
            if len(chosen) < int(budget):
                for c in classes:
                    if int(c) not in chosen:
                        chosen.append(int(c))
                    if len(chosen) >= int(budget):
                        break

            return copy_subset(oracle_atlas, chosen)

        if policy == "dmr_oracle_greedy":
            # Upper bound: greedily pick classes that minimize DMR w.r.t oracle_atlas.
            # This uses oracle knowledge and is intended for evaluation only.
            chosen: List[int] = []
            base = copy.deepcopy(cur_state)
            seen_map = self.edge_clients_seen.get(eid, {})

            def dmr_for_state(state: Prototypes) -> float:
                return dmr_avg(
                    state,
                    oracle_atlas,
                    seen_map,
                    current_round,
                    total_rounds,
                    self.tau1p,
                    self.tau2p,
                    self.phase1,
                    self.phase2,
                )

            remaining = [c for c in classes]
            for _ in range(budget):
                best_c = None
                best_dmr = float("inf")
                for c in remaining:
                    temp = copy.deepcopy(base)
                    temp.update(copy_subset(oracle_atlas, [c]))
                    val = dmr_for_state(temp)
                    if val < best_dmr:
                        best_dmr = val
                        best_c = c
                if best_c is None:
                    break
                chosen.append(best_c)
                base.update(copy_subset(oracle_atlas, [best_c]))
                remaining.remove(best_c)
            return copy_subset(oracle_atlas, chosen)

        # fallback
        chosen = classes[:budget]
        return copy_subset(oracle_atlas, chosen)

    def maybe_send_updates(self, round_idx: int, oracle_atlas: Prototypes, total_rounds: int) -> None:
        """Schedule messages for delivery at round_idx + delay, subject to trigger."""
        if not oracle_atlas:
            return

        delay = int(self.cfg.delay_rounds)
        deliver_round = int(round_idx + delay)

        for eid in self.edge_state.keys():
            do_send = True
            trig = str(self.cfg.trigger)

            if trig == "ace":
                cur_ace = ace_avg(self.edge_state[eid], oracle_atlas)
                do_send = cur_ace > float(self.cfg.ace_threshold)
            elif trig == "dcas_risk":
                # Decision-critical (proxy) trigger: send only when stale/partial atlas is likely to
                # change curriculum decisions (under current phase).
                risk_sum_topB, risk_max = self._risk_stats_for_edge(
                    eid=int(eid),
                    oracle_atlas=oracle_atlas,
                    current_round=int(round_idx),
                    total_rounds=int(total_rounds),
                )
                self.risk_sum_topB_log.setdefault(int(round_idx), {})[int(eid)] = float(risk_sum_topB)
                self.risk_max_log.setdefault(int(round_idx), {})[int(eid)] = float(risk_max)

                do_send = float(risk_sum_topB) > float(getattr(self.cfg, "risk_threshold", 0.0))

                # optional cooldown to avoid over-triggering
                cd = int(getattr(self.cfg, "cooldown_rounds", 0))
                if do_send and cd > 0:
                    last = int(self.last_trigger_round.get(int(eid), -10**9))
                    if int(round_idx) - last <= cd:
                        do_send = False
                if do_send:
                    self.last_trigger_round[int(eid)] = int(round_idx)

            elif trig == "dmr_oracle":
                cur_dmr = dmr_avg(
                    self.edge_state[eid],
                    oracle_atlas,
                    self.edge_clients_seen.get(eid, {}),
                    int(round_idx),
                    int(total_rounds),
                    self.tau1p,
                    self.tau2p,
                    self.phase1,
                    self.phase2,
                )
                do_send = cur_dmr > float(self.cfg.dmr_threshold)

            if not do_send:
                continue

            # instrumentation: how many edges triggered at this round
            self.triggered_edges[int(round_idx)] = int(self.triggered_edges.get(int(round_idx), 0) + 1)

            payload = self._select_classes_for_edge(eid, oracle_atlas, int(round_idx), int(total_rounds))
            self.pending.append((deliver_round, int(eid), payload))

            b = protos_nbytes(payload)
            self.sent_bytes[round_idx] = int(self.sent_bytes.get(round_idx, 0) + b)

    def deliver_due(self, round_idx: int) -> int:
        """Deliver any pending updates scheduled for round_idx. Returns bytes delivered."""
        due = [(dr, eid, p) for (dr, eid, p) in self.pending if dr == int(round_idx)]
        if not due:
            self.delivered_bytes[round_idx] = int(self.delivered_bytes.get(round_idx, 0))
            return 0

        remaining = [(dr, eid, p) for (dr, eid, p) in self.pending if dr != int(round_idx)]
        self.pending = remaining

        delivered = 0
        for _, eid, payload in due:
            # merge update into edge atlas state
            if payload:
                self.edge_state[eid].update(copy.deepcopy(payload))
                # mark freshness per class at this edge
                for c in payload.keys():
                    self.last_delivered_round[eid][int(c)] = int(round_idx)
                delivered += protos_nbytes(payload)

        self.delivered_bytes[round_idx] = int(self.delivered_bytes.get(round_idx, 0) + delivered)
        return int(delivered)

    # --------------------
    # Decision-critical risk proxy (DCAS-lite)
    # --------------------
    def _edge_sensitivity_scores(
        self,
        eid: int,
        atlas: Prototypes,
        current_round: int,
        total_rounds: int,
    ) -> Dict[int, float]:
        """Approximate decision sensitivity per class for this edge.

        High when a class (void or influential seen) lies near the *active* curriculum
        boundary. This uses only the edge's current atlas view + the client seen-sets
        (no oracle knowledge).

        Returns: {class_id: sensitivity_score}
        """
        if not atlas:
            return {}

        # Determine which boundary is active for the current curriculum phase.
        progress = float(current_round) / float(max(1, total_rounds))
        if progress < float(self.phase1):
            boundary_mode: Optional[str] = "tau1"
        elif progress < float(self.phase2):
            boundary_mode = "tau2"
        else:
            boundary_mode = None  # all-allowed phase => distances don't affect unlock set

        if boundary_mode is None:
            return {}

        eps = 1e-6
        scores: Dict[int, float] = {}
        counts: Dict[int, int] = {}

        seen_map = self.edge_clients_seen.get(int(eid), {}) or {}
        for _, seen_classes in seen_map.items():
            seen_present = [c for c in seen_classes if c in atlas]
            if not seen_present:
                continue

            void_classes = [c for c in atlas.keys() if c not in seen_classes]
            if not void_classes:
                continue

            dist_triplets: List[Tuple[int, float, int]] = []
            for v in void_classes:
                mu_v = _to_np(atlas[int(v)]["mean"]).astype(np.float64)
                best = float("inf")
                best_s = -1
                for s in seen_present:
                    mu_s = _to_np(atlas[int(s)]["mean"]).astype(np.float64)
                    d = float(np.linalg.norm(mu_v - mu_s))
                    if d < best:
                        best = d
                        best_s = int(s)
                if best_s >= 0 and np.isfinite(best):
                    dist_triplets.append((int(v), float(best), int(best_s)))

            if not dist_triplets:
                continue

            dvals = np.asarray([d for _, d, _ in dist_triplets], dtype=np.float64)
            tau1 = float(np.percentile(dvals, float(self.tau1p)))
            tau2 = float(np.percentile(dvals, float(self.tau2p)))
            boundary = tau1 if boundary_mode == "tau1" else tau2

            for v, d, s_nn in dist_triplets:
                margin = abs(float(d) - float(boundary))
                sens = 1.0 / (margin + eps)

                scores[v] = float(scores.get(v, 0.0) + sens)
                counts[v] = int(counts.get(v, 0) + 1)

                # also credit the nearest seen class (it influences d(k,v))
                if s_nn >= 0:
                    scores[s_nn] = float(scores.get(s_nn, 0.0) + 0.30 * sens)
                    counts[s_nn] = int(counts.get(s_nn, 0) + 1)

        out: Dict[int, float] = {}
        for c, val in scores.items():
            out[int(c)] = float(val) / float(max(1, counts.get(int(c), 1)))
        return out


    def _utility_scores_for_edge(
        self,
        eid: int,
        oracle_atlas: Prototypes,
        current_round: int,
        total_rounds: int,
    ) -> Dict[int, float]:
        """Learning-utility proxy per class for DCAS-G.

        Unlike the decision-risk proxy, this *excludes* boundary sensitivity and focuses on:
          - prototype drift / missingness (refresh stale state)
          - void criticality at the edge (helps void recovery)
          - age since last delivery (coverage / anti-starvation)
        """
        classes = [int(c) for c in oracle_atlas.keys()]
        if not classes:
            return {}

        cur_state = self.edge_state.get(int(eid), {}) or {}
        void_counts = self.edge_void_counts.get(int(eid), {}) or {}
        n_clients = max(1, int(self.edge_num_clients.get(int(eid), 0)))

        def class_drift(c: int) -> float:
            if c not in oracle_atlas:
                return 0.0
            mu_o = _to_np(oracle_atlas[int(c)]["mean"]).astype(np.float64)
            if c not in cur_state:
                return float(np.linalg.norm(mu_o) + 1.0)
            mu_e = _to_np(cur_state[int(c)]["mean"]).astype(np.float64)
            return float(np.linalg.norm(mu_e - mu_o))

        out: Dict[int, float] = {}
        for c in classes:
            drift = float(class_drift(int(c))) + 1e-12

            crit = float(void_counts.get(int(c), 0)) / float(n_clients)  # [0,1]
            crit_factor = 0.10 + crit

            last = self.last_delivered_round.get(int(eid), {}).get(int(c), -1)
            age = int(current_round) - int(last) if int(last) >= 0 else int(current_round) + 1
            age_factor = 1.0 + min(float(age), 50.0) / 10.0

            out[int(c)] = drift * crit_factor * age_factor

        return out

    def _risk_scores_for_edge(
        self,
        eid: int,
        oracle_atlas: Prototypes,
        current_round: int,
        total_rounds: int,
    ) -> Dict[int, float]:
        """Decision-risk proxy per class for this edge (DCAS-lite).

        risk(c) increases when:
          - prototype drift is large (ACE per class) OR class is missing at the edge
          - the class is void-critical for many clients at the edge
          - the class is near the active curriculum boundary (phase-dependent)
          - the class hasn't been refreshed recently (age)
        """
        classes = [int(c) for c in oracle_atlas.keys()]
        if not classes:
            return {}

        cur_state = self.edge_state.get(int(eid), {}) or {}
        void_counts = self.edge_void_counts.get(int(eid), {}) or {}
        n_clients = max(1, int(self.edge_num_clients.get(int(eid), 0)))

        sens = self._edge_sensitivity_scores(int(eid), cur_state, int(current_round), int(total_rounds))

        def class_drift(c: int) -> float:
            if c not in oracle_atlas:
                return 0.0
            mu_o = _to_np(oracle_atlas[int(c)]["mean"]).astype(np.float64)
            if c not in cur_state:
                # missing at edge: treat as high drift proportional to prototype norm
                return float(np.linalg.norm(mu_o) + 1.0)
            mu_e = _to_np(cur_state[int(c)]["mean"]).astype(np.float64)
            return float(np.linalg.norm(mu_e - mu_o))

        out: Dict[int, float] = {}
        for c in classes:
            drift = float(class_drift(int(c)))

            crit = float(void_counts.get(int(c), 0)) / float(n_clients)  # [0,1]
            crit_factor = 0.10 + crit

            last = self.last_delivered_round.get(int(eid), {}).get(int(c), -1)
            age = int(current_round) - int(last) if int(last) >= 0 else int(current_round) + 1
            age_factor = 1.0 + min(float(age), 50.0) / 10.0

            s = float(sens.get(int(c), 0.0))
            sens_factor = 1.0 + float(np.tanh(s / 5.0))  # bounded

            out[int(c)] = drift * crit_factor * age_factor * sens_factor

        return out

    def _risk_stats_for_edge(
        self,
        eid: int,
        oracle_atlas: Prototypes,
        current_round: int,
        total_rounds: int,
    ) -> Tuple[float, float]:
        """Return (risk_sum_topB, risk_max) for this edge."""
        risk = self._risk_scores_for_edge(int(eid), oracle_atlas, int(current_round), int(total_rounds))
        if not risk:
            return 0.0, 0.0

        vals = list(risk.values())
        risk_max = float(max(vals)) if vals else 0.0

        B = int(getattr(self.cfg, "class_budget", -1))
        if B < 0:
            risk_sum_topB = float(np.sum(vals))
        else:
            top = sorted(vals, reverse=True)[: max(1, B)]
            risk_sum_topB = float(np.sum(top))

        return float(risk_sum_topB), float(risk_max)

    def round_risk_stats(self, round_idx: int) -> Dict[str, float]:
        """Convenience accessor for logging/debugging."""
        rs = self.risk_sum_topB_log.get(int(round_idx), {}) or {}
        rm = self.risk_max_log.get(int(round_idx), {}) or {}
        return {
            "risk_sum_topB_mean": float(np.mean(list(rs.values()))) if rs else 0.0,
            "risk_max_mean": float(np.mean(list(rm.values()))) if rm else 0.0,
            "triggered_edges": float(self.triggered_edges.get(int(round_idx), 0)),
        }

