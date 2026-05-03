"""
student/algoritmoSmartEcoRutas.py
==================================
SmartEcoRutas VRP Solver — UPCT / Lhicarsa competition.

Pipeline
--------
1. Clarke-Wright savings  — greedy deterministic construction
2. Route merge pass        — aggressively reduce route count
3. Local search            — relocate, or-opt, 2-opt, route-merge
4. ILS loop               — double-bridge perturbation of best + LS until deadline
5. Final route merge pass  — squeeze last route reductions before output

Internal representation
-----------------------
Routes are stored as  list[list[int]]  where every int is a *container
position* — an index into self.cont_uids / self.cont_mi.
BASE and DUMP nodes are implicit; to_uid_route() inserts them on export.

Scoring (primary → tiebreaker)
-------------------------------
1. Minimum total routes   — drives every design decision
2. Minimum total travel time
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np


# =============================================================================
# Public entry point
# =============================================================================

def solve(problem, time_limit_s: float, seed: int | None = None) -> list[list[str]]:
    """
    Called by the framework.  Returns a list of routes:
        [["BASE", "c_001", ..., "DUMP", "BASE"], ...]
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Keep 2 s safety margin so we never time-out mid-export
    deadline = time.time() + time_limit_s - 2.0
    return _Solver(problem).run(deadline)


# =============================================================================
# Solver
# =============================================================================

class _Solver:
    """
    Encapsulates all problem data and the solving pipeline.

    Containers are referenced by their *position* (index 0 ... n-1) into
    self.cont_uids / self.cont_mi, not by UID string.  This avoids repeated
    dict lookups and lets us use plain Python lists throughout the hot paths.
    """

    # ── tuning knobs ──────────────────────────────────────────────────────────
    K_RELOC          = 35   # nearest-neighbour count for relocate / intra-route ops
    SOLOMON_S        = 50   # budget for multi-start Solomon I1 construction (seconds)
    SOLOMON_LS_S     = 5   # LS budget per Solomon candidate before geo-pipeline
    SOLOMON_K_INSERT = 25   # insertion candidate positions per container
    N_SOL_CANDIDATES = 3    # top-K Solomon solutions fed into geo-pipeline slices
    MAX_RESTARTS     = 3    # double-bridge restarts per slice (cap; LNS tried first)

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, problem):
        p = problem

        self.p      = p
        self.BASE   = p.base_uid()
        self.DUMP   = p.dump_uid()
        self.CAP    = int(p.max_containers_before_dump)
        self.MAX_T  = float(p.route_max_work_s)
        self.svc_c  = float(p.service_time_container_s)
        self.svc_d  = float(p.service_time_dump_s)

        # Container arrays
        self.cont_uids: list[str] = p.containers_uids()
        self.n = len(self.cont_uids)
        self.cont_mi: list[int]   = [p.uid_to_index(u) for u in self.cont_uids]
        self.mi_to_pos: dict[int, int] = {mi: pos for pos, mi in enumerate(self.cont_mi)}

        self.base_mi: int = p.base_index
        self.dump_mi: int = p.dump_index

        # Travel-time matrix as Python list-of-lists for fast scalar access
        # (avoids numpy overhead on repeated small T[i][j] lookups in hot loops)
        self.T: list[list[float]] = p.T.tolist()

        # Precomputed k-nearest neighbour positions
        print("[Solver] Precomputing neighbour positions ...")
        self._nbr_pos: list[list[int]] = self._precompute_nbr_pos(self.K_RELOC)
        print(f"[Solver] Ready — {self.n} containers.")

    # ------------------------------------------------------------------
    # Precomputation helpers
    # ------------------------------------------------------------------

    def _precompute_nbr_pos(self, k: int) -> list[list[int]]:
        """Return, for each container position, a list of k nearest positions."""
        result: list[list[int]] = []
        for pos, uid in enumerate(self.cont_uids):
            nbrs = self.p.k_nearest(uid, k, only_containers=True)
            result.append([self.mi_to_pos[self.p.uid_to_index(u)] for u, _ in nbrs])
        return result


    def route_time(self, route: list[int]) -> float:
        """
        Total time (travel + service) for a container-position sequence.
        DUMP nodes are inserted implicitly every CAP containers.
        """
        T     = self.T
        base  = self.base_mi
        dump  = self.dump_mi
        cap   = self.CAP
        svc_c = self.svc_c
        svc_d = self.svc_d
        cont  = self.cont_mi

        total = 0.0
        prev  = base
        load  = 0
        for pos in route:
            mi = cont[pos]
            if load == cap:
                total += T[prev][dump] + svc_d
                prev   = dump
                load   = 0
            total += T[prev][mi] + svc_c
            prev   = mi
            load  += 1
        # Final compulsory DUMP → BASE
        total += T[prev][dump] + svc_d + T[dump][base]
        return total

    # ------------------------------------------------------------------
    # Solution cost  (lexicographic: fewer routes > less time)
    # ------------------------------------------------------------------

    def cost(self, routes: list[list[int]]) -> tuple[int, float]:
        return len(routes), sum(self.route_time(r) for r in routes)

    # ------------------------------------------------------------------
    # Phase-2/3/5 helpers (PHASES_2-5.md §1.3, §1.4)
    # ------------------------------------------------------------------

    def route_travel(self, route: list[int]) -> float:
        """Travel-only time for a route (no per-container or per-DUMP service)."""
        T     = self.T
        base  = self.base_mi
        dump  = self.dump_mi
        cap   = self.CAP
        cont  = self.cont_mi

        if not route:
            return 0.0
        total = 0.0
        prev  = base
        load  = 0
        for pos in route:
            mi = cont[pos]
            if load == cap:
                total += T[prev][dump] + T[dump][mi]
                prev   = mi
                load   = 1
            else:
                total += T[prev][mi]
                prev   = mi
                load  += 1
        total += T[prev][dump] + T[dump][base]
        return total

    def total_travel(self, routes: list[list[int]]) -> float:
        """Sum of travel-only time across all routes (PHASES_2-5.md §1.3)."""
        return sum(self.route_travel(r) for r in routes)

    # ------------------------------------------------------------------
    # Output conversion
    # ------------------------------------------------------------------

    def to_uid_route(self, pos_list: list[int]) -> list[str]:
        """Convert internal position list → full UID route with BASE/DUMP."""
        route = [self.BASE]
        load  = 0
        for pos in pos_list:
            if load == self.CAP:
                route.append(self.DUMP)
                load = 0
            route.append(self.cont_uids[pos])
            load += 1
        route.append(self.DUMP)
        route.append(self.BASE)
        return route

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Clarke-Wright savings construction
    # ══════════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════════
    # Route-merge pass   (directly attacks primary objective)
    # ══════════════════════════════════════════════════════════════════════════

    def route_merge_pass(self, routes: list[list[int]]) -> list[list[int]]:
        """
        Try every ordered pair (A, B): if A+B is feasible, merge them.
        Repeat until no more merges are possible.
        Each successful merge removes one route from the solution.
        """
        improved = True
        while improved:
            improved = False
            m        = len(routes)
            found    = False
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    merged = routes[i] + routes[j]
                    if self.route_time(merged) <= self.MAX_T:
                        routes = [r for k, r in enumerate(routes) if k != i and k != j]
                        routes.append(merged)
                        improved = True
                        found    = True
                        break
                if found:
                    break
        return routes

    # ══════════════════════════════════════════════════════════════════════════
    # Local search
    # ══════════════════════════════════════════════════════════════════════════

    def local_search(
        self,
        routes:   list[list[int]],
        deadline: float,
    ) -> list[list[int]]:
        """
        Operators applied in priority order (first-improvement):

        1. route_merge  — combine two routes entirely  → −1 route
        2. relocate     — move one container inter-route (neighbour-guided)
        3. or_opt(2)    — move two consecutive containers inter-route
        4. 2opt_intra   — reverse a segment within one route (travel time)
        """
        times    = [self.route_time(r) for r in routes]
        improved = True

        while improved and time.time() < deadline:
            improved = False

            routes, times, ok = self._op_route_merge(routes, times)
            if ok:
                improved = True
                continue

            routes, times, ok = self._op_relocate(routes, times)
            if ok:
                improved = True
                continue

            routes, times, ok = self._op_or_opt(routes, times, seg=2)
            if ok:
                improved = True
                continue

            routes, times, ok = self._op_2opt(routes, times)
            if ok:
                improved = True

        return routes

    # ── operator: route merge ─────────────────────────────────────────────────

    def _op_route_merge(
        self,
        routes: list[list[int]],
        times:  list[float],
    ) -> tuple[list[list[int]], list[float], bool]:
        """Merge ordered pair (i, j) if feasible — first hit wins."""
        m     = len(routes)
        MAX_T = self.MAX_T
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                merged = routes[i] + routes[j]
                t      = self.route_time(merged)
                if t <= MAX_T:
                    nr = [r  for k, r  in enumerate(routes) if k != i and k != j]
                    nt = [tt for k, tt in enumerate(times)  if k != i and k != j]
                    nr.append(merged)
                    nt.append(t)
                    return nr, nt, True
        return routes, times, False

    # ── operator: relocate ────────────────────────────────────────────────────

    def _op_relocate(
        self,
        routes: list[list[int]],
        times:  list[float],
    ) -> tuple[list[list[int]], list[float], bool]:
        """
        Move one container from route i → route j (first improvement).

        Candidate destination routes come from the precomputed k-nearest
        neighbours of the container being moved, so we skip routes that
        are geographically distant (and almost certainly won't improve).

        Improvement is judged lexicographically: (n_routes, total_time).
        Eliminating a route (ri becomes empty) always wins.
        """
        MAX_T = self.MAX_T
        m     = len(routes)
        old_n = m
        old_t = sum(times)

        # Build container-position → route-index map
        pos_to_ri: dict[int, int] = {}
        for ri, r in enumerate(routes):
            for pos in r:
                pos_to_ri[pos] = ri

        for i in range(m):
            ri = routes[i]
            for k_in, pos in enumerate(ri):
                # What route i looks like without this container
                new_ri   = ri[:k_in] + ri[k_in + 1:]
                ri_empty = len(new_ri) == 0
                t_new_ri = self.route_time(new_ri) if new_ri else 0.0

                # Candidate routes: those containing a nearest neighbour of pos
                cand_js: set[int] = set()
                for nbr_pos in self._nbr_pos[pos]:
                    j = pos_to_ri.get(nbr_pos)
                    if j is not None and j != i:
                        cand_js.add(j)

                for j in cand_js:
                    rj = routes[j]
                    # Find cheapest feasible insertion position in rj
                    best_t_j = float("inf")
                    best_ins = -1
                    for ins in range(len(rj) + 1):
                        new_rj = rj[:ins] + [pos] + rj[ins:]
                        t = self.route_time(new_rj)
                        if t < best_t_j and t <= MAX_T:
                            best_t_j = t
                            best_ins = ins

                    if best_ins < 0:
                        continue  # no feasible insertion

                    new_n = old_n - 1 if ri_empty else old_n
                    new_t = (old_t
                             - times[i] - times[j]
                             + (t_new_ri if not ri_empty else 0.0)
                             + best_t_j)

                    if (new_n, new_t) < (old_n, old_t):
                        new_rj = rj[:best_ins] + [pos] + rj[best_ins:]
                        nr: list[list[int]] = []
                        nt: list[float]     = []
                        for k2 in range(m):
                            if k2 == i:
                                if not ri_empty:
                                    nr.append(new_ri); nt.append(t_new_ri)
                            elif k2 == j:
                                nr.append(new_rj);    nt.append(best_t_j)
                            else:
                                nr.append(routes[k2]); nt.append(times[k2])
                        return nr, nt, True

        return routes, times, False

    # ── operator: or-opt ──────────────────────────────────────────────────────

    def _op_or_opt(
        self,
        routes: list[list[int]],
        times:  list[float],
        seg:    int = 2,
    ) -> tuple[list[list[int]], list[float], bool]:
        """
        Move a block of `seg` consecutive containers from route i → route j.
        Same lexicographic improvement criterion as relocate.
        """
        MAX_T = self.MAX_T
        m     = len(routes)
        old_n = m
        old_t = sum(times)

        for i in range(m):
            ri = routes[i]
            if len(ri) < seg:
                continue
            for k_s in range(len(ri) - seg + 1):
                segment  = ri[k_s: k_s + seg]
                new_ri   = ri[:k_s] + ri[k_s + seg:]
                ri_empty = len(new_ri) == 0
                t_new_ri = self.route_time(new_ri) if new_ri else 0.0

                for j in range(m):
                    if i == j:
                        continue
                    rj = routes[j]
                    best_t_j = float("inf")
                    best_ins = -1
                    for ins in range(len(rj) + 1):
                        new_rj = rj[:ins] + segment + rj[ins:]
                        t = self.route_time(new_rj)
                        if t < best_t_j and t <= MAX_T:
                            best_t_j = t
                            best_ins = ins

                    if best_ins < 0:
                        continue

                    new_n = old_n - 1 if ri_empty else old_n
                    new_t = (old_t
                             - times[i] - times[j]
                             + (t_new_ri if not ri_empty else 0.0)
                             + best_t_j)

                    if (new_n, new_t) < (old_n, old_t):
                        new_rj = rj[:best_ins] + segment + rj[best_ins:]
                        nr: list[list[int]] = []
                        nt: list[float]     = []
                        for k2 in range(m):
                            if k2 == i:
                                if not ri_empty:
                                    nr.append(new_ri); nt.append(t_new_ri)
                            elif k2 == j:
                                nr.append(new_rj);    nt.append(best_t_j)
                            else:
                                nr.append(routes[k2]); nt.append(times[k2])
                        return nr, nt, True

        return routes, times, False

    # ── operator: 2-opt intra-route ───────────────────────────────────────────

    def _op_2opt(
        self,
        routes: list[list[int]],
        times:  list[float],
    ) -> tuple[list[list[int]], list[float], bool]:
        """
        Intra-route 2-opt: reverse a segment [a..b] to reduce travel time.
        For each route, finds the single best reversal (best-improvement
        within each route), then returns immediately on the first route
        where improvement is found.
        """
        MAX_T = self.MAX_T
        for i, ri in enumerate(routes):
            n_r = len(ri)
            if n_r < 3:
                continue
            t_best   = times[i]
            best_rev = None
            for a in range(n_r - 1):
                for b in range(a + 2, n_r):
                    candidate = ri[:a] + ri[a: b + 1][::-1] + ri[b + 1:]
                    t         = self.route_time(candidate)
                    if t < t_best and t <= MAX_T:
                        t_best   = t
                        best_rev = candidate
            if best_rev is not None:
                nr = list(routes); nr[i] = best_rev
                nt = list(times);  nt[i] = t_best
                return nr, nt, True

        return routes, times, False


    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — multi-start CW branch
    # ══════════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Solomon I1 insertion branch
    # ══════════════════════════════════════════════════════════════════════════

    def _solomon_i1(self, positions: list[int], seed: int,
                    lam: float = 1.0) -> list[list[int]]:
        """
        Solomon's I1 insertion heuristic adapted for time-constrained VRP.

        Builds routes one at a time:
          - Each route is seeded with a far-from-BASE container.
          - At each step, insert the unassigned container u with the highest
            c2(u) = lam * T[BASE→u] − best_delta(u), where best_delta is the
            minimum feasible time increase across all candidate insertion
            positions.  High c2 means u is both far away (urgent to insert
            early) and cheap to slot in now.
          - When no container fits the current route, close it and open a new
            one seeded with the farthest remaining unassigned container.

        Candidate positions are limited to neighbours of u already in the
        route (via _nbr_pos) plus both endpoints, keeping cost O(n·K·route).
        """
        T, cont  = self.T, self.cont_mi
        base_mi  = self.base_mi
        MAX_T    = self.MAX_T
        K        = self.SOLOMON_K_INSERT
        nbr_pos  = self._nbr_pos

        # Sort by distance from BASE descending for tie-breaking new seeds
        unassigned: list[int] = sorted(
            positions, key=lambda p: -T[base_mi][cont[p]])
        unassigned = [p for p in unassigned if p != seed]

        routes:   list[list[int]] = []
        current:  list[int]       = [seed]
        cur_time: float           = self.route_time([seed])
        in_route: set[int]        = {seed}

        while unassigned:
            best_u:     int | None  = None
            best_ins:   int         = 0
            best_c2:    float       = -float('inf')
            best_delta: float       = 0.0

            for u in unassigned:
                b_delta: float    = float('inf')
                b_ins:   int | None = None

                # Candidate insertion slots: adjacent to u's neighbours in route
                # + both endpoints
                slots: set[int] = {0, len(current)}
                for nbr in nbr_pos[u][:K]:
                    if nbr in in_route:
                        idx = current.index(nbr)
                        slots.add(idx)
                        slots.add(idx + 1)

                for ins in slots:
                    candidate = current[:ins] + [u] + current[ins:]
                    t = self.route_time(candidate)
                    if t <= MAX_T:
                        delta = t - cur_time
                        if delta < b_delta:
                            b_delta, b_ins = delta, ins

                if b_ins is None:
                    continue

                c2 = lam * T[base_mi][cont[u]] - b_delta
                if c2 > best_c2:
                    best_c2, best_u, best_ins, best_delta = c2, u, b_ins, b_delta

            if best_u is not None:
                current.insert(best_ins, best_u)
                cur_time += best_delta
                in_route.add(best_u)
                unassigned.remove(best_u)
            else:
                routes.append(current)
                # New seed: farthest remaining (first in sorted list)
                new_seed = unassigned[0]
                current  = [new_seed]
                cur_time = self.route_time([new_seed])
                in_route = {new_seed}
                unassigned.remove(new_seed)

        if current:
            routes.append(current)
        return routes

    def _multi_start_solomon(
        self, deadline: float, n_keep: int | None = None,
    ) -> list[list[list[int]]]:
        """
        Run _solomon_i1() repeatedly with varied seeds and λ values.
        Returns the top n_keep solutions sorted by cost (best first).

        Seed strategy:
          Runs 0..n_keep-1 : n_keep farthest containers from BASE —
                             deterministic, spread across geographic extremes.
          Runs n_keep+     : random from top-25% farthest containers, weighted
                             by distance from BASE.

        λ rotates through [1.0, 0.5, 2.0] to vary the urgency/cost balance.
        """
        if n_keep is None:
            n_keep = self.N_SOL_CANDIDATES
        all_pos  = list(range(self.n))
        T, cont  = self.T, self.cont_mi
        base_mi  = self.base_mi

        # Deterministic seeds: n_keep farthest containers from BASE
        sorted_far  = sorted(all_pos, key=lambda p: -T[base_mi][cont[p]])
        det_seeds   = sorted_far[:n_keep]

        # Random pool: top 25% farthest containers
        top_pool    = sorted_far[:max(1, self.n // 4)]
        top_weights = [T[base_mi][cont[p]] for p in top_pool]

        lams = [1.0, 0.5, 2.0]
        # Sorted list of (cost, solution), best first; bounded to n_keep
        top_k: list[tuple[tuple[int, float], list[list[int]]]] = []
        run = 0

        while time.time() < deadline:
            lam  = lams[run % len(lams)]
            seed = (det_seeds[run] if run < len(det_seeds)
                    else random.choices(top_pool, weights=top_weights)[0])

            routes = self._solomon_i1(all_pos, seed=seed, lam=lam)
            routes = self.route_merge_pass(routes)
            c = self.cost(routes)

            top_k.append((c, [list(r) for r in routes]))
            top_k.sort(key=lambda x: x[0])
            top_k = top_k[:n_keep]

            is_best = bool(top_k) and c == top_k[0][0]
            marker  = "* new best" if is_best else "         "
            print(f"[Sol run {run:3d}] {marker} {c[0]}r {c[1]/3600:.3f}h"
                  f"  lam={lam} seed={seed}")
            run += 1

        best_c = top_k[0][0] if top_k else (0, 0.0)
        print(f"[Solomon multi-start] {run} runs → best={best_c[0]}r"
              f" {best_c[1]/3600:.3f}h  keeping top-{len(top_k)}")
        return [sol for _, sol in top_k]

    # ══════════════════════════════════════════════════════════════════════════
    # ALNS — startup: DBSCAN clustering
    # ══════════════════════════════════════════════════════════════════════════


    def _save_uid_snapshot(self, tag: str, routes: list[list[int]]) -> None:
        """Save UID routes for a pipeline stage so visualize.py can draw them."""
        instance_name = getattr(self.p, "subproblem", None) or self.p.instance_dir.name
        out_dir = Path.cwd() / "extra_algorithm_output" / instance_name
        out_dir.mkdir(parents=True, exist_ok=True)
        n_r, total_t = self.cost(routes)
        payload = {
            "tag":         tag,
            "n_routes":    n_r,
            "total_time_s": round(total_t, 1),
            "routes":      [self.to_uid_route(r) for r in routes],
        }
        with open(out_dir / f"{tag}_routes.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)

    # ══════════════════════════════════════════════════════════════════════════

    def run(self, deadline: float) -> list[list[str]]:
        """
        Multi-slice pipeline:
          1. Solomon multi-start (SOLOMON_S s) → top N_SOL_CANDIDATES solutions
          2. Remaining time divided equally among candidates
          3. Each slice: SOLOMON_LS_S of local search + geo-aware pipeline
             At end of each slice the best result is logged and snapshotted.
          4. Global best across all slices is returned.
        """
        t0 = time.time()

        # ── Phase 1: multi-start Solomon I1 → top-N candidates ───────────────
        t_sol    = time.time() + self.SOLOMON_S
        top_sols = self._multi_start_solomon(t_sol)
        if not top_sols:
            raise RuntimeError("Solomon multi-start produced no solutions")

        # Save the best construction result for visualize.py
        self._save_uid_snapshot("solomon_construction", top_sols[0])

        # ── Divide remaining time equally among candidates ────────────────────
        n_cands   = len(top_sols)
        remaining = deadline - time.time()
        slice_s   = remaining / n_cands

        global_best:   list[list[int]]   = top_sols[0]
        global_best_c: tuple[int, float] = self.cost(global_best)

        for k, sol in enumerate(top_sols):
            is_last   = (k == n_cands - 1)
            slice_end = min(deadline, time.time() + slice_s)

            # Local search on this candidate
            t_ls = min(slice_end, time.time() + self.SOLOMON_LS_S)
            sol  = self.local_search(sol, t_ls)
            snap_tag = "solomon_phase1" if k == 0 else f"sol_cand_{k+1}_phase1"
            self._save_uid_snapshot(snap_tag, sol)
            c_ls = self.cost(sol)
            print(f"[Slice {k+1}/{n_cands}] after LS: "
                  f"{c_ls[0]}r {c_ls[1]/3600:.3f}h  "
                  f"({time.time()-t0:.1f}s elapsed)")

            # Geo-aware pipeline for this slice
            sol = self._s_driver(sol, slice_end, is_last_slice=is_last)
            c   = self.cost(sol)

            # Log and snapshot end of slice
            self._save_uid_snapshot(f"slice_{k+1}_best", sol)
            print(f"[Slice {k+1}/{n_cands} END] "
                  f"best={c[0]}r {c[1]/3600:.3f}h  "
                  f"elapsed={time.time()-t0:.1f}s")

            if c < global_best_c:
                global_best   = sol
                global_best_c = c

        print(f"[Final] routes={global_best_c[0]}, "
              f"total_time={global_best_c[1]:.0f}s  "
              f"({time.time()-t0:.1f}s elapsed)")
        self._save_extra_result(global_best)
        return [self.to_uid_route(r) for r in global_best]

    # ══════════════════════════════════════════════════════════════════════════
    # Post-Solomon pipeline — geography-aware operators (S1..S5)
    # ══════════════════════════════════════════════════════════════════════════

    def _s_precompute(self, routes):
        self._s_x = [self.p.uid_to_node(u).x for u in self.cont_uids]
        self._s_y = [self.p.uid_to_node(u).y for u in self.cont_uids]

    def _s_centroid(self, route):
        if not route:
            return (0.0, 0.0)
        xs = self._s_x; ys = self._s_y
        n = len(route)
        return (sum(xs[p] for p in route) / n, sum(ys[p] for p in route) / n)

    def _s_euclid(self, pa, pb):
        dx = pa[0] - pb[0]; dy = pa[1] - pb[1]
        return (dx*dx + dy*dy) ** 0.5

    def _s_isolation(self, c, ri, others):
        """High value → c is geometrically misplaced in others[ri]."""
        T = self.T; cont = self.cont_mi; mi_c = cont[c]
        r = others[ri]
        same = [x for x in r if x != c]
        min_same = min(T[mi_c][cont[x]] for x in same) if same else 0.0
        best_other = float("inf")
        for rj, r2 in enumerate(others):
            if rj == ri:
                continue
            for x in r2:
                v = T[mi_c][cont[x]]
                if v < best_other:
                    best_other = v
        if best_other == float("inf"):
            return 0.0
        return min_same / (best_other + 1e-9)

    def _s_best_insert(self, c, route):
        """(route_time, ins_idx) of cheapest feasible insertion, or None."""
        MAX_T = self.MAX_T; best = None
        for ins in range(len(route) + 1):
            t = self.route_time(route[:ins] + [c] + route[ins:])
            if t <= MAX_T and (best is None or t < best[0]):
                best = (t, ins)
        return best

    def _best_insert_any(self, c: int, route: list[int]) -> tuple[float, int]:
        """Cheapest insertion position regardless of feasibility (no MAX_T guard)."""
        best_t = float("inf"); best_ins = 0
        for ins in range(len(route) + 1):
            t = self.route_time(route[:ins] + [c] + route[ins:])
            if t < best_t:
                best_t = t; best_ins = ins
        return best_t, best_ins

    # ── intra-route helpers ───────────────────────────────────────────────────

    def _intra_2opt_route(self, route):
        MAX_T = self.MAX_T; n_r = len(route)
        if n_r < 3:
            return route
        t_best = self.route_time(route); best_rev = None
        for a in range(n_r - 1):
            for b in range(a + 2, n_r):
                cand = route[:a] + route[a:b+1][::-1] + route[b+1:]
                t = self.route_time(cand)
                if t < t_best and t <= MAX_T:
                    t_best = t; best_rev = cand
        return best_rev if best_rev is not None else route

    def _intra_oropt_route(self, route, seg):
        MAX_T = self.MAX_T; nbr = self._nbr_pos
        n_r = len(route)
        if n_r < seg + 2:
            return route
        improved = True; cur = route
        while improved:
            improved = False; n_r = len(cur)
            t_orig = self.route_time(cur)
            idx_map = {p: i for i, p in enumerate(cur)}
            for k_s in range(n_r - seg + 1):
                segment = cur[k_s: k_s + seg]
                base = cur[:k_s] + cur[k_s + seg:]
                slots = {0, len(base)}
                for cand_pos in nbr[segment[0]]:
                    i_b = idx_map.get(cand_pos)
                    if i_b is not None:
                        slots.add(i_b); slots.add(i_b + 1)
                for ins in slots:
                    if ins == k_s:
                        continue
                    cand = base[:ins] + segment + base[ins:]
                    t = self.route_time(cand)
                    if t < t_orig - 1e-6 and t <= MAX_T:
                        cur = cand; improved = True; break
                if improved:
                    break
        return cur

    # ── infeasibility-aware intra helpers (accept any improvement, no MAX_T guard)

    def _intra_2opt_infeas(self, route):
        """Intra 2-opt iterated to convergence, no MAX_T guard.

        Each pass finds the best single reversal. Repeats until no improvement.
        Used during compression to shave down routes that are tens–hundreds of
        seconds over MAX_T without the MAX_T guard blocking partial gains.
        """
        cur = route
        while True:
            n_r = len(cur)
            if n_r < 3:
                return cur
            t_best = self.route_time(cur); best_rev = None
            for a in range(n_r - 1):
                for b in range(a + 2, n_r):
                    cand = cur[:a] + cur[a:b+1][::-1] + cur[b+1:]
                    t = self.route_time(cand)
                    if t < t_best - 1e-6:
                        t_best = t; best_rev = cand
            if best_rev is None:
                return cur
            cur = best_rev

    def _intra_oropt_full_infeas(self, route, seg):
        """Full-slot or-opt (all insertion positions, not just k-NN).

        O(n²) per pass but n is typically 50–200, fast enough. Essential for
        shaving last 100–500s of overload where k-NN slots miss the optimum.
        """
        n_r = len(route)
        if n_r < seg + 2:
            return route
        improved = True; cur = route
        while improved:
            improved = False
            n_r = len(cur)
            t_orig = self.route_time(cur)
            for k_s in range(n_r - seg + 1):
                segment = cur[k_s: k_s + seg]
                base = cur[:k_s] + cur[k_s + seg:]
                for ins in range(len(base) + 1):
                    if ins == k_s:
                        continue
                    cand = base[:ins] + segment + base[ins:]
                    t = self.route_time(cand)
                    if t < t_orig - 1e-6:
                        cur = cand; improved = True; break
                if improved:
                    break
        return cur

    def _deep_intra_polish_infeas(self, route):
        """Alternate 2-opt + or-opt(1,2,3) full-slot until converged.

        Stronger than running them once each — catches the case where a
        2-opt reversal unlocks an or-opt gain and vice versa. Intended for
        routes just over MAX_T where every saved second matters.
        """
        cur = route
        while True:
            t_before = self.route_time(cur)
            cur = self._intra_2opt_infeas(cur)
            for seg in (1, 2, 3):
                cur = self._intra_oropt_full_infeas(cur, seg)
            t_after = self.route_time(cur)
            if t_after >= t_before - 1e-6:
                return cur

    def _intra_oropt_infeas(self, route, seg):
        """Intra or-opt variant without the MAX_T guard (see _intra_2opt_infeas)."""
        nbr = self._nbr_pos
        n_r = len(route)
        if n_r < seg + 2:
            return route
        improved = True; cur = route
        while improved:
            improved = False; n_r = len(cur)
            t_orig = self.route_time(cur)
            idx_map = {p: i for i, p in enumerate(cur)}
            for k_s in range(n_r - seg + 1):
                segment = cur[k_s: k_s + seg]
                base = cur[:k_s] + cur[k_s + seg:]
                slots = {0, len(base)}
                for cand_pos in nbr[segment[0]]:
                    i_b = idx_map.get(cand_pos)
                    if i_b is not None:
                        slots.add(i_b); slots.add(i_b + 1)
                for ins in slots:
                    if ins == k_s:
                        continue
                    cand = base[:ins] + segment + base[ins:]
                    t = self.route_time(cand)
                    if t < t_orig - 1e-6:
                        cur = cand; improved = True; break
                if improved:
                    break
        return cur

    # ── S1: intra-route polish ────────────────────────────────────────────────

    def _s1_intra_polish(self, routes, deadline):
        order = list(range(len(routes)))
        random.shuffle(order)
        out = [list(r) for r in routes]
        for r_idx in order:
            if time.time() >= deadline:
                break
            r = out[r_idx]
            for seg in (1, 2, 3):
                if time.time() >= deadline:
                    break
                r = self._intra_oropt_route(r, seg)
            r = self._intra_2opt_route(r)
            if self.route_time(r) <= self.MAX_T:
                out[r_idx] = r
        return out

    # ── S2: cross-route 2-opt ─────────────────────────────────────────────────

    def _s2_cross_2opt(self, routes, deadline):
        MAX_T = self.MAX_T; nbr = self._nbr_pos
        out = [list(r) for r in routes]
        for _sweep in range(3):
            if time.time() >= deadline:
                break
            sweep_end = min(deadline, time.time() + 25)
            n = len(out)
            centroids = [self._s_centroid(r) for r in out]
            dists = [self._s_euclid(centroids[a], centroids[b])
                     for a in range(n) for b in range(a+1, n)]
            if not dists:
                break
            median_d = sorted(dists)[len(dists) // 2]
            pos_loc = {}
            for ri, r in enumerate(out):
                for ki, pos in enumerate(r):
                    pos_loc[pos] = (ri, ki)
            improved_any = False
            done = False
            for a in range(n):
                if time.time() >= sweep_end:
                    break
                for b in range(a+1, n):
                    if time.time() >= sweep_end:
                        break
                    if self._s_euclid(centroids[a], centroids[b]) > 2.0 * median_d + 1e-9:
                        continue
                    A = out[a]; B = out[b]
                    LA = len(A); LB = len(B)
                    if LA < 1 or LB < 1:
                        continue
                    tA_t = self.route_travel(A); tB_t = self.route_travel(B)
                    found = False
                    for i in range(LA - 1):
                        for cand in nbr[A[i]]:
                            loc = pos_loc.get(cand)
                            if loc is None or loc[0] != b:
                                continue
                            j = loc[1]
                            if j >= LB - 1:
                                continue
                            A2 = A[:i+1] + B[j+1:]
                            B2 = B[:j+1] + A[i+1:]
                            if not A2 or not B2:
                                continue
                            if self.route_time(A2) > MAX_T or self.route_time(B2) > MAX_T:
                                continue
                            if self.route_travel(A2) + self.route_travel(B2) < tA_t + tB_t - 1e-6:
                                out[a] = A2; out[b] = B2
                                pos_loc = {}
                                for ri, r in enumerate(out):
                                    for ki, pos in enumerate(r):
                                        pos_loc[pos] = (ri, ki)
                                centroids[a] = self._s_centroid(A2)
                                centroids[b] = self._s_centroid(B2)
                                improved_any = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        done = True
                        break
                if done:
                    break
            if not improved_any:
                break
        return out

    # ── S3: ejection-chain route elimination ─────────────────────────────────

    def _s3_ejection_chain(self, routes, target_idx, deadline, max_chain=20):
        if len(routes) < 2 or target_idx >= len(routes):
            return None
        others = [list(r) for i, r in enumerate(routes) if i != target_idx]
        bank = list(routes[target_idx])
        ejected_once = set(); touched = set(); steps = 0
        T = self.T; cont = self.cont_mi
        while bank and steps < max_chain and time.time() < deadline:
            c = bank.pop(0)
            def _prox(ri, c=c):
                r = others[ri]
                return min(T[cont[c]][cont[x]] for x in r) if r else float("inf")
            candidates = sorted(range(len(others)), key=_prox)[:5]
            placed = False
            for ri in candidates:
                res = self._s_best_insert(c, others[ri])
                if res is not None:
                    others[ri].insert(res[1], c); touched.add(ri)
                    placed = True; break
            if placed:
                steps += 1; continue
            if not candidates:
                return None
            ri = candidates[0]; r = others[ri]
            if not r:
                return None
            victim = max(r, key=lambda x: self._s_isolation(x, ri, others))
            if victim in ejected_once:
                return None
            ejected_once.add(victim)
            r.remove(victim)
            res = self._s_best_insert(c, r)
            if res is None:
                return None
            r.insert(res[1], c); touched.add(ri)
            bank.append(victim); steps += 1
        if bank:
            return None
        for ri in touched:
            polished = self._intra_2opt_route(others[ri])
            if self.route_time(polished) > self.MAX_T:
                return None
            others[ri] = polished
        for r in others:
            if self.route_time(r) > self.MAX_T:
                return None
        return [r for r in others if r]

    def _s3_force_eliminate(self, routes, target_idx, deadline):
        if len(routes) < 2 or target_idx >= len(routes):
            return None
        others = [list(r) for i, r in enumerate(routes) if i != target_idx]
        touched = set()
        for c in routes[target_idx]:
            best_ri = -1; best_t = float("inf"); best_ins = -1
            for ri, r in enumerate(others):
                for ins in range(len(r) + 1):
                    t = self.route_time(r[:ins] + [c] + r[ins:])
                    if t < best_t:
                        best_t = t; best_ri = ri; best_ins = ins
            if best_ri < 0:
                return None
            others[best_ri].insert(best_ins, c); touched.add(best_ri)
        for ri in list(touched):
            if time.time() >= deadline:
                break
            r = others[ri]
            for seg in (1, 2, 3):
                r = self._intra_oropt_route(r, seg)
            others[ri] = self._intra_2opt_route(r)
        for r in others:
            if self.route_time(r) > self.MAX_T:
                return None
        return others

    # ── S4: sector-boundary swap ──────────────────────────────────────────────

    def _s4_sector_swap(self, routes, deadline):
        import itertools
        out = [list(r) for r in routes]; n = len(out)
        if n < 2:
            return out
        centroids = [self._s_centroid(r) for r in out]
        xs = self._s_x; ys = self._s_y
        pairs = sorted(itertools.combinations(range(n), 2),
                       key=lambda p: -self._s_euclid(centroids[p[0]], centroids[p[1]]))
        for (a, b) in pairs:
            if time.time() >= deadline:
                break
            Ca, Cb = centroids[a], centroids[b]
            boundary = sorted(
                [(self._s_euclid((xs[x], ys[x]), Cb), x) for x in out[a]
                 if self._s_euclid((xs[x], ys[x]), Cb) < self._s_euclid((xs[x], ys[x]), Ca)]
            )
            for _, x in boundary[:6]:
                if time.time() >= deadline:
                    break
                new_a = [p for p in out[a] if p != x]
                ins = self._s_best_insert(x, out[b])
                if ins is None:
                    continue
                new_b = out[b][:ins[1]] + [x] + out[b][ins[1]:]
                if self.route_travel(new_a) + self.route_travel(new_b) < \
                        self.route_travel(out[a]) + self.route_travel(out[b]) - 1e-6:
                    out[a] = new_a; out[b] = new_b
                    centroids[a] = self._s_centroid(new_a)
                    centroids[b] = self._s_centroid(new_b)
                    break
        return [r for r in out if r]

    # ── S5: final merge + polish ──────────────────────────────────────────────

    def _s5_final(self, routes, deadline):
        out = self.route_merge_pass([list(r) for r in routes])
        return self._s1_intra_polish(out, deadline)

    # ── double-bridge perturbation ────────────────────────────────────────────

    def _split_into_routes(self, sequence: list[int]) -> list[list[int]]:
        """Greedy feasibility split: fill a route until time exceeded, then new route."""
        routes: list[list[int]] = []
        chunk:  list[int]       = []
        for pos in sequence:
            trial = chunk + [pos]
            if self.route_time(trial) <= self.MAX_T:
                chunk = trial
            else:
                if chunk:
                    routes.append(chunk)
                chunk = [pos]
        if chunk:
            routes.append(chunk)
        return routes

    def _double_bridge_perturb(self, routes: list[list[int]]) -> list[list[int]]:
        """4-opt double-bridge: reconnect flattened sequence as A+C+B+D.
        Cannot be undone by any 2-opt move — guarantees escape from local optima."""
        flat = [pos for r in routes for pos in r]
        n    = len(flat)
        if n < 8:
            return [list(r) for r in routes]
        cuts     = sorted(random.sample(range(1, n), 3))
        a, b, c  = cuts
        new_flat = flat[:a] + flat[b:c] + flat[a:b] + flat[c:]
        return self._split_into_routes(new_flat)

    # ── Infeasible ruin-recreate (infeasibility_search.md) ───────────────────

    def _rr_score(self, routes: list[list[int]], i: int) -> float:
        """Higher score = better ruin-recreate target.

        Combines (a) aggregate slack in other routes, (b) how well target's
        containers spread across multiple other routes via k-NN (absorbability),
        and (c) a mild size prior. Penalises isolated outlier routes whose
        removal dumps work that has nowhere to go.
        """
        r = routes[i]
        if not r:
            return -1e9
        MAX_T = self.MAX_T
        n = len(routes)
        others_slack = 0.0
        other_pos: set[int] = set()
        for j in range(n):
            if j == i:
                continue
            rj = routes[j]
            others_slack += max(0.0, MAX_T - self.route_time(rj))
            other_pos.update(rj)
        target_work = self.route_time(r)
        k_per = max(1, len(self._nbr_pos[r[0]]))
        absorb_hits = 0
        for c in r:
            nbrs = self._nbr_pos[c]
            for p in nbrs:
                if p in other_pos:
                    absorb_hits += 1
        absorb = absorb_hits / (len(r) * k_per)
        slack_ratio = others_slack / max(1.0, target_work)
        return (slack_ratio * 2.0) + (absorb * 3.0) \
               - (len(r) / max(1, self.p.container_count))

    def _compress_2opt_star(self, others, over, MAX_T) -> bool:
        """Stage 3d — inter-route 2-opt* (tail exchange).

        For each overloaded ri and each other rj, try swapping the tails after
        cut points (ki, kj). Accept the first pair of cuts whose resulting
        routes strictly reduce pairwise overload.
        """
        for ri in over:
            r_i = others[ri]
            t_i = self.route_time(r_i)
            if t_i <= MAX_T:
                continue
            ov_i = t_i - MAX_T
            for rj in range(len(others)):
                if rj == ri:
                    continue
                r_j = others[rj]
                t_j = self.route_time(r_j)
                ov_j = max(0.0, t_j - MAX_T)
                n_i = len(r_i); n_j = len(r_j)
                for ki in range(1, n_i):
                    for kj in range(1, n_j):
                        new_i = r_i[:ki] + r_j[kj:]
                        new_j = r_j[:kj] + r_i[ki:]
                        if not new_i or not new_j:
                            continue
                        t_ni = self.route_time(new_i)
                        t_nj = self.route_time(new_j)
                        new_ov = (max(0.0, t_ni - MAX_T)
                                  + max(0.0, t_nj - MAX_T))
                        if new_ov < ov_i + ov_j - 1e-6:
                            others[ri] = new_i
                            others[rj] = new_j
                            return True
        return False

    def _compress_merge_split(self, others, over, MAX_T) -> bool:
        """Stage 3e — merge most-overloaded route with its nearest feasible
        neighbour (by centroid distance), nearest-neighbour reorder the
        combined container list, then re-split into routes. Accept if the
        result has ≤ 2 routes all feasible, and pairwise overload decreases.
        """
        if not over:
            return False
        ri = max(over, key=lambda i: self.route_time(others[i]))
        r_i = others[ri]
        ci = self._s_centroid(r_i)
        best_rj = -1; best_d = float("inf")
        for rj in range(len(others)):
            if rj == ri:
                continue
            cj = self._s_centroid(others[rj])
            d = self._s_euclid(ci, cj)
            if d < best_d:
                best_d = d; best_rj = rj
        if best_rj < 0:
            return False
        r_j = others[best_rj]
        t_i = self.route_time(r_i); t_j = self.route_time(r_j)
        ov_pair = max(0.0, t_i - MAX_T) + max(0.0, t_j - MAX_T)
        # Nearest-neighbour reorder of the combined pool
        pool = list(r_i) + list(r_j)
        if not pool:
            return False
        T = self.T; cont = self.cont_mi
        ordered = [pool[0]]
        remaining = set(pool[1:])
        while remaining:
            last = ordered[-1]
            nxt = min(remaining, key=lambda x: T[cont[last]][cont[x]])
            ordered.append(nxt); remaining.remove(nxt)
        new_routes = self._split_into_routes(ordered)
        if not new_routes or len(new_routes) > 2:
            return False
        new_ov = sum(max(0.0, self.route_time(r) - MAX_T) for r in new_routes)
        if new_ov < ov_pair - 1e-6:
            # replace ri with first new, rj with second (or empty)
            others[ri] = new_routes[0]
            others[best_rj] = new_routes[1] if len(new_routes) > 1 else []
            return True
        return False

    def _compress_chain_eject(self, others, over, MAX_T,
                              max_attempts: int = 200) -> bool:
        """Stage 3f — cheap ejection chain. Move a container from a feasible
        rj to some third route rk, then try to relocate a segment from ri
        into the freed slot of rj. Accept if total overload strictly drops.
        """
        attempts = 0
        feasible_rj = [rj for rj, r in enumerate(others)
                       if self.route_time(r) <= MAX_T and r]
        slack_rank = sorted(
            feasible_rj,
            key=lambda rj: self.route_time(others[rj]),
        )
        for ri in over:
            r_i = others[ri]
            t_i = self.route_time(r_i)
            if t_i <= MAX_T or not r_i:
                continue
            ov_i = t_i - MAX_T
            for rj in slack_rank:
                if rj == ri:
                    continue
                r_j = others[rj]
                t_j = self.route_time(r_j)
                # Try ejecting one container from rj into some rk (fastest slot)
                for cj_idx, c_out in enumerate(r_j):
                    if attempts >= max_attempts:
                        return False
                    attempts += 1
                    r_j_wo = r_j[:cj_idx] + r_j[cj_idx + 1:]
                    # find rk that absorbs c_out cheaply and stays feasible
                    best_rk = -1; best_t_rk = float("inf"); best_ins_k = 0
                    for rk in slack_rank:
                        if rk == ri or rk == rj:
                            continue
                        r_k = others[rk]
                        for ins in range(len(r_k) + 1):
                            t = self.route_time(r_k[:ins] + [c_out] + r_k[ins:])
                            if t <= MAX_T and t < best_t_rk:
                                best_t_rk = t; best_rk = rk; best_ins_k = ins
                    if best_rk < 0:
                        continue
                    # Now try to relocate best single container from ri into r_j_wo
                    moved = False
                    for ci_idx, c_in in enumerate(r_i):
                        r_i_wo = r_i[:ci_idx] + r_i[ci_idx + 1:]
                        best_t_j2 = float("inf"); best_ins_j = 0
                        for ins in range(len(r_j_wo) + 1):
                            t = self.route_time(r_j_wo[:ins] + [c_in]
                                                + r_j_wo[ins:])
                            if t < best_t_j2:
                                best_t_j2 = t; best_ins_j = ins
                        t_i_new = self.route_time(r_i_wo) if r_i_wo else 0.0
                        new_ov = (max(0.0, t_i_new - MAX_T)
                                  + max(0.0, best_t_j2 - MAX_T))
                        old_ov = ov_i + max(0.0, t_j - MAX_T)
                        if new_ov < old_ov - 1e-6:
                            # Commit: remove c_out from rj→rk, move c_in from ri→rj_wo
                            r_k = others[best_rk]
                            others[best_rk] = (r_k[:best_ins_k] + [c_out]
                                               + r_k[best_ins_k:])
                            others[rj] = (r_j_wo[:best_ins_j] + [c_in]
                                          + r_j_wo[best_ins_j:])
                            others[ri] = r_i_wo
                            return True
                    if moved:
                        return True
        return False

    def _infeasible_ruin_recreate(
        self, routes: list[list[int]], deadline: float,
        target_idx: int | None = None,
    ) -> list[list[int]] | None:
        """
        Route-removal with controlled infeasibility:

        1. Ruin    — remove target route (adaptive _rr_score), place containers
                     in a bank.
        2. Expand  — insert every bank container into remaining routes using
                     regret-2 insertion with a per-route overload cap and
                     progressive λ penalty (no catastrophic concentration).
        3. Compress— layered stages 3a–3f (intra polish, relocate, swap,
                     2-opt*, merge-split, chain-eject) with stall-driven
                     acceptance relaxation.

        Returns n-1 feasible routes on success, None on failure.
        """
        if len(routes) < 2:
            return None

        MAX_T = self.MAX_T

        # ── Step 1: Ruin (Fix A — _rr_score) ──────────────────────────────────
        if target_idx is None:
            tgt_idx = max(range(len(routes)),
                          key=lambda i: self._rr_score(routes, i))
        else:
            tgt_idx = target_idx
        print(f"[INFEAS-RR] Targeting route {tgt_idx} "
              f"({len(routes[tgt_idx])} containers, "
              f"t={self.route_time(routes[tgt_idx])/3600:.3f}h)")
        bank   = list(routes[tgt_idx])
        others = [list(r) for i, r in enumerate(routes) if i != tgt_idx]

        # ── Step 2: Infeasible expansion (Fix B — regret-2, safe cap) ─────────
        # Per-route hard overload cap prevents catastrophic concentration.
        # Progressive λ (1.0→9.0) lets early placements spread, forces late
        # placements to prefer less-loaded destinations.
        total_bank = len(bank)
        EXPAND_OV_CAP_BASE = max(2000.0, 0.15 * MAX_T)

        while bank:
            if time.time() >= deadline:
                return None
            placed = total_bank - len(bank)
            _lam = 1.0 + 8.0 * (placed / max(1, total_bank))

            best_c    = None; best_ri = -1; best_ins_i = -1; best_regret = -1e18
            current_cap = EXPAND_OV_CAP_BASE

            for c in bank:
                nbrs     = set(self._nbr_pos[c])
                cand_ris = [ri for ri, r in enumerate(others)
                            if any(p in nbrs for p in r)]
                if not cand_ris:
                    cand_ris = list(range(len(others)))

                # Hard cap with exponential relaxation until at least one dest qualifies
                hits: list[tuple[float, int, int]] = []
                cap_here = current_cap
                while not hits:
                    for ri in cand_ris:
                        t_after, ins = self._best_insert_any(c, others[ri])
                        if (t_after - MAX_T) > cap_here:
                            continue
                        cost = t_after + _lam * max(0.0, t_after - MAX_T)
                        hits.append((cost, ri, ins))
                    if not hits:
                        cap_here *= 1.5
                        if cap_here > 5.0 * MAX_T:
                            # extreme fallback — accept any destination
                            for ri in cand_ris:
                                t_after, ins = self._best_insert_any(c, others[ri])
                                cost = t_after + _lam * max(0.0, t_after - MAX_T)
                                hits.append((cost, ri, ins))
                            break

                hits.sort()
                regret = (hits[1][0] - hits[0][0]) if len(hits) >= 2 else 1e9
                if regret > best_regret:
                    best_regret = regret
                    best_c      = c
                    best_ri     = hits[0][1]
                    best_ins_i  = hits[0][2]

            if best_c is None:
                return None

            others[best_ri].insert(best_ins_i, best_c)
            bank.remove(best_c)

        # ── Step 3: Compression (Fixes C1–C6, D) ──────────────────────────────
        # Layered stages 3a–3f with stall-driven acceptance relaxation:
        #   3a intra polish (infeas variants drop MAX_T guard)
        #   3b segment relocate (seg 1,2,3)
        #   3c cross-swap
        #   3d inter-route 2-opt* (tail exchange)
        #   3e merge-and-resplit
        #   3f chain eject
        t_compress_start = time.time()

        _init_over = [i for i, r in enumerate(others) if self.route_time(r) > MAX_T]
        _init_ov   = sum(max(0.0, self.route_time(r) - MAX_T) for r in others)
        _over_str  = "  ".join(f"r{i}:{self.route_time(others[i]):.0f}s"
                               for i in _init_over)
        print(f"[COMPRESS init]  total_ov={_init_ov:.0f}s  "
              f"over={len(_init_over)}: {_over_str}")

        _iter = 0
        stall_count = 0
        tiny_progress_streak = 0
        allow_worsen_used_this_iter = False
        # Tabu search: block recently seen (ri_state, rj_state) pairs from worsening
        # moves to prevent oscillation cycles. Each entry expires after TABU_TENURE iters.
        _TABU_TENURE = 8
        _tabu_set: set   = set()
        _tabu_queue: list = []   # list of (iter_added, signature)
        while time.time() < deadline:
            over = [i for i, r in enumerate(others) if self.route_time(r) > MAX_T]
            if not over:
                break

            _ov_before = sum(max(0.0, self.route_time(r) - MAX_T) for r in others)
            _iter += 1
            allow_worsen_used_this_iter = False

            # Expire tabu entries older than TABU_TENURE iterations.
            while _tabu_queue and _tabu_queue[0][0] + _TABU_TENURE <= _iter:
                _, _exp_sig = _tabu_queue.pop(0)
                _tabu_set.discard(_exp_sig)

            # 3a: Intra-route polish — use infeas variants on overloaded routes,
            # and pre-polish a few feasible destinations (C1 + C2).
            # Small overloads (<2000s) get deep polish (iterates 2-opt +
            # full-slot or-opt to convergence) to shave tens–hundreds of
            # seconds back under MAX_T.
            c3a_tried = 0
            c3a_improved = 0
            for ri in over:
                if time.time() >= deadline:
                    break

                r = others[ri]
                before = self.route_time(r)

                ov_r = before - MAX_T

                if ov_r <= 2000.0:
                    new_r = self._deep_intra_polish_infeas(r)
                else:
                    for seg in (1, 2, 3):
                        r = self._intra_oropt_infeas(r, seg)
                    new_r = self._intra_2opt_infeas(r)

                after = self.route_time(new_r)

                c3a_tried += 1
                if after < before:
                    c3a_improved += 1

                others[ri] = new_r

            # C2: pre-polish 5 feasible routes with least slack
            #also optimize feasible routes
            #makes later distribution easier
            feasible_by_slack = sorted(
                [i for i, r in enumerate(others)
                 if self.route_time(r) <= MAX_T and r],
                key=lambda i: (MAX_T - self.route_time(others[i])),
            )[:5]
            for ri in feasible_by_slack:
                others[ri] = self._intra_2opt_route(others[ri])

            over = [i for i, r in enumerate(others) if self.route_time(r) > MAX_T]
            if not over:
                break

            relaxed = stall_count >= 2          # C4: accept travel-shortening
            allow_worsen = stall_count >= 4     # C4: one worsening move OK

            # 3b: Segment relocation (seg=1,2,3) — multi-node moves, global
            # overload criterion, dest_order computed fresh per route.
            progress  = False
            _3b_gen   = 0   # routes where seg_moves list was non-empty
            _3b_tried = 0   # (ri, rj) destination pairs evaluated
            _3b_acc   = {1: 0, 2: 0, 3: 0}  # accepted moves per seg size
            for seg_size in (1, 2, 3):
                if progress or time.time() >= deadline:
                    break
                for ri in over:
                    if time.time() >= deadline:
                        break
                    t_ri = self.route_time(others[ri])
                    if t_ri <= MAX_T:
                        continue
                    r     = others[ri]
                    n_r   = len(r)
                    if n_r < seg_size:
                        continue
                    ov_ri = max(0.0, t_ri - MAX_T)

                    # Build (saving, k_start, seg_block, r_without, t_without)
                    seg_moves: list[tuple[float, int, list, list, float]] = []
                    for k_s in range(n_r - seg_size + 1):
                        seg_block = r[k_s: k_s + seg_size]
                        r_wo      = r[:k_s] + r[k_s + seg_size:]
                        t_wo      = self.route_time(r_wo) if r_wo else 0.0
                        saving    = t_ri - t_wo
                        if saving > 1e-6:
                            seg_moves.append((saving, k_s, seg_block, r_wo, t_wo))
                    seg_moves.sort(reverse=True)

                    if seg_moves:
                        _3b_gen += 1

                    # Fresh dest_order for this ri (fix: stale priorities)
                    dest_order = sorted(
                        range(len(others)),
                        key=lambda rj: self.route_time(others[rj]),
                    )

                    moved = False
                    for saving, k_s, seg_block, r_wo, t_wo in seg_moves:
                        new_ov_ri = max(0.0, t_wo - MAX_T)

                        for rj in dest_order:
                            if rj == ri:
                                continue
                            rj_r      = others[rj]
                            t_rj_old  = self.route_time(rj_r)
                            old_ov_rj = max(0.0, t_rj_old - MAX_T)
                            _3b_tried += 1

                            # Best insertion of the whole segment into rj
                            best_t_rj = float("inf"); best_ins = -1
                            for ins in range(len(rj_r) + 1):
                                t = self.route_time(
                                    rj_r[:ins] + seg_block + rj_r[ins:])
                                if t < best_t_rj:
                                    best_t_rj = t; best_ins = ins
                            new_ov_rj = max(0.0, best_t_rj - MAX_T)

                            old_pair = ov_ri + old_ov_rj
                            new_pair = new_ov_ri + new_ov_rj
                            accept = new_pair < old_pair - 1e-6
                            # C4 relaxed: equal overload + strict travel improvement
                            if not accept and relaxed \
                                    and abs(new_pair - old_pair) < 1e-6:
                                if (t_wo + best_t_rj) < (t_ri + t_rj_old) - 1e-6:
                                    accept = True
                            # C4 allow_worsen: one worsening move / iter if travel drops
                            if (not accept and allow_worsen
                                    and not allow_worsen_used_this_iter
                                    and new_pair < old_pair + 0.10 * MAX_T):
                                if (t_wo + best_t_rj) < (t_ri + t_rj_old) - 1e-6:
                                    accept = True
                                    allow_worsen_used_this_iter = True

                            if accept:
                                others[ri] = r_wo
                                others[rj] = (rj_r[:best_ins]
                                              + seg_block
                                              + rj_r[best_ins:])
                                _3b_acc[seg_size] += 1
                                progress = True
                                moved    = True
                                break

                        if moved:
                            break  # one move per ri; recalc savings next iteration

            # 3b worsening escape: first-found worsening relocation with delta < 100s.
            # Ordered by most-overloaded ri first, most-slack rj first to find useful
            # moves quickly. Fires whenever total_ov < 2000s, regardless of progress.
            _3b_worsening_acc = 0
            _ov_now_3b = sum(max(0.0, self.route_time(r) - MAX_T) for r in others)
            if _ov_now_3b < 2000.0:
                _EPSILON = 100.0
                _found_3b_w = False
                _over_by_ov = sorted(over,
                                     key=lambda i: self.route_time(others[i]),
                                     reverse=True)
                _dest_by_slack = sorted(
                    range(len(others)),
                    key=lambda j: self.route_time(others[j]),
                )
                for seg_size in (1, 2, 3):
                    if _found_3b_w:
                        break
                    for ri in _over_by_ov:
                        if _found_3b_w:
                            break
                        r   = others[ri]
                        n_r = len(r)
                        if n_r < seg_size:
                            continue
                        t_ri  = self.route_time(r)
                        ov_ri = max(0.0, t_ri - MAX_T)
                        for k_s in range(n_r - seg_size + 1):
                            if _found_3b_w:
                                break
                            seg_block = r[k_s: k_s + seg_size]
                            r_wo      = r[:k_s] + r[k_s + seg_size:]
                            t_wo      = self.route_time(r_wo) if r_wo else 0.0
                            new_ov_ri = max(0.0, t_wo - MAX_T)
                            for rj in _dest_by_slack:
                                if rj == ri:
                                    continue
                                rj_r      = others[rj]
                                old_ov_rj = max(0.0, self.route_time(rj_r) - MAX_T)
                                best_t_rj = float("inf"); best_ins = -1
                                for ins in range(len(rj_r) + 1):
                                    t = self.route_time(
                                        rj_r[:ins] + seg_block + rj_r[ins:])
                                    if t < best_t_rj:
                                        best_t_rj = t; best_ins = ins
                                new_ov_rj = max(0.0, best_t_rj - MAX_T)
                                delta = (new_ov_ri + new_ov_rj) - (ov_ri + old_ov_rj)
                                if 1e-6 < delta < _EPSILON:
                                    _new_rj = (rj_r[:best_ins]
                                               + seg_block
                                               + rj_r[best_ins:])
                                    _sig = (tuple(r_wo), tuple(_new_rj), ri, rj)
                                    if _sig in _tabu_set:
                                        continue  # tabu: skip this move
                                    others[ri] = r_wo
                                    others[rj] = _new_rj
                                    _tabu_set.add(_sig)
                                    _tabu_queue.append((_iter, _sig))
                                    _3b_worsening_acc += 1
                                    _3b_acc[seg_size] += 1
                                    progress    = True
                                    _found_3b_w = True
                                    break

            # 3c: Cross-swap — runs whenever progress is False OR total_ov < 2000s.
            # When complementary (ov < 2000), limits scope to top-3 most overloaded
            # routes to keep each iteration cheap (~3× fewer pairs than full scan).
            _3c_tried = 0
            _3c_acc   = 0
            _3c_worsening_acc = 0
            _ov_now_3c = sum(max(0.0, self.route_time(r) - MAX_T) for r in others)
            if not progress or _ov_now_3c < 2000.0:
                _3c_progress = False
                # Limit candidate source routes: all when falling back (no progress),
                # top-3 most overloaded when running as complement to 3b.
                _3c_sources = over if not progress else sorted(
                    over,
                    key=lambda i: self.route_time(others[i]),
                    reverse=True,
                )[:3]
                for ri in _3c_sources:
                    if time.time() >= deadline:
                        break
                    t_ri  = self.route_time(others[ri])
                    if t_ri <= MAX_T:
                        continue
                    ov_ri = max(0.0, t_ri - MAX_T)
                    r     = others[ri]
                    dest_order = sorted(
                        range(len(others)),
                        key=lambda rj: self.route_time(others[rj]),
                    )
                    for rj in dest_order:
                        if rj == ri:
                            continue
                        rj_r    = others[rj]
                        ov_rj   = max(0.0, self.route_time(rj_r) - MAX_T)
                        swapped = False
                        for ci, c_i in enumerate(r):
                            for cj, c_j in enumerate(rj_r):
                                _3c_tried += 1
                                new_ri_r = r[:ci] + [c_j] + r[ci + 1:]
                                new_rj_r = rj_r[:cj] + [c_i] + rj_r[cj + 1:]
                                t_new_ri = self.route_time(new_ri_r)
                                t_new_rj = self.route_time(new_rj_r)
                                if (max(0.0, t_new_ri - MAX_T)
                                        + max(0.0, t_new_rj - MAX_T)
                                        < ov_ri + ov_rj - 1e-6):
                                    others[ri] = new_ri_r
                                    others[rj] = new_rj_r
                                    _3c_acc     += 1
                                    progress     = True
                                    _3c_progress = True
                                    swapped      = True
                                    break
                            if swapped:
                                break
                        if swapped:
                            break

                # 3c worsening escape: first-found worsening swap with delta < 100s.
                # Ordered: most-overloaded ri first, most-slack rj first.
                _ov_now_3c_w = sum(max(0.0, self.route_time(r) - MAX_T)
                                   for r in others)
                if not _3c_progress and _ov_now_3c_w < 2000.0:
                    _EPSILON = 100.0
                    _found_3c_w = False
                    _over_3c = sorted(over,
                                      key=lambda i: self.route_time(others[i]),
                                      reverse=True)
                    _dest_3c  = sorted(
                        range(len(others)),
                        key=lambda j: self.route_time(others[j]),
                    )
                    for ri in _over_3c:
                        if _found_3c_w:
                            break
                        t_ri  = self.route_time(others[ri])
                        if t_ri <= MAX_T:
                            continue
                        ov_ri = max(0.0, t_ri - MAX_T)
                        r     = others[ri]
                        for rj in _dest_3c:
                            if _found_3c_w or rj == ri:
                                continue
                            rj_r  = others[rj]
                            ov_rj = max(0.0, self.route_time(rj_r) - MAX_T)
                            for ci, c_i in enumerate(r):
                                if _found_3c_w:
                                    break
                                for cj, c_j in enumerate(rj_r):
                                    new_ri_r  = r[:ci] + [c_j] + r[ci + 1:]
                                    new_rj_r  = rj_r[:cj] + [c_i] + rj_r[cj + 1:]
                                    new_ov_ri = max(0.0,
                                        self.route_time(new_ri_r) - MAX_T)
                                    new_ov_rj = max(0.0,
                                        self.route_time(new_rj_r) - MAX_T)
                                    delta = (new_ov_ri + new_ov_rj) - (ov_ri + ov_rj)
                                    if 1e-6 < delta < _EPSILON:
                                        _sig = (tuple(new_ri_r), tuple(new_rj_r),
                                                ri, rj)
                                        if _sig in _tabu_set:
                                            continue  # tabu: skip
                                        others[ri] = new_ri_r
                                        others[rj] = new_rj_r
                                        _tabu_set.add(_sig)
                                        _tabu_queue.append((_iter, _sig))
                                        _3c_worsening_acc += 1
                                        progress    = True
                                        _found_3c_w = True
                                        break

            # 3d: inter-route 2-opt* (tail exchange)
            _3d_acc = 0
            if not progress and time.time() < deadline:
                over_now = [i for i, r in enumerate(others)
                            if self.route_time(r) > MAX_T]
                if self._compress_2opt_star(others, over_now, MAX_T):
                    _3d_acc = 1; progress = True

            # 3e: merge-and-resplit killer
            _3e_acc = 0
            if not progress and time.time() < deadline:
                over_now = [i for i, r in enumerate(others)
                            if self.route_time(r) > MAX_T]
                if self._compress_merge_split(others, over_now, MAX_T):
                    _3e_acc = 1; progress = True

            # 3f: chain eject
            _3f_acc = 0
            if not progress and time.time() < deadline:
                over_now = [i for i, r in enumerate(others)
                            if self.route_time(r) > MAX_T]
                if self._compress_chain_eject(others, over_now, MAX_T):
                    _3f_acc = 1; progress = True

            _ov_after = sum(max(0.0, self.route_time(r) - MAX_T) for r in others)
            _3b_acc_str = "/".join(str(_3b_acc[s]) for s in (1, 2, 3))
            _suffix = "" if progress else "  ← NO PROGRESS"
            print(f"[COMPRESS iter {_iter:2d}]  "
                  f"total_ov {_ov_before:.0f}→{_ov_after:.0f}s  "
                  f"over={len(over)}  "
                  f"3a: tried={c3a_tried} acc={c3a_improved}  "
                  f"3b: gen={_3b_gen} tried={_3b_tried} acc={_3b_acc_str}(s1/s2/s3) w={_3b_worsening_acc}  "
                  f"3c: tried={_3c_tried} acc={_3c_acc} w={_3c_worsening_acc}  "
                  f"3d={_3d_acc} 3e={_3e_acc} 3f={_3f_acc}"
                  f"{_suffix}")

            # Stall + progress tracking (Fix D)
            improved = _ov_after < _ov_before - 1.0
            if improved:
                stall_count = 0
                frac_drop = (_ov_before - _ov_after) / max(1.0, _ov_before)
                if frac_drop < 0.005:
                    tiny_progress_streak += 1
                else:
                    tiny_progress_streak = 0
            else:
                stall_count += 1

            if not progress:
                break  # exhausted all move types — no further overload reduction
            if tiny_progress_streak >= 3 and stall_count >= 4:
                break  # grinding with no meaningful progress — free budget

        # ── Step 4: Accept or reject ──────────────────────────────────────────
        t_compress = time.time() - t_compress_start
        still_over = [i for i, r in enumerate(others) if self.route_time(r) > MAX_T]
        if still_over:
            details = "  ".join(
                f"r{i}:{self.route_time(others[i]):.0f}s"
                for i in still_over
            )
            print(f"[INFEAS-RR] Compression FAILED after {t_compress:.1f}s — "
                  f"{len(still_over)} route(s) still infeasible: {details}")
            return None
        feasible = [r for r in others if r]
        print(f"[INFEAS-RR] Compression SUCCESS → {len(feasible)}r "
              f"in {t_compress:.1f}s")
        return feasible

    # ── LNS route elimination ─────────────────────────────────────────────────

    def _lns_route_eliminate(
        self, routes: list[list[int]], deadline: float,
    ) -> list[list[int]] | None:
        """
        Remove the shortest route and insert all its containers into the
        remaining routes using regret-2 insertion (highest-regret container
        placed first — the one whose best slot degrades most if delayed).

        After a first-pass failure, polishes modified routes with intra or-opt
        + 2-opt to free time slack, then retries the unplaced containers.
        Returns a one-fewer-route solution on success, None otherwise.
        """
        if len(routes) < 2:
            return None

        tgt_idx = min(
            range(len(routes)),
            key=lambda i: (len(routes[i]), self.route_time(routes[i])),
        )
        target   = routes[tgt_idx]
        others   = [list(r) for i, r in enumerate(routes) if i != tgt_idx]
        unplaced = list(target)
        modified: set[int] = set()

        for attempt in range(2):
            while unplaced:
                if time.time() >= deadline:
                    return None

                best_c = None; best_ri = -1; best_ins_i = -1; best_regret = -1e18

                for c in unplaced:
                    nbrs = set(self._nbr_pos[c])
                    cand_ris = [ri for ri, r in enumerate(others)
                                if any(p in nbrs for p in r)]
                    if not cand_ris:
                        cand_ris = list(range(len(others)))

                    hits: list[tuple[float, int, int]] = []
                    for ri in cand_ris:
                        res = self._s_best_insert(c, others[ri])
                        if res is not None:
                            hits.append((res[0], ri, res[1]))

                    if not hits:
                        continue  # no feasible slot for this container yet

                    hits.sort()
                    regret = (hits[1][0] - hits[0][0]) if len(hits) >= 2 else 1e9
                    if regret > best_regret:
                        best_regret  = regret
                        best_c       = c
                        best_ri      = hits[0][1]
                        best_ins_i   = hits[0][2]

                if best_c is None:
                    break  # no container can be placed this pass

                others[best_ri].insert(best_ins_i, best_c)
                unplaced.remove(best_c)
                modified.add(best_ri)

            if not unplaced:
                break  # success — skip second attempt

            if attempt == 0:
                # Polish modified routes to free time slack, then retry
                for ri in modified:
                    if time.time() >= deadline:
                        return None
                    r = others[ri]
                    for seg in (1, 2, 3):
                        r = self._intra_oropt_route(r, seg)
                    others[ri] = self._intra_2opt_route(r)

        if unplaced:
            return None  # some containers could not be placed

        # Final polish on modified routes
        for ri in modified:
            if time.time() >= deadline:
                break
            r = others[ri]
            for seg in (1, 2, 3):
                r = self._intra_oropt_route(r, seg)
            others[ri] = self._intra_2opt_route(r)
            if self.route_time(others[ri]) > self.MAX_T:
                return None

        return [r for r in others if r]

    # ── driver ────────────────────────────────────────────────────────────────

    def _s_driver(self, best_phase1, deadline, is_last_slice: bool = True):
        # Last slice reserves 14 s for S5 final polish; other slices only 3 s.
        SAFETY_S       = 14 if is_last_slice else 3
        drive_deadline = deadline - SAFETY_S
        final_deadline = deadline - 3   # used only when is_last_slice
        t0 = time.time()
        self._s_precompute(best_phase1)

        cur = [list(r) for r in best_phase1]
        best_seen = [list(r) for r in best_phase1]
        best_c = self.cost(best_seen)
        print(f"[S-init] start from {best_c[0]}r {best_c[1]/3600:.3f}h")

        # Stage 0: opening polish
        cur = self.route_merge_pass(cur)
        cur = self._s1_intra_polish(cur, min(drive_deadline, time.time() + 25))
        c0 = self.cost(cur)
        if c0 < best_c:
            best_seen = [list(r) for r in cur]; best_c = c0
            print(f"[S0] polish → {best_c[0]}r {best_c[1]/3600:.3f}h")
            self._save_uid_snapshot("s0_polish", cur)

        stall      = 0
        restarts   = 0
        elim_fails = 0
        while time.time() < drive_deadline:
            improved = False

            # Stage A: ejection-chain burst
            order = sorted(range(len(cur)),
                           key=lambda i: (len(cur[i]), self.route_time(cur[i])))
            attempt_deadline = min(drive_deadline, time.time() + 90)
            for tgt in order:
                if time.time() >= attempt_deadline:
                    break
                new = self._s3_ejection_chain(cur, tgt, time.time() + 4)
                if new is not None:
                    cur = new; c = self.cost(cur)
                    print(f"[S3 v] tgt={tgt} → {c[0]}r {c[1]/3600:.3f}h")
                    if c < best_c:
                        best_seen = [list(r) for r in cur]; best_c = c
                        self._save_uid_snapshot("s3_eliminate", cur)
                    improved = True; break

            # Stage A': force-eliminate fallback on smallest route
            if not improved and time.time() < drive_deadline and len(cur) > 1:
                order2 = sorted(range(len(cur)),
                                key=lambda i: (len(cur[i]), self.route_time(cur[i])))
                cand = self._s3_force_eliminate(cur, order2[0], time.time() + 15)
                if cand is not None and self.cost(cand) < self.cost(cur):
                    cur = cand; c = self.cost(cur)
                    print(f"[S3c v] force+polish → {c[0]}r {c[1]/3600:.3f}h")
                    if c < best_c:
                        best_seen = [list(r) for r in cur]; best_c = c
                    improved = True

            # Stage B: cross-route 2-opt
            b_end = min(drive_deadline, time.time() + 40)
            t_before = self.total_travel(cur)
            cur = self._s2_cross_2opt(cur, b_end)
            t_after = self.total_travel(cur)
            if t_after < t_before - 1e-3:
                print(f"[S2] travel {t_before/3600:.3f} → {t_after/3600:.3f} h")
                if self.cost(cur) < best_c:
                    best_seen = [list(r) for r in cur]; best_c = self.cost(cur)
                improved = True

            # Stage C: sector swap
            c_end = min(drive_deadline, time.time() + 20)
            cur = self._s4_sector_swap(cur, c_end)
            cc = self.cost(cur)
            if cc < best_c:
                best_seen = [list(r) for r in cur]; best_c = cc
                print(f"[S4 v] → {best_c[0]}r {best_c[1]/3600:.3f}h")
                improved = True

            # Stage D: intra polish
            d_end = min(drive_deadline, time.time() + 15)
            cur = self._s1_intra_polish(cur, d_end)
            if self.cost(cur) < best_c:
                best_seen = [list(r) for r in cur]; best_c = self.cost(cur)
                improved = True

            if improved:
                stall      = 0
                elim_fails = 0
            else:
                stall += 1
                if stall >= 2 and time.time() < drive_deadline:
                    stall = 0
                    # Try LNS route elimination before restarting
                    lns_budget = min(drive_deadline - time.time(), 45.0)
                    if lns_budget > 5.0:
                        elim = self._lns_route_eliminate(
                            cur, time.time() + lns_budget)
                        if elim is not None:
                            c_lns = self.cost(elim)
                            cur   = elim
                            print(f"[LNS-elim] → {c_lns[0]}r {c_lns[1]/3600:.3f}h")
                            if c_lns < best_c:
                                best_seen = [list(r) for r in cur]
                                best_c    = c_lns
                            elim_fails = 0
                        else:
                            elim_fails += 1
                            # Primary escape: infeasible ruin-recreate on up to
                            # 5 target routes (2 min each), shortest first
                            n_try = min(5, len(cur))
                            order_rr = sorted(
                                range(len(cur)),
                                key=lambda i: -self._rr_score(cur, i),
                            )[:n_try]
                            print(f"[INFEAS-RR] Entering infeasible "
                                  f"ruin-recreate stage — {len(cur)}r, "
                                  f"trying {n_try} target(s)")
                            rr_success = False
                            for tgt in order_rr:
                                remaining = drive_deadline - time.time()
                                if remaining <= 10.0:
                                    break
                                budget = min(120.0, remaining - 5.0)
                                inf_result = self._infeasible_ruin_recreate(
                                    cur, time.time() + budget,
                                    target_idx=tgt)
                                if inf_result is not None:
                                    c_inf = self.cost(inf_result)
                                    cur   = inf_result
                                    if c_inf < best_c:
                                        best_seen = [list(r) for r in cur]
                                        best_c    = c_inf
                                    elim_fails = 0
                                    rr_success = True
                                    break
                            # Last resort: double-bridge diversity kick
                            if not rr_success and restarts < self.MAX_RESTARTS:
                                restarts += 1
                                cur = self._double_bridge_perturb(best_seen)
                                cur = self.route_merge_pass(cur)
                                cur = self._s1_intra_polish(
                                    cur, min(drive_deadline, time.time() + 10))
                                c_r = self.cost(cur)
                                print(f"[S-restart #{restarts}] dbl-bridge → "
                                      f"{c_r[0]}r {c_r[1]/3600:.3f}h")
                                elim_fails = 0

        # Stage Z: final polish on best snapshot (last slice only)
        if is_last_slice:
            final = self._s5_final(best_seen, final_deadline)
            c_final = self.cost(final)
            if c_final < best_c:
                best_seen, best_c = final, c_final
        print(f"[S-driver end] {best_c[0]}r {best_c[1]/3600:.3f}h "
              f"  restarts={restarts}  ({time.time()-t0:.1f}s elapsed)")
        return best_seen

    def _save_extra_result(self, best: list[list[int]]) -> None:
        """
        Save per-route summary to extra_algorithm_output/<instance>/extra_result.json.
        """
        instance_name = getattr(self.p, "subproblem", None) or \
                        self.p.instance_dir.name

        out_dir = Path.cwd() / "extra_algorithm_output" / instance_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "extra_result.json"

        n     = len(best)
        times = [self.route_time(r) for r in best]

        routes_out = []
        for idx, (r, t) in enumerate(zip(best, times), start=1):
            routes_out.append({
                "route_index":  idx,
                "n_containers": len(r),
                "total_s":      round(t, 3),
                "total_h":      round(t / 3600, 6),
            })

        payload = {
            "instance":            instance_name,
            "n_routes":            n,
            "limit_s":             self.MAX_T,
            "limit_h":             round(self.MAX_T / 3600, 6),
            "capacity_limit":      self.CAP,
            "dump_service_s":      self.svc_d,
            "container_service_s": self.svc_c,
            "routes":              routes_out,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[Extra] Saved {out_path}")

        # ── save full solution routes for visualize.py ────────────────────────
        # Writes the final UID routes so visualize.py --from-solution can
        # reproduce the exact same maps without re-running the solver.
        uid_routes = [self.to_uid_route(r) for r in best]
        solution_payload = {
            "instance":  instance_name,
            "n_routes":  n,
            "routes":    uid_routes,   # list[list[str]] — full UID sequences
        }
        sol_path = out_dir / "solution.json"
        with open(sol_path, "w", encoding="utf-8") as f:
            json.dump(solution_payload, f, indent=2)
        print(f"[Extra] Saved {sol_path}")