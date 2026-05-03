"""
visualize.py — Route visualization for SmartEcoRutas  (ALNS pipeline)
======================================================================

Runs the solver in "visualization mode" for every instance found under
the data directory, captures snapshots at the key moments of the ALNS
pipeline, and saves everything to  output/<instance_name>/

Two modes
---------
  Default (solver mode):
    Runs the full solver for every instance and saves snapshots.
    Total time ≈ 4 × --time-limit.

  --from-solution (replay mode):
    Reads the solution already saved by the solver in
        extra_algorithm_output/<INSTANCE>/solution.json
    and draws the final map without re-running the solver.
    Instant — no computation, identical to what run.py produced.

Snapshots saved per instance (solver mode)
------------------------------------------
  snapshot_cw.png            after Clarke-Wright (before merge)
  snapshot_phase1.png        after Phase 1 local search
  snapshot_clusters.png      geographic clustering (city vs villages)
  snapshot_alns_best.png     best solution found by ALNS
  convergence.png            route count + total time vs ALNS iteration
  route_times.png            bar chart of each route's time vs limit

Snapshots saved per instance (replay mode)
------------------------------------------
  snapshot_final.png         the exact solution from the last run.py run

Usage
-----
    python visualize.py                                # solver mode, default data/
    python visualize.py --data-dir data --time-limit 900 --seed 0

    python visualize.py --from-solution                # replay mode
    python visualize.py --from-solution --extra-dir extra_algorithm_output

    --data-dir    parent folder with 4 instance subfolders  (default: data)
    --extra-dir   folder with extra_algorithm_output structure (default: extra_algorithm_output)
    --time-limit  solver budget in seconds per instance  (default: 120)
    --seed        random seed  (default: 0)
    --dpi         PNG resolution  (default: 150)
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from framework.problem_instance import ProblemInstance          # noqa: E402
from student.algoritmoSmartEcoRutas import _Solver              # noqa: E402

INSTANCES = ["LATERAL_CARTON", "LATERAL_ENVASE", "LATERAL_RESTO", "TRASERA_RESTO"]


# =============================================================================
# Drawing helpers
# =============================================================================

def _get_cmap(name: str):
    """Compatibility wrapper for matplotlib colormaps (3.7+ deprecation)."""
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        return cm.get_cmap(name)


def _route_colors(n: int) -> list:
    cmap = _get_cmap("tab20" if n <= 20 else "hsv")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _draw_route_map(
    ax: plt.Axes,
    solver: "_VisualizingSolver",
    routes: list[list[int]],
    title: str,
) -> None:
    p      = solver.p
    colors = _route_colors(len(routes))

    def coords(uid: str):
        node = p.uid_to_node(uid)
        return node.lon, node.lat

    for route, col in zip(routes, colors):
        uid_route = solver.to_uid_route(route)
        xs = [coords(u)[0] for u in uid_route]
        ys = [coords(u)[1] for u in uid_route]
        ax.plot(xs, ys, "-", color=col, linewidth=0.7, alpha=0.75, zorder=2)
        cx = [coords(u)[0] for u in uid_route if u not in (solver.BASE, solver.DUMP)]
        cy = [coords(u)[1] for u in uid_route if u not in (solver.BASE, solver.DUMP)]
        ax.scatter(cx, cy, s=6, color=col, zorder=3, linewidths=0)

    bx, by = coords(solver.BASE)
    dx, dy = coords(solver.DUMP)
    ax.plot(bx, by, marker="*", markersize=12, color="#2563eb",
            zorder=5, linestyle="none", label="BASE")
    ax.plot(dx, dy, marker="D", markersize=7,  color="#6b7280",
            zorder=5, linestyle="none", label="DUMP")

    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("longitude", fontsize=8)
    ax.set_ylabel("latitude",  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    n_rt    = len(routes)
    handles = [mpatches.Patch(color=colors[i], label=f"Route {i+1}")
               for i in range(min(n_rt, 12))]
    if n_rt > 12:
        handles.append(mpatches.Patch(color="white",
                                      label=f"... +{n_rt-12} more", linewidth=0))
    handles += [
        plt.Line2D([0], [0], marker="*", color="#2563eb",
                   linestyle="none", markersize=8, label="BASE"),
        plt.Line2D([0], [0], marker="D", color="#6b7280",
                   linestyle="none", markersize=6, label="DUMP"),
    ]
    ax.legend(handles=handles, fontsize=6, ncol=2,
              loc="upper left", framealpha=0.7)


def _save_snapshot(
    solver: "_VisualizingSolver",
    routes: list[list[int]],
    path: Path,
    title: str,
    dpi: int,
) -> None:
    n_r, total_t = solver.cost(routes)
    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_route_map(ax, solver, routes, title)
    fig.suptitle(f"{n_r} routes   |   total time {total_t/3600:.2f} h",
                 fontsize=9, y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    [viz] {path.name}")


def _save_cluster_map(solver: "_VisualizingSolver", out_dir: Path, dpi: int) -> None:
    """
    Draw a map coloring containers by DBSCAN cluster.
    City cluster = blue, village clusters = distinct colours, noise = grey.
    """
    p           = solver.p
    clabel      = solver._cluster_label
    city_cl     = solver._city_cluster

    # Gather unique non-city, non-noise cluster labels
    village_lbls = sorted({l for l in clabel if l != -1 and l != city_cl})
    # Assign a colour per village cluster
    vcmap   = _get_cmap("Set1")
    v_color = {lbl: vcmap(i / max(len(village_lbls) - 1, 1))
               for i, lbl in enumerate(village_lbls)}

    fig, ax = plt.subplots(figsize=(9, 7))

    city_xs, city_ys       = [], []
    noise_xs, noise_ys     = [], []
    village_pts: dict[int, tuple[list, list]] = {l: ([], []) for l in village_lbls}

    for pos, uid in enumerate(solver.cont_uids):
        node = p.uid_to_node(uid)
        lbl  = clabel[pos]
        if lbl == city_cl:
            city_xs.append(node.lon); city_ys.append(node.lat)
        elif lbl == -1:
            noise_xs.append(node.lon); noise_ys.append(node.lat)
        else:
            village_pts[lbl][0].append(node.lon)
            village_pts[lbl][1].append(node.lat)

    ax.scatter(city_xs,  city_ys,  s=5, color="#2563eb", alpha=0.6,
               zorder=3, linewidths=0, label=f"City cluster ({len(city_xs)})")
    ax.scatter(noise_xs, noise_ys, s=5, color="#9ca3af", alpha=0.5,
               zorder=2, linewidths=0, label=f"Noise ({len(noise_xs)})")
    for lbl, (xs, ys) in village_pts.items():
        ax.scatter(xs, ys, s=7, color=v_color[lbl], alpha=0.8,
                   zorder=4, linewidths=0, label=f"Village cl.{lbl} ({len(xs)})")

    # BASE and DUMP
    base_node = p.uid_to_node(solver.BASE)
    dump_node = p.uid_to_node(solver.DUMP)
    ax.plot(base_node.lon, base_node.lat, marker="*", markersize=14,
            color="#1e3a8a", zorder=6, linestyle="none", label="BASE")
    ax.plot(dump_node.lon, dump_node.lat, marker="D", markersize=8,
            color="#6b7280", zorder=6, linestyle="none", label="DUMP")

    n_city    = len(city_xs)
    n_village = sum(len(v[0]) for v in village_pts.values())
    n_noise   = len(noise_xs)
    ax.set_title(
        f"DBSCAN geographic clustering\n"
        f"city={n_city}  village={n_village}  noise={n_noise}  "
        f"(city_radius={solver.CITY_RADIUS_S:.0f}s)",
        fontsize=10, pad=6,
    )
    ax.set_xlabel("longitude", fontsize=8)
    ax.set_ylabel("latitude",  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.legend(fontsize=6, ncol=2, loc="upper left", framealpha=0.7)

    fig.tight_layout()
    path = out_dir / "snapshot_clusters.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    [viz] {path.name}")


def _save_convergence(history: list[dict], out_dir: Path, dpi: int) -> None:
    if not history:
        return
    labels   = [h["label"]      for h in history]
    n_routes = [h["routes"]     for h in history]
    times_h  = [h["total_time"] / 3600 for h in history]
    xs = range(len(history))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    phase1_routes = n_routes[0]
    ax1.axhline(phase1_routes, color="#888780", linewidth=1,
                linestyle=":", label=f"Phase 1 baseline ({phase1_routes})")

    ax1.step(xs, n_routes, where="post", color="#534AB7", linewidth=1.5)
    ax1.scatter(xs, n_routes, s=25, color="#534AB7", zorder=4)
    ax1.set_ylabel("route count", fontsize=9)
    ax1.set_title("convergence — ALNS iterations", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, linewidth=0.3, alpha=0.4)

    prev = None
    for i, nr in enumerate(n_routes):
        if prev is not None and nr < prev:
            ax1.annotate(f"−{prev-nr}", (i, nr),
                         textcoords="offset points", xytext=(0, 7),
                         fontsize=7, color="#534AB7", ha="center")
        prev = nr

    ax2.plot(xs, times_h, color="#1D9E75", linewidth=1.5)
    ax2.scatter(xs, times_h, s=25, color="#1D9E75", zorder=4)
    ax2.set_ylabel("total time (h)", fontsize=9)
    ax2.set_xlabel("snapshot", fontsize=9)
    ax2.grid(True, linewidth=0.3, alpha=0.4)

    step = max(1, len(labels) // 20)
    ax2.set_xticks(list(xs)[::step])
    ax2.set_xticklabels(labels[::step], rotation=35, ha="right", fontsize=7)

    # separator between Phase 1 and ALNS
    ax1.axvline(0.5, color="#888780", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.axvline(0.5, color="#888780", linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.text(0.55, ax1.get_ylim()[1] * 0.98, "ALNS →",
             fontsize=7, color="#888780", va="top")

    fig.tight_layout(rect=[0.06, 0.10, 1.0, 1.0])
    fig.savefig(out_dir / "convergence.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    [viz] convergence.png")


def _save_route_times(
    solver: "_VisualizingSolver",
    routes: list[list[int]],
    out_dir: Path,
    dpi: int,
) -> None:
    times  = [solver.route_time(r) for r in routes]
    n_r    = len(times)
    colors = _route_colors(n_r)
    limit  = solver.MAX_T

    fig, ax = plt.subplots(figsize=(max(8, n_r * 0.45), 4))
    bars = ax.bar(range(n_r), [t / 3600 for t in times],
                  color=colors, width=0.7, zorder=2, linewidth=0)
    ax.axhline(limit / 3600, color="#dc2626", linewidth=1.2,
               linestyle="--", label=f"limit ({limit/3600:.1f} h)")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{100*t/limit:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("route index", fontsize=9)
    ax.set_ylabel("time (hours)", fontsize=9)
    ax.set_title("route time utilisation vs limit", fontsize=10)
    ax.set_xticks(range(n_r))
    ax.set_xticklabels([f"R{i+1}" for i in range(n_r)], fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax.set_ylim(0, limit / 3600 * 1.15)
    fig.tight_layout()
    fig.savefig(out_dir / "route_times.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    [viz] route_times.png")


# =============================================================================
# Instrumented solver — ALNS pipeline with snapshot hooks
# =============================================================================

class _VisualizingSolver(_Solver):
    """
    Mirrors _Solver.run() exactly but injects snapshot calls at:
      - after Clarke-Wright
      - after Phase 1 local search
      - after DBSCAN clustering (cluster map)
      - each ALNS iteration (via _iter_hook callback → convergence chart)
      - final best solution
    """

    def __init__(self, problem, out_dir: Path, dpi: int):
        super().__init__(problem)
        self.out_dir  = out_dir
        self.dpi      = dpi
        self.history: list[dict] = []

    def run(self, deadline: float) -> list[list[str]]:
        # ── Phase 1 ──────────────────────────────────────────────────────────
        t0     = time.time()
        routes = self.clarke_wright(alpha=0.0)
        print(f"  [Phase 1] CW: {len(routes)} routes ({time.time()-t0:.1f}s)")

        self._snap(routes, "cw", "After Clarke-Wright (before merge)")

        routes = self.route_merge_pass(routes)
        print(f"  [Phase 1] After merge pass: {len(routes)} routes")

        t_ls1  = min(deadline, time.time() + self.PHASE1_S)
        routes = self.local_search(routes, t_ls1)
        best   = routes
        best_c = self.cost(best)
        print(f"  [Phase 1] After local search: routes={best_c[0]}, "
              f"time={best_c[1]:.0f}s")

        self._snap(routes, "phase1", "Phase 1 — after local search")
        self._record("Phase 1", best_c)

        # ── Cluster map (uses DBSCAN data computed in __init__) ───────────────
        _save_cluster_map(self, self.out_dir, self.dpi)

        # ── Phase 2 — ALNS ───────────────────────────────────────────────────
        best = self._alns_run(best, deadline, _iter_hook=self._record_alns_iter)
        best_c = self.cost(best)

        # ── Phase 3 — final merge ─────────────────────────────────────────────
        best   = self.route_merge_pass(best)
        best_c = self.cost(best)
        print(f"  [Final] routes={best_c[0]}, time={best_c[1]:.0f}s")

        self._snap(best, "alns_best", "Best ALNS result (final solution)")

        # Supporting charts
        _save_convergence(self.history, self.out_dir, self.dpi)
        _save_route_times(self, best, self.out_dir, self.dpi)

        return [self.to_uid_route(r) for r in best]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _snap(self, routes: list[list[int]], tag: str, title: str) -> None:
        n_r, total_t = self.cost(routes)
        ts           = time.strftime("%H:%M:%S")
        full_title   = f"{title}\n{n_r} routes | {total_t/3600:.2f} h | {ts}"
        _save_snapshot(self, routes,
                       self.out_dir / f"snapshot_{tag}.png",
                       full_title, self.dpi)

    def _record(self, label: str, cost: tuple[int, float]) -> None:
        self.history.append({
            "label":      label,
            "routes":     cost[0],
            "total_time": cost[1],
        })

    def _record_alns_iter(
        self,
        it:     int,
        best_c: tuple[int, float],
        marker: str,
        d_name: str,
        r_name: str,
    ) -> None:
        self._record(f"A{it}({d_name}+{r_name})", best_c)


# =============================================================================
# Entry point — runs all 4 instances
# =============================================================================

def _replay_instance(
    name: str,
    inst_dir: Path,
    extra_dir: Path,
    out_dir: Path,
    dpi: int,
) -> None:
    """
    Load a previously saved solution.json and draw the final route map.
    The instance data is still needed for coordinates — but the solver
    never runs.  This produces the exact map for the solution in report.json.
    """
    sol_path = extra_dir / name / "solution.json"
    if not sol_path.exists():
        print(f"  [skip] {sol_path} not found — run the solver first.")
        return

    with open(sol_path, encoding="utf-8") as f:
        payload = json.load(f)

    uid_routes: list[list[str]] = payload["routes"]
    n_routes = len(uid_routes)
    print(f"  Loaded {n_routes} routes from {sol_path.name}")

    problem = ProblemInstance.load_from_dir(
        inst_dir, precompute_neighbors=False, verbose=False
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    total_t = sum(problem.total_time_route_uids(r) for r in uid_routes)

    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_route_map_uid(ax, problem, uid_routes,
                        f"{name} — final solution (from run.py)")
    fig.suptitle(
        f"{n_routes} routes   |   total time {total_t/3600:.2f} h   |   "
        f"loaded from solution.json",
        fontsize=9, y=0.98,
    )
    fig.tight_layout()
    out_path = out_dir / "snapshot_final.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    [viz] {out_path.name}")


def _draw_route_map_uid(
    ax: plt.Axes,
    problem,
    uid_routes: list[list[str]],
    title: str,
) -> None:
    """Draw a route map directly from UID route sequences (no internal solver needed)."""
    base_uid = problem.base_uid()
    dump_uid = problem.dump_uid()
    colors   = _route_colors(len(uid_routes))

    def coords(uid: str):
        node = problem.uid_to_node(uid)
        return node.lon, node.lat

    for route, col in zip(uid_routes, colors):
        xs = [coords(u)[0] for u in route]
        ys = [coords(u)[1] for u in route]
        ax.plot(xs, ys, "-", color=col, linewidth=0.7, alpha=0.75, zorder=2)
        cx = [coords(u)[0] for u in route if u not in (base_uid, dump_uid)]
        cy = [coords(u)[1] for u in route if u not in (base_uid, dump_uid)]
        ax.scatter(cx, cy, s=6, color=col, zorder=3, linewidths=0)

    bx, by = coords(base_uid)
    dx, dy = coords(dump_uid)
    ax.plot(bx, by, marker="*", markersize=12, color="#2563eb",
            zorder=5, linestyle="none", label="BASE")
    ax.plot(dx, dy, marker="D", markersize=7,  color="#6b7280",
            zorder=5, linestyle="none", label="DUMP")

    n_rt    = len(uid_routes)
    handles = [mpatches.Patch(color=colors[i], label=f"Route {i+1}")
               for i in range(min(n_rt, 12))]
    if n_rt > 12:
        handles.append(mpatches.Patch(color="white",
                                      label=f"... +{n_rt-12} more", linewidth=0))
    handles += [
        plt.Line2D([0], [0], marker="*", color="#2563eb",
                   linestyle="none", markersize=8, label="BASE"),
        plt.Line2D([0], [0], marker="D", color="#6b7280",
                   linestyle="none", markersize=6, label="DUMP"),
    ]
    ax.legend(handles=handles, fontsize=6, ncol=2,
              loc="upper left", framealpha=0.7)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("longitude", fontsize=8)
    ax.set_ylabel("latitude",  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize SmartEcoRutas ALNS solver — all instances",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="parent folder containing the 4 instance subfolders",
    )
    parser.add_argument(
        "--extra-dir", type=Path, default=Path("extra_algorithm_output"),
        help="folder written by the solver (contains solution.json per instance)",
    )
    parser.add_argument(
        "--from-solution", action="store_true",
        help="replay mode: load saved solution.json instead of re-running the solver",
    )
    parser.add_argument("--time-limit", type=float, default=120.0,
                        help="solver budget in seconds per instance (solver mode only)")
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--dpi",        type=int,   default=150)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    found = []
    for name in INSTANCES:
        p = args.data_dir / name
        if p.exists():
            found.append((name, p))
        else:
            print(f"[skip] {p} not found")

    if not found:
        print(f"No instances found under {args.data_dir}. Check --data-dir.")
        return

    # ── replay mode: load saved solution, draw map, done ─────────────────────
    if args.from_solution:
        print("[viz] Replay mode — loading saved solutions from solution.json")
        for name, inst_dir in found:
            print(f"\n  {name}")
            out_dir = REPO_ROOT / "output" / name
            _replay_instance(name, inst_dir, args.extra_dir, out_dir, args.dpi)
        print("\n[viz] Done.")
        return

    # ── solver mode: run solver, save snapshots ───────────────────────────────
    for name, inst_dir in found:
        print(f"\n{'='*60}")
        print(f"  Instance: {name}")
        print(f"{'='*60}")

        out_dir = REPO_ROOT / "output" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output  : {out_dir}")

        problem  = ProblemInstance.load_from_dir(inst_dir)
        solver   = _VisualizingSolver(problem, out_dir, args.dpi)
        deadline = time.time() + args.time_limit - 2.0
        solver.run(deadline)

        print(f"  Files saved:")
        for f in sorted(out_dir.iterdir()):
            print(f"    {f.name}")

    print("\n[viz] All instances done.")


if __name__ == "__main__":
    main()