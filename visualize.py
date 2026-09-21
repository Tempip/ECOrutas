"""
visualize.py — figures for the SmartEcoRutas solver
====================================================
Every figure in the README is produced by this script, from two sources:

- extra_algorithm_output/<INSTANCE>/: the solutions that
  student/algoritmoSmartEcoRutas.py writes (solution.json, extra_result.json and
  the *_routes.json pipeline snapshots). They feed the route maps.
- results/final_run.json: per-route results of our final run under the official
  protocol, transcribed from its log. It feeds the route-duration chart and the
  results table. Without it, the chart falls back to the saved solutions.

Usage
-----
    python visualize.py                  # figures from the saved output
    python visualize.py --run            # run the solver first (15 min per instance)
    python visualize.py --instances LATERAL_CARTON --theme light
    python visualize.py --social         # also draw the 1280x640 repository card

Output (default folder: docs/img/)
------
    routes_<instance>_<theme>.png   one small map per route, that route highlighted
    utilisation_<theme>.png         duration of every route as a share of the shift
    social-preview.png              repository card for GitHub / LinkedIn (--social)

The results table of the README is printed at the end, in Markdown.

Authors: Pedro José Rodrigues Souza and Illia Pastushenko.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
INSTANCES = ["LATERAL_CARTON", "LATERAL_ENVASE", "LATERAL_RESTO", "TRASERA_RESTO"]
SPECIAL = {"BASE", "DUMP"}

# Chart tokens: one accent for the data, neutral inks for everything else.
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "context": "#cfcdc6", "accent": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "context": "#46463f", "accent": "#3987e5",
    },
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.titlelocation": "left",
})


def px(value: float, dpi: int) -> float:
    """Convert screen pixels to points so line widths stay crisp at any DPI."""
    return value * 72.0 / dpi


# =============================================================================
# Loading
# =============================================================================

def load_instance(data_dir: Path, extra_dir: Path, name: str) -> dict | None:
    """Return coordinates, final routes, per-route summary and snapshots, or None."""
    folder = extra_dir / name
    sol_path, res_path = folder / "solution.json", folder / "extra_result.json"
    if not (sol_path.exists() and res_path.exists()):
        return None

    nodes = pd.read_csv(data_dir / name / "nodes.csv", usecols=["uid", "kind", "lon", "lat"])
    coords = {u: (lo, la) for u, lo, la in zip(nodes["uid"], nodes["lon"], nodes["lat"])}
    containers = nodes[nodes["kind"] == "container"]

    solution = json.loads(sol_path.read_text(encoding="utf-8"))
    result = json.loads(res_path.read_text(encoding="utf-8"))
    routes = solution["routes"]
    summary = result["routes"]
    if len(routes) != len(summary):
        raise ValueError(f"{name}: solution.json and extra_result.json disagree on route count")

    snapshots = {}
    for snap in folder.glob("*_routes.json"):
        data = json.loads(snap.read_text(encoding="utf-8"))
        snapshots[data["tag"]] = (data["n_routes"], data["total_time_s"])

    return {
        "name": name,
        "coords": coords,
        "container_xy": (containers["lon"].to_numpy(), containers["lat"].to_numpy()),
        "routes": routes,
        "summary": summary,
        "limit_s": float(result["limit_s"]),
        "snapshots": snapshots,
    }


def load_final_run(path: Path, names: list[str]) -> list[dict]:
    """Per-route results of the final run, shaped like load_instance() output."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = []
    for name in names:
        inst = data["instances"].get(name)
        if inst is None:
            continue
        runs.append({
            "name": name,
            "limit_s": float(data["route_limit_s"]),
            "summary": [{"total_s": r["total_h"] * 3600} for r in inst["routes"]],
            "containers": inst["containers"],
            "construction": inst["construction"],
            "final": inst["final"],
        })
    return runs


# =============================================================================
# Figure 1 — one small map per route
# =============================================================================

def plot_route_grid(inst: dict, theme: str, out_path: Path, dpi: int) -> None:
    T = THEMES[theme]
    routes, summary, coords = inst["routes"], inst["summary"], inst["coords"]
    n = len(routes)
    cols = 4 if n > 9 else 3
    rows = math.ceil(n / cols)

    cx, cy = inst["container_xy"]
    base, dump = coords["BASE"], coords["DUMP"]
    xs_all = list(cx) + [base[0], dump[0]]
    ys_all = list(cy) + [base[1], dump[1]]
    pad_x = (max(xs_all) - min(xs_all)) * 0.04
    pad_y = (max(ys_all) - min(ys_all)) * 0.04
    xlim = (min(xs_all) - pad_x, max(xs_all) + pad_x)
    ylim = (min(ys_all) - pad_y, max(ys_all) + pad_y)
    aspect = 1.0 / math.cos(math.radians(sum(ylim) / 2))

    span_x = (xlim[1] - xlim[0]) / aspect
    span_y = ylim[1] - ylim[0]
    panel_w = 3.0
    panel_h = panel_w * span_y / span_x + 0.35
    header_h = 1.45
    fig = plt.figure(figsize=(cols * panel_w, rows * panel_h + header_h), dpi=dpi)
    fig.patch.set_facecolor(T["surface"])
    top = 1 - header_h / (rows * panel_h + header_h)
    gs = fig.add_gridspec(rows, cols, left=0.02, right=0.98, bottom=0.02, top=top,
                          wspace=0.06, hspace=0.28)

    stop_size = px(3.2, dpi) ** 2 * 4
    context_size = px(2.2, dpi) ** 2 * 4
    for i in range(rows * cols):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.set_facecolor(T["surface"])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(T["grid"])
            spine.set_linewidth(px(1, dpi))
        if i >= n:
            ax.axis("off")
            continue

        ax.scatter(cx, cy, s=context_size, color=T["context"], linewidths=0, zorder=1)
        path = [coords[u] for u in routes[i]]
        ax.plot([p[0] for p in path], [p[1] for p in path], color=T["accent"],
                linewidth=px(1.4, dpi), alpha=0.9, solid_joinstyle="round",
                solid_capstyle="round", zorder=2)
        stops = [coords[u] for u in routes[i] if u not in SPECIAL]
        ax.scatter([p[0] for p in stops], [p[1] for p in stops], s=stop_size,
                   color=T["accent"], linewidths=0, zorder=3)
        ax.scatter(*base, s=px(9, dpi) ** 2 * 4, marker="s", color=T["ink"],
                   edgecolors=T["surface"], linewidths=px(1.5, dpi), zorder=4)
        ax.scatter(*dump, s=px(10, dpi) ** 2 * 4, marker="^", color=T["ink"],
                   edgecolors=T["surface"], linewidths=px(1.5, dpi), zorder=4)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect(aspect)

        info = summary[i]
        share = 100 * info["total_s"] / inst["limit_s"]
        ax.set_title(f"Route {i + 1}", fontsize=8.5, fontweight="bold", color=T["ink"],
                     pad=14)
        ax.text(0, 1.015, f"{info['n_containers']} stops · {info['total_h']:.2f} h · "
                f"{share:.1f}% of shift", transform=ax.transAxes, fontsize=6.8,
                color=T["ink2"], va="bottom", ha="left")

    n_cont = len(cx)
    head_y = 1 - 0.30 / (rows * panel_h + header_h)
    fig.text(0.02, head_y, f"{inst['name']}: {n} routes cover {n_cont:,} containers",
             fontsize=13, fontweight="bold", color=T["ink"], va="top")
    fig.text(0.02, head_y - 0.42 / (rows * panel_h + header_h),
             "Each panel highlights one route. Lines join consecutive stops in a straight "
             "line, including trips to the dump; real trucks follow the street network.",
             fontsize=8, color=T["ink2"], va="top")

    handles = [
        Line2D([0], [0], color=T["accent"], linewidth=1.6, marker="o", markersize=3.5,
               label="this route"),
        Line2D([0], [0], color=T["context"], linestyle="none", marker="o", markersize=3.5,
               label="other containers"),
        Line2D([0], [0], color=T["ink"], linestyle="none", marker="s", markersize=5,
               label="base"),
        Line2D([0], [0], color=T["ink"], linestyle="none", marker="^", markersize=5.5,
               label="dump"),
    ]
    legend = fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False,
                        fontsize=7.5, bbox_to_anchor=(0.985, head_y + 0.004),
                        handletextpad=0.4, columnspacing=1.2)
    for text in legend.get_texts():
        text.set_color(T["ink2"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=T["surface"])
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path}")


# =============================================================================
# Figure 2 — route duration as a share of the working-time limit
# =============================================================================

def plot_utilisation(instances: list[dict], theme: str, out_path: Path, dpi: int,
                     subtitle: str = "Each dot is one route of the final solution.") -> None:
    T = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8.2, 0.78 * len(instances) + 1.75), dpi=dpi)
    fig.patch.set_facecolor(T["surface"])
    ax.set_facecolor(T["surface"])

    all_shares = []
    for row, inst in enumerate(instances):
        y = len(instances) - 1 - row
        shares = sorted(100 * r["total_s"] / inst["limit_s"] for r in inst["summary"])
        all_shares += shares
        # Deterministic vertical spread so dots with similar values stay visible.
        offsets = [((k % 5) - 2) * 0.075 for k in range(len(shares))]
        ax.scatter(shares, [y + o for o in offsets], s=px(8, dpi) ** 2 * 4,
                   color=T["accent"], edgecolors=T["surface"], linewidths=px(2, dpi),
                   zorder=3)
        ax.text(1.005, y, f"{len(shares)} routes · median {statistics.median(shares):.1f}%",
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                fontsize=8, color=T["ink2"], va="center", ha="left")

    span = 100 - min(all_shares)
    step = 1 if span <= 6 else 2 if span <= 12 else 5 if span <= 30 else 10
    low = math.floor((min(all_shares) - step / 2) / step) * step
    ax.set_xlim(low, 100 + step * 0.35)
    ax.set_ylim(-0.6, len(instances) - 0.4)
    ax.set_yticks(range(len(instances)))
    ax.set_yticklabels([inst["name"] for inst in reversed(instances)], fontsize=8.5,
                       color=T["ink"])
    ticks = list(range(int(low), 101, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks], fontsize=8, color=T["muted"])
    ax.tick_params(axis="both", length=0, pad=6)
    ax.grid(axis="x", color=T["grid"], linewidth=px(1, dpi), zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(T["axis"])
    ax.spines["bottom"].set_linewidth(px(1, dpi))

    ax.axvline(100, color=T["ink2"], linewidth=px(1.5, dpi), zorder=2)
    ax.text(100, len(instances) - 0.45, "shift limit", fontsize=7.5, color=T["ink2"],
            ha="right", va="bottom")

    fig.text(0.015, 0.975, "Route duration as a share of the working-time limit",
             fontsize=12, fontweight="bold", color=T["ink"], va="top")
    fig.text(0.015, 0.975 - 0.34 / fig.get_figheight(), subtitle,
             fontsize=8.5, color=T["ink2"], va="top")
    fig.subplots_adjust(left=0.2, right=0.79, top=1 - 0.95 / fig.get_figheight(),
                        bottom=0.42 / fig.get_figheight())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=T["surface"])
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path}")


# =============================================================================
# Repository card (GitHub social preview, 1280 x 640)
# =============================================================================

def plot_social_preview(inst: dict, out_path: Path) -> None:
    T = THEMES["dark"]
    dpi = 200
    fig = plt.figure(figsize=(1280 / dpi, 640 / dpi), dpi=dpi)
    fig.patch.set_facecolor(T["surface"])

    ax = fig.add_axes([0.56, 0.08, 0.41, 0.84])
    ax.set_facecolor(T["surface"])
    ax.axis("off")
    cx, cy = inst["container_xy"]
    coords = inst["coords"]
    ax.scatter(cx, cy, s=1.2, color="#6b6b63", linewidths=0, zorder=1)
    for route in inst["routes"]:
        path = [coords[u] for u in route]
        ax.plot([p[0] for p in path], [p[1] for p in path], color=T["accent"],
                linewidth=0.45, alpha=0.75, zorder=2)
    ax.set_aspect(1.0 / math.cos(math.radians(float(cy.mean()))))

    fig.text(0.06, 0.70, "SmartEcoRutas", fontsize=26, fontweight="bold", color=T["ink"])
    fig.text(0.06, 0.525, "Vehicle routing for waste\ncollection in Cartagena",
             fontsize=10.5, color=T["ink2"], linespacing=1.35)
    fig.text(0.06, 0.36, "Winning solution · Retos-UPCT 2026", fontsize=10,
             fontweight="bold", color=T["accent"])
    fig.text(0.06, 0.285, "Challenge sponsored by Lhicarsa", fontsize=9, color=T["ink2"])
    fig.text(0.06, 0.12, "Pedro José Rodrigues Souza · Illia Pastushenko", fontsize=8.5,
             color=T["muted"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=T["surface"])
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path}")


# =============================================================================
# Summary table
# =============================================================================

def summary_table(instances: list[dict]) -> str:
    lines = [
        "| Instance | Containers | Routes after construction | Final routes "
        "| Total route time (h) | Median route vs shift limit |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for inst in instances:
        built = inst["snapshots"].get("solomon_construction", (None, None))[0]
        shares = [100 * r["total_s"] / inst["limit_s"] for r in inst["summary"]]
        total_h = sum(r["total_s"] for r in inst["summary"]) / 3600
        lines.append(
            f"| `{inst['name']}` | {len(inst['container_xy'][0]):,} | {built if built else '—'} "
            f"| **{len(inst['routes'])}** | {total_h:.1f} | {statistics.median(shares):.1f}% |"
        )
    return "\n".join(lines)


def results_table(runs: list[dict]) -> str:
    """The README results table: best construction -> final solution, per instance."""
    lines = [
        "| Instance | Containers | Routes | Total route time | Driving time |",
        "|---|---:|---:|---:|---:|",
    ]
    sums = {"containers": 0, "r0": 0, "r1": 0, "h0": 0.0, "h1": 0.0, "drive": 0.0}
    for run in runs:
        built, final = run["construction"], run["final"]
        h1 = final["total_time_s"] / 3600
        drive = final["travel_time_s"] / 3600
        lines.append(
            f"| `{run['name']}` | {run['containers']:,} | {built['routes']} → **{final['routes']}** "
            f"| {built['total_time_h']:.1f} h → **{h1:.1f} h** | {drive:.1f} h |"
        )
        for key, value in (("containers", run["containers"]), ("r0", built["routes"]),
                           ("r1", final["routes"]), ("h0", built["total_time_h"]),
                           ("h1", h1), ("drive", drive)):
            sums[key] += value
    if len(runs) > 1:
        lines.append(
            f"| **Total** | **{sums['containers']:,}** | {sums['r0']} → **{sums['r1']}** "
            f"| {sums['h0']:.1f} h → **{sums['h1']:.1f} h** | **{sums['drive']:.1f} h** |"
        )
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the figures for the SmartEcoRutas solver.")
    parser.add_argument("--data-dir", default="data", help="instance folders (default: data)")
    parser.add_argument("--extra-dir", default="extra_algorithm_output",
                        help="solver output folder (default: extra_algorithm_output)")
    parser.add_argument("--results", default="results/final_run.json",
                        help="per-route results of the final run (default: results/final_run.json)")
    parser.add_argument("--out-dir", default="docs/img", help="where figures go (default: docs/img)")
    parser.add_argument("--instances", nargs="*", default=INSTANCES)
    parser.add_argument("--theme", choices=["light", "dark", "both"], default="both")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--run", action="store_true",
                        help="run the solver through run.py before drawing")
    parser.add_argument("--time-limit-min", type=float, default=15.0,
                        help="solver budget per instance when --run is given (default: 15)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--social", action="store_true",
                        help="also draw the 1280x640 repository card")
    args = parser.parse_args()

    data_dir = (REPO_ROOT / args.data_dir).resolve()
    extra_dir = (REPO_ROOT / args.extra_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()

    if args.run:
        cmd = [sys.executable, "run.py", "--instances", *args.instances, "--no-geo",
               "--time-limit-min", str(args.time_limit_min), "--seed", str(args.seed)]
        print("[viz] running:", " ".join(cmd))
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    loaded = []
    for name in args.instances:
        inst = load_instance(data_dir, extra_dir, name)
        if inst is None:
            print(f"[viz] {name}: no solution.json/extra_result.json in {extra_dir} — no map "
                  "(run with --run first)")
            continue
        loaded.append(inst)
    final_run = [] if args.run else load_final_run((REPO_ROOT / args.results).resolve(),
                                                   args.instances)
    if not loaded and not final_run:
        sys.exit("[viz] nothing to draw")

    # The duration chart uses the final run when available, else the saved solutions.
    if final_run:
        chart_data = final_run
        subtitle = ("Each dot is one route. Final run under the official protocol: "
                    "15 minutes per instance, seed 0.")
    else:
        chart_data = loaded
        subtitle = "Each dot is one route of the saved solution."

    themes = ["light", "dark"] if args.theme == "both" else [args.theme]
    for theme in themes:
        for inst in loaded:
            plot_route_grid(inst, theme, out_dir / f"routes_{inst['name'].lower()}_{theme}.png",
                            args.dpi)
        plot_utilisation(chart_data, theme, out_dir / f"utilisation_{theme}.png", args.dpi,
                         subtitle)
    if args.social and loaded:
        plot_social_preview(loaded[0], out_dir / "social-preview.png")

    if final_run:
        print("\nFinal run (README results table):\n")
        print(results_table(final_run))
    if loaded:
        print("\nSaved solutions in extra_algorithm_output/:\n")
        print(summary_table(loaded))


if __name__ == "__main__":
    main()
