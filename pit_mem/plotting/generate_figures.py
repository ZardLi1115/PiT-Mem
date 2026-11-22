"""
PVLDB Best-Paper Visual Paradigm Figure Generator for PiT-Mem (Major Revision).
  - 100% Type-42 TrueType vector PDF embedding (zero Type-3 bitmaps)
  - STIX / Serif academic typography aligned with ACM `acmart` Linux Libertine
  - Top-centered multi-column shared legend strips (zero curve occlusion)
  - Continuous single-path mitered vector arrows (`Stealth`, zero broken T-junctions)
  - Colorblind-safe high-contrast palette + grayscale-distinct markers & vector hatches
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle, Rectangle
import numpy as np
import pandas as pd

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.6,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.3,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.85,
    "axes.edgecolor": "#222222",
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "legend.fontsize": 7.8,
    "axes.grid": True,
    "grid.alpha": 0.40,
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "grid.color": "#AAB2BD",
})

PALETTE = {
    "PiT-Mem": "#C0392B",
    "PiT-Mem (Full)": "#C0392B",
    "PiT-Mem (Lazy-Branch MVCC)": "#C0392B",
    "RAP-Compact (Ours)": "#C0392B",
    "Grid-SQ8-Scan": "#6A3D9A",
    "PiT-Mem-ScanOnly": "#6A3D9A",
    "PiT-Mem-GraphOnly": "#D95F02",
    "ACORN-BiTemp": "#2B5C8F",
    "PG-BiTemp": "#1F4E79",
    "Zep/Graphiti": "#1B7837",
    "Zep/Graphiti (Community)": "#1B7837",
    "SeRF-Adapt": "#8C510A",
    "DSG-1D": "#018571",
    "HNSW-Post": "#4D4D4D",
    "HNSW-Post (ef=1024)": "#3B3B3B",
    "Mem0-Overwrite": "#2B5C8F",
    "Naive-GraphRAG": "#7F8C8D",
    "Eager-LLM-Consolidate": "#1F4E79",
    "Unpack-Only (No Repair)": "#D95F02",
    "SeRF-Static-Rebuild": "#8C510A",
    "Parquet-Zstd + Medoid": "#1F4E79",
    "SQLite-BTree + Medoid": "#018571",
    "LLM-Abstractive-Summary": "#6A3D9A",
    "Centroid-MeanPool": "#1B7837",
    "Narrative-Truncation": "#5D6D7E",
}

MARKERS = {
    "PiT-Mem": "o",
    "Grid-SQ8-Scan": "p",
    "PiT-Mem-ScanOnly": "p",
    "PiT-Mem-GraphOnly": "D",
    "ACORN-BiTemp": "X",
    "PG-BiTemp": "s",
    "Zep/Graphiti": "d",
    "SeRF-Adapt": "v",
    "DSG-1D": "P",
    "HNSW-Post": "^",
    "HNSW-Post (ef=1024)": "^",
    "RAP-Compact (Ours)": "o",
    "Parquet-Zstd + Medoid": "s",
    "SQLite-BTree + Medoid": "P",
    "LLM-Abstractive-Summary": "v",
    "Centroid-MeanPool": "D",
    "Narrative-Truncation": "^",
}

LINESTYLES = {
    "PiT-Mem": "-",
    "Grid-SQ8-Scan": "-.",
    "PiT-Mem-ScanOnly": "-.",
    "PiT-Mem-GraphOnly": "--",
    "ACORN-BiTemp": "--",
    "PG-BiTemp": "-",
    "Zep/Graphiti": ":",
    "SeRF-Adapt": "-",
    "DSG-1D": "-.",
    "HNSW-Post": ":",
    "HNSW-Post (ef=1024)": ":",
}


def _save_both(fig: plt.Figure, fig_dir: str, stem: str, preview_dir: str | None = None) -> None:
    pdf_path = os.path.join(fig_dir, f"{stem}.pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
        fig.savefig(os.path.join(preview_dir, f"{stem}.png"), dpi=260, bbox_inches="tight", pad_inches=0.03)


def _add_better_badge(ax: plt.Axes, text: str, loc: str = "upper right") -> None:
    pos_map = {
        "upper right": (0.965, 0.935, "right", "top"),
        "upper left": (0.035, 0.935, "left", "top"),
        "lower right": (0.965, 0.065, "right", "bottom"),
        "lower left": (0.035, 0.065, "left", "bottom"),
    }
    x, y, ha, va = pos_map[loc]
    ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va,
        fontsize=7.2, fontweight="bold", color="#1B4F72",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#EBF5FB", edgecolor="#5DADE2", linewidth=0.65, alpha=0.94),
        zorder=8,
    )


def _callout_arrow(
    color: str,
    lw: float = 0.95,
    rad: float = 0.0,
    shrinkA: float = 0.0,
    shrinkB: float = 3.2,
) -> dict:
    kw: dict = {
        "arrowstyle": ArrowStyle("-|>", head_length=0.50, head_width=0.22),
        "color": color,
        "lw": lw,
        "shrinkA": shrinkA,
        "shrinkB": shrinkB,
        "joinstyle": "miter",
        "capstyle": "butt",
    }
    if abs(rad) > 1e-6:
        kw["connectionstyle"] = f"arc3,rad={rad}"
    return kw


# ==============================================================================
# FIGURE 1: Motivational Pathology Quadrant & Bidirectional Leakage
# ==============================================================================
def plot_fig1_motivation(results_dir: str, fig_dir: str, preview_dir: str | None = None) -> None:
    df1 = pd.read_csv(os.path.join(results_dir, "exp1_scalability_qps_recall.csv"))
    df4 = pd.read_csv(os.path.join(results_dir, "exp4_leakage_and_audit.csv"))

    sub_sp = df1[df1["regime"] == "sparse_1pct"] if "regime" in df1.columns else df1
    mot_scale = int(sub_sp["scale"].max())
    sub = sub_sp[sub_sp["scale"] == mot_scale]
    sc_lbl = f"{mot_scale // 1_000_000}M" if mot_scale >= 1_000_000 else f"{mot_scale // 1_000}K"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.25, 2.75), gridspec_kw={"width_ratios": [1.08, 1.0]})
    fig.subplots_adjust(wspace=0.31)

    max_p99 = max(80.0, float(sub["p99_lat_ms"].max()) * 1.9)
    ax1.axvspan(0.0, 70.0, color="#FDEDEC", alpha=0.40, zorder=0)
    ax1.axhspan(10.0, max_p99 * 1.2, xmin=0.62, xmax=1.0, color="#EBF5FB", alpha=0.45, zorder=0)
    rect_sla = Rectangle((88.0, 0.06), 17.5, 5.0, facecolor="#D5F5E3", edgecolor="#27AE60",
                         linestyle="--", linewidth=0.85, alpha=0.65, zorder=1)
    ax1.add_patch(rect_sla)
    ax1.text(96.8, 1.55, "SLA\nZone", ha="center", va="top", fontsize=6.5,
             fontweight="bold", color="#196F3D", zorder=3)
    ax1.text(24.0, 2.4, "Percolation\nCollapse", ha="center", va="center",
             fontsize=6.8, style="italic", color="#922B21", zorder=2)

    systems_m = ["PiT-Mem", "PG-BiTemp", "ACORN-BiTemp", "SeRF-Adapt", "DSG-1D", "HNSW-Post", "Zep/Graphiti"]

    pit_pt, pg_pt = None, None
    other_pts = []
    for s in systems_m:
        rows = sub[sub["system"] == s]
        if len(rows) == 0:
            continue
        if s == "PiT-Mem" and len(rows[rows["param_ef"] == 96]) > 0:
            pick = rows[rows["param_ef"] == 96].iloc[0]
        else:
            pick = rows.sort_values(["recall_at_10", "qps"], ascending=[False, False]).iloc[0]
        rx = float(pick["recall_at_10"]) * 100.0
        py = float(pick["p99_lat_ms"])
        is_ours = (s == "PiT-Mem")
        ax1.scatter(
            [rx], [py],
            s=90 if is_ours else 46,
            marker=MARKERS.get(s, "o"),
            color=PALETTE.get(s, "#333"),
            edgecolors="white",
            linewidths=0.85,
            zorder=7 if is_ours else 5,
        )
        if is_ours:
            ax1.annotate(
                f"PiT-Mem\n({py:.2f}ms, {rx:.0f}%)",
                xy=(rx, py), xytext=(38.0, max(0.09, min(0.65, py * 0.45))),
                fontsize=6.8, fontweight="bold", color="#922B21",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#C0392B", lw=0.65, alpha=0.95),
                arrowprops=_callout_arrow("#C0392B", lw=0.95, shrinkA=0.0, shrinkB=4.0),
                zorder=8,
            )
            pit_pt = (rx, py)
        else:
            other_pts.append((py, rx, s))
            if s == "PG-BiTemp":
                pg_pt = (rx, py)

    disp_m1 = {
        "PG-BiTemp": "PG-BiTemp",
        "ACORN-BiTemp": "ACORN-Style (C++)",
        "SeRF-Adapt": "SeRF-Style 1D",
        "DSG-1D": "DSG-Style 1D",
        "HNSW-Post": "HNSW-Flat (C++)",
        "Zep/Graphiti": "Graphiti-Style",
    }
    # Sort baseline points by log-y so staggered left-side callouts never overlap
    other_pts.sort(key=lambda t: t[0], reverse=True)
    y_slots = np.geomspace(max_p99 * 0.62, max(4.5, max_p99 * 0.045), max(1, len(other_pts)))
    for (py, rx, s), y_lbl in zip(other_pts, y_slots):
        x_lbl = max(14.0, min(48.0, rx - 44.0))
        ax1.annotate(
            disp_m1.get(s, s), xy=(rx, py), xytext=(x_lbl, y_lbl),
            fontsize=5.9, color="#222222", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="#7F8C8D", lw=0.55, shrinkA=1.5, shrinkB=2.5),
            zorder=8,
        )

    if pit_pt and pg_pt and pg_pt[1] > pit_pt[1] * 1.5:
        speedup = pg_pt[1] / max(1e-4, pit_pt[1])
        ax1.annotate(
            "", xy=(pit_pt[0], pit_pt[1]), xytext=(pg_pt[0], pg_pt[1]),
            arrowprops=_callout_arrow("#C0392B", lw=1.05, rad=-0.16, shrinkA=4.5, shrinkB=5.8),
            zorder=6,
        )
        mid_y = np.sqrt(pit_pt[1] * pg_pt[1]) * 1.15
        ax1.text(
            79.0, mid_y, f"{speedup:.1f}× Lower P99",
            fontsize=6.3, fontweight="bold", color="#C0392B", ha="center",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#C0392B", lw=0.6, alpha=0.95),
            zorder=9,
        )

    ax1.set_yscale("log")
    ax1.set_xlim(8, 106)
    ax1.set_ylim(0.06, max_p99)
    ax1.set_xlabel("Recall@10 (%)")
    ax1.set_ylabel("P99 Tail Latency (ms, log)")
    ax1.set_title(f"(a) AS-OF Pathology (N={sc_lbl}, $\\sigma\\approx 1\\%$)")
    _add_better_badge(ax1, "Better ↘", "upper left")

    pa = df4[df4["protocol"] == "Protocol_A_Post2025"].copy()
    zep_key = "Zep/Graphiti (Community)" if "Zep/Graphiti (Community)" in pa["system"].values else "Zep/Graphiti"
    s4 = ["PiT-Mem (Full)", zep_key, "Mem0-Overwrite", "Naive-GraphRAG"]
    pa = pa.set_index("system").reindex([s for s in s4 if s in pa["system"].values]).reset_index()
    x2 = np.arange(len(pa))
    w = 0.35

    b_lcr = ax2.bar(
        x2 - w / 2, pa["lcr_pct"], width=w,
        color="#D95F02", edgecolor="#1A1A1A", linewidth=0.7, hatch="///",
        label="Look-Ahead Leak (LCR)", zorder=4,
    )
    b_tnt = ax2.bar(
        x2 + w / 2, pa["stale_taint_pct"], width=w,
        color="#2B5C8F", edgecolor="#1A1A1A", linewidth=0.7, hatch="\\\\\\",
        label="Stale Summary Taint", zorder=4,
    )

    for bars, col in [(b_lcr, "#A04000"), (b_tnt, "#1B4F72")]:
        for bar in bars:
            h = bar.get_height()
            txt = "0%" if h < 0.05 else f"{h:.0f}%"
            ax2.text(
                bar.get_x() + bar.get_width() / 2, h + 1.8, txt,
                ha="center", va="bottom", fontsize=6.5,
                fontweight="bold" if h < 0.05 else "normal",
                color="#C0392B" if h < 0.05 else col, zorder=6,
            )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(["PiT-Mem\n(Ours)", "Latest-Sum.\n(Graphiti-st.)", "Overwrite\n(Mem0-st.)", "Append-Only\n(GraphRAG)"], fontsize=6.7)
    ax2.set_ylabel("Contamination Rate (%)")
    ax2.set_ylim(0, 142)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_title("(b) Dual-Axis Policy Leakage (SEC)")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=6.7, handlelength=1.35)
    _add_better_badge(ax2, "Better ↓", "upper right")

    _save_both(fig, fig_dir, "fig1_motivation", preview_dir)
    plt.close(fig)


# ==============================================================================
# FIGURE 2: Flagship Double-Column Architecture & Physical Layout Blueprint
# ==============================================================================
def _build_fig2_tikz_source(tok_save_avg: float, max_comp: float) -> str:
    grid_cmds = []
    gx, gy, cs = 50.20, 30.85, 2.28
    for r in range(5):
        for c in range(5):
            cx = gx + c * cs
            cy = gy + r * cs
            if r == 4 or c == 4:
                grid_cmds.append(f"  \\futurecell{{{cx:.2f}}}{{{cy:.2f}}}{{{cs:.2f}}}")
            elif r == 0 and c == 1:
                grid_cmds.append(
                    f"  \\filldraw[fill=RedFill, draw=RedStroke, line width=0.50pt] "
                    f"({cx:.2f}, {cy:.2f}) rectangle ({cx + cs:.2f}, {cy + cs:.2f});"
                )
            elif r == 3 or c == 3:
                grid_cmds.append(
                    f"  \\filldraw[fill=white, draw=BlueStroke, densely dashed, line width=0.55pt] "
                    f"({cx:.2f}, {cy:.2f}) rectangle ({cx + cs:.2f}, {cy + cs:.2f});"
                )
            else:
                grid_cmds.append(
                    f"  \\filldraw[fill=BlueFill, draw=BlueStroke, line width=0.50pt] "
                    f"({cx:.2f}, {cy:.2f}) rectangle ({cx + cs:.2f}, {cy + cs:.2f});"
                )
    grid_tex = "\n".join(grid_cmds)

    bits = [1, 0, 1, 1, 0, 0, 1, 0]
    lane_cmds = []
    for lane, alive in enumerate(bits):
        lx = 45.60 + lane * 4.18
        fc = "BlueFill" if alive else "RedFill"
        ec = "BlueStroke" if alive else "RedStroke"
        tc = "BlueText" if alive else "RedText"
        lbl = "Live" if alive else "Dead"
        lane_cmds.append(
            f"  \\filldraw[fill={fc}, draw={ec}, line width=0.58pt] "
            f"({lx:.2f}, 9.40) rectangle ({lx + 3.82:.2f}, 17.60);\n"
            f"  \\node[font=\\fontsize{{5.1}}{{6.0}}\\selectfont\\bfseries, text={tc}, align=center] "
            f"at ({lx + 1.91:.2f}, 13.50) {{L{lane}\\\\{lbl}}};"
        )
    lanes_tex = "\n".join(lane_cmds)

    return rf"""\documentclass[tikz,border=1.0pt]{{standalone}}
\usepackage[T1]{{fontenc}}
\usepackage{{libertine}}
\usepackage[libertine]{{newtxmath}}
\usepackage[scaled=0.85]{{inconsolata}}
\usepackage{{anyfontsize}}
\usetikzlibrary{{arrows.meta,calc}}

\definecolor{{SlateDark}}{{HTML}}{{1E293B}}
\definecolor{{SlateMid}}{{HTML}}{{475569}}
\definecolor{{SlateBorder}}{{HTML}}{{94A3B8}}
\definecolor{{SlateRule}}{{HTML}}{{CBD5E1}}
\definecolor{{BgLane}}{{HTML}}{{F8FAFC}}
\definecolor{{BgNeutral}}{{HTML}}{{F1F5F9}}
\definecolor{{FutureFill}}{{HTML}}{{E2E8F0}}
\definecolor{{BlueStroke}}{{HTML}}{{1F4E79}}
\definecolor{{BlueFill}}{{HTML}}{{EBF3FA}}
\definecolor{{BlueText}}{{HTML}}{{102A43}}
\definecolor{{RedStroke}}{{HTML}}{{B03A2E}}
\definecolor{{RedFill}}{{HTML}}{{FDEDEC}}
\definecolor{{RedText}}{{HTML}}{{78281F}}

\newcommand{{\drawstage}}[4]{{%
  \fill[BgLane, rounded corners=1.65mm] (#1, 0.0) rectangle ({{#1+#2}}, 53.6);
  \path[fill=SlateDark] (#1, 49.1) {{[rounded corners=1.65mm] -- (#1, 53.6) -- ({{#1+#2}}, 53.6)}} -- ({{#1+#2}}, 49.1) -- cycle;
  \draw[SlateDark, line width=1.05pt, rounded corners=1.65mm] (#1, 0.0) rectangle ({{#1+#2}}, 53.6);
  \node[font=\fontsize{{7.1}}{{8.3}}\selectfont\bfseries, text=white, anchor=center] at ({{#1+#2/2}}, 51.35) {{#3.\hspace{{0.42em}}#4}};
}}

\newcommand{{\drawsubcard}}[5]{{%
  \filldraw[fill=white, draw=SlateBorder, line width=0.80pt, rounded corners=1.15mm] (#1, #2) rectangle ({{#1+#3}}, {{#2+#4}});
  \node[font=\fontsize{{6.25}}{{7.4}}\selectfont\bfseries, text=SlateDark, anchor=center] at ({{#1+#3/2}}, {{#2+#4-1.85}}) {{#5}};
  \draw[SlateRule, line width=0.65pt] ({{#1+1.25}}, {{#2+#4-3.45}}) -- ({{#1+#3-1.25}}, {{#2+#4-3.45}});
}}

\newcommand{{\futurecell}}[3]{{%
  \begin{{scope}}
    \fill[FutureFill] (#1, #2) rectangle ({{#1+#3}}, {{#2+#3}});
    \clip (#1, #2) rectangle ({{#1+#3}}, {{#2+#3}});
    \draw[SlateMid, line width=0.42pt]
      ({{#1-0.45*#3}}, #2) -- ({{#1+0.55*#3}}, {{#2+#3}})
      ({{#1+0.02*#3}}, #2) -- ({{#1+1.02*#3}}, {{#2+#3}})
      ({{#1+0.49*#3}}, #2) -- ({{#1+1.49*#3}}, {{#2+#3}});
  \end{{scope}}
  \draw[SlateMid, line width=0.60pt] (#1, #2) rectangle ({{#1+#3}}, {{#2+#3}});
}}

\begin{{document}}
\begin{{tikzpicture}}[
  x=1.26mm, y=1.26mm,
  flow/.style={{
    -{{Stealth[length=5.3pt, width=3.9pt, inset=0.90pt]}},
    draw=SlateDark,
    line width=1.05pt,
    line join=miter,
    line cap=butt,
    shorten <=0pt,
    shorten >=0pt
  }},
  forkflow/.style={{
    {{Stealth[length=5.3pt, width=3.9pt, inset=0.90pt]}}-{{Stealth[length=5.3pt, width=3.9pt, inset=0.90pt]}},
    draw=SlateDark,
    line width=1.05pt,
    line join=miter,
    line cap=butt,
    shorten <=0pt,
    shorten >=0pt
  }},
  stageflow/.style={{
    -{{Stealth[length=6.4pt, width=4.8pt, inset=1.05pt]}},
    draw=SlateDark,
    line width=1.42pt,
    line join=miter,
    line cap=butt,
    shorten <=0pt,
    shorten >=0pt
  }},
  dualaxis/.style={{
    {{Stealth[length=4.7pt, width=3.4pt, inset=0.75pt]}}-{{Stealth[length=4.7pt, width=3.4pt, inset=0.75pt]}},
    draw=SlateDark,
    line width=0.88pt,
    line join=miter,
    line cap=butt
  }}
]

% STAGE 1: Bi-Temporal Model & NVMe WAL (x = 0.0 .. 38.6)
\drawstage{{0.0}}{{38.6}}{{1}}{{Bi-Temporal Model \& NVMe WAL}}

\drawsubcard{{1.4}}{{25.2}}{{35.8}}{{22.3}}{{2D Bi-Temporal Stabbing Plane}}
\fill[BgNeutral] (6.4, 28.8) rectangle (22.4, 39.4);
\draw[SlateDark, densely dashed, line width=0.62pt] (6.4, 39.4) -- (22.4, 39.4) -- (22.4, 28.8);
\filldraw[fill=RedFill, draw=RedStroke, line width=0.62pt] (7.5, 30.5) rectangle (18.5, 36.6);
\node[font=\fontsize{{5.1}}{{6.0}}\selectfont, text=RedText, align=center] at (13.0, 33.55) {{$m_1$ (Restated)\\$t_k^e = t_k^{{\mathrm{{now}}}}$}};
\filldraw[fill=BlueFill, draw=BlueStroke, line width=0.62pt] (18.5, 30.5) rectangle (33.8, 36.6);
\node[font=\fontsize{{5.1}}{{6.0}}\selectfont, text=BlueText, align=center] at (26.15, 33.55) {{$m_1'$ (Active Branch)\\$[t_k^{{\mathrm{{now}}}}, \infty)$}};
\draw[dualaxis] (6.4, 42.9) -- (6.4, 28.8) -- (35.6, 28.8);
\filldraw[fill=SlateDark, draw=white, line width=0.45pt] (22.4, 39.4) circle[radius=0.82mm];
\node[font=\fontsize{{5.5}}{{6.5}}\selectfont\bfseries, text=SlateDark, anchor=west] at (23.0, 40.5) {{$Q(\mathbf{{q}}, t_v^*, t_k^*)$}};
\node[font=\fontsize{{5.3}}{{6.2}}\selectfont, text=SlateMid] at (21.0, 26.95) {{Transaction Time $t_k$}};
\node[font=\fontsize{{5.3}}{{6.2}}\selectfont, text=SlateMid, rotate=90] at (3.85, 35.8) {{Valid Time $t_v$}};

\drawsubcard{{1.4}}{{1.5}}{{35.8}}{{22.3}}{{Operators \& NVMe Durability}}
\fill[BlueStroke] (2.65, 17.25) rectangle (3.30, 17.90);
\node[font=\fontsize{{5.15}}{{6.2}}\selectfont, text=SlateDark, anchor=west] at (3.70, 17.58) {{64-bit ID $\cdot$ $\mathbf{{EVOLVE}}(t_v^{{\mathrm{{chg}}}})$ slice}};
\fill[BlueStroke] (2.65, 13.15) rectangle (3.30, 13.80);
\node[font=\fontsize{{5.15}}{{6.2}}\selectfont, text=SlateDark, anchor=west] at (3.70, 13.48) {{$\mathbf{{RESTATE}}(t_k^{{\mathrm{{now}}}})$: $O(1)$ death + taint}};
\fill[BlueStroke] (2.65, 9.05) rectangle (3.30, 9.70);
\node[font=\fontsize{{5.15}}{{6.2}}\selectfont, text=SlateDark, anchor=west] at (3.70, 9.38) {{POSIX \texttt{{mmap}} Snapshot + Cold FP32}};
\fill[BlueStroke] (2.65, 4.95) rectangle (3.30, 5.60);
\node[font=\fontsize{{5.15}}{{6.2}}\selectfont, text=SlateDark, anchor=west] at (3.70, 5.28) {{Append-Only CRC32 Mutation WAL}};

% STAGE 2: Streaming Macro-Segments & AVX2 Layout (x = 42.8 .. 101.4)
\drawstage{{42.8}}{{58.6}}{{2}}{{Streaming Macro-Segments \& AVX2 Layout}}

\drawsubcard{{44.2}}{{25.2}}{{23.4}}{{22.3}}{{Causal Macro-Segments}}
{grid_tex}
\filldraw[fill=BlueFill, draw=BlueStroke, line width=0.48pt] (45.6, 28.25) rectangle (46.85, 29.35);
\node[font=\fontsize{{5.0}}{{5.9}}\selectfont, text=SlateDark, anchor=west] at (47.15, 28.80) {{Interior}};
\filldraw[fill=white, draw=BlueStroke, densely dashed, line width=0.50pt] (56.2, 28.25) rectangle (57.45, 29.35);
\node[font=\fontsize{{5.0}}{{5.9}}\selectfont, text=SlateDark, anchor=west] at (57.75, 28.80) {{Boundary}};
\filldraw[fill=RedFill, draw=RedStroke, line width=0.48pt] (45.6, 26.15) rectangle (46.85, 27.25);
\node[font=\fontsize{{5.0}}{{5.9}}\selectfont, text=SlateDark, anchor=west] at (47.15, 26.70) {{Pruned}};
\futurecell{{56.2}}{{26.15}}{{1.18}}
\node[font=\fontsize{{5.0}}{{5.9}}\selectfont, text=SlateDark, anchor=west] at (57.75, 26.70) {{Future}};

\drawsubcard{{68.8}}{{25.2}}{{31.2}}{{22.3}}{{32-Bit Intra-Seg Edge \& Budgets}}
\filldraw[fill=BgNeutral, draw=SlateMid, line width=0.58pt] (70.2, 38.3) rectangle (75.4, 42.4);
\node[font=\fontsize{{5.35}}{{6.3}}\selectfont\bfseries, text=SlateDark] at (72.8, 40.35) {{4b $g_k$}};
\filldraw[fill=BgNeutral, draw=SlateMid, line width=0.58pt] (75.4, 38.3) rectangle (80.6, 42.4);
\node[font=\fontsize{{5.35}}{{6.3}}\selectfont\bfseries, text=SlateDark] at (78.0, 40.35) {{4b $g_v$}};
\filldraw[fill=BlueFill, draw=BlueStroke, line width=0.58pt] (80.6, 38.3) rectangle (98.6, 42.4);
\node[font=\fontsize{{5.3}}{{6.2}}\selectfont\bfseries, text=BlueText] at (89.6, 40.35) {{24b Seg Local ID}};

\filldraw[fill=BlueFill, draw=BlueStroke, line width=0.62pt] (70.2, 30.5) rectangle (85.2, 36.6);
\node[font=\fontsize{{5.15}}{{6.1}}\selectfont\bfseries, text=BlueText, align=center] at (77.7, 33.55) {{$\mathcal{{N}}_{{\mathrm{{pre}}}}$ ($M_{{\mathrm{{pre}}}}{{=}}16/32$)\\Causal Prefix}};
\filldraw[fill=BgNeutral, draw=SlateMid, line width=0.62pt] (85.2, 30.5) rectangle (98.6, 36.6);
\node[font=\fontsize{{5.15}}{{6.1}}\selectfont, text=SlateDark, align=center] at (91.9, 33.55) {{$\mathcal{{N}}_{{\mathrm{{fwd}}}}$ ($M_{{\mathrm{{fwd}}}}{{=}}12$)\\Tag Bridge}};
\node[font=\fontsize{{5.05}}{{6.0}}\selectfont, text=SlateMid] at (84.4, 27.55) {{Bounded 416K cap $\cdot$ Vitter lag reservoir}};

\drawsubcard{{44.2}}{{1.5}}{{55.8}}{{22.3}}{{64B-Aligned Cell-Local SQ8 + SoA Death Table (1,032 B/rec Core Arrays)}}
{lanes_tex}
\draw[flow] (78.68, 13.50) -- (83.60, 13.50);
\filldraw[fill=BgNeutral, draw=SlateDark, line width=0.65pt, rounded corners=0.85mm] (83.60, 9.40) rectangle (98.60, 17.60);
\node[font=\fontsize{{5.25}}{{6.3}}\selectfont\bfseries, text=SlateDark, align=center] at (91.10, 13.50) {{\texttt{{tzcnt}} Mask +\\7$\times$8b \texttt{{maddubs}}\\SQ8 Kernel}};
\node[font=\fontsize{{5.2}}{{6.1}}\selectfont, text=SlateMid] at (72.1, 5.25) {{Branchless \texttt{{\_mm256\_max\_epu32}} + \texttt{{cmpeq\_epi32}} (zero global SQ8 duplication)}};

% STAGE 3: Analytical Cost Router (x = 105.6 .. 139.2)
\drawstage{{105.6}}{{33.6}}{{3}}{{Analytical Cost Router}}

\drawsubcard{{107.0}}{{36.6}}{{30.8}}{{10.9}}{{Stratified SIMD Pilot}}
\node[font=\fontsize{{5.20}}{{6.2}}\selectfont, text=SlateDark, align=center] at (122.4, 40.1) {{6 AVX2 blocks/cell ($<0.4\,\mu$s)\\Compare $\mathcal{{C}}_{{\mathrm{{graph}}}}$ vs.\ $\mathcal{{C}}_{{\mathrm{{scan}}}}(N_{{\mathrm{{slots}}}})$}};

\filldraw[fill=BlueFill, draw=BlueStroke, line width=0.65pt, rounded corners=0.95mm] (107.0, 17.8) rectangle (121.8, 30.8);
\node[font=\fontsize{{5.3}}{{6.4}}\selectfont\bfseries, text=BlueText, align=center] at (114.4, 24.3) {{Causal Graph\\{{\normalfont\fontsize{{4.95}}{{5.95}}\selectfont ($\mathcal{{C}}_{{\mathrm{{graph}}}} < \mathcal{{C}}_{{\mathrm{{scan}}}}$)}} \\[0.5pt]Concave $\sqrt{{s_m}}$\\Multi-Seg Beam}};

\filldraw[fill=white, draw=SlateDark, line width=0.65pt, rounded corners=0.95mm] (123.0, 17.8) rectangle (137.8, 30.8);
\node[font=\fontsize{{5.3}}{{6.4}}\selectfont\bfseries, text=SlateDark, align=center] at (130.4, 24.3) {{AVX2 SQ8 Scan\\{{\normalfont\fontsize{{4.95}}{{5.95}}\selectfont ($\mathcal{{C}}_{{\mathrm{{scan}}}} \le \mathcal{{C}}_{{\mathrm{{graph}}}}$)}} \\[0.5pt]Cell-Pruned\\Contiguous}};

\drawsubcard{{107.0}}{{1.5}}{{30.8}}{{10.9}}{{Exact FP32 Reranker}}
\node[font=\fontsize{{5.20}}{{6.25}}\selectfont\bfseries, text=BlueText, align=center] at (122.4, 5.0) {{4-way FMA (\texttt{{mmap}} FP32, top-$R$)\\$\rightarrow$ Snapshot-Isolated Top-$k$}};

\draw[SlateDark, line width=1.05pt, line cap=butt] (122.40, 36.60) -- (122.40, 33.80);
\draw[forkflow] (114.40, 30.80) -- (114.40, 33.80) -- (130.40, 33.80) -- (130.40, 30.80);

\draw[SlateDark, line width=1.05pt, line join=miter, line cap=butt]
  (114.40, 17.80) -- (114.40, 15.20) -- (130.40, 15.20) -- (130.40, 17.80);
\draw[flow] (122.40, 15.20) -- (122.40, 12.40);

% STAGE 4: Provenance MVCC & Compaction (x = 143.4 .. 182.0)
\drawstage{{143.4}}{{38.6}}{{4}}{{Provenance MVCC \& Compaction}}

\drawsubcard{{144.8}}{{25.2}}{{35.8}}{{22.3}}{{PhaseFair MVCC + Adaptive $k_f$}}
\filldraw[fill=RedFill, draw=RedStroke, line width=0.60pt, rounded corners=0.75mm] (145.9, 33.6) rectangle (154.9, 41.8);
\node[font=\fontsize{{4.9}}{{5.9}}\selectfont\bfseries, text=RedText, align=center] at (150.4, 37.7) {{Tainted $r$\\Adaptive $2^r k_f$}};
\draw[flow] (154.9, 37.7) -- (158.2, 37.7);

\filldraw[fill=BgNeutral, draw=SlateMid, line width=0.60pt, rounded corners=0.75mm] (158.2, 33.6) rectangle (167.2, 41.8);
\node[font=\fontsize{{4.9}}{{5.9}}\selectfont\bfseries, text=SlateDark, align=center] at (162.7, 37.7) {{Unpack +\\FP32 Top-$k$}};
\draw[flow] (167.2, 37.7) -- (170.5, 37.7);

\filldraw[fill=BlueFill, draw=BlueStroke, line width=0.60pt, rounded corners=0.75mm] (170.5, 33.6) rectangle (179.5, 41.8);
\node[font=\fontsize{{4.9}}{{5.9}}\selectfont\bfseries, text=BlueText, align=center] at (175.0, 37.7) {{Atomic CAS\\Hot ($\ge 3$) $r'$}};

\node[font=\fontsize{{5.1}}{{6.15}}\selectfont, text=SlateDark, align=center] at (162.7, 28.9) {{Frontier-clean $|O|\le k$ $\cdot$ {tok_save_avg:.1f}\% est.\ token savings\\0.00\% LCR \& 0.0\% Stale Taint}};

\drawsubcard{{144.8}}{{1.5}}{{35.8}}{{22.3}}{{Unified Tiered RAP-Compact}}
\filldraw[fill=FutureFill, draw=SlateDark, line width=0.58pt] (145.9, 9.8) rectangle (151.1, 18.0);
\node[font=\fontsize{{4.95}}{{5.95}}\selectfont\bfseries, text=SlateDark, align=center] at (148.5, 13.9) {{32B\\Hdr}};
\filldraw[fill=RedFill, draw=RedStroke, line width=0.58pt] (151.1, 9.8) rectangle (162.7, 18.0);
\node[font=\fontsize{{4.95}}{{5.95}}\selectfont\bfseries, text=RedText, align=center] at (156.9, 13.9) {{56B Slots\\(IEEE-754)}};
\filldraw[fill=BgNeutral, draw=SlateMid, line width=0.58pt] (162.7, 9.8) rectangle (168.3, 18.0);
\node[font=\fontsize{{4.95}}{{5.95}}\selectfont\bfseries, text=SlateDark, align=center] at (165.5, 13.9) {{Dict\\Pool}};
\filldraw[fill=BlueFill, draw=BlueStroke, line width=0.58pt] (168.3, 9.8) rectangle (179.5, 18.0);
\node[font=\fontsize{{4.95}}{{5.95}}\selectfont\bfseries, text=BlueText, align=center] at (173.9, 13.9) {{Unit SQ8\\+ Entity Idx}};

\node[font=\fontsize{{5.1}}{{6.15}}\selectfont, text=SlateDark, align=center] at (162.7, 5.25) {{Unified Hot+Cold $[0,4]$ range merge ($<25\,\mu$s)\\Up to {max_comp:.1f}$\times$ compression (lossless slots)}};

\foreach \xa/\xb in {{38.6/42.8, 101.4/105.6, 139.2/143.4}} {{
  \draw[stageflow] (\xa, 36.35) -- (\xb, 36.35);
  \draw[stageflow] (\xa, 12.65) -- (\xb, 12.65);
}}

\end{{tikzpicture}}
\end{{document}}
"""


def plot_fig2_architecture(results_dir: str, fig_dir: str, preview_dir: str | None = None) -> None:
    df3 = pd.read_csv(os.path.join(results_dir, "exp3_restate_ingestion.csv"))
    df5 = pd.read_csv(os.path.join(results_dir, "exp5_rap_compact_ablation.csv"))
    tok_save_avg = float(df3[df3["system"] == "PiT-Mem (Lazy-Branch MVCC)"]["token_savings_pct"].mean())
    max_comp = float(df5[df5["method"] == "RAP-Compact (Ours)"]["compression_ratio"].max())

    os.makedirs(fig_dir, exist_ok=True)
    pdf_out = os.path.join(fig_dir, "fig2_architecture.pdf")
    pdflatex_bin = shutil.which("pdflatex")
    if pdflatex_bin is None:
        raise RuntimeError("pdflatex is required to compile native TikZ fig2_architecture.pdf")

    tex_src = _build_fig2_tikz_source(tok_save_avg, max_comp)
    with tempfile.TemporaryDirectory(prefix="pitmem_fig2_") as tmpd:
        tex_path = os.path.join(tmpd, "fig2_architecture.tex")
        with open(tex_path, "w", encoding="utf-8") as fh:
            fh.write(tex_src)
        proc = subprocess.run(
            [pdflatex_bin, "-interaction=nonstopmode", "-halt-on-error", "fig2_architecture.tex"],
            cwd=tmpd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        built_pdf = os.path.join(tmpd, "fig2_architecture.pdf")
        if proc.returncode != 0 or not os.path.exists(built_pdf):
            raise RuntimeError(f"TikZ compilation of fig2_architecture failed:\n{proc.stdout[-2500:]}")
        shutil.copy2(built_pdf, pdf_out)

    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
        try:
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(pdf_out)
            bmp = doc[0].render(scale=4.5)
            bmp.to_pil().save(os.path.join(preview_dir, "fig2_architecture.png"))
            doc.close()
        except Exception:
            pass


# ==============================================================================
# FIGURE 3: Exp 1 Multi-Regime & Multi-Scale Recall-Throughput Pareto Frontier
#   Row 1: Workload 1A (`moderate_12pct`, 100% Causal Graph Regime)
#   Row 2: Workload 1B (`sparse_1pct`, Sparse Historical Audit Regime)
# ==============================================================================
def plot_fig3_exp1_scalability(results_dir: str, fig_dir: str, preview_dir: str | None = None) -> None:
    df = pd.read_csv(os.path.join(results_dir, "exp1_scalability_qps_recall.csv"))
    if "regime" not in df.columns:
        df["regime"] = "sparse_1pct"
    regimes = [r for r in ["moderate_12pct", "sparse_1pct"] if r in df["regime"].unique()]
    scales = sorted(df["scale"].unique())
    nrows, ncols = len(regimes), len(scales)

    fig, axes = plt.subplots(nrows, ncols, figsize=(10.4, 4.65 if nrows == 2 else 2.75), sharey=False, squeeze=False)
    fig.subplots_adjust(wspace=0.23, hspace=0.36, top=0.86 if nrows == 2 else 0.80)

    sys_order = [
        "PiT-Mem", "Grid-SQ8-Scan", "ACORN-BiTemp", "PG-BiTemp",
        "SeRF-Adapt", "DSG-1D", "Zep/Graphiti", "HNSW-Post",
    ]
    disp_e1 = {
        "PiT-Mem": "PiT-Mem",
        "Grid-SQ8-Scan": "Grid-SQ8-Scan",
        "ACORN-BiTemp": "ACORN-Style (C++)",
        "PG-BiTemp": "PG-BiTemp",
        "SeRF-Adapt": "SeRF-Style 1D",
        "DSG-1D": "DSG-Style 1D",
        "Zep/Graphiti": "Graphiti-Style",
        "HNSW-Post": "HNSW-Flat (C++)",
    }
    legend_handles = {}

    panel_idx = 0
    for r_i, reg in enumerate(regimes):
        reg_tag = "W1A Graph" if reg == "moderate_12pct" else "W1B Sparse"
        for c_i, sc in enumerate(scales):
            ax = axes[r_i][c_i]
            sub = df[(df["regime"] == reg) & (df["scale"] == sc)]
            if len(sub) == 0:
                continue
            sig_val = float(sub["selectivity"].iloc[0]) * 100.0

            ax.axvspan(0.90, 1.03, color="#D5F5E3", alpha=0.42, zorder=0)
            ax.axvline(0.90, color="#27AE60", linestyle="--", linewidth=0.72, alpha=0.8, zorder=1)

            for sys_name in sys_order:
                grp = sub[sub["system"] == sys_name]
                if len(grp) == 0:
                    continue
                grp_s = grp.sort_values(["recall_at_10", "qps"])
                is_ours = (sys_name == "PiT-Mem")
                lbl_s = disp_e1.get(sys_name, sys_name)
                (ln,) = ax.plot(
                    grp_s["recall_at_10"],
                    grp_s["qps"],
                    marker=MARKERS.get(sys_name, "o"),
                    linestyle=LINESTYLES.get(sys_name, "-"),
                    color=PALETTE.get(sys_name, "#333333"),
                    linewidth=2.15 if is_ours else 1.35,
                    markersize=5.2 if is_ours else 4.2,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    label=lbl_s,
                    zorder=6 if is_ours else 4,
                )
                if lbl_s not in legend_handles:
                    legend_handles[lbl_s] = ln

            pit_sub = sub[sub["system"] == "PiT-Mem"]
            if reg == "moderate_12pct":
                cmp_sub = sub[sub["system"] == "Grid-SQ8-Scan"]
                cmp_lbl = "vs. Scan"
            else:
                # Only annotate PG-BiTemp speedup at N >= 500K (avoid magnifying small-scale 100K client/server ratios)
                cmp_sub = sub[sub["system"] == "PG-BiTemp"] if sc >= 500_000 else pd.DataFrame()
                cmp_lbl = "vs. PG"

            if len(cmp_sub) > 0 and len(pit_sub) > 0:
                if reg == "moderate_12pct" and len(cmp_sub[cmp_sub["param_ef"] == 64]) > 0:
                    cmp_best = cmp_sub[cmp_sub["param_ef"] == 64].iloc[0]
                else:
                    cmp_best = cmp_sub.sort_values(["recall_at_10", "qps"], ascending=[False, False]).iloc[0]
                if reg != "moderate_12pct" and len(pit_sub[pit_sub["param_ef"] == 96]) > 0:
                    pit_pick = pit_sub[pit_sub["param_ef"] == 96].iloc[0]
                else:
                    sla_thr = 0.92 if reg == "moderate_12pct" else 0.95
                    pit_best = pit_sub[pit_sub["recall_at_10"] >= sla_thr]
                    pit_pick = pit_best.sort_values("qps").iloc[-1] if len(pit_best) > 0 else pit_sub.iloc[-1]
                sp = float(pit_pick["qps"]) / max(1e-6, float(cmp_best["qps"]))
                sp_fmt = f"{sp:.0f}×" if sp >= 100.0 else f"{sp:.1f}×"
                if sp >= 1.25:
                    ax.text(
                        0.04, 0.25,
                        f"{sp_fmt} QPS {cmp_lbl}",
                        transform=ax.transAxes,
                        fontsize=6.6, fontweight="bold", color="#C0392B",
                        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#C0392B", lw=0.6, alpha=0.92),
                        zorder=7,
                    )

            ax.set_yscale("log")
            if r_i == nrows - 1:
                ax.set_xlabel("Recall@10")
            if c_i == 0:
                ax.set_ylabel("Throughput (QPS, log)")
            panel_chr = chr(ord("a") + panel_idx)
            panel_idx += 1
            sc_lbl = f"{sc // 1_000_000}M" if sc >= 1_000_000 else f"{sc // 1_000}K"
            ax.set_title(f"({panel_chr}) {reg_tag}: N={sc_lbl} ($\\sigma={sig_val:.1f}\\%$)", fontsize=8.8)
            ax.set_xlim(max(0.05, float(sub["recall_at_10"].min()) - 0.05), 1.03)
            _add_better_badge(ax, "Better ↗", "lower left")

    fig.legend(
        list(legend_handles.values()), list(legend_handles.keys()),
        loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=min(8, len(legend_handles)), frameon=True, facecolor="white", edgecolor="#CCCCCC",
        fontsize=7.6, columnspacing=1.05, handletextpad=0.45,
    )
    _save_both(fig, fig_dir, "fig3_exp1_scalability", preview_dir)
    plt.close(fig)


# ==============================================================================
# FIGURE 4: Exp 2 Selectivity Sweep & Crossover Regimes
# ==============================================================================
def plot_fig4_exp2_selectivity(results_dir: str, fig_dir: str, preview_dir: str | None = None) -> None:
    df = pd.read_csv(os.path.join(results_dir, "exp2_selectivity_sweep.csv"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.25, 2.95))
    fig.subplots_adjust(wspace=0.29, top=0.74)

    scan_key = "Grid-SQ8-Scan" if "Grid-SQ8-Scan" in df["system"].values else "PiT-Mem-ScanOnly"
    order_sys = [
        "PiT-Mem", scan_key, "PiT-Mem-GraphOnly",
        "ACORN-BiTemp", "PG-BiTemp", "HNSW-Post (ef=1024)", "SeRF-Adapt", "DSG-1D",
    ]
    short_lbl = {
        "PiT-Mem": "PiT-Mem (Hybrid)",
        scan_key: "Grid-SQ8-Scan",
        "PiT-Mem-GraphOnly": "PiT-Mem (GraphOnly)",
        "ACORN-BiTemp": "ACORN-Style (C++)",
        "PG-BiTemp": "PG-BiTemp",
        "HNSW-Post (ef=1024)": "HNSW-Flat (C++)",
        "SeRF-Adapt": "SeRF-Style 1D",
        "DSG-1D": "DSG-Style 1D",
    }

    pit_df_e2 = df[df["system"] == "PiT-Mem"].sort_values("actual_sigma", ascending=False)
    g_sigs = pit_df_e2[pit_df_e2["graph_query_pct"] >= 50.0]["actual_sigma"] * 100.0
    s_sigs = pit_df_e2[pit_df_e2["graph_query_pct"] < 50.0]["actual_sigma"] * 100.0
    if len(g_sigs) > 0 and len(s_sigs) > 0:
        x_cross = float(np.sqrt(float(g_sigs.min()) * float(s_sigs.max())))
    else:
        x_cross = 2.2

    for ax in (ax1, ax2):
        ax.axvspan(55.0, x_cross, color="#EBF5FB", alpha=0.40, zorder=0)
        ax.axvspan(x_cross, 0.035, color="#FEF9E7", alpha=0.48, zorder=0)
        ax.axvline(x_cross, color="#5D6D7E", linestyle="--", linewidth=0.85, zorder=2)

    handles, labels = [], []
    for s in order_sys:
        grp = df[df["system"] == s].sort_values("actual_sigma", ascending=False)
        if len(grp) == 0:
            continue
        is_ours = (s == "PiT-Mem")
        lw = 2.25 if is_ours else (1.65 if s in (scan_key, "PiT-Mem-GraphOnly") else 1.25)
        ls = LINESTYLES.get(s, "-")
        mk = MARKERS.get(s, "o")
        col = PALETTE.get(s, "#444")
        sig_pct = grp["actual_sigma"] * 100.0

        (l1,) = ax1.plot(
            sig_pct, grp["p99_lat_ms"], linestyle=ls, linewidth=lw, marker=mk,
            markersize=4.8 if is_ours else 3.8, markeredgecolor="white", markeredgewidth=0.5,
            color=col, zorder=6 if is_ours else 4,
        )
        ax2.plot(
            sig_pct, grp["recall_at_10"] * 100.0, linestyle=ls, linewidth=lw, marker=mk,
            markersize=4.8 if is_ours else 3.8, markeredgecolor="white", markeredgewidth=0.5,
            color=col, zorder=6 if is_ours else 4,
        )
        handles.append(l1)
        labels.append(short_lbl.get(s, s))

    max_lat = max(60.0, float(df["p99_lat_ms"].max()) * 3.8)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(55.0, 0.035)
    ax1.set_ylim(0.08, max_lat)
    ax1.set_xlabel("Snapshot Selectivity $\\sigma$ (%, log)")
    ax1.set_ylabel("P99 Tail Latency (ms, log)")
    ax1.set_title("(a) P99 Latency & Router Crossover")
    ax1.text(0.22, 0.94, "Graph Regime", transform=ax1.transAxes, fontsize=6.2, color="#1F618D",
             ha="center", va="top", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#AED6F1", lw=0.5, alpha=0.94), zorder=7)
    ax1.text(0.73, 0.94, "Scan Regime", transform=ax1.transAxes, fontsize=6.2, color="#7D6608",
             ha="center", va="top", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#F9E79F", lw=0.5, alpha=0.94), zorder=7)
    _add_better_badge(ax1, "Better ↓", "lower left")

    ax2.set_xscale("log")
    ax2.set_xlim(55.0, 0.035)
    ax2.set_ylim(-4, 108)
    ax2.set_xlabel("Snapshot Selectivity $\\sigma$ (%, log)")
    ax2.set_ylabel("Recall@10 (%)")
    ax2.set_title("(b) Recall Robustness & Collapse")
    hp_grp = df[df["system"] == "HNSW-Post (ef=1024)"].sort_values("actual_sigma", ascending=False)
    if len(hp_grp) > 0:
        hp_broad_rec = float(hp_grp["recall_at_10"].iloc[0]) * 100.0
        hp_narrow_rec = float(hp_grp["recall_at_10"].iloc[-1]) * 100.0
        hp_narrow_sig = float(hp_grp["actual_sigma"].iloc[-1]) * 100.0
        ax2.annotate(
            f"HNSW-Flat Collapse\n({hp_broad_rec:.0f}%→{hp_narrow_rec:.0f}%)",
            xy=(hp_narrow_sig, hp_narrow_rec), xytext=(0.75, 14.0),
            fontsize=6.3, fontweight="bold", color="#3B3B3B",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#666", lw=0.55),
            arrowprops=_callout_arrow("#3B3B3B", lw=0.90, shrinkA=0.0, shrinkB=3.0), zorder=7,
        )
    _add_better_badge(ax2, "Better ↑", "lower left")

    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=4, frameon=True, facecolor="white", edgecolor="#CCCCCC",
        fontsize=6.9, columnspacing=0.85, handletextpad=0.4,
    )
    _save_both(fig, fig_dir, "fig4_exp2_selectivity", preview_dir)
    plt.close(fig)


# ==============================================================================
# FIGURE 5: Exp 3 Restatement Write Speed, End-to-End Write+Read Tokens & U-Curve
# ==============================================================================
def plot_fig5_exp3_restate(results_dir: str, fig_dir: str, preview_dir: str | None = None) -> None:
    df = pd.read_csv(os.path.join(results_dir, "exp3_restate_ingestion.csv"))
    th_path = os.path.join(results_dir, "exp3_theta_ablation.csv")
    has_th = os.path.exists(th_path)

    if has_th:
        df_th = pd.read_csv(th_path)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.3, 2.70))
        fig.subplots_adjust(wspace=0.29)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.25, 2.75))
        fig.subplots_adjust(wspace=0.31)
        ax3 = None

    # Panel (a): Write Throughput (Measured C++ Attempts vs. Calibrated Rebuild / Eager Models)
    sys_map = [
        ("PiT-Mem (Lazy-Branch MVCC)", "PiT-Mem (Measured C++)", "o", "-"),
        ("SeRF-Static-Rebuild", "Static Rebuild (Est.)", "v", "--"),
        ("Eager-LLM-Consolidate", "Eager-LLM (Modeled)", "s", ":"),
    ]
    for raw_s, disp_s, mk, ls in sys_map:
        g = df[df["system"] == raw_s].sort_values("restate_ratio")
        if len(g) == 0:
            continue
        is_ours = ("PiT-Mem" in raw_s)
        ax1.plot(
            g["restate_ratio"] * 100.0, g["ops_per_sec"],
            marker=mk, linestyle=ls, linewidth=2.2 if is_ours else 1.45,
            markersize=5.2 if is_ours else 4.4, markeredgecolor="white", markeredgewidth=0.6,
            color=PALETTE.get(raw_s, "#333"), label=disp_s, zorder=6 if is_ours else 4,
        )

    ax1.set_yscale("log")
    ax1.set_ylim(max(0.2, float(df["ops_per_sec"].min()) * 0.45), float(df["ops_per_sec"].max()) * 14.0)
    ax1.set_xlabel("RESTATE Intensity (% of Writes)")
    ax1.set_ylabel("Write Rate (Ops/s, log)")
    ax1.set_title("(a) Write Rate (C++ vs. Est.)")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CCC", fontsize=5.8)
    _add_better_badge(ax1, "Better ↑", "upper left")

    # Panel (b): Calibrated Accounting Model Token Cost across Restatement Ratios
    ratios = sorted(df["restate_ratio"].unique())
    x = np.arange(len(ratios))
    w = 0.26
    e_toks = [float(df[(df["restate_ratio"] == r) & (df["system"] == "Eager-LLM-Consolidate")]["llm_tokens_used"].iloc[0]) / 1e3 for r in ratios]
    u_sub = df[df["system"] == "Unpack-Only (No Repair)"]
    u_toks = [float(u_sub[u_sub["restate_ratio"] == r]["llm_tokens_used"].iloc[0]) / 1e3 for r in ratios] if len(u_sub) > 0 else e_toks
    l_toks = [float(df[(df["restate_ratio"] == r) & (df["system"] == "PiT-Mem (Lazy-Branch MVCC)")]["llm_tokens_used"].iloc[0]) / 1e3 for r in ratios]
    savs = [float(df[(df["restate_ratio"] == r) & (df["system"] == "PiT-Mem (Lazy-Branch MVCC)")]["token_savings_pct"].iloc[0]) for r in ratios]

    ax2.bar(x - w, e_toks, width=w, color="#1F4E79", edgecolor="#1A1A1A", lw=0.65, hatch="///", label="Eager (Model)", zorder=4)
    ax2.bar(x, u_toks, width=w, color="#D95F02", edgecolor="#1A1A1A", lw=0.65, hatch="...", label="Unpack-Only", zorder=4)
    b_l = ax2.bar(x + w, l_toks, width=w, color="#C0392B", edgecolor="#1A1A1A", lw=0.65, label="PiT-Mem (W+R)", zorder=5)

    for bar, sv in zip(b_l, savs):
        ax2.text(
            bar.get_x() + bar.get_width() * 0.85, bar.get_height() + max(e_toks) * 0.02,
            f"-{sv:.0f}%", ha="center", va="bottom", fontsize=5.4, fontweight="bold", color="#C0392B", zorder=7,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(r * 100)}%" for r in ratios])
    ax2.set_ylim(0, max(max(e_toks), max(u_toks)) * 1.28)
    ax2.set_xlabel("RESTATE Intensity (% of Writes)")
    ax2.set_ylabel("Modeled Tokens ($\\times 10^3$)")
    ax2.set_title("(b) Modeled Write+Read Tokens")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CCC", fontsize=6.0)
    _add_better_badge(ax2, "Better ↓", "upper right")

    # Panel (c): Hotness Threshold U-Curve Decomposition (Calibrated Token Accounting Model)
    if ax3 is not None:
        tx = np.arange(len(df_th))
        w_tok = df_th["write_consolidation_tokens"].to_numpy(dtype=float) / 1e3
        r_tok = df_th["extra_read_prompt_tokens"].to_numpy(dtype=float) / 1e3
        tot_tok = w_tok + r_tok
        ax3.bar(tx, w_tok, width=0.52, color="#1F4E79", edgecolor="#1A1A1A", lw=0.65, label="Write Consolidation", zorder=4)
        ax3.bar(tx, r_tok, width=0.52, bottom=w_tok, color="#E67E22", edgecolor="#1A1A1A", lw=0.65, hatch="///", label="Read Unpack Overhead", zorder=4)
        ax3.plot(tx, tot_tok, color="#C0392B", marker="o", linewidth=1.6, markersize=4.6, zorder=6)

        best_i = int(np.argmin(tot_tok))
        ax3.annotate(
            f"Model Min\n(-{float(df_th['total_savings_pct'].iloc[best_i]):.0f}%)",
            xy=(best_i, tot_tok[best_i]),
            xytext=(max(0.6, best_i - 0.6), max(tot_tok) * 0.68),
            fontsize=6.3, fontweight="bold", color="#C0392B",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#C0392B", lw=0.6),
            arrowprops=_callout_arrow("#C0392B", lw=0.95, shrinkA=0.0, shrinkB=3.0),
            zorder=7,
        )
        t_lbls = ["0\n(Eager)", "1", "3", "5", "10", "$\\infty$\n(Unpack)"]
        ax3.set_xticks(tx)
        ax3.set_xticklabels(t_lbls[:len(tx)], fontsize=7.0)
        ax3.set_ylim(0, max(tot_tok) * 1.25)
        ax3.set_xlabel("Hotness Threshold $\\theta_{\\mathrm{hot}}$")
        ax3.set_ylabel("Modeled Tokens ($\\times 10^3$)")
        ax3.set_title("(c) Modeled $\\theta_{\\mathrm{hot}}$ Curve (10%)")
        ax3.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CCC", fontsize=6.2)
        _add_better_badge(ax3, "Better ↓", "upper right")

    _save_both(fig, fig_dir, "fig5_exp3_restate", preview_dir)
    plt.close(fig)


# ==============================================================================
# FIGURE 6: Exp 4 Dual-Protocol Look-Ahead Contamination (LCR) & Grounded Audit F1
# ==============================================================================
def plot_fig6_exp4_leakage(results_dir: str, fig_dir: str, preview_dir: str | None = None) -> None:
    df = pd.read_csv(os.path.join(results_dir, "exp4_leakage_and_audit.csv"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.25, 2.95))
    fig.subplots_adjust(wspace=0.28, top=0.80)

    pa = df[df["protocol"] == "Protocol_A_Post2025"].reset_index(drop=True)
    pb = df[df["protocol"] == "Protocol_B_Counterfactual"].reset_index(drop=True)
    systems = list(pa["system"])
    disp_names = ["PiT-Mem\n(Full)", "PiT-Mem\n(NoLazy)", "Latest-Sum.\n(Graphiti)", "Overwrite\n(Mem0)", "Append\n(GraphRAG)"]
    x = np.arange(len(systems))
    w = 0.36

    b1 = ax1.bar(x - w / 2, pa["lcr_pct"], width=w, color="#C0392B", edgecolor="#1A1A1A", lw=0.7,
                 label="Protocol A (120 Post-2025 SEC Cases)", zorder=4)
    b2 = ax1.bar(x + w / 2, pb["lcr_pct"], width=w, color="#1F4E79", edgecolor="#1A1A1A", lw=0.7, hatch="///",
                 label="Protocol B (120 Counterfactual Cases)", zorder=4)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        txt = "0%" if h < 0.05 else f"{h:.0f}"
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, txt, ha="center", va="bottom",
                 fontsize=6.0, fontweight="bold" if h < 0.05 else "normal",
                 color="#C0392B" if h < 0.05 else "#222")

    ax1.set_xticks(x)
    ax1.set_xticklabels(disp_names[:len(x)], fontsize=6.1)
    ax1.set_ylabel("Look-Ahead Contamination (LCR, %)")
    ax1.set_ylim(0, 122)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_title("(a) Pre-Restatement Look-Ahead (LCR)")
    _add_better_badge(ax1, "Better ↓", "upper left")

    f1 = ax2.bar(x - w / 2, pa["audit_f1_pct"], width=w, color="#C0392B", edgecolor="#1A1A1A", lw=0.7, zorder=4)
    f2 = ax2.bar(x + w / 2, pb["audit_f1_pct"], width=w, color="#1F4E79", edgecolor="#1A1A1A", lw=0.7, hatch="///", zorder=4)

    for bar in list(f1) + list(f2):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}", ha="center", va="bottom", fontsize=6.0, color="#222")

    ax2.set_xticks(x)
    ax2.set_xticklabels(disp_names[:len(x)], fontsize=6.1)
    ax2.set_ylabel("Grounded Restatement Audit F1 (%)")
    ax2.set_ylim(0, 114)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_title("(b) Grounded Restatement Audit F1")
    _add_better_badge(ax2, "Better ↑", "upper right")

    fig.legend(
        [b1, b2],
        ["Protocol A (120 Post-2025 SEC XBRL Cases)", "Protocol B (120 Entity-Anonymized Cases)"],
        loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=2, frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=7.2,
    )
    _save_both(fig, fig_dir, "fig6_exp4_leakage", preview_dir)
    plt.close(fig)


# ==============================================================================
# FIGURE 7: Exp 5 RAP-Compact vs. Parquet-Zstd, SQLite-BTree & Lossy Baselines
# ==============================================================================
def plot_fig7_exp5_compact(results_dir: str, fig_dir: str, preview_dir: str | None = None) -> None:
    df = pd.read_csv(os.path.join(results_dir, "exp5_rap_compact_ablation.csv"))
    cov_df = df[df["dataset"] == "Protocol_B_Covenants"] if "dataset" in df.columns else df
    sec_df = df[df["dataset"] == "Wild_SEC_EDGAR_Filings"] if "dataset" in df.columns else pd.DataFrame()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.25, 2.95))
    fig.subplots_adjust(wspace=0.31, top=0.74)

    methods_order = [
        ("RAP-Compact (Ours)", "RAP-Compact (Res. C++, SQ8)", "-"),
        ("Parquet-Zstd + Medoid", "Parquet-Zstd (File Read, FP32)", "--"),
        ("SQLite-BTree + Medoid", "SQLite-BTree (In-Mem, FP32)", "-."),
        ("LLM-Abstractive-Summary", "LLM-Summary (Trunc. Prompt)", "--"),
        ("Centroid-MeanPool", "Centroid-MeanPool", "-."),
        ("Narrative-Truncation", "Narrative-Truncation", ":"),
    ]
    handles, labels = [], []

    for raw_m, disp_m, ls in methods_order:
        g = cov_df[cov_df["method"] == raw_m].sort_values("target_budget")
        if len(g) == 0:
            continue
        is_ours = ("RAP-Compact" in raw_m)
        lw = 2.15 if is_ours else 1.4
        mk = MARKERS.get(raw_m, "o")
        col = PALETTE.get(raw_m, "#333")
        y_vals = g["covenant_qa_acc_pct"].to_numpy(dtype=float)

        # Plot exact unshifted accuracy values; use concentric hollow/solid markers for overlapping 100% lossless curves
        if raw_m == "SQLite-BTree + Medoid":
            ms, mfc, mec, mew, zo = 6.8, "none", col, 1.25, 4
        elif raw_m == "Parquet-Zstd + Medoid":
            ms, mfc, mec, mew, zo = 5.6, "white", col, 1.15, 5
        elif is_ours:
            ms, mfc, mec, mew, zo = 4.3, col, "white", 0.55, 6
        else:
            ms, mfc, mec, mew, zo = 4.2, col, "white", 0.55, 4

        (l1,) = ax1.plot(
            g["target_budget"] * 100.0, y_vals,
            marker=mk, linestyle=ls, linewidth=lw, markersize=ms,
            markerfacecolor=mfc, markeredgecolor=mec, markeredgewidth=mew,
            color=col, zorder=zo,
        )
        handles.append(l1)
        labels.append(disp_m)

    ax1.axhspan(94.0, 106.0, color="#D5F5E3", alpha=0.40, zorder=0)
    ax1.set_xlabel("Target Compaction Budget $\\beta$ (%)")
    ax1.set_ylabel("Template Anchor Check (%)")
    ax1.set_xlim(12.5, 52.5)
    ax1.set_ylim(-4, 107)
    ax1.set_xticks([15, 25, 35, 50])
    ax1.set_title("(a) Template Slot Preservation")
    _add_better_badge(ax1, "Better ↖", "lower right")

    # Panel (b): Compression Ratio vs. Hybrid Point-in-Time Query Latency (us, log)
    lossless_methods = ["RAP-Compact (Ours)", "SQLite-BTree + Medoid", "Parquet-Zstd + Medoid"]
    for m_name in lossless_methods:
        g = cov_df[cov_df["method"] == m_name].sort_values("compression_ratio")
        if len(g) == 0:
            continue
        is_ours = ("RAP-Compact" in m_name)
        ax2.plot(
            g["compression_ratio"], g["query_lat_us"],
            marker=MARKERS.get(m_name, "o"), linestyle=LINESTYLES.get(m_name, "-"),
            linewidth=2.1 if is_ours else 1.4, markersize=5.5 if is_ours else 4.5,
            markeredgecolor="white", markeredgewidth=0.6, color=PALETTE.get(m_name, "#333"),
            zorder=6 if is_ours else 4,
        )

    if len(sec_df) > 0:
        sr = sec_df[sec_df["method"] == "RAP-Compact (Ours)"]
        if len(sr) > 0:
            sx = float(sr["compression_ratio"].iloc[0])
            sy = float(sr["query_lat_us"].iloc[0])
            ax2.scatter([sx], [sy], s=95, marker="*", color="#C0392B", edgecolors="#1A1A1A", linewidths=0.7, zorder=8)
            ax2.annotate(
                f"XBRL Templated\n({sx:.1f}×, {sy:.1f}$\\mu$s)",
                xy=(sx, sy), xytext=(max(4.0, sx * 0.35), sy * 6.5),
                fontsize=6.2, fontweight="bold", color="#922B21",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#C0392B", lw=0.6),
                arrowprops=_callout_arrow("#C0392B", lw=0.90, shrinkA=0.0, shrinkB=3.5),
                zorder=8,
            )

    rap25 = cov_df[(cov_df["method"] == "RAP-Compact (Ours)") & (cov_df["target_budget"] == 0.25)]
    sql25 = cov_df[(cov_df["method"] == "SQLite-BTree + Medoid") & (cov_df["target_budget"] == 0.25)]
    if len(rap25) > 0 and len(sql25) > 0:
        sp_sql = float(sql25["query_lat_us"].iloc[0]) / max(1e-3, float(rap25["query_lat_us"].iloc[0]))
        ax2.text(
            0.66, 0.56, f"{sp_sql:.1f}× vs. SQLite\n(In-Mem, $\\beta$=25%)",
            transform=ax2.transAxes, ha="center", fontsize=6.2, fontweight="bold", color="#C0392B",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#C0392B", lw=0.6, alpha=0.95),
            zorder=7,
        )

    ax2.set_yscale("log")
    ax2.set_xlabel("Compression Ratio ($\\times$ Raw Footprint)")
    ax2.set_ylabel("Query Latency ($\\mu$s, log)")
    ax2.set_ylim(3.0, max(20000.0, float(cov_df["query_lat_us"].max()) * 2.5))
    ax2.set_title("(b) Slot-Backed Store Latency")
    _add_better_badge(ax2, "Better ↘", "upper left")

    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=3, frameon=True, facecolor="white", edgecolor="#CCCCCC",
        fontsize=6.2, columnspacing=0.65, handletextpad=0.35,
    )
    _save_both(fig, fig_dir, "fig7_exp5_compact", preview_dir)
    plt.close(fig)


def generate_all_figures(results_dir: str, fig_dir: str, preview_dir: str | None = None) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    plot_fig1_motivation(results_dir, fig_dir, preview_dir)
    plot_fig2_architecture(results_dir, fig_dir, preview_dir)
    plot_fig3_exp1_scalability(results_dir, fig_dir, preview_dir)
    plot_fig4_exp2_selectivity(results_dir, fig_dir, preview_dir)
    plot_fig5_exp3_restate(results_dir, fig_dir, preview_dir)
    plot_fig6_exp4_leakage(results_dir, fig_dir, preview_dir)
    plot_fig7_exp5_compact(results_dir, fig_dir, preview_dir)
    print(f"[Plotting] Generated all 7 PVLDB Best-Paper style vector PDF figures -> {fig_dir}")


if __name__ == "__main__":
    import sys
    r_dir = sys.argv[1] if len(sys.argv) > 1 else "results/colab_pro"
    f_dir = sys.argv[2] if len(sys.argv) > 2 else "paper/figures"
    p_dir = sys.argv[3] if len(sys.argv) > 3 else None
    generate_all_figures(r_dir, f_dir, p_dir)
