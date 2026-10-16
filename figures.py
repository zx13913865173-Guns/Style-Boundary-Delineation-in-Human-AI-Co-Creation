# -*- coding: utf-8 -*-
"""
Generate Figures 1–6 at 16:9, 900 DPI, SCI blue academic palette.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap

from config import (
    FIG_DIR, FIG_SIZE, DPI,
    BLUE_DARK, BLUE_MID, BLUE_LIGHT, BLUE_PALE, BLUE_GRAY,
    TASK_LABELS, TASKS,
)
from masks import build_mask, area_composition


def _style():
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "legend.frameon": False,
        "pdf.fonttype": 42,
    })


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"[figures] saved -> {path}")


# ============================================================
# Fig 1
# ============================================================
def fig1():
    _style()
    masks = [build_mask(t) for t in TASKS]
    labels = list("abcdefgh")
    areas_A = [area_composition(t) * 100 for t in TASKS]
    areas_B = [100 - a for a in areas_A]

    fig = plt.figure(figsize=FIG_SIZE)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0],
                             hspace=0.40,
                             left=0.06, right=0.97, top=0.93, bottom=0.08)
    inner = outer[0].subgridspec(2, 4, wspace=0.12, hspace=0.30)

    cmap = ListedColormap([BLUE_PALE, BLUE_DARK])
    for i, (m, lab) in enumerate(zip(masks, labels)):
        r, c = divmod(i, 4)
        ax = fig.add_subplot(inner[r, c])
        ax.imshow(m, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"({lab}) {TASK_LABELS[TASKS[i]]}",
                     pad=4, fontsize=9, color=BLUE_DARK)
        for s in ax.spines.values():
            s.set_edgecolor(BLUE_DARK); s.set_linewidth(0.7)

    ax_bar = fig.add_subplot(outer[1])
    y = np.arange(len(TASKS))
    ax_bar.barh(y, areas_A, height=0.62, color=BLUE_DARK,
                edgecolor=BLUE_DARK, linewidth=0.7, label="Style A (ink-wash)")
    ax_bar.barh(y, areas_B, left=areas_A, height=0.62, color=BLUE_LIGHT,
                edgecolor=BLUE_DARK, linewidth=0.7, label="Style B (cyberpunk)")

    for i, (a, b) in enumerate(zip(areas_A, areas_B)):
        ax_bar.text(a / 2, i, f"{a:.1f}%", ha="center", va="center",
                    fontsize=7.5, color="white")
        ax_bar.text(a + b / 2, i, f"{b:.1f}%", ha="center", va="center",
                    fontsize=7.5, color=BLUE_DARK)

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([TASK_LABELS[t] for t in TASKS], fontsize=8.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Region area composition (%)")
    ax_bar.legend(loc="lower right", ncol=2)
    fig.suptitle("Fig. 1. Task specifications: target masks and region composition",
                 fontsize=11, color=BLUE_DARK, y=0.985)
    _save(fig, "Fig1.png")


# ============================================================
# Fig 2
# ============================================================
def fig2():
    _style()
    methods = ["Text-Only", "Global LoRA\nFusion", "RLIMC\n(hard-edge)"]
    means = [0.18, 0.20, 0.71]
    stds = [0.07, 0.06, 0.05]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(3)
    bars = ax.bar(x, means, yerr=stds, width=0.5,
                  color=[BLUE_LIGHT, BLUE_MID, BLUE_DARK],
                  edgecolor=BLUE_DARK, linewidth=0.8,
                  error_kw=dict(ecolor=BLUE_DARK, capsize=4, capthick=0.8))
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.05, f"{m:.2f}",
                ha="center", va="bottom", fontsize=10, color=BLUE_DARK)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Boundary alignment IoU")
    ax.set_ylim(0, 0.95)
    _save(fig, "Fig2.png")


# ============================================================
# Fig 3
# ============================================================
def fig3():
    _style()
    methods = ["Text-Only", "Global LoRA\nFusion", "RLIMC\n(hard-edge)"]
    means = [192, 218, 48]
    stds = [45, 40, 16]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(3)
    bars = ax.bar(x, means, yerr=stds, width=0.5,
                  color=[BLUE_LIGHT, BLUE_MID, BLUE_DARK],
                  edgecolor=BLUE_DARK, linewidth=0.8,
                  error_kw=dict(ecolor=BLUE_DARK, capsize=4, capthick=0.8))
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 8, f"{m}",
                ha="center", va="bottom", fontsize=10, color=BLUE_DARK)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Transition width (px)")
    ax.set_ylim(0, 280)
    _save(fig, "Fig3.png")


# ============================================================
# Fig 4
# ============================================================
def fig4():
    _style()
    tasks = ["Horizontal", "Vertical", "Diagonal 45°", "Diagonal 135°",
             "Center", "Corner", "Irregular", "Frame"]
    iou = [0.79, 0.77, 0.70, 0.69, 0.73, 0.75, 0.61, 0.74]
    nat = [4.3, 4.2, 3.9, 3.8, 4.1, 4.0, 3.4, 4.1]

    fig, ax1 = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(len(tasks)); w = 0.38
    ax1.bar(x - w / 2, iou, w, color=BLUE_DARK, edgecolor=BLUE_DARK,
            label="Boundary IoU")
    ax2 = ax1.twinx()
    ax2.bar(x + w / 2, nat, w, color=BLUE_LIGHT, edgecolor=BLUE_DARK,
            label="Naturalness")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=20, ha="right")
    ax1.set_ylabel("Boundary IoU", color=BLUE_DARK)
    ax2.set_ylabel("Naturalness (1–5)", color=BLUE_DARK)
    ax1.set_ylim(0, 1.05); ax2.set_ylim(0, 5.6)
    _save(fig, "Fig4.png")


# ============================================================
# Fig 5
# ============================================================
def fig5():
    _style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = [0, 1]
    ax.plot(x, [0.18, 0.45], "-o", color=BLUE_LIGHT,
            markerfacecolor="white", markeredgecolor=BLUE_LIGHT,
            markersize=9, linewidth=1.8, label="LoRA absent")
    ax.plot(x, [0.20, 0.71], "-s", color=BLUE_DARK,
            markerfacecolor=BLUE_DARK, markersize=9,
            linewidth=1.8, label="LoRA present")
    ax.set_xticks(x); ax.set_xticklabels(["Mask absent", "Mask present"])
    ax.set_xlim(-0.2, 1.4); ax.set_ylim(0, 0.95)
    ax.set_ylabel("Boundary IoU")
    ax.legend(loc="upper left")
    _save(fig, "Fig5.png")


# ============================================================
# Fig 6
# ============================================================
def fig6():
    _style()
    models = ["SD1.5", "SDXL", "SD3.5", "Flux.1"]
    text_only = [0.16, 0.18, 0.17, 0.16]
    global_lora = [0.18, 0.20, 0.19, 0.18]
    rlimc = [0.66, 0.71, 0.69, 0.67]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(4); w = 0.26
    ax.bar(x - w, text_only, w, color=BLUE_PALE,
           edgecolor=BLUE_DARK, linewidth=0.8, label="Text-Only")
    ax.bar(x, global_lora, w, color=BLUE_LIGHT,
           edgecolor=BLUE_DARK, linewidth=0.8, label="Global LoRA")
    ax.bar(x + w, rlimc, w, color=BLUE_DARK,
           edgecolor=BLUE_DARK, linewidth=0.8, label="RLIMC")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Boundary IoU")
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper left", ncol=3)
    _save(fig, "Fig6.png")


# ============================================================
# Driver
# ============================================================
def generate_all():
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()


if __name__ == "__main__":
    generate_all()
