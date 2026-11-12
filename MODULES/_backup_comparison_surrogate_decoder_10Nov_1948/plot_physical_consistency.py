#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/comparison_surrogate_decoder/plot_physical_consistency.py

Figure for R1.4 (from physical_consistency.py): physical laws that the eigenvalue solver satisfies by construction and the
surrogate decoder does not.
  (a) frequency of mode 2 vs the stiffness factor of zone 1 (other factors fixed), eigenvalue solver vs surrogate: the
      sweep with the most decreasing steps of the surrogate among 300 random states x 7 parameters;
  (b) share of cases in which a 5 % stiffer zone or bearing gets a LOWER frequency (impossible: Rayleigh's principle);
  (c) mean difference between the frequencies of mirror-image damage scenarios (identical in the symmetric bridge).

    python MODULES/comparison_surrogate_decoder/plot_physical_consistency.py
"""
import os
import sys

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.path.join(PROJECT_ROOT, "Output", "comparison_surrogate_decoder", "decoder_checks_06Nov-FINAL5modes_H10-30-50-70-80")
FIG = os.path.join(PROJECT_ROOT, "MODULES", "review29oct_fixed", "Latex_docs", "figures_final", "Fig_surrogate_physics")
EIGEN, SURR = "#2a78d6", "#eb6834"            # categorical slots 1 and 2 of the validated reference palette
INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#e2e1dc", "#ecebe7"
MODE = 2

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8, "axes.titlesize": 8,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6})


def style(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


LABELS = ["Eigenvalue solver", "Surrogate, 14,000 sim.", "Surrogate, 2,000 sim.", "Surrogate, 500 sim."]


def hbars(ax, values, title, fmt, show_labels):
    """Horizontal bars, one per decoder (eigenvalue solver first, at the top); value printed at the end of each bar."""
    y = np.arange(len(values))
    ax.barh(y, values, height=0.56, color=[EIGEN] + [SURR] * (len(values) - 1), edgecolor="white", linewidth=1.0)
    top = max(values)
    for yi, v in zip(y, values):
        ax.text(v + 0.03 * top, yi, "0" if abs(v) < 1e-9 else fmt.format(v), ha="left", va="center", color=INK,
                fontsize=7)
    ax.set_yticks(y, LABELS[:len(values)] if show_labels else [""] * len(values))
    ax.set_ylim(len(values) - 0.5, -0.5)                       # eigenvalue solver at the top
    ax.set_xlim(0, 1.3 * top)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", color=INK)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    pc = pd.read_csv(os.path.join(SRC, "physical_consistency.csv"))
    sw = pd.read_csv(os.path.join(SRC, "monotonicity_sweep.csv"))
    z, fe, fs = sw["z_d"].values, sw[f"f{MODE}_eigen"].values, sw[f"f{MODE}_surrogate"].values

    fig = plt.figure(figsize=(7.1, 2.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.25, 0.62, 0.95, 0.10, 0.95], wspace=0.0)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 4])]
    # (a) sweep: shade where the surrogate frequency DEcreases while the stiffness increases
    dec = np.diff(fs) < 0
    for i in np.where(dec)[0]:
        ax[0].axvspan(z[i], z[i + 1], color=SHADE, lw=0)
    ax[0].plot(z, fe, color=EIGEN, lw=2.0, solid_capstyle="round")
    ax[0].plot(z, fs, color=SURR, lw=2.0, ls=(0, (4, 2)))
    k = int(np.argmin(np.abs(z - 0.80)))
    ax[0].text(z[k], fe[k] - 0.07, "Eigenvalue solver", color=INK, va="top", ha="left", fontsize=7)
    ax[0].text(z[k] - 0.02, fs[k] + 0.05, "Surrogate", color=INK, va="bottom", ha="right", fontsize=7)
    ax[0].set_xlim(z[0], z[-1])
    ax[0].set_ylim(min(fe.min(), fs.min()) - 0.1, max(fe.max(), fs.max()) + 0.12)
    ax[0].set_xlabel(r"Stiffness factor of zone 1, $z_1$")
    ax[0].set_ylabel(fr"Frequency of mode {MODE} [Hz]")
    ax[0].set_title("(a) Frequency vs stiffness", loc="left", color=INK)
    ax[0].text(0.97, 0.04, "shaded: the surrogate frequency\ndecreases as the zone gets stiffer",
               transform=ax[0].transAxes, va="bottom", ha="right", color=INK2, fontsize=6.5)
    style(ax[0])
    ax[0].grid(True, axis="x", color=GRID, lw=0.6)
    # (b) monotonicity, (c) symmetry
    hbars(ax[1], pc["monotonicity_violations_pct"].tolist(), "(b) Stiffer, but lower\nfrequency [% of cases]",
          "{:.1f}%", True)
    hbars(ax[2], pc["mirror_freq_diff_mean_pct"].tolist(), "(c) Mirror-image scenarios:\nfrequency difference [%]",
          "{:.2f}%", False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.84)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}.{ext}", dpi=300)
        fig.savefig(os.path.join(SRC, f"Fig_surrogate_physics.{ext}"), dpi=300)
    print(f"[OK] {FIG}.pdf / .png")


if __name__ == "__main__":
    main()
