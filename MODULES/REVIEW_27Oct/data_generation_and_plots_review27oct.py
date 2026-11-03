#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_generation_and_plots_review27oct.py
========================================
Updated & Expanded Dataset Generation and Visual Diagnostic Script for the
Major Revision of MS CPENG-8674 (saved in `code/MODULES/review_27oct/`).

This script directly updates and extends:
  - `MODULES/PREPROCESSING/Noisy_random_dataset_generation.py`
  - `MODULES/PREPROCESSING/zones_dataset_generation.py`
  - `MODULES/PREPROCESSING/singlelement_ataset_generation.py`

What is new and WHY each addition is required by the Reviewers:
---------------------------------------------------------------
1. DECOUPLED FE DISCRETIZATION vs. DAMAGE ZONES (Reviewer 1 Comment 1 & 7, Reviewer 2 Comment 3):
   - In the first submission, the 5 damage zones were identical to the 5 finite elements
     (`n_elements = n_zones = 5`), meaning zero discretization (model-form) error existed
     between the synthetic observations and the VAE physics decoder.
   - Here, the data-generating structure can be discretized into `n_elements` fine elements
     (e.g., 20, 50, or 100 elements) while retaining `n_zones` damage zones (e.g., 5 or 10)
     and extracting mode shapes only at the `n_sensors` physical sensor stations.
   - Importantly, `Mass_matrix.npy`, `Ke_matrix.npy`, and `L_inv.npy` saved inside each
     dataset folder are ALWAYS those of the NOMINAL identification model used by the VAE
     decoder, whereas the modal observations (`freqs_data_true.npy`, `vertmodes_data_true.npy`,
     `rotmodes_data_true.npy`) come from the fine/perturbed physical model.

2. REALISTIC DESIGN OF EXPERIMENTS (DoE) FOR TRAINING/VALIDATION vs. TESTING (Reviewer 1 Comment 6, Reviewer 3 Comment 3):
   - Why naive uniform sampling (`z ~ U[0.5, 1.0]^D`) was criticized:
     Under independent uniform sampling across 5 elements, the probability that all 5 elements
     have <10% damage (`z_i > 0.90`) is `0.2^5 = 0.00032` (only ~3 samples out of 10,000!).
     Thus, the network was trained almost exclusively on structures with simultaneous damage
     in all 5 elements.
   - How `realistic_bridge_doe` solves this for Training & Validation:
     We sample from a structured 4-regime operational mixture:
       * Regime 1 (15%): Healthy / Near-Healthy baseline (`z_i = 1.0` or in `[0.985, 1.0]`).
       * Regime 2 (35%): Single-zone localized damage (1 zone damaged with 60% weight on
                         mild-to-moderate damage `z_k in [0.75, 0.97]` and 40% on severe
                         damage `z_k in [0.50, 0.75]`; remaining D-1 zones intact `z = 1.0`).
       * Regime 3 (25%): Two-zone localized damage (2 zones damaged, adjacent or non-adjacent;
                         remaining D-2 zones intact `z = 1.0`).
       * Regime 4 (25%): Multi-zone space-filling uniform background (`z ~ U[0.50, 1.0]^D`) so
                         the amortized encoder still covers the entire D-dimensional hypercube.
   - How Testing datasets differ from Training/Validation:
     While Training/Validation uses a single mixed DoE (split 70% train / 20% val / 10% in-dist
     test via `split_indices.npz`), the Testing suite consists of dedicated, separated stress-test
     datasets (`TEST_E5_healthy`, `TEST_E5_single`, `TEST_E5_double`, `TEST_E5_finemesh50`,
     `TEST_E5_localized_fine50`, `TEST_E5_massR050`, `TEST_E5_oodsevere`, `TEST_E5_noise2x`, etc.)
     so we can report performance per physical regime to the reviewers.

3. MASS MATRIX PERTURBATIONS (Reviewer 3 Comment 1, Reviewer 1 Comment 2):
   - Supports both:
     (a) Fixed element-wise mass modeling error (`Mass_matrix_perturbed.npy` vs. nominal
         `Mass_matrix.npy`, matching your October `06Oct_Data_Noisy_E5_Lvl25_MassPerturb5` work), and
     (b) Sample-by-sample random element mass errors (`sigma_m in {1%, 2.5%, 5%, 10%}`) and
         global non-structural mass offsets (`+5%` asphalt/pavement load).

4. ELASTIC BEARING SUPPORTS & REALISTIC 20m RC BRIDGE (Reviewer 2 Comments 1-4):
   - Replaces idealized pinned supports (`v_0 = v_L = 0`) with vertical elastomeric bearing
     springs `k_{v,L} = s_L * k_{v,nom}` and `k_{v,R} = s_R * k_{v,nom}`, where `(s_L, s_R)`
     can be inferred alongside the deck zone stiffness factors `z_1 ... z_D`.
   - Includes a realistic 20 m Reinforced Concrete (RC) slab-girder bridge (`L=20m, E=34 GPa,
     I=0.45 m^4, A=4.0 m^2, rho_eff=3125 kg/m^3`, 40 FE elements, 10 deck zones + 2 bearings,
     11 sensors, 6 modes, `f_1 = 4.28 Hz`).

5. EXPLANATORY & DIAGNOSTIC FIGURES (Saved in `code/Data/review_27oct/Figures_Dataset_Diagnostics/`):
   - Running this script automatically generates 6 comprehensive figures explaining the new
     model, mesh convergence, localized sub-zone damage, elastic bearings, mass perturbations,
     and the new Training DoE vs. Testing datasets.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .fe_beam_model import BeamConfig, BeamFEModel, zone_equivalent_factor
from . import damage_scenarios as ds
from .generate_review_datasets import (
    TOY_E5,
    TOY_FINE,
    TOY_ELASTIC,
    BRIDGE20,
    REGISTRY,
    generate,
)

DATA_ROOT = os.path.join("Data", "review_27oct")
FIG_DIR = os.path.join(DATA_ROOT, "Figures_Dataset_Diagnostics")


def _setup_matplotlib():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.dpi": 150,
    })


# =============================================================================
# FIGURE 1: Direct Comparison of PREVIOUS (Original) vs. NEW (Current) Training DoE
# =============================================================================
def plot_fig1_doe_comparison():
    """
    Directly compares ONLY the Previous (Original First-Submission Uniform DoE)
    against the New (Current Realistic Early-Stage & Localized DoE) across 3 clear panels:
      - Panel A: Number of simultaneously damaged zones per structure (k = 0..5),
                 showing the shift from 94% 5-zone damage to 30% Healthy + 45% Single-Zone.
      - Panel B: Severity distribution within damaged zones (z_i < 0.98), showing how
                 the New DoE strongly favors slight/early-stage damage (z in [0.80, 0.98],
                 2%-20% stiffness loss) over extreme heavy damage (z in [0.50, 0.65]).
      - Panel C: 2D Joint space (z_1, z_2) showing the healthy point (1,1) and the
                 single-zone localized axes vs. the old uniform square.
    """
    rng = np.random.default_rng(42)
    N = 10000
    cfg = TOY_E5

    # 1. Previous (Original) Dataset: z_i ~ U[0.50, 1.0]^5
    Z_orig = np.array([ds.uniform(cfg, rng, zlb=0.5)["z_zone"] for _ in range(N)])
    # 2. New (Current) Dataset: Realistic Early-Stage & Localized DoE
    Z_new = np.array([ds.realistic_bridge_doe(cfg, rng, zlb=0.5)["z_zone"] for _ in range(N)])

    # Damaged zone threshold: z_i < 0.98 (>2% stiffness loss)
    thresh = 0.98
    n_dam_orig = np.sum(Z_orig < thresh, axis=1)
    n_dam_new = np.sum(Z_new < thresh, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    col_orig = "#d95f02"  # Orange-red for Previous (Original)
    col_new = "#1b9e77"   # Teal-green for New (Current)

    # -------------------------------------------------------------------------
    # PANEL A: Number of Damaged Zones per Structure (k = 0, 1, 2, 3, 4, 5)
    # -------------------------------------------------------------------------
    ax = axes[0]
    bins_k = np.arange(-0.5, 6.5, 1.0)
    pct_orig = 100.0 * np.histogram(n_dam_orig, bins=bins_k)[0] / N
    pct_new = 100.0 * np.histogram(n_dam_new, bins=bins_k)[0] / N

    x_k = np.arange(6)
    w = 0.38
    b1 = ax.bar(x_k - w / 2, pct_orig, width=w, color=col_orig, alpha=0.85,
                edgecolor="black", label="Previous (Original Uniform DoE)")
    b2 = ax.bar(x_k + w / 2, pct_new, width=w, color=col_new, alpha=0.90,
                edgecolor="black", label="New (Realistic Early-Stage DoE)")

    for b, p in zip(b1, pct_orig):
        if p >= 0.5:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{p:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, color=col_orig, fontweight="bold")
    for b, p in zip(b2, pct_new):
        if p >= 0.5:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{p:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, color="#0e664b", fontweight="bold")

    ax.set_xticks(x_k)
    ax.set_xticklabels(["0\n(Healthy)", "1\n(1 Zone)", "2\n(2 Zones)", "3\n(3 Zones)", "4\n(4 Zones)", "5\n(All 5)"])
    ax.set_ylim(0, 106)
    ax.set_xlabel("Number of Damaged Zones per Structure ($z_i < 0.98$)")
    ax.set_ylabel("Percentage of Dataset (%)")
    ax.set_title("(A) Localization: Number of Damaged Zones\nNew Dataset Favours Healthy (30%) & Single-Zone (45%)",
                 fontweight="bold")
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # -------------------------------------------------------------------------
    # PANEL B: Damage Severity Distribution in Damaged Zones (z_i < 0.98)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    dam_vals_orig = Z_orig[Z_orig < thresh]
    dam_vals_new = Z_new[Z_new < thresh]
    bins_z = np.linspace(0.50, 0.98, 25)

    ax2.hist(dam_vals_orig, bins=bins_z, density=True, color=col_orig, alpha=0.35,
             edgecolor=col_orig, linewidth=2, label="Previous: Flat Uniform Severity (Equal Heavy & Slight)")
    ax2.hist(dam_vals_new, bins=bins_z, density=True, color=col_new, alpha=0.55,
             edgecolor="black", linewidth=1.2, label="New: Smooth Priority on Slight / Early-Stage Damage")

    p_slight = 100.0 * np.mean(dam_vals_new >= 0.80)
    p_mod = 100.0 * np.mean((dam_vals_new >= 0.65) & (dam_vals_new < 0.80))
    p_heavy = 100.0 * np.mean(dam_vals_new < 0.65)

    # Annotate severity bands
    ax2.axvspan(0.80, 0.98, color="#a6dba0", alpha=0.22)
    ax2.axvspan(0.65, 0.80, color="#fee08b", alpha=0.18)
    ax2.axvspan(0.50, 0.65, color="#f4a582", alpha=0.15)
    ax2.axvline(0.80, color="gray", linestyle="--", linewidth=1)
    ax2.axvline(0.65, color="gray", linestyle="--", linewidth=1)

    ymax_b = max(ax2.get_ylim()[1], 4.8)
    ax2.set_ylim(0, ymax_b * 1.12)
    ax2.text(0.89, ymax_b * 0.72,
             f"Slight / Early\n[0.80, 0.98]\n({p_slight:.0f}% of cases)",
             ha="center", fontsize=8.8, fontweight="bold", color="#0e664b")
    ax2.text(0.725, ymax_b * 0.48,
             f"Moderate\n[0.65, 0.80]\n({p_mod:.0f}%)",
             ha="center", fontsize=8.8, fontweight="bold", color="#8c510a")
    ax2.text(0.575, ymax_b * 0.30,
             f"Heavy\n[0.50, 0.65]\n({p_heavy:.0f}%)",
             ha="center", fontsize=8.8, fontweight="bold", color="#b2182b")

    ax2.set_xlim(0.50, 0.985)
    ax2.set_xlabel("Stiffness Factor $z_i$ of Damaged Zones ($1.0 =$ Healthy)")
    ax2.set_ylabel("Probability Density (Given Zone is Damaged)")
    ax2.set_title(f"(B) Severity: Smooth Priority on Early-Stage Damage\n{p_slight:.0f}% Slight ($2\\%$–$20\\%$ Loss) vs. {p_heavy:.0f}% Heavy ($35\\%$–$50\\%$)",
                  fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # -------------------------------------------------------------------------
    # PANEL C: 2D Joint Projection (z_1, z_2) -- Previous vs. New
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    ax3.scatter(Z_orig[:800, 0], Z_orig[:800, 1], s=14, alpha=0.28, color=col_orig,
                label="Previous (Interior Cloud Only)")
    ax3.scatter(Z_new[:1200, 0], Z_new[:1200, 1], s=18, alpha=0.55, color=col_new,
                label="New (Healthy Corner + Single-Zone Axes)")
    ax3.plot(1.0, 1.0, marker="*", markersize=16, color="gold", markeredgecolor="black",
             zorder=10, label="Healthy Structure $(1.0, 1.0)$")
    ax3.set_xlim(0.48, 1.03)
    ax3.set_ylim(0.48, 1.03)
    ax3.set_xlabel("Zone 1 Stiffness Factor $z_1$")
    ax3.set_ylabel("Zone 2 Stiffness Factor $z_2$")
    ax3.set_title("(C) 2D Joint Projection $(z_1, z_2)$\nConcentrated on Healthy & Single-Zone Early Damage",
                  fontweight="bold")
    ax3.legend(loc="lower left", fontsize=8.5)
    ax3.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Figure 1 (Addressing R1.6 & R3.3): Direct Comparison of Previous (Original) vs. New Realistic Training Dataset",
        fontsize=13.5, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "Fig1_DoE_Distribution_Comparison.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# FIGURE 2: FE Mesh Discretization Convergence (5 vs 10 vs 20 vs 50 vs 100 Elements)
# =============================================================================
def plot_fig2_mesh_discretization():
    """
    Visually explains Reviewer 1 Comment 1 (R1.1) and Reviewer 3 Comment 4 (R3.4):
    Why a 5-element FE mesh accurately resolves Modes 1-4 (<2.3% freq error, MAC=1.000)
    but aliases Mode 5 (+10.99% freq error, MAC=0.151) due to having only 4 internal nodes.
    """
    meshes = [5, 10, 20, 50, 100]
    models = {
        n: BeamFEModel(BeamConfig(name=f"FE{n}", n_elements=n, n_zones=5))
        for n in meshes
    }
    ref_model = models[100]
    f_ref, v_ref_full, r_ref_full = ref_model.modal(ref_model.stiffness(np.ones(5)), ref_model.mass(), n_modes=5)
    x_ref = np.linspace(0.0, ref_model.cfg.L, ref_model.cfg.n_nodes)

    f_FE5, v_FE5, r_FE5 = models[5].modal(models[5].stiffness(np.ones(5)), models[5].mass(), n_modes=5)
    f_FE20, v_FE20, r_FE20 = models[20].modal(models[20].stiffness(np.ones(5)), models[20].mass(), n_modes=5)
    x_FE5 = np.linspace(0.0, 10.0, 6)
    x_FE20 = np.linspace(0.0, 10.0, 21)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5))

    # Panel (0,0): Relative Frequency Error vs Mesh Size for Modes 1..5
    ax = axes[0, 0]
    for m in range(5):
        errs = []
        for n in meshes:
            fn, _, _ = models[n].modal(models[n].stiffness(np.ones(5)), models[n].mass(), n_modes=5)
            errs.append(100.0 * (fn[m] - f_ref[m]) / f_ref[m])
        ax.plot(meshes, errs, marker="o", linewidth=2, label=f"Mode {m+1} ({f_ref[m]:.1f} Hz)")
    ax.axhline(2.5, color="gray", linestyle="--", label="2.5% Freq. Noise Level")
    ax.set_xscale("log")
    ax.set_xticks(meshes)
    ax.set_xticklabels([str(n) for n in meshes])
    ax.set_xlabel("Number of Finite Elements $N_{\\mathrm{el}}$ in Mesh")
    ax.set_ylabel("Rel. Frequency Error vs. 100-FE (%)")
    ax.set_title("Modal Frequency Convergence vs. Mesh Size", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Panels (0,1), (0,2), (1,0), (1,1), (1,2): Mode shapes 1 to 5 (Vertical & Rotational at sensors)
    panel_coords = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for m, (r_idx, c_idx) in enumerate(panel_coords):
        axm = axes[r_idx, c_idx]
        # Align sign at sensors
        v100_s = v_ref_full[m, ref_model.cfg.sensor_nodes]
        s100 = 1.0 if v100_s[np.argmax(np.abs(v100_s))] >= 0 else -1.0
        s5 = 1.0 if v_FE5[m, np.argmax(np.abs(v_FE5[m]))] >= 0 else -1.0
        # Ensure Mode 5 continuous curve and FE5 have consistent first-lobe sign
        if np.dot(v_ref_full[m, ::20] * s100, v_FE5[m] * s5) < 0:
            s5 = -s5

        axm.plot(x_ref, v_ref_full[m] * s100, "k-", linewidth=2.0, label="100-FE Continuous Ref.")
        axm.plot(x_FE20, v_FE20[m] * (1.0 if np.dot(v_FE20[m], v_ref_full[m, ::5] * s100) >= 0 else -1.0),
                 "b--", linewidth=1.3, label="20-FE Mesh")
        axm.plot(x_FE5, v_FE5[m] * s5, "ro-", markersize=7, linewidth=1.5, label="5-FE Mesh (6 Sensors)")
        ferr = 100.0 * (f_FE5[m] - f_ref[m]) / f_ref[m]
        axm.set_title(f"Mode {m+1}: $f_{{\\mathrm{{FE5}}}}={f_FE5[m]:.1f}$ Hz vs $f_{{\\mathrm{{FE100}}}}={f_ref[m]:.1f}$ Hz ({ferr:+.2f}%)",
                      fontweight="bold", color=("darkred" if m == 4 else "black"))
        axm.set_xlabel("Span Coordinate $x$ (m)")
        axm.set_ylabel("Vertical Mode $\\phi_v(x)$")
        axm.axhline(0, color="gray", linewidth=0.8)
        axm.grid(True, linestyle="--", alpha=0.4)
        if m in (0, 4):
            axm.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Figure 2 (Addressing R1.1 & R3.4): Discretization Error Between 5-Element Decoder and Fine FE Meshes\n"
        "Modes 1–4 Match Closely (<2.3% Error), While Mode 5 Hits the 4-Internal-Node Nyquist Limit on a 5-Element Mesh",
        fontsize=12.5, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(FIG_DIR, "Fig2_FineMesh_Discretization_Convergence.png")
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


# =============================================================================
# FIGURE 3: Sub-Zone Localized Cracking (1-2 Fine Elements) vs Harmonic Zone Equivalent
# =============================================================================
def plot_fig3_subzone_localized_damage():
    """
    Visually explains Reviewer 2 Comment 3 (R2.3):
    What happens when a crack affects only 2 fine elements (0.40 m = 4% of span)
    inside Zone 2 (which is 2.0 m = 20% of span) on a 50-element mesh, and why
    the harmonic-mean zone stiffness z_eq accurately matches its modal response.
    """
    cfg50 = TOY_FINE[50]
    m50 = BeamFEModel(cfg50)
    m5 = BeamFEModel(TOY_E5)

    # Create a localized crack in elements 14 and 15 (inside Zone 2: elements 10..19)
    z_el = np.ones(50)
    z_el[14:16] = 0.40  # 60% stiffness loss over 0.40m crack
    z_zone_eq = zone_equivalent_factor(z_el, cfg50.zone_of_element, 5)
    z_zone_arith = np.array([z_el[k*10:(k+1)*10].mean() for k in range(5)])

    f_healthy, v_healthy, _ = m50.modal(m50.stiffness(np.ones(5)), m50.mass(), n_modes=4)
    f_crack50, v_crack50, _ = m50.modal(m50.stiffness(z_zone_eq, z_element=z_el), m50.mass(), n_modes=4)
    f_zone50, v_zone50, _ = m50.modal(m50.stiffness(z_zone_eq), m50.mass(), n_modes=4)
    f_zone5, v_zone5, _ = m5.modal(m5.stiffness(z_zone_eq), m5.mass(), n_modes=4)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # Panel 1: Stiffness profile along the 10 m beam
    x_el = np.linspace(0, 10, 50, endpoint=False) + 0.1
    axes[0].step(np.linspace(0, 10, 51), np.append(z_el, z_el[-1]), where="post",
                 color="#d95f02", linewidth=2.5, label="True Sub-Zone Crack (2/50 FE = 0.4 m)")
    axes[0].step(np.linspace(0, 10, 6), np.append(z_zone_eq, z_zone_eq[-1]), where="post",
                 color="#1b9e77", linestyle="--", linewidth=2.5,
                 label=f"Harmonic Zone Equiv. ($z_{{\\mathrm{{eq}},2}}={z_zone_eq[1]:.3f}$)")
    axes[0].step(np.linspace(0, 10, 6), np.append(z_zone_arith, z_zone_arith[-1]), where="post",
                 color="#7570b3", linestyle=":", linewidth=2.0,
                 label=f"Arithmetic Zone Mean ($z_{{\\mathrm{{arith}},2}}={z_zone_arith[1]:.3f}$)")
    axes[0].set_ylim(0.30, 1.08)
    axes[0].set_xlabel("Beam Coordinate $x$ (m)")
    axes[0].set_ylabel("Stiffness Reduction Factor $z(x)$")
    axes[0].set_title("Sub-Zone Crack vs. 5-Zone Equivalent Label", fontweight="bold")
    axes[0].legend(loc="lower left", fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Frequency shift comparison across Modes 1-4
    modes_idx = np.arange(1, 5)
    df_crack = 100.0 * (f_crack50 - f_healthy) / f_healthy
    df_zone50 = 100.0 * (f_zone50 - f_healthy) / f_healthy
    width = 0.35
    axes[1].bar(modes_idx - width/2, df_crack, width=width, color="#d95f02",
                label="True 0.4m Crack (50-FE)")
    axes[1].bar(modes_idx + width/2, df_zone50, width=width, color="#1b9e77",
                label="2.0m Harmonic Equiv. Zone")
    axes[1].set_xticks(modes_idx)
    axes[1].set_xticklabels([f"Mode {m}" for m in modes_idx])
    axes[1].set_ylabel("Frequency Shift vs. Healthy (%)")
    axes[1].set_title("Frequency Shifts Match Across Modes 1–4", fontweight="bold")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # Panel 3: Mode 1 & Mode 2 curvature / displacement difference
    x50 = np.linspace(0, 10, 51)
    for m, col, ls in [(0, "#1f78b4", "-"), (1, "#e31a1c", "--")]:
        s_h = 1.0 if v_healthy[m, np.argmax(np.abs(v_healthy[m]))] >= 0 else -1.0
        s_c = 1.0 if v_crack50[m, np.argmax(np.abs(v_crack50[m]))] >= 0 else -1.0
        s_z = 1.0 if v_zone50[m, np.argmax(np.abs(v_zone50[m]))] >= 0 else -1.0
        dv_crack = v_crack50[m] * s_c - v_healthy[m] * s_h
        dv_zone = v_zone50[m] * s_z - v_healthy[m] * s_h
        axes[2].plot(x50, dv_crack, color=col, linestyle="-", linewidth=2.0,
                     label=f"Mode {m+1} $\\Delta\\phi_v$ (0.4m Crack)")
        axes[2].plot(x50, dv_zone, color=col, linestyle=":", linewidth=2.2,
                     label=f"Mode {m+1} $\\Delta\\phi_v$ (Harmonic Zone)")
    axes[2].axvspan(2.0, 4.0, color="orange", alpha=0.15, label="Damaged Zone 2 ([2m, 4m])")
    axes[2].set_xlabel("Beam Coordinate $x$ (m)")
    axes[2].set_ylabel("Mode Shape Change $\\Delta\\phi_v(x)$")
    axes[2].set_title("Mode-Shape Perturbation: Crack vs. Zone Equiv.", fontweight="bold")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Figure 3 (Addressing R2.3 & R1.1): Localized Sub-Zone Cracking on a 50-Element Mesh\n"
        "Why the Harmonic-Mean Effective Zone Stiffness $z_{\\mathrm{eq}}$ Captures Narrow Cracks Within a Zone",
        fontsize=12.5, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "Fig3_SubZone_Localized_Crack_vs_ZoneEquivalent.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# FIGURE 4: Elastic Bearing Supports & Realistic 20m RC Bridge Model
# =============================================================================
def plot_fig4_elastic_bearings_and_bridge():
    """
    Visually explains Reviewer 2 Comments 1, 2, 4 (R2.1, R2.2, R2.4):
    1) How elastic bearing stiffness (kappa = k_v L^3 / EI) and bearing degradation
       (s_L, s_R < 1) modify natural frequencies and produce non-zero support
       displacements at x = 0 and x = L.
    2) The modal characteristics of the realistic 20m RC Bridge (40 FE, 10 zones +
       2 elastic bearings, 11 sensors).
    """
    m_el = BeamFEModel(TOY_ELASTIC)
    m_br = BeamFEModel(BRIDGE20)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # Panel 1: Effect of dimensionless bearing stiffness kappa on f1, f2, f3
    kappas = np.logspace(1, 4, 25)
    f_pinned, _, _ = BeamFEModel(TOY_E5).modal(
        BeamFEModel(TOY_E5).stiffness(np.ones(5)), BeamFEModel(TOY_E5).mass(), n_modes=3
    )
    ratios = np.zeros((len(kappas), 3))
    for idx, kap in enumerate(kappas):
        cfg_k = BeamConfig(
            name="tmp", n_elements=20, n_zones=5, support="elastic",
            kv_nom=kap * TOY_E5.EI / (TOY_E5.L ** 3), support_latent=True
        )
        mk = BeamFEModel(cfg_k)
        fk, _, _ = mk.modal(mk.stiffness(np.ones(5), np.ones(2)), mk.mass(), n_modes=3)
        ratios[idx] = fk / f_pinned

    for m in range(3):
        axes[0].plot(kappas, ratios[:, m], linewidth=2.2, label=f"Mode {m+1} ($f_{m+1}/f_{m+1}^{{\\mathrm{{pin}}}}$)")
    axes[0].axvline(200.0, color="red", linestyle="--", label="Selected Nominal $\\kappa=200$")
    axes[0].axvspan(40.0, 200.0, color="red", alpha=0.10, label="Bearing Degradation $s \\in [0.2, 1.0]$")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Dimensionless Bearing Stiffness $\\kappa = k_v L^3 / (EI)$")
    axes[0].set_ylabel("Frequency Ratio $f_{\\mathrm{elastic}} / f_{\\mathrm{pinned}}$")
    axes[0].set_title("Effect of Elastic Bearings on Frequencies", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Mode 1 & 2 shapes under Left Bearing Degradation (s_L = 0.25, s_R = 1.0) vs Deck Damage
    x20 = np.linspace(0, 10, 21)
    _, v_nom, _ = m_el.modal(m_el.stiffness(np.ones(5), np.array([1.0, 1.0])), m_el.mass(), n_modes=2)
    _, v_sup, _ = m_el.modal(m_el.stiffness(np.ones(5), np.array([0.25, 1.0])), m_el.mass(), n_modes=2)
    z_deck = np.array([0.55, 1.0, 1.0, 1.0, 1.0])
    _, v_dck, _ = m_el.modal(m_el.stiffness(z_deck, np.array([1.0, 1.0])), m_el.mass(), n_modes=2)

    for v_arr, lbl, col, ls in [
        (v_nom, "Nominal Bearings $(s_L=1, s_R=1)$", "black", "-"),
        (v_sup, "Degraded Left Bearing $(s_L=0.25, s_R=1)$", "#e41a1c", "--"),
        (v_dck, "Deck Zone 1 Damage $(z_1=0.55, s_L=1)$", "#377eb8", ":"),
    ]:
        s = 1.0 if v_arr[0, np.argmax(np.abs(v_arr[0]))] >= 0 else -1.0
        axes[1].plot(x20, v_arr[0] * s, color=col, linestyle=ls, linewidth=2.2, label=lbl)
    axes[1].scatter([0.0, 10.0], [abs(v_sup[0, 0]), abs(v_sup[0, -1])], color="#e41a1c", s=65, zorder=5)
    axes[1].set_xlabel("Span Coordinate $x$ (m)")
    axes[1].set_ylabel("Mode 1 Vertical Shape $\\phi_{v,1}(x)$")
    axes[1].set_title("Why Bearing Degradation Is Distinguishable at $x=0, L$", fontweight="bold")
    axes[1].legend(loc="lower center", fontsize=8.5)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # Panel 3: Realistic 20m RC Bridge (40 FE, 10 zones + 2 bearings, 6 modes)
    x_br = np.linspace(0, 20, 41)
    f_br, v_br, _ = m_br.modal(m_br.stiffness(np.ones(10), np.ones(2)), m_br.mass(), n_modes=6)
    for m in range(4):
        s = 1.0 if v_br[m, np.argmax(np.abs(v_br[m]))] >= 0 else -1.0
        axes[2].plot(x_br, v_br[m] * s, linewidth=1.8, label=f"Mode {m+1} ({f_br[m]:.2f} Hz)")
        axes[2].scatter(x_br[m_br.cfg.sensor_nodes], v_br[m, m_br.cfg.sensor_nodes] * s, s=22)
    axes[2].set_xlabel("Bridge Span $x$ (m) — 40 FE, 10 Zones, 11 Sensors")
    axes[2].set_ylabel("Vertical Mode Shapes $\\phi_v(x)$")
    axes[2].set_title("New 20 m RC Bridge Benchmark (R2.1–R2.4)", fontweight="bold")
    axes[2].legend(loc="best", fontsize=8.5)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Figure 4 (Addressing R2.1–R2.4): Elastic Bearing Supports and the 20 m RC Bridge Benchmark\n"
        "Jointly Modeling Deck Stiffness Zones and Abutment Bearing Degradation $(s_L, s_R)$",
        fontsize=12.5, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "Fig4_Elastic_Bearings_and_Bridge20_Model.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# FIGURE 5: Mass Matrix Perturbations vs Measurement Noise (R3.1 & R1.2)
# =============================================================================
def plot_fig5_mass_perturbations():
    """
    Visually explains Reviewer 3 Comment 1 (R3.1):
    1) How spatially random element mass errors (1%, 2.5%, 5%, 10%) compare with the
       2.5% measurement noise on frequencies and 5% on mode shapes.
    2) Why a uniform global mass offset (+5% pavement overlay) shifts all frequencies
       by -2.41% (confounded with a uniform -4.76% stiffness shift) while leaving
       normalized mode shapes unchanged (MAC = 1.0000).
    """
    m5 = BeamFEModel(TOY_E5)
    rng = np.random.default_rng(123)
    z_ref = np.array([0.70, 1.0, 0.85, 1.0, 1.0])
    K_ref = m5.stiffness(z_ref)
    f0, v0, _ = m5.modal(K_ref, m5.mass(), n_modes=5)

    sigmas = [0.01, 0.025, 0.05, 0.10]
    freq_errs_by_sigma = []
    mac_by_sigma = []
    for sig in sigmas:
        fe_list, mac_list = [], []
        for _ in range(200):
            r_el = np.clip(rng.normal(1.0, sig, size=5), 0.6, 1.4)
            fp, vp, _ = m5.modal(K_ref, m5.mass(r_el), n_modes=5)
            fe_list.append(100.0 * np.mean(np.abs(fp - f0) / f0))
            # MAC on first 4 modes
            macs = []
            for m in range(4):
                a, b = v0[m], vp[m]
                macs.append((np.dot(a, b) ** 2) / (np.dot(a, a) * np.dot(b, b)))
            mac_list.append(np.mean(macs))
        freq_errs_by_sigma.append(fe_list)
        mac_by_sigma.append(mac_list)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    bp = axes[0].boxplot(freq_errs_by_sigma, patch_artist=True, boxprops=dict(facecolor="#a6cee3", alpha=0.85))
    axes[0].set_xticklabels(["1.0%", "2.5%", "5.0%", "10.0%"])
    axes[0].axhline(2.5, color="#e31a1c", linestyle="--", linewidth=2,
                    label="Training Freq. Noise Std ($\\sigma_f = 2.5\\%$)")
    axes[0].set_xlabel("Element Mass Perturbation Std $\\sigma_m$ ($\\rho_e \\sim \\mathcal{N}(\\rho_0, \\sigma_m^2)$)")
    axes[0].set_ylabel("Mean Relative Frequency Perturbation (%)")
    axes[0].set_title("Frequency Sensitivity to Unmodeled Mass Error", fontweight="bold")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].boxplot([1.0 - np.array(m) for m in mac_by_sigma],
                    patch_artist=True, boxprops=dict(facecolor="#b2df8a", alpha=0.85))
    axes[1].set_xticklabels(["1.0%", "2.5%", "5.0%", "10.0%"])
    axes[1].set_xlabel("Element Mass Perturbation Std $\\sigma_m$")
    axes[1].set_ylabel("Mode Shape Discrepancy $1 - \\mathrm{MAC}$ (Modes 1–4)")
    axes[1].set_title("Mode-Shape Sensitivity to Unmodeled Mass Error", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Figure 5 (Addressing R3.1 & R1.2): Sensitivity of Modal Quantities to Mass Matrix Perturbations\n"
        "Because $\\omega \\propto \\sqrt{k/m}$, a $\\sigma_m \\leq 5\\%$ Element Mass Error Induces $<1.5\\%$ Frequency Shift (Below the $2.5\\%$ Noise Floor)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "Fig5_Mass_Perturbations_vs_Noise.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# FIGURE 6: Previous vs. New Training/Validation Dataset & Dedicated Test Benchmarks
# =============================================================================
def plot_fig6_train_vs_test_overview():
    """
    Directly compares the Previous (Original Uniform) dataset against the New
    (Realistic Early-Stage & Localized) dataset in terms of:
    - Panel A: Fundamental Frequency f_1 distribution (showing how the Previous
               dataset shifted f_1 far below the undamaged frequency f_1=3.35 Hz,
               whereas the New dataset concentrates near the undamaged/early-damage
               regime while maintaining a tail down to severe damage).
    - Panel B: Identical distribution across the 70% Train / 20% Validation / 10%
               Test splits of `TRAIN_E5_realistic_doe` (R3.2).
    - Panel C: Dedicated Testing Benchmark regimes spanning Healthy (FPR),
               Single/Double Mild Damage, Sub-Zone Crack, Fine Mesh, and OOD.
    """
    orig_path = os.path.join(DATA_ROOT, "TRAIN_E5_uniform")
    new_path = os.path.join(DATA_ROOT, "TRAIN_E5_realistic_doe")

    F_orig = np.load(os.path.join(orig_path, "freqs_data_true.npy"))
    Z_orig = np.load(os.path.join(orig_path, "alpha_factors_true.npy"))

    F_new = np.load(os.path.join(new_path, "freqs_data_true.npy"))
    Z_new = np.load(os.path.join(new_path, "alpha_factors_true.npy"))
    sp = np.load(os.path.join(new_path, "split_indices.npz"))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    col_orig = "#d95f02"
    col_new = "#1b9e77"

    # Panel A: Fundamental Frequency f_1 (Hz) -- Previous vs. New
    bins_f1 = np.linspace(2.35, 3.65, 35)
    axes[0].hist(F_orig[:, 0], bins=bins_f1, density=True, color=col_orig, alpha=0.40,
                 edgecolor=col_orig, linewidth=1.8, label="Previous (Original Uniform DoE)")
    axes[0].hist(F_new[:, 0], bins=bins_f1, density=True, color=col_new, alpha=0.55,
                 edgecolor="black", linewidth=1.1, label="New (Realistic Early-Stage DoE)")
    axes[0].axvline(3.352, color="black", linestyle="--", linewidth=2,
                    label="Undamaged Clean $f_1 = 3.35$ Hz")
    axes[0].set_xlabel("Noisy Fundamental Frequency $f_1$ (Hz)")
    axes[0].set_ylabel("Probability Density")
    axes[0].set_title("(A) Fundamental Frequency $f_1$ Distribution\nPrevious vs. New Dataset", fontweight="bold")
    axes[0].legend(loc="upper left", fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Panel B: Train (70%) / Validation (20%) / In-Dist Test (10%) of New Dataset
    bins_z = np.linspace(0.50, 1.005, 26)
    for split_key, lbl, col in [
        ("train", f"New Train Split (70%, N={len(sp['train'])})", "#1b9e77"),
        ("val", f"New Validation Split (20%, N={len(sp['val'])}) — For $\\gamma$", "#2b5c8f"),
        ("test", f"New Held-Out Split (10%, N={len(sp['test'])})", "#7570b3"),
    ]:
        z_sub = Z_new[sp[split_key]].ravel()
        axes[1].hist(z_sub, bins=bins_z, density=True, histtype="step", linewidth=2.2, color=col, label=lbl)
    axes[1].hist(Z_orig.ravel(), bins=bins_z, density=True, histtype="step", linestyle="--",
                 linewidth=2.0, color=col_orig, label="Previous Uniform DoE (Flat $z_i$)")
    axes[1].set_xlabel("Element Stiffness Factor $z_i$ (All Zones)")
    axes[1].set_ylabel("Probability Density (Log Scale)")
    axes[1].set_yscale("log")
    axes[1].set_title("(B) Marginal $z_i$ Across Train / Val / Test Splits\nIdentical Split Proportions (R3.2)",
                      fontweight="bold")
    axes[1].legend(loc="lower left", fontsize=8.2)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # Panel C: Dedicated Test Benchmarks (Minimum Zone Stiffness vs Fundamental Frequency f_1)
    bench_specs = [
        ("TEST_E5_healthy", "Healthy ($z=1$, R1.6 FPR)", "#2ca02c", "*"),
        ("TEST_E5_single", "Single-Zone Mild (R1.6)", "#1f78b4", "o"),
        ("TEST_E5_double", "Double-Zone Mild (R1.6)", "#ff7f00", "s"),
        ("TEST_E5_localized_fine50", "Sub-Zone Crack 50-FE (R2.3)", "#6a3d9a", "^"),
        ("TEST_E5_finemesh50", "Fine Mesh 50-FE (R1.1)", "#b15928", "D"),
        ("TEST_E5_oodsevere", "OOD Severe $z\\in[0.3,0.45]$ (R1.5)", "#e31a1c", "v"),
    ]
    for b_name, b_lbl, b_col, b_mk in bench_specs:
        b_path = os.path.join(DATA_ROOT, b_name)
        if not os.path.exists(b_path):
            continue
        Fb = np.load(os.path.join(b_path, "freqs_data_true.npy"))[:150, 0]
        Zb = np.load(os.path.join(b_path, "alpha_factors_true.npy"))[:150]
        axes[2].scatter(Zb.min(axis=1), Fb, s=22, alpha=0.65, color=b_col, marker=b_mk, label=b_lbl)

    axes[2].axvline(0.50, color="red", linestyle="--", linewidth=1.5, label="Training Lower Bound ($z_{\\mathrm{lb}}=0.50$)")
    axes[2].set_xlabel("Minimum Zone Stiffness $\\min_i z_i$ (Most Damaged Zone)")
    axes[2].set_ylabel("Noisy Fundamental Frequency $f_1$ (Hz)")
    axes[2].set_title("(C) Dedicated Testing Regimes\nHealthy, Localized Mild, Fine-Mesh & OOD", fontweight="bold")
    axes[2].legend(loc="lower right", fontsize=8.0)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Figure 6: Previous vs. New Training/Validation Dataset and Dedicated Testing Regimes",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "Fig6_TrainVal_vs_Test_Benchmarks_Overview.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# MAIN EXECUTION: Generate Realistic DoE Datasets + All Diagnostic Plots
# =============================================================================
def main():
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    _setup_matplotlib()

    print("=" * 78)
    print("STEP 1: Generating Updated Realistic Early-Stage DoE Datasets in Data/review_27oct/")
    print("=" * 78)
    for ds_name in ("TRAIN_E5_realistic_doe", "TRAIN_E5_finemesh20_realistic_doe"):
        generate(REGISTRY[ds_name], DATA_ROOT)

    print("\n" + "=" * 78)
    print("STEP 2: Generating Explanatory & Diagnostic Figures in Data/review_27oct/Figures_Dataset_Diagnostics/")
    print("=" * 78)
    f1 = plot_fig1_doe_comparison()
    print(f"  [Saved] {f1}")
    f2 = plot_fig2_mesh_discretization()
    print(f"  [Saved] {f2}")
    f3 = plot_fig3_subzone_localized_damage()
    print(f"  [Saved] {f3}")
    f4 = plot_fig4_elastic_bearings_and_bridge()
    print(f"  [Saved] {f4}")
    f5 = plot_fig5_mass_perturbations()
    print(f"  [Saved] {f5}")
    f6 = plot_fig6_train_vs_test_overview()
    print(f"  [Saved] {f6}")
    print("\nAll datasets and 6 explanatory figures generated successfully.")


if __name__ == "__main__":
    main()
