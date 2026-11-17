#!/usr/bin/env python3
"""
R3.4: corner plots (VAE vs reference posterior) of the same two scenarios of the test dataset (50 elements, same design
of experiments) identified with 5 modes and with 3 modes, in the parameters where the damage is located.
Reference posteriors (SMC, 4000 particles) are read from the caches of the two data sets:
  Data/13Nov_FINAL_SlabS10_7D_DoE30/reference_posteriors/GT_test_fine_uniform_P{pos}_*.npz         (computed if missing)
  Data/13Nov_FINAL_SlabS10_7D_DoE30_3modes/reference_posteriors/GT_test_fine_uniform_P{pos}_*.npz  (copied from goliat)
Usage (project root): python MODULES/review29oct_fixed/modes_corner.py
"""
import os, sys
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from MODULES.review29oct_fixed.posterior_families import family_from_preds
from MODULES.review29oct_fixed.reference_posterior import compute_reference_posterior, cache_path
import MODULES.review29oct_fixed.plotting as P

O = os.path.join(ROOT, "Output", "baseline_GaussCopula_GaussMarginals")
SETUPS = {"5 modes": (os.path.join(O, "GCLogN_13Nov-DoE30_7D_Gamma1.0_Conv6000"),
                      os.path.join(ROOT, "Data", "13Nov_FINAL_SlabS10_7D_DoE30")),
          "3 modes": (os.path.join(O, "GCLogN_13Nov-DoE30-3modes_7D_Gamma1.0_Conv6000"),
                      os.path.join(ROOT, "Data", "13Nov_FINAL_SlabS10_7D_DoE30_3modes"))}
CASES = {3: r"Mild damage in zone 2, $z_2^{\mathrm{true}}=0.78$", 1: r"Severe damage in end zone 1, $z_1^{\mathrm{true}}=0.56$"}
DIMS = [0, 1, 3, 5]                     # z1, z2, z4, z_bL: where the damage of both scenarios is located or spread
SPLIT, LB = "test_fine_uniform", 0.45
OUT = os.path.join(ROOT, "MODULES", "review29oct_fixed", "Latex_docs", "figures_final", "Fig_modes_corner")

z = np.load(os.path.join(SETUPS["5 modes"][1], f"alphas_{SPLIT}.npy")).astype(float)


def reference(setup, pos):
    run, data = SETUPS[setup]
    if os.path.isfile(os.path.join(data, "problem_config.json")):
        info = np.load(os.path.join(run, "Problem_info.npy"), allow_pickle=True).item()
        x, _ = compute_reference_posterior(data, pos, float(info["gamma"]) ** 2, info["obs_types"], 4000, split=SPLIT)
        return x.astype(float)
    path = cache_path(data, pos, 1.0, "fv", 4000, SPLIT)          # only the cache was copied from goliat
    d = np.load(path, allow_pickle=True)
    assert np.allclose(d["info"].item()["z_true"], z[pos], atol=1e-6), "scenario mismatch"
    return d["samples"].astype(float)


@P.pub_style
def figure(samples):
    n = len(DIMS)
    fig = plt.figure(figsize=(7.0, 7.4))
    outer = fig.add_gridspec(2, 2, left=0.08, right=0.985, bottom=0.12, top=0.94, wspace=0.16, hspace=0.30)
    grid = np.linspace(LB, 1.0, 300)
    k = 0
    for r0, pos in enumerate(CASES):
        for c0, setup in enumerate(SETUPS):
            x_ref, x_q = samples[setup, pos]
            zt = z[pos]
            gs = outer[r0, c0].subgridspec(n, n, wspace=0.17, hspace=0.12)
            for r in range(n):
                for c in range(r + 1):
                    ax = fig.add_subplot(gs[r, c])
                    dr, dc = DIMS[r], DIMS[c]
                    if r == c:
                        k_ref, k_q = P.kde_1d(x_ref[:, dr], grid, LB), P.kde_1d(x_q[:, dr], grid, LB)
                        ax.fill_between(grid, k_ref, color=P.REF_FILL, lw=0)
                        ax.plot(grid, k_ref, color=P.REF_LINE, ls=P.REF_DASH, lw=0.8)
                        ax.plot(grid, k_q, color=P.VAE_LINE, lw=1.1)
                        ax.axvline(zt[dr], color=P.TRUE_COLOR, lw=0.9)
                        ax.set_ylim(0, 1.05 * max(k_ref.max(), k_q.max()))
                        ax.set_yticks([])
                        P.despine(ax, ("top", "left", "right"))
                    else:
                        P._overlay_panel(ax, x_ref, x_q, dc, dr, zt, LB)
                        ax.set_yticks([0.6, 0.8, 1.0])
                    ax.set_xlim(LB, 1.0)
                    ax.set_xticks([0.6, 0.8, 1.0])
                    ax.tick_params(labelbottom=r == n - 1, labelleft=c == 0 and r > 0, labelsize=7)
                    if r == n - 1:
                        ax.set_xlabel(P.LABELS_7D[dc], labelpad=1)
                    if c == 0 and r > 0:
                        ax.set_ylabel(P.LABELS_7D[dr], rotation=0, ha="right", va="center")
            ax_t = fig.add_subplot(gs[0, 1:])
            ax_t.axis("off")
            w1 = np.mean([wasserstein_distance(x_ref[:, d], x_q[:, d]) for d in range(x_ref.shape[1])])
            ax_t.text(0.02, 0.95, f"({chr(97 + k)}) Test case {pos}, {setup}\n{CASES[pos]}\nMean $W_1$ = {w1:.4f}",
                      transform=ax_t.transAxes, ha="left", va="top", fontsize=8.5)
            print(f"P{pos} {setup}: mean W1 (7 parameters) {w1:.4f}")
            k += 1
    handles = [mpatches.Patch(facecolor=P.REF_FILL, edgecolor=P.REF_LINE, ls=P.REF_DASH, label=P.REF_LABEL),
               mlines.Line2D([], [], color=P.VAE_LINE, lw=1.4, label=P.VAE_LABEL),
               mpatches.Patch(facecolor="#FDCF5F", edgecolor="none", label="Reference posterior: joint density"),
               mlines.Line2D([], [], color=P.TRUE_COLOR, lw=0.9, marker="*", ms=8, mec=P.TRUE_EDGE, mew=0.4,
                             label=P.TRUE_LABEL)]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    for e in ("pdf", "png"):
        fig.savefig(f"{OUT}.{e}", dpi=300)
    plt.close(fig)
    print("saved", OUT)


samples = {}
for setup, (run, data) in SETUPS.items():
    fam = family_from_preds(np.load(os.path.join(run, f"Test_predicted_props_{SPLIT}.npy"), allow_pickle=True).item(), LB)
    for pos in CASES:
        samples[setup, pos] = (reference(setup, pos), fam.sample(4000, seed=pos, idx=[pos])[0])
figure(samples)
