#!/usr/bin/env python3
"""
Panels of Fig. 6 (joint posteriors, VAE vs reference) redrawn at their printed size (3.45 in wide, half of a 7 in
figure), so that the lettering prints at a legible size (ASCE). Same data and samples as selected_cases_figures.py:
reference posterior from the SMC caches, VAE posterior sampled with family_from_preds(...).sample(4000, seed=pos).
The legend and the colour bar are omitted: the caption of Fig. 6 explains them.
Usage (project root): python MODULES/review29oct_fixed/overlay_print_size.py --out <folder>
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from MODULES.review29oct_fixed.posterior_families import family_from_preds
from MODULES.review29oct_fixed.reference_posterior import compute_reference_posterior
import MODULES.review29oct_fixed.plotting as P

RUN = os.path.join(ROOT, "Output", "baseline_GaussCopula_GaussMarginals", "GCLogN_13Nov-DoE30_7D_Gamma1.0_Conv6000")
DATA = os.path.join(ROOT, "Data", "13Nov_FINAL_SlabS10_7D_DoE30")
LB = 0.45
PANELS = {"P3": ("test_fine_uniform", 3), "P10": ("test_fine_uniform", 10),   # test dataset
          "P474": ("test_fine", 474), "P376": ("test_fine", 376)}              # test dataset B

RC = {**P.PAPER_RC, "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
      "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0,
      "ytick.major.size": 2.0, "xtick.major.pad": 1.2, "ytick.major.pad": 1.2, "axes.labelpad": 1.5,
      "lines.linewidth": 0.9, "pdf.fonttype": 42}


def overlay(x_ref, x_vae, z_true, path, width=3.45):
    n = x_ref.shape[1]
    left, right, bottom, top, gap = 0.40, 0.04, 0.34, 0.06, 0.08
    height = width - left - right + bottom + top
    with plt.style.context(["default", RC]):
        fig = plt.figure(figsize=(width, height))
        gs = fig.add_gridspec(n, n, left=left / width, right=1 - right / width, bottom=bottom / height,
                              top=1 - top / height, wspace=gap, hspace=gap)
        grid = np.linspace(LB, 1.0, 300)
        for r in range(n):
            for c in range(r + 1):
                ax = fig.add_subplot(gs[r, c])
                if r == c:
                    k_ref, k_q = P.kde_1d(x_ref[:, r], grid, LB), P.kde_1d(x_vae[:, r], grid, LB)
                    ax.fill_between(grid, k_ref, color=P.REF_FILL, lw=0)
                    ax.plot(grid, k_ref, color=P.REF_LINE, ls=P.REF_DASH, lw=0.6)
                    ax.plot(grid, k_q, color=P.VAE_LINE, lw=0.9)
                    ax.axvline(z_true[r], color=P.TRUE_COLOR, lw=0.7)
                    ax.set_ylim(0, 1.05 * max(k_ref.max(), k_q.max()))
                    ax.set_yticks([])
                    P.despine(ax, ("top", "left", "right"))
                else:
                    X, Y, Z = P.density_2d(x_ref[:, c], x_ref[:, r], LB)
                    ax.imshow(Z / (Z.max() + 1e-300), origin="lower", extent=[LB, 1, LB, 1], aspect="auto",
                              cmap=P.REF_CMAP, vmin=0, vmax=1, interpolation="bilinear")
                    ax.contour(X, Y, Z, levels=P.hpd_levels(Z), colors=[P.REF_LINE], linestyles=[P.REF_DASH], linewidths=0.5)
                    X, Y, Z = P.density_2d(x_vae[:, c], x_vae[:, r], LB)
                    ax.contour(X, Y, Z, levels=P.hpd_levels(Z), colors=[P.VAE_LINE], linewidths=0.7)
                    ax.plot(z_true[c], z_true[r], marker="*", color=P.TRUE_COLOR, ms=5, mec="white", mew=0.3,
                            ls="none", zorder=10, clip_on=False)
                    ax.set_ylim(LB, 1)
                    ax.set_yticks([0.6, 1.0], ["0.6", "1"])
                ax.set_xlim(LB, 1)
                ax.set_xticks([0.6, 1.0], ["0.6", "1"])
                ax.tick_params(labelbottom=r == n - 1, labelleft=(c == 0 and r > 0))
                if r == n - 1:
                    ax.set_xlabel(P.LABELS_7D[c])
                if c == 0 and r > 0:
                    ax.set_ylabel(P.LABELS_7D[r], rotation=0, ha="right", va="center")
        # legend in the empty upper-right triangle
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(facecolor=P.REF_FILL, edgecolor=P.REF_LINE, ls=P.REF_DASH, lw=0.6,
                           label=r"Reference posterior $p(\mathbf{z}\,|\,\mathbf{m})$"),
            mpatches.Patch(facecolor="#FDCF5F", edgecolor="none", label="Reference posterior: joint density"),
            mlines.Line2D([], [], color=P.VAE_LINE, lw=1.0, label=r"VAE posterior $q_{\zeta_\theta}(\mathbf{z}\,|\,\mathbf{m})$"),
            mlines.Line2D([], [], color=P.TRUE_COLOR, lw=0.7, marker="*", ms=5, mec="white", mew=0.3,
                          label=r"True state $\mathbf{z}^{\mathrm{true}}$"),
            mlines.Line2D([], [], color="none", label="Contours: 50% and 90% of the mass"),
        ]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1 - right / width, 1 - top / height),
                   fontsize=7, frameon=False, handlelength=1.6, handletextpad=0.5, labelspacing=0.35, borderaxespad=0)
        fig.savefig(path)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    info = np.load(os.path.join(RUN, "Problem_info.npy"), allow_pickle=True).item()
    gamma2, obs = float(info["gamma"]) ** 2, info["obs_types"]
    for name, (split, pos) in PANELS.items():
        preds = np.load(os.path.join(RUN, f"Test_predicted_props_{split}.npy"), allow_pickle=True).item()
        x_q = family_from_preds(preds, LB).sample(4000, seed=pos, idx=[pos])[0]
        x_gt, _ = compute_reference_posterior(DATA, pos, gamma2, obs, 4000, split=split)
        z = np.load(os.path.join(DATA, f"alphas_{split}.npy")).astype(float)[pos]
        out = os.path.join(args.out, f"Fig_overlay_{name}.pdf")
        overlay(x_gt.astype(float), x_q, z, out)
        print("saved", out, "| true state", np.round(z, 2))


if __name__ == "__main__":
    main()
