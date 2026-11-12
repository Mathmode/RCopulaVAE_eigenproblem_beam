#!/usr/bin/env python3
"""
Figures of the representative test cases selected with test_case_analysis.py (reference model, 50-element test data):
reference (exact) posterior by SMC, posterior of the proposed method, and
  <run>/cases/P{pos}_Overlay.png     corner plot, proposed method vs reference posterior (as the paper figures)
  <run>/cases/P{pos}_1D.png          marginal overlays
  <run>/cases/P{pos}_Violin.png      violins of the proposed method (as the paper figures)
  <run>/cases/P{pos}_GT.png, _VAE.png single corner plots
  <run>/cases/Fig_cases_well.png     decision violins of the well-identified cases (2 x 2)
  <run>/cases/Fig_cases_fail.png     decision violins of the failure cases (1 x 3)
  <run>/cases/cases_metrics.csv      agreement with the reference posterior, decisions, errors
Usage (project root): python MODULES/review29oct_fixed/selected_cases_figures.py [--workers 4]
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from scipy.stats import norm, wasserstein_distance

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed import plotting as P
from MODULES.review29oct_fixed.posterior_families import family_from_preds
from MODULES.review29oct_fixed.reference_posterior import compute_many, compute_reference_posterior

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RUN = os.path.join(ROOT, "Output", "baseline_GaussCopula_GaussMarginals", "GCLogN_06Nov-5modes_7D_Gamma1.0_Conv6000")
DATA = os.path.join(ROOT, "Data", "06Nov_FINAL_SlabS10_7D_5sensors_5modes")
OUT = os.path.join(RUN, "cases")
LB, Z_C, Z_H, Z_R = 0.45, 0.84, 0.95, 0.80
SPLIT = "test_fine"
CASES = {  # position in the test dataset: (group, short title)
    1096: ("well", "Healthy bridge"),
    891: ("well", r"Light damage in zone 3, $z_3^*=0.85$"),
    695: ("well", r"Damage in the right bearing, $z_{b,R}^*=0.74$"),
    944: ("well", r"Damage in zones 2 and 3, $z_2^*=0.67$, $z_3^*=0.61$"),
    688: ("fail", "Healthy bridge, false positive in end zone 1"),
    462: ("fail", r"Damage at the right support (zone 5), $z_5^*=0.50$"),
    407: ("fail", r"Damage in end zone 1, $z_1^*=0.71$"),
}
LAB = [r"$z_1$", r"$z_2$", r"$z_3$", r"$z_4$", r"$z_5$", r"$z_{b,L}$", r"$z_{b,R}$"]
XS = np.array([0, 1, 2, 3, 4, 5.7, 6.7])
COL_H, EDGE_H, COL_D, EDGE_D = "#a9cbe8", "#1d4e89", "#f0a04b", "#8a4a00"


def decision_violins(ax, smp, z_true, mean, lo, hi, title):
    ax.axhspan(LB, Z_C, color="#fde3e1", lw=0, zorder=0)
    ax.axhline(Z_C, ls="--", color="#b03a2e", lw=0.9, zorder=1)
    for d in range(7):
        dam = mean[d] < Z_C
        vp = ax.violinplot(smp[:, d], positions=[XS[d]], widths=0.78, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(COL_D if dam else COL_H); b.set_edgecolor(EDGE_D if dam else EDGE_H)
            b.set_alpha(1.0); b.set_linewidth(0.8); b.set_zorder(2)
        ax.plot([XS[d]] * 2, [lo[d], hi[d]], color="k", lw=1.3, zorder=3)
        ax.plot(XS[d], mean[d], "o", ms=4.6, mfc="white", mec="k", mew=0.8, zorder=4)
        ax.plot(XS[d], z_true[d], "*", ms=10.5, color="#c0392b", mec="white", mew=0.5, zorder=5)
        fp, fn = z_true[d] >= Z_H and dam, z_true[d] < Z_R and not dam
        if fp or fn:
            ax.text(XS[d], 1.035, "FP" if fp else "FN", ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color="#b03a2e")
    ax.set_xticks(XS); ax.set_xticklabels(LAB, fontsize=9)
    ax.set_ylim(LB, 1.09); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_title(title, fontsize=8.6, pad=4)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    tr = ax.get_xaxis_transform()
    for x0, x1, txt in ((-0.3, 4.3, "deck zones"), (5.4, 7.0, "bearings")):
        ax.plot([x0, x1], [-0.15, -0.15], color="0.5", lw=0.7, transform=tr, clip_on=False)
        ax.text(0.5 * (x0 + x1), -0.17, txt, ha="center", va="top", fontsize=7.5, color="0.35", transform=tr)


LEGEND = [Patch(fc=COL_H, ec=EDGE_H, label=r"Posterior, declared healthy ($\hat z_d\geq z_c$)"),
          Patch(fc=COL_D, ec=EDGE_D, label=r"Posterior, declared damaged ($\hat z_d<z_c$)"),
          Patch(fc="#fde3e1", ec="none", label=f"Damage region, $z<z_c={Z_C:.2f}$"),
          Line2D([], [], color="k", lw=1.3, label="95% credible interval"),
          Line2D([], [], ls="", marker="o", mfc="white", mec="k", label=r"Posterior mean $\hat z_d$"),
          Line2D([], [], ls="", marker="*", ms=10, color="#c0392b", label=r"True value $z_d^*$"),
          Line2D([], [], ls="", marker="$FP$", ms=12, color="#b03a2e", label="False positive / negative")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--particles", type=int, default=4000)
    ap.add_argument("--violins-only", action="store_true", help="only the composite decision-violin figures")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    info = np.load(os.path.join(RUN, "Problem_info.npy"), allow_pickle=True).item()
    gamma2, obs = float(info["gamma"]) ** 2, info["obs_types"]
    preds = np.load(os.path.join(RUN, "Test_predicted_props_test_fine.npy"), allow_pickle=True).item()
    fam = family_from_preds(preds, LB)
    z = np.load(os.path.join(DATA, f"alphas_{SPLIT}.npy")).astype(np.float64)
    pos_list = list(CASES)
    print("--- reference posteriors (SMC) ---")
    compute_many(DATA, pos_list, gamma2, obs, args.particles, args.workers, False, SPLIT)

    mu, s = preds["test_logz_loc"].astype(float), preds["test_logz_scale"].astype(float)
    a, b = (np.log(LB) - mu) / s, -mu / s
    Zn = norm.cdf(b) - norm.cdf(a)
    mean = np.exp(mu + s ** 2 / 2) * (norm.cdf(b - s) - norm.cdf(a - s)) / Zn
    q = lambda pr: np.exp(mu + s * norm.ppf(norm.cdf(a) + pr * Zn))
    lo, hi = q(0.025), q(0.975)

    rows, samples = [], {}
    for pos in pos_list:
        if args.violins_only:
            samples[pos] = fam.sample(4000, seed=pos, idx=[pos])[0]
            continue
        x_gt, _ = compute_reference_posterior(DATA, pos, gamma2, obs, args.particles, split=SPLIT)
        x_gt = x_gt.astype(np.float64)
        x_q = fam.sample(4000, seed=pos, idx=[pos])[0]
        samples[pos] = x_q
        zt = z[pos]
        lo_gt, hi_gt = np.quantile(x_gt, [0.025, 0.975], axis=0)
        dec = mean[pos] < Z_C
        rows.append({"pos": pos, "case": CASES[pos][1].replace("$", ""), "group": CASES[pos][0],
                     **{f"{n}_true": zt[d] for d, n in enumerate(C.PARAM_NAMES)},
                     **{f"{n}_mean": mean[pos, d] for d, n in enumerate(C.PARAM_NAMES)},
                     **{f"{n}_ref_mean": x_gt[:, d].mean() for d, n in enumerate(C.PARAM_NAMES)},
                     "declared_damaged": " ".join(n for d, n in enumerate(C.PARAM_NAMES) if dec[d]),
                     "FP": int(np.sum((zt >= Z_H) & dec)), "FN": int(np.sum((zt < Z_R) & ~dec)),
                     "zstar_in_VAE95": int(np.sum((zt >= lo[pos]) & (zt <= hi[pos]))),
                     "zstar_in_ref95": int(np.sum((zt >= lo_gt) & (zt <= hi_gt))),
                     "mean_abs_diff_mean_VAE_ref": float(np.mean(np.abs(x_q.mean(0) - x_gt.mean(0)))),
                     "mean_std_ratio_VAE_ref": float(np.mean(x_q.std(0) / x_gt.std(0))),
                     "mean_W1_VAE_ref": float(np.mean([wasserstein_distance(x_gt[:, d], x_q[:, d]) for d in range(7)]))})
        tag = f"Test case {pos}: {CASES[pos][1]}"
        P.corner_overlay(x_gt, x_q, zt, list(range(C.N_DIMS)), LB, os.path.join(OUT, f"P{pos}_Overlay.png"))
        P.strip_1d(x_gt, x_q, zt, LB, os.path.join(OUT, f"P{pos}_1D.png"))
        P.physical_profile(x_q, zt, LB, os.path.join(OUT, f"P{pos}_Violin.png"))
        P.corner_single(x_gt, zt, LB, f"{tag}: reference posterior", "relative posterior density (reference)",
                        os.path.join(OUT, f"P{pos}_GT.png"))
        P.corner_single(x_q, zt, LB, f"{tag}: proposed method", "relative posterior density (proposed method)",
                        os.path.join(OUT, f"P{pos}_VAE.png"))
        print(f"  P{pos}: {CASES[pos][1]} | decisions: {rows[-1]['declared_damaged'] or 'none'} | FP {rows[-1]['FP']} FN "
              f"{rows[-1]['FN']} | z* in 95%: VAE {rows[-1]['zstar_in_VAE95']}/7, ref {rows[-1]['zstar_in_ref95']}/7 | "
              f"W1 {rows[-1]['mean_W1_VAE_ref']:.4f}")
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "cases_metrics.csv"), index=False, float_format="%.4f")
    # contrast for case 462: the same structure in the 10-element test data (same equivalent zone factor, uniform)
    preds10 = np.load(os.path.join(RUN, "Test_predicted_props.npy"), allow_pickle=True).item()
    fam10 = family_from_preds(preds10, LB)
    mu10, s10 = preds10["test_logz_loc"][462].astype(float), preds10["test_logz_scale"][462].astype(float)
    a10, b10 = (np.log(LB) - mu10) / s10, -mu10 / s10
    Z10 = norm.cdf(b10) - norm.cdf(a10)
    m10 = np.exp(mu10 + s10 ** 2 / 2) * (norm.cdf(b10 - s10) - norm.cdf(a10 - s10)) / Z10
    q10 = lambda pr: np.exp(mu10 + s10 * norm.ppf(norm.cdf(a10) + pr * Z10))
    extra = ("Same structure, damage uniform in zone 5", fam10.sample(4000, seed=462, idx=[462])[0], m10, q10(0.025), q10(0.975))

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    for group, fname in (("well", "Fig_cases_well"), ("fail", "Fig_cases_fail")):
        sel = [p for p in pos_list if CASES[p][0] == group]
        fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4), sharey=True)
        axs = axs.ravel()
        for k, (ax, pos) in enumerate(zip(axs, sel)):
            decision_violins(ax, samples[pos], z[pos], mean[pos], lo[pos], hi[pos],
                             f"({chr(97 + k)}) Test case {pos}" + "\n" + CASES[pos][1])
        if group == "fail":
            t, smp10, mm, ll, hh = extra
            decision_violins(axs[3], smp10, z[462], mm, ll, hh,
                             "(d) Test case 462, 10-element data" + "\n" + t)
        for ax in axs[::2]:
            ax.set_ylabel("Stiffness factor $z$")
        fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=7.4, frameon=False)
        fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=3.0)
        for e in ("png", "pdf"):
            fig.savefig(os.path.join(OUT, f"{fname}.{e}"), dpi=300)
        plt.close(fig)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()
