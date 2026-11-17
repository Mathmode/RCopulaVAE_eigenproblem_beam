#!/usr/bin/env python3
"""
Figures of the representative test cases selected with test_case_analysis.py (reference model, 50-element test data):
reference (exact) posterior by SMC, posterior of the proposed method, and
  <run>/cases/P{pos}_Overlay.png     corner plot, proposed method vs reference posterior (as the paper figures)
  <run>/cases/P{pos}_1D.png          marginal overlays
  <run>/cases/P{pos}_Violin.png      violins of the proposed method (as the paper figures)
  <run>/cases/P{pos}_GT.png, _VAE.png single corner plots
  <run>/cases/Fig_cases_well.png     decision violins of the well-identified cases (2 x 2)
  <run>/cases/Fig_cases_fail.png     decision violins of the two representative failure cases (1 x 2)
  <run>/cases/cases_metrics.csv      agreement with the reference posterior, decisions, errors
Usage (project root): python MODULES/review29oct_fixed/selected_cases_figures.py [--set DoE30|06Nov] [--workers 4]
--set chooses the run, its data set and the selected cases (DoE30: data set of the letter since 13 Nov 2026).
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

LB, Z_C, Z_H, Z_R = 0.45, 0.84, 0.95, 0.80
SPLIT = "test_fine"
_B = os.path.join(ROOT, "Output", "baseline_GaussCopula_GaussMarginals")
CASE_SETS = {
    # 06Nov data set (letter until 13 Nov 2026)
    "06Nov": {"run": os.path.join(_B, "GCLogN_06Nov-5modes_7D_Gamma1.0_Conv6000"),
              "data": os.path.join(ROOT, "Data", "06Nov_FINAL_SlabS10_7D_5sensors_5modes"),
              "cases": {1096: ("well", "Healthy bridge"),
                        891: ("well", r"Light damage in zone 3, $z_3^{\mathrm{true}}=0.85$"),
                        695: ("well", r"Damage in the right bearing, $z_{b,R}^{\mathrm{true}}=0.74$"),
                        944: ("well", r"Damage in zones 2 and 3, $z_2^{\mathrm{true}}=0.67$, $z_3^{\mathrm{true}}=0.61$"),
                        688: ("fail", "Healthy bridge, false positive in end zone 1"),
                        462: ("fail", r"Damage next to the right support, $z_5^{\mathrm{true}}=0.50$"),
                        407: ("fail", r"Damage in end zone 1, $z_1^{\mathrm{true}}=0.71$")},
              "fig": {"well": [1096, 891, 695, 944], "fail": [688, 462]}},
    # DoE30 data set (letter since 13 Nov 2026): successful cases with mild damage, representative wrong decisions
    "DoE30": {"run": os.path.join(_B, "GCLogN_13Nov-DoE30_7D_Gamma1.0_Conv6000"),
              "data": os.path.join(ROOT, "Data", "13Nov_FINAL_SlabS10_7D_DoE30"),
              "cases": {534: ("well", "Healthy bridge"),
                        443: ("well", r"Mild damage in zone 4, $z_4^{\mathrm{true}}=0.74$"),
                        84: ("well", r"Mild damage in the right bearing, $z_{b,R}^{\mathrm{true}}=0.77$"),
                        474: ("well", r"Mild damage in zones 2 and 5, $z_2^{\mathrm{true}}=0.70$, $z_5^{\mathrm{true}}=0.71$"),
                        486: ("fail", "Healthy bridge"),
                        1127: ("fail", r"Mild damage in end zone 1, $z_1^{\mathrm{true}}=0.74$"),
                        376: ("fail", r"Mild damage in end zone 1, $z_1^{\mathrm{true}}=0.71$")},
              # letter figures (13 Nov, tau = 0.84): fail cases 486 and 376
              "fig": {"well": [534, 443, 84, 474], "fail": [486, 376]},
              # tau: smallest threshold detecting 95 % of the relevant damage of the validation set (decision_analysis.py)
              "tau": 0.86},
}
CASE_SETS["06Nov"]["tau"] = 0.84            # equal-error-rate threshold of the 06Nov letter
# DoE30, test set A (FE model of the training, damage spread over each zone): main test set of the letter since 13 Nov
CASE_SETS["DoE30A"] = {"run": CASE_SETS["DoE30"]["run"], "data": CASE_SETS["DoE30"]["data"], "split": "test", "tau": 0.84,
                       "cases": {534: ("well", "Healthy bridge"),
                                 443: ("well", r"Mild damage in zone 4, $z_4^{\mathrm{true}}=0.74$"),
                                 84: ("well", r"Mild damage in the right bearing, $z_{b,R}^{\mathrm{true}}=0.77$"),
                                 474: ("well", r"Mild damage in zones 2 and 5, $z_2^{\mathrm{true}}=0.70$, $z_5^{\mathrm{true}}=0.71$"),
                                 486: ("fail", "Healthy bridge, false positive in end zone 5"),
                                 622: ("fail", r"Mild damage in end zone 5, $z_5^{\mathrm{true}}=0.74$, missed")},
                       "fig": {"well": [534, 443, 84, 474], "fail": [486, 622]}}
# DoE30, test dataset of the letter (50 elements, same design of experiments): benchmark scenarios with light/mild damage
CASE_SETS["DoE30A50"] = {"run": CASE_SETS["DoE30"]["run"], "data": CASE_SETS["DoE30"]["data"], "split": "test_fine_uniform",
                         "tau": 0.84,
                         "cases": {3: ("well", r"Mild damage in zone 2, $z_2^{\mathrm{true}}=0.78$"),
                                   10: ("well", r"Light damage in zone 4, $z_4^{\mathrm{true}}=0.90$"),
                                   6: ("fail", r"Triple damage, $z_2^{\mathrm{true}}=0.54$, $z_4^{\mathrm{true}}=0.72$, $z_{b,R}^{\mathrm{true}}=0.64$"),
                                   7: ("fail", "Five damaged components")},
                         "fig": {"well": [3, 10], "fail": [6, 7]}}
PRED_FILE = {"test_fine": "Test_predicted_props_test_fine.npy", "test": "Test_predicted_props.npy",
             "test_fine_uniform": "Test_predicted_props_test_fine_uniform.npy"}
RUN = DATA = OUT = None          # set in main() from --set
CASES, FIG_CASES = {}, {}
LAB = [r"$z_1$", r"$z_2$", r"$z_3$", r"$z_4$", r"$z_5$", r"$z_{b,L}$", r"$z_{b,R}$"]
XS = np.array([0, 1, 2, 3, 4, 5.7, 6.7])
COL_H, EDGE_H, COL_D, EDGE_D = "#a9cbe8", "#1d4e89", "#f0a04b", "#8a4a00"
COL_U, EDGE_U = "#d9d9d9", "#555555"
Z_SIG, P_LO, P_HI = 0.80, 0.10, 0.90   # significant damage (loss > 20 %) and verdict limits on P(z_d < 0.80 | m)


def decision_violins(ax, smp, z_true, mean, lo, hi, title, pd):
    """Violins coloured by the verdict on P_d = P(z_d < 0.80 | m): healthy (P_d < 0.1), uncertain, damaged (P_d > 0.9)."""
    ax.axhspan(LB, Z_SIG, color="#fde3e1", lw=0, zorder=0)
    ax.axhline(Z_SIG, ls="--", color="#b03a2e", lw=0.9, zorder=1)
    for d in range(7):
        p = pd[d]
        fc, ec = (COL_H, EDGE_H) if p < P_LO else ((COL_D, EDGE_D) if p > P_HI else (COL_U, EDGE_U))
        vp = ax.violinplot(smp[:, d], positions=[XS[d]], widths=0.78, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(fc); b.set_edgecolor(ec)
            b.set_alpha(1.0); b.set_linewidth(0.8); b.set_zorder(2)
        ax.plot([XS[d]] * 2, [lo[d], hi[d]], color="k", lw=1.3, zorder=3)
        ax.plot(XS[d], mean[d], "o", ms=4.6, mfc="white", mec="k", mew=0.8, zorder=4)
        ax.plot(XS[d], z_true[d], "*", ms=10.5, color="#c0392b", mec="white", mew=0.5, zorder=5)
        ax.text(XS[d], 1.035, f"{p:.2f}", ha="center", va="bottom", fontsize=7, color="0.15")
    ax.set_xticks(XS); ax.set_xticklabels(LAB, fontsize=9)
    ax.set_ylim(LB, 1.09); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_title(title, fontsize=8.6, pad=9)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    tr = ax.get_xaxis_transform()
    for x0, x1, txt in ((-0.3, 4.3, "deck zones"), (5.4, 7.0, "bearings")):
        ax.plot([x0, x1], [-0.15, -0.15], color="0.5", lw=0.7, transform=tr, clip_on=False)
        ax.text(0.5 * (x0 + x1), -0.17, txt, ha="center", va="top", fontsize=7.5, color="0.35", transform=tr)


def legend_handles():
    """Legend of the decision-violin figures (built after the threshold Z_C is set)."""
    return [Patch(fc=COL_H, ec=EDGE_H, label=r"No significant damage: $P_d<0.1$"),
            Patch(fc=COL_U, ec=EDGE_U, label=r"Uncertain: $0.1\leq P_d\leq0.9$"),
            Patch(fc=COL_D, ec=EDGE_D, label=r"Significant damage: $P_d>0.9$"),
            Patch(fc="#fde3e1", ec="none", label=r"Significant damage region, $z<\tau$"),
            Line2D([], [], color="k", lw=1.3, label="95% credible interval"),
            Line2D([], [], ls="", marker="o", mfc="white", mec="k", label=r"Posterior mean $\hat z_d$"),
            Line2D([], [], ls="", marker="*", ms=10, color="#c0392b", label=r"True value $z_d^{\mathrm{true}}$")]


LEGEND = legend_handles()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--particles", type=int, default=4000)
    ap.add_argument("--violins-only", action="store_true", help="only the composite decision-violin figures")
    ap.add_argument("--set", default="DoE30", choices=sorted(CASE_SETS), help="run, data set and selected cases")
    ap.add_argument("--tau", type=float, default=None, help="decision threshold (default: the one of the case set)")
    args = ap.parse_args()
    global RUN, DATA, OUT, CASES, FIG_CASES, Z_C, LEGEND, SPLIT
    cs = CASE_SETS[args.set]
    RUN, DATA, CASES, FIG_CASES = cs["run"], cs["data"], cs["cases"], cs["fig"]
    Z_C = args.tau if args.tau is not None else cs["tau"]
    SPLIT = cs.get("split", "test_fine")
    LEGEND = legend_handles()
    OUT = os.path.join(RUN, {"test_fine": "cases", "test": "cases_testA", "test_fine_uniform": "cases_testA50"}[SPLIT])
    os.makedirs(OUT, exist_ok=True)
    info = np.load(os.path.join(RUN, "Problem_info.npy"), allow_pickle=True).item()
    gamma2, obs = float(info["gamma"]) ** 2, info["obs_types"]
    preds = np.load(os.path.join(RUN, PRED_FILE[SPLIT]), allow_pickle=True).item()
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
    pdam = (norm.cdf((np.log(Z_SIG) - mu) / s) - norm.cdf(a)) / Zn     # P(z_d < 0.80 | m), closed form

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
    # contrast for the failure cases (not plotted): the same structures in the 10-element test data (test set A), where
    # the same equivalent damage is spread uniformly over each zone
    preds10 = np.load(os.path.join(RUN, "Test_predicted_props.npy"), allow_pickle=True).item()
    for pos in FIG_CASES["fail"]:
        mu10, s10 = preds10["test_logz_loc"][pos].astype(float), preds10["test_logz_scale"][pos].astype(float)
        a10, b10 = (np.log(LB) - mu10) / s10, -mu10 / s10
        Z10 = norm.cdf(b10) - norm.cdf(a10)
        m10 = np.exp(mu10 + s10 ** 2 / 2) * (norm.cdf(b10 - s10) - norm.cdf(a10 - s10)) / Z10
        print(f"  case {pos} in the 10-element data (uniform damage in each zone): posterior means {np.round(m10, 3)}")

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm", "pdf.fonttype": 42})  # TrueType, no Type 3
    for group, fname in (("well", "Fig_cases_well"), ("fail", "Fig_cases_fail")):
        sel = FIG_CASES[group]
        if len(sel) == 4:
            fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4), sharey=True)
            rect, pad = (0, 0.1, 1, 1), 3.0
        else:
            fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=True)
            rect, pad = (0, 0.17, 1, 1), 1.0
        axs = np.ravel(axs)
        for k, (ax, pos) in enumerate(zip(axs, sel)):
            decision_violins(ax, samples[pos], z[pos], mean[pos], lo[pos], hi[pos],
                             f"({chr(97 + k)}) Test case {pos}" + "\n" + CASES[pos][1], pdam[pos])
        for ax in axs[::2]:
            ax.set_ylabel("Stiffness factor $z$")
        fig.legend(handles=LEGEND, loc="lower center", ncol=4, fontsize=7.2, frameon=False)
        fig.tight_layout(rect=rect, h_pad=pad)
        for e in ("png", "pdf"):
            fig.savefig(os.path.join(OUT, f"{fname}.{e}"), dpi=300)
        plt.close(fig)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()
