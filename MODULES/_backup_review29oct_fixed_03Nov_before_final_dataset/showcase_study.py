#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed/showcase_study.py

Curated scenarios that explain what the method can and cannot identify, with robust statistics: every scenario is
repeated with R independent as-built structures (mass and stiffness deviations) and measurement-noise draws, computed
with the refined 50-element model (test set B conditions). The VAE posterior is computed for all replicas (amortized,
no retraining), the exact posterior (SMC) for the first few, which are used to check the VAE and to draw the figures.

Scenarios (undamaged parameters 0.98):
  S01 healthy bridge                          false alarms
  S02 zone 2 = 0.60                           localized, severe
  S03 zone 3 = 0.70 (mid-span)                localized, mid-span zone (seen by modes 1, 3 and 5 only)
  S04 zone 4 = 0.85                           mild damage, detection limit
  S05 zones 2 and 3 = 0.65                    adjacent damage: negative correlation (trade-off)
  S06 zones 1 and 5 = 0.65                    symmetric damage: mode shapes separate left and right
  S07 left bearing = 0.55                     bearing alone: not observable with vertical bending modes
  S08 left bearing = 0.60, zone 1 = 0.65      bearing and end zone
  S09 zones 1, 3, 4 = 0.70, 0.60, 0.75        multiple damage
  S10 0.2 m defect at mid-span (stiffness x0.3)       localized defect inside zone 3
  S11 0.2 m defect next to the left support (x0.3)    defect where the bending curvature vanishes

Outputs in <run>/Showcase/:
  Fig_showcase_posteriors     exact posterior (red) vs VAE (filled) for the representative replica of every scenario
                              (among the replicas with an exact posterior, the one whose VAE posterior mean is closest
                              to the median of the posterior means of all replicas)
  Fig_showcase_robustness     VAE 95 % intervals and means of all replicas, with detection / false-alarm rates
  Fig_showcase_correlations   correlation matrices, exact vs VAE, for the scenarios with correlated parameters
  Fig_showcase_pairs          2-D posteriors of the most correlated pairs (exact vs VAE, 50 % and 90 % HPD)
  showcase_summary.csv        per scenario: detection, false alarms, coverage, interval width (VAE and exact)

    python MODULES/review29oct_fixed/showcase_study.py --run-folder KSVAE_fixed_01Nov-5modes_7D_3KS_obs-fv_Gamma1.0_Ep1500
"""

import os
import sys
import argparse

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from multiprocessing import Pool

from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.reference_posterior import load_problem_config
from MODULES.review29oct_fixed.localized_damage_study import (simulate_scenario, _exact_posterior, vae_posteriors,
                                                              plot_grid, HEALTHY, THRESHOLD)

SCENARIOS = [
    ("S01", "healthy bridge", {}, []),
    ("S02", "zone 2 = 0.60 (localized, severe)", {1: 0.60}, []),
    ("S03", "zone 3 = 0.70 (mid-span)", {2: 0.70}, []),
    ("S04", "zone 4 = 0.85 (mild)", {3: 0.85}, []),
    ("S05", "zones 2 and 3 = 0.65 (adjacent)", {1: 0.65, 2: 0.65}, []),
    ("S06", "zones 1 and 5 = 0.65 (symmetric)", {0: 0.65, 4: 0.65}, []),
    ("S07", "left bearing = 0.55", {5: 0.55}, []),
    ("S08", "left bearing = 0.60, zone 1 = 0.65", {5: 0.60, 0: 0.65}, []),
    ("S09", "zones 1, 3, 4 = 0.70, 0.60, 0.75", {0: 0.70, 2: 0.60, 3: 0.75}, []),
    ("S10", "0.2 m defect at mid-span (x0.3)", {}, [(2, [5], 0.30)]),
    ("S11", "0.2 m defect at the left support (x0.3)", {}, [(0, [0], 0.30)]),
]
CORR_SCENARIOS = ["S05", "S06", "S08", "S09"]
# physically meaningful pair of each scenario: the two damaged parameters (or damaged zone and bearing)
PAIRS = {"S05": (1, 2), "S06": (0, 4), "S08": (0, 5), "S09": (2, 3)}


def build():
    out = []
    for sid, label, mods, defect in SCENARIOS:
        z = np.full(C.N_DIMS, HEALTHY)
        for k, v in mods.items():
            z[k] = v
        out.append({"id": sid, "name": sid, "label": f"{sid}: {label}", "kind": "showcase", "z": z, "defect": defect})
    return out


def interval_stats(samples, z_ref):
    """95 % equal-tailed intervals; declared damaged = upper bound < threshold."""
    lo, hi = np.quantile(samples, [0.025, 0.975], axis=0)
    return lo, hi, samples.mean(0), hi < THRESHOLD, (z_ref >= lo) & (z_ref <= hi)


def plot_robustness(scen, vae, z_refs, lb, save_path):
    import matplotlib.pyplot as plt
    from MODULES.review29oct_fixed.plotting import fixed_style, save_figure, LABELS_7D

    @fixed_style
    def draw():
        n = len(scen)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.4 * nrows), sharey=True, squeeze=False)
        for a, (i, sc) in zip(axes.ravel(), enumerate(scen)):
            R = len(vae[i])
            offs = np.linspace(-0.32, 0.32, R)
            det = np.zeros(C.N_DIMS)
            cov = np.zeros(C.N_DIMS)
            for r in range(R):
                lo, hi, m, declared, inside = interval_stats(vae[i][r], z_refs[i][r])
                det += declared
                cov += inside
                for d in range(C.N_DIMS):
                    col = "steelblue" if d < C.N_ZONES else "#2CA02C"
                    a.plot([d + 1 + offs[r]] * 2, [lo[d], hi[d]], color=col, alpha=0.45, lw=1.1)
                    a.plot(d + 1 + offs[r], m[d], ".", color="k", ms=2.5)
            zr = np.mean(z_refs[i], axis=0)
            a.plot(np.arange(1, C.N_DIMS + 1), zr, "*", color="#FFD700", mec="black", ms=11, zorder=10)
            a.axhline(THRESHOLD, color="#D62728", ls=":", lw=1)
            truly = zr < THRESHOLD
            txt = []
            if truly.any():
                txt.append("detected: " + ", ".join(f"{LABELS_7D[d]} {det[d] / R:.0%}" for d in np.flatnonzero(truly)))
            fa = det[~truly].sum() / (R * max(1, (~truly).sum()))
            txt.append(f"false alarms: {fa:.1%} | coverage: {cov.sum() / (R * C.N_DIMS):.0%}")
            a.text(0.02, 0.03, "\n".join(txt), transform=a.transAxes, fontsize=8.5, va="bottom",
                   bbox=dict(fc="white", ec="0.7", alpha=0.9))
            a.set_title(sc["label"], fontsize=10.5)
            a.set_xticks(np.arange(1, C.N_DIMS + 1), LABELS_7D, fontsize=9)
            a.set_ylim(lb - 0.02, 1.02)
            a.axvline(5.5, color="0.5", ls="-.", lw=0.8)
            a.grid(axis="y", ls=":", alpha=0.5)
        for a in axes.ravel()[n:]:
            a.axis("off")
        for a in axes[:, 0]:
            a.set_ylabel("stiffness factor z")
        fig.suptitle(f"VAE 95 % credible intervals (bars) and posterior means (dots) of {len(vae[0])} independent "
                     f"replicas per scenario (as-built structure and noise), refined 50-element data; star = true "
                     f"value; dotted line = damage threshold {THRESHOLD}", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_figure(fig, save_path)
        plt.close(fig)
    draw()


def plot_correlations(scen, exact, vae, rep, save_path):
    import matplotlib.pyplot as plt
    from MODULES.review29oct_fixed.plotting import fixed_style, save_figure, LABELS_7D

    @fixed_style
    def draw():
        ids = [i for i, s in enumerate(scen) if s["id"] in CORR_SCENARIOS]
        fig, axes = plt.subplots(2, len(ids), figsize=(4.3 * len(ids), 8.2), squeeze=False)
        im = None
        for c, i in enumerate(ids):
            for r, (x, who) in enumerate(((exact[i][rep[i]], "exact posterior (SMC)"), (vae[i][rep[i]], "VAE"))):
                M = np.corrcoef(x.T)
                np.fill_diagonal(M, np.nan)
                a = axes[r, c]
                im = a.imshow(M, cmap="RdBu_r", vmin=-0.8, vmax=0.8)
                for (p, q), v in np.ndenumerate(M):
                    if np.isfinite(v):
                        a.text(q, p, f"{v:.2f}", ha="center", va="center", fontsize=7)
                a.set_xticks(range(C.N_DIMS), LABELS_7D, fontsize=8.5)
                a.set_yticks(range(C.N_DIMS), LABELS_7D, fontsize=8.5)
                a.set_title(f"{scen[i]['id']}, {who}", fontsize=10.5)
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="correlation coefficient")
        fig.suptitle("Posterior correlation matrices (representative replica): " +
                     "; ".join(scen[i]["label"] for i in ids), fontsize=10.5)
        save_figure(fig, save_path)
        plt.close(fig)
    draw()


def plot_pairs(scen, exact, vae, z_refs, rep, lb, save_path):
    import matplotlib.pyplot as plt
    from MODULES.review29oct_fixed.plotting import fixed_style, save_figure, LABELS_7D, density_2d, hpd_levels, RED, BLUE

    @fixed_style
    def draw():
        ids = [i for i, s in enumerate(scen) if s["id"] in CORR_SCENARIOS]
        fig, axes = plt.subplots(1, len(ids), figsize=(4.4 * len(ids), 4.6), squeeze=False)
        for a, i in zip(axes[0], ids):
            p, q = PAIRS[scen[i]["id"]]
            Rm = np.corrcoef(exact[i][rep[i]].T)
            means = np.array([v.mean(0) for v in vae[i]])
            a.plot(means[:, p], means[:, q], "o", color="0.45", ms=3.5, alpha=0.8, zorder=3)
            for x, col in ((exact[i][rep[i]], RED), (vae[i][rep[i]], BLUE)):
                X, Y, Z = density_2d(x[:, p], x[:, q], lb)
                lev = hpd_levels(Z, (0.9, 0.5))
                a.contour(X, Y, Z, levels=lev, colors=col, linestyles=["--", "-"][:len(lev)], linewidths=1.6)
            zr = z_refs[i][rep[i]]
            a.plot(zr[p], zr[q], "*", color="#FFD700", mec="black", ms=14, zorder=10)
            a.set_xlim(lb, 1)
            a.set_ylim(lb, 1)
            a.set_aspect("equal")
            a.set_xlabel(LABELS_7D[p])
            a.set_ylabel(LABELS_7D[q])
            a.set_title(f"{scen[i]['id']}: r = {Rm[p, q]:+.2f} (exact)", fontsize=11)
            a.grid(ls=":", alpha=0.5)
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([], [], color=RED, label="exact posterior (SMC)"),
                            Line2D([], [], color=BLUE, label="VAE"),
                            Line2D([], [], color="k", ls="-", label="50 % HPD"),
                            Line2D([], [], color="k", ls="--", label="90 % HPD"),
                            Line2D([], [], marker="*", color="#FFD700", mec="black", ls="", ms=12, label="true value"),
                            Line2D([], [], marker="o", color="0.45", ls="", ms=5,
                                   label="VAE posterior means of all replicas")],
                   loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle("Damaged pair of each scenario: posterior of the representative replica (contours) and posterior "
                     "means of all replicas (dots)", fontsize=12)
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        save_figure(fig, save_path)
        plt.close(fig)
    draw()


def main():
    parser = argparse.ArgumentParser(description="Curated showcase scenarios with replicas: exact and VAE posteriors.")
    parser.add_argument("--run-folder", type=str, required=True)
    parser.add_argument("--replicas", type=int, default=30, help="independent as-built structures + noise per scenario")
    parser.add_argument("--exact-replicas", type=int, default=5, help="replicas with the exact (SMC) posterior")
    parser.add_argument("--n-particles", type=int, default=4000)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--replot", action="store_true", help="redraw the figures from Showcase/showcase_posteriors.npz")
    args, _ = parser.parse_known_args()

    from MODULES.review29oct_fixed.postprocessing import load_run_info
    run_folder = args.run_folder if os.path.isabs(args.run_folder) else os.path.join(C.OUTPUT_DIR, args.run_folder)
    info = load_run_info(run_folder)
    data_dir = info["data_dir"]
    pcfg = load_problem_config(data_dir)
    lb = float(pcfg["lbound"])
    out_dir = os.path.join(run_folder, "Showcase")
    os.makedirs(out_dir, exist_ok=True)
    scen = build()
    R, RE = args.replicas, min(args.exact_replicas, args.replicas)
    print(f"=== Showcase: {len(scen)} scenarios x {R} replicas (exact posterior for {RE}), data set "
          f"{os.path.basename(data_dir)}, refined 50-element model ===")

    cache = os.path.join(out_dir, "showcase_posteriors.npz")
    if args.replot:
        d = np.load(cache, allow_pickle=True)
        vae, exact, z_refs = list(d["vae"]), list(d["exact"]), list(d["z_refs"])
        R, RE = vae[0].shape[0], exact[0].shape[0]
        return finish(scen, vae, exact, z_refs, R, RE, lb, out_dir)

    obs, z_refs = [], []
    for i, sc in enumerate(scen):
        zr = []
        for r in range(R):
            f, v, rr, z_ref = simulate_scenario(sc, pcfg, args.seed + 1000 * i + r)
            obs.append((i, r, f, v, rr))
            zr.append(z_ref)
        z_refs.append(np.array(zr))

    print("--- VAE posteriors of all replicas ---")
    q_all = vae_posteriors(run_folder, data_dir, tuple(np.stack([o[k] for o in obs]) for k in (2, 3, 4)),
                           args.n_samples, info)
    vae = [[None] * R for _ in scen]
    for (i, r, *_), q in zip(obs, q_all):
        vae[i][r] = q

    workers = args.workers or max(1, (os.cpu_count() or 2) // 2)
    jobs = [(k, data_dir, o[2], o[3], o[4], args.n_particles) for k, o in enumerate(obs) if o[1] < RE]
    print(f"--- exact posteriors of {len(jobs)} observations (SMC, {workers} workers) ---")
    with Pool(min(workers, len(jobs))) as pool:
        res = dict(pool.map(_exact_posterior, jobs))
    exact = [[None] * RE for _ in scen]
    for k, s in res.items():
        i, r = obs[k][0], obs[k][1]
        exact[i][r] = s

    np.savez_compressed(cache, vae=np.array([[v[:2000] for v in vs] for vs in vae], dtype=np.float32),
                        exact=np.array([[e[:2000] for e in es] for es in exact], dtype=np.float32),
                        z_refs=np.array(z_refs))
    finish(scen, vae, exact, z_refs, R, RE, lb, out_dir)


def finish(scen, vae, exact, z_refs, R, RE, lb, out_dir):
    rows = []
    for i, sc in enumerate(scen):
        row = {"scenario": sc["id"], "description": sc["label"]}
        for tag, post, n in (("VAE", vae[i], R), ("exact", exact[i], RE)):
            truly = z_refs[i][0] < THRESHOLD
            det = np.zeros(C.N_DIMS)
            cov = np.zeros(C.N_DIMS)
            width = []
            for r in range(n):
                lo, hi, m, declared, inside = interval_stats(post[r], z_refs[i][r])
                det += declared
                cov += inside
                width.append(hi - lo)
            row[f"{tag}_detection_rate"] = float(det[truly].sum() / (n * truly.sum())) if truly.any() else np.nan
            row[f"{tag}_false_alarm_rate"] = float(det[~truly].sum() / (n * (~truly).sum()))
            row[f"{tag}_coverage95"] = float(cov.sum() / (n * C.N_DIMS))
            row[f"{tag}_width95_deck"] = float(np.mean(np.array(width)[:, :C.N_ZONES]))
            row[f"{tag}_width95_bearings"] = float(np.mean(np.array(width)[:, C.N_ZONES:]))
        diff = [np.abs(vae[i][r].mean(0) - exact[i][r].mean(0)).max() for r in range(RE)]
        ratio = [vae[i][r].std(0) / exact[i][r].std(0) for r in range(RE)]
        row["VAE_vs_exact_max_mean_diff"] = float(np.max(diff))
        row["VAE_vs_exact_std_ratio_min"] = float(np.min(ratio))
        row["VAE_vs_exact_std_ratio_max"] = float(np.max(ratio))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "showcase_summary.csv"), index=False, float_format="%.4f")

    # representative replica of each scenario: among those with the exact posterior, the one whose VAE posterior mean
    # is closest to the median of the posterior means of all replicas (the figures state this choice)
    rep = []
    for i in range(len(scen)):
        means = np.array([v.mean(0) for v in vae[i]])
        med = np.median(means, axis=0)
        rep.append(int(np.argmin([np.linalg.norm(means[r] - med) for r in range(RE)])))
    for i, sc in enumerate(scen):
        sc["z_ref"] = z_refs[i][rep[i]]
    refs0 = [exact[i][rep[i]] for i in range(len(scen))]
    vae0 = [vae[i][rep[i]] for i in range(len(scen))]
    plot_grid(scen, list(range(len(scen))), refs0, vae0, lb, 4, 3,
              "Showcase scenarios, representative replica: exact posterior (red, dashed) and VAE (filled), "
              "refined 50-element data", os.path.join(out_dir, "Fig_showcase_posteriors.png"), panel_w=5.0)
    plot_robustness(scen, vae, z_refs, lb, os.path.join(out_dir, "Fig_showcase_robustness.png"))
    plot_correlations(scen, exact, vae, rep, os.path.join(out_dir, "Fig_showcase_correlations.png"))
    plot_pairs(scen, exact, vae, z_refs, rep, lb, os.path.join(out_dir, "Fig_showcase_pairs.png"))

    pd.set_option("display.width", 220)
    print(df[["scenario", "VAE_detection_rate", "VAE_false_alarm_rate", "VAE_coverage95", "exact_coverage95",
              "VAE_width95_deck", "VAE_width95_bearings", "VAE_vs_exact_max_mean_diff",
              "VAE_vs_exact_std_ratio_min", "VAE_vs_exact_std_ratio_max"]].round(3).to_string(index=False))
    print(f"\nFigures and CSV in {out_dir}")


if __name__ == "__main__":
    main()
