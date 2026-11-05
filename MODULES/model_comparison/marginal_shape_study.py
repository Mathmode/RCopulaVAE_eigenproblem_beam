#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/model_comparison/marginal_shape_study.py

Why do more Kumaraswamy components help when the exact posteriors are unimodal? (Results_analysis, Sec. 9.)

A Kumaraswamy density f(u) = a b u^(a-1) (1 - u^a)^(b-1) with a, b > 1 (the range of the encoder,
GC_KSmix_architectures.positive_shape_parameter) has one mode and vanishes at both ends of the support. The marginal of an
exact posterior is different when the likelihood is still high at a bound of the prior: it is a smooth curve cut by the
bound, with a positive density there. The script measures how much this matters, on exact (SMC) marginals of the
benchmarks and of the test structures drawn from the prior (cached in <data_dir>/reference_posteriors/; test set A):

  1. Best fit of every exact marginal by each family, independent of the encoder (maximum likelihood on the SMC samples):
     one Kumaraswamy, a mixture of K = 2, 3, 5, and a truncated Gaussian. E_p[log q] differences are differences of
     KL(p || q) and need no density estimate of p.
  2. The trained encoders: E_p[log q_d] of the marginals of each run on the same samples, and the bound behaviour of
     the K = 1 run.
  3. Every marginal is classified by its shape (density at the bounds relative to its peak, position of the mode).
  4. Figure with representative marginals: exact, K = 1, K = 3 with its weighted components, truncated Gaussian.

No training; about 30 minutes on 6 cores the first time (fits kept in marginals.csv, --refit repeats them).
Results: Output/model_comparison/<data set>/marginal_shapes/
    python MODULES/model_comparison/marginal_shape_study.py
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

import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.special import betaln, expit, ndtr, softmax
from scipy.stats import skew

from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.posterior_families import family_from_preds
from MODULES.review29oct_fixed.reference_posterior import compute_reference_posterior, load_problem_config
from MODULES.review29oct_fixed.plotting import kde_1d

LOG_MIN = 1e-300
CLASSES = {"cut_mode": "cut by a bound, interior mode", "pile": "piled against a bound (no interior mode)",
           "interior": "interior bell, far from the bounds"}


# =============================================================================
# 1. FAMILIES OF MARGINALS, ON u = (z - lbound) / (1 - lbound)
# =============================================================================
def ks_logpdf(u, a, b, w):
    """log density of a Kumaraswamy mixture at u (S,); a, b, w (K,)."""
    uu = u[None, :]
    a_, b_, w_ = a[:, None], b[:, None], w[:, None]
    log_ua = a_ * np.log(uu)
    log1m = np.log(-np.expm1(log_ua))                                  # log(1 - u^a), stable next to the bounds
    log_comp = np.log(w_ + LOG_MIN) + np.log(a_) + np.log(b_) + (a_ - 1.0) * np.log(uu) + (b_ - 1.0) * log1m
    m = np.max(log_comp, axis=0)
    return m + np.log(np.sum(np.exp(log_comp - m), axis=0))


def tn_logpdf(u, mu, sig):
    """log density of a Gaussian (mu, sig) truncated to [0, 1] at u (S,)."""
    x = (u - mu) / sig
    z = np.maximum(ndtr((1.0 - mu) / sig) - ndtr(-mu / sig), 1e-300)
    return -0.5 * x ** 2 - 0.5 * np.log(2.0 * np.pi) - np.log(sig) - np.log(z)


def fit_ks(u, K, rng, restarts):
    """Maximum likelihood of a K-component Kumaraswamy mixture with the range of the encoder (a, b > 1.0009)."""
    lo = np.log(1e-3)

    def unpack(p):
        return 1.0 + np.exp(p[:K]) + 1e-3, 1.0 + np.exp(p[K:2 * K]) + 1e-3, softmax(np.append(p[2 * K:], 0.0))

    def nll(p):
        a, b, w = unpack(p)
        return -np.mean(ks_logpdf(u, a, b, w))

    bounds = [(lo, 7.0)] * (2 * K) + [(-8.0, 8.0)] * (K - 1)
    best = None
    for _ in range(restarts):
        p0 = np.concatenate([rng.uniform(0.0, 4.0, K), rng.uniform(-3.0, 4.0, K), rng.normal(0.0, 1.0, K - 1)])
        r = minimize(nll, p0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun, unpack(best.x)


def fit_tn(u):
    """Maximum likelihood of a truncated Gaussian on [0, 1] with the range of the encoder: mu in (0, 1)."""
    def nll(p):
        return -np.mean(tn_logpdf(u, expit(p[0]), np.exp(p[1])))

    m, s = np.clip(np.mean(u), 0.02, 0.98), max(np.std(u), 1e-3)
    starts = [(np.log(m / (1.0 - m)), np.log(s)), (3.0, np.log(s)), (-3.0, np.log(s))]
    best = min((minimize(nll, np.array(p0), method="L-BFGS-B", bounds=[(-10.0, 10.0), (np.log(1e-3), np.log(5.0))])
                for p0 in starts), key=lambda r: r.fun)
    return -best.fun, (expit(best.x[0]), np.exp(best.x[1]))


# =============================================================================
# 2. SHAPE OF AN EXACT MARGINAL
# =============================================================================
def classify_shape(z, lbound, n_grid=201):
    """Class of a marginal from its reflected KDE: 'cut' = density at a bound >= 25 % of the peak."""
    grid = np.linspace(lbound, 1.0, n_grid)
    pdf = kde_1d(z, grid, lbound)
    peak, mode = np.max(pdf), grid[int(np.argmax(pdf))]
    at_lo, at_hi = pdf[0] / peak, pdf[-1] / peak
    margin = 0.03 * (1.0 - lbound)
    if at_lo < 0.25 and at_hi < 0.25:
        return "interior", at_lo, at_hi, mode
    near_bound = (mode <= lbound + margin) if at_lo >= at_hi else (mode >= 1.0 - margin)
    return ("pile" if near_bound else "cut_mode"), at_lo, at_hi, mode


def _analyse(job):
    """Fits of one exact marginal."""
    key, z, lbound, seed = job
    u = np.clip((z - lbound) / (1.0 - lbound), 1e-6, 1.0 - 1e-6)
    rng = np.random.default_rng(seed)
    cls, at_lo, at_hi, mode = classify_shape(z, lbound)
    row = {"key": key, "class": cls, "density_at_lower": at_lo, "density_at_upper": at_hi, "mode": mode}
    row["ll_tn"], _ = fit_tn(u)
    for K, restarts in ((1, 2), (2, 4), (3, 6), (5, 8)):
        row[f"ll_ks{K}"], _ = fit_ks(u, K, rng, restarts)
    return row


def _best_single_ks(job):
    """Parameters of the best single Kumaraswamy of one exact marginal."""
    key, z, lbound, seed = job
    u = np.clip((z - lbound) / (1.0 - lbound), 1e-6, 1.0 - 1e-6)
    _, (a, b, _) = fit_ks(u, 1, np.random.default_rng(seed), 3)
    return key, float(a[0]), float(b[0])


def ks_skewness(a, b):
    """Skewness of a Kumaraswamy distribution from its raw moments E[u^n] = b B(1 + n / a, b)."""
    m1, m2, m3 = (np.exp(np.log(b) + betaln(1.0 + n / a, b)) for n in (1, 2, 3))
    var = m2 - m1 ** 2
    return (m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3) / var ** 1.5


def count_modes(z, lbound, n_grid=401, prominence=0.10):
    """Modes of the reflected KDE of z with a prominence of at least `prominence` of the peak (bounds included)."""
    pdf = kde_1d(z, np.linspace(lbound, 1.0, n_grid), lbound)
    return len(find_peaks(np.concatenate(([0.0], pdf, [0.0])), prominence=prominence * np.max(pdf))[0])


def shape_statistics(df, samples, lb, workers):
    """Adds moments and number of modes of the exact marginals and the moments of the best single Kumaraswamy."""
    jobs = [((int(p), int(d)), samples[int(p)][:, int(d)], lb, 1000 * int(p) + int(d)) for p, d in zip(df["pos"], df["d"])]
    with Pool(workers) as pool:
        fits = {k: (a, b) for k, a, b in pool.map(_best_single_ks, jobs, chunksize=16)}
    rows = []
    for (p, d), z, _, _ in jobs:
        u = (z - lb) / (1.0 - lb)
        a, b = fits[(p, d)]
        rows.append({"mean_u": u.mean(), "skew_exact": skew(z), "skew_ln_z": skew(np.log(z)), "a_ks1": a, "b_ks1": b,
                     "skew_ks1": ks_skewness(a, b), "n_modes": count_modes(z, lb)})
    return pd.concat([df.drop(columns=[c for c in rows[0] if c in df]).reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def report(df, fams, lb, out_dir):
    """Tables of the study: gaps by class and by location, decomposition of the K = 1 deficit, density at the bound."""
    fit_cols = ["ll_ks1", "ll_ks2", "ll_ks3", "ll_tn"]
    table = []
    for cls in list(CLASSES) + ["all"]:
        d_ = df[df["class"] == cls] if cls != "all" else df
        row = {"class": CLASSES.get(cls, "all marginals"), "n": len(d_), "share": len(d_) / len(df)}
        row.update({f"best {c[3:]}": (d_["ll_ks5"] - d_[c]).mean() for c in fit_cols})
        row.update({f"trained {k}": (d_["ll_trained_KS3"] - d_[f"ll_trained_{k}"]).mean() for k in ("KS1", "TN", "GMM")})
        table.append(row)
    summary = pd.DataFrame(table)
    summary.to_csv(os.path.join(out_dir, "summary_by_class.csv"), index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\nKL gaps [nats per marginal]. best: KL(p || best member of the family) - KL(p || best K = 5 mixture).\n"
          "trained: KL(p || trained marginal) - KL(p || trained K = 3 marginal).")
    print(summary.round(4).to_string(index=False))

    bell = df[df["class"] == "interior"]
    rows = []
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        g = bell[(bell["mean_u"] >= lo) & (bell["mean_u"] < hi)]
        if len(g):
            rows.append({"mean of u": f"[{lo}, {hi})", "n": len(g), "skew exact": g["skew_exact"].median(),
                         "skew best K=1": g["skew_ks1"].median(), "best K=1 gap": (g["ll_ks5"] - g["ll_ks1"]).mean(),
                         "best TN gap": (g["ll_ks5"] - g["ll_tn"]).mean(), "trained K=1 gap": (g["ll_trained_KS3"] - g["ll_trained_KS1"]).mean()})
    by_loc = pd.DataFrame(rows)
    by_loc.to_csv(os.path.join(out_dir, "interior_bells_by_location.csv"), index=False, float_format="%.4f")
    print("\nInterior bells by location (skewness: median; gaps as above)")
    print(by_loc.round(4).to_string(index=False))
    print(f"Interior bells: skewness of z {bell['skew_exact'].median():+.3f}, of ln z {bell['skew_ln_z'].median():+.3f}; "
          f"|skew ln z| < |skew z| in {np.mean(np.abs(bell['skew_ln_z']) < np.abs(bell['skew_exact'])):.0%}; correlation of "
          f"|skew exact - skew K=1| with the best K=1 gap {np.corrcoef(np.abs(bell['skew_exact'] - bell['skew_ks1']), bell['ll_ks5'] - bell['ll_ks1'])[0, 1]:.2f}")

    per = 7.0                                                # sums over the 7 parameters of a scenario
    total = per * (df["ll_trained_KS3"] - df["ll_trained_KS1"]).mean()
    family = per * (df["ll_ks3"] - df["ll_ks1"]).mean()
    amort = {k: per * (df[c] - df[f"ll_trained_{k}"]).mean() for k, c in (("KS1", "ll_ks1"), ("KS3", "ll_ks3"), ("TN", "ll_tn"))}
    print(f"\nDeficit of trained K = 1 against trained K = 3 [nats per scenario]: {total:.3f} = family limit (best K = 3 - best "
          f"K = 1) {family:.3f} ({family / total:.0%}) + excess of the training gap {amort['KS1'] - amort['KS3']:.3f}.")
    print("Training gap (best member - trained encoder) [nats per scenario]: " + ", ".join(f"{k} {v:.3f}" for k, v in amort.items()))

    print(f"\nExact marginals with more than one mode of prominence >= 10 % of the peak: {int((df['n_modes'] > 1).sum())} of {len(df)}")
    cut = df[(df["class"] == "cut_mode") & (df["density_at_upper"] >= 0.25)]
    grid = np.linspace(0.001, 0.999, 400)
    ratios = {k: [] for k in ("KS1", "KS3", "TN")}
    for p, d in zip(cut["pos"], cut["d"]):
        for k in ratios:
            pdf = fams[k].pdf_grid(slice(int(p), int(p) + 1), grid)[0, int(d)]
            ratios[k].append(np.interp(0.99, grid, pdf) / np.max(pdf))
    print(f"Marginals cut at z = 1 with an interior mode ({len(cut)}): exact density at the bound / peak, mean "
          f"{cut['density_at_upper'].mean():.2f}; trained density at u = 0.99 / peak, median: "
          + ", ".join(f"{k} {np.median(v):.2f}" for k, v in ratios.items()))
    und = np.load(os.path.join(C.DATA_DIR, "alphas_test.npy")) >= C.DAMAGE_THRESHOLD
    b1 = fams["KS1"].b[:, 0, :]
    print(f"K = 1 run, test set A: median b of the undamaged parameters {np.median(b1[und]):.2f}, of the damaged {np.median(b1[~und]):.2f}")


# =============================================================================
# 3. FIGURE
# =============================================================================
def pick_example(df, cond, how="max"):
    """Benchmark marginal that meets cond (any scenario if no benchmark does): largest or median gap of the best K = 1."""
    pool = df[cond & (df["subset"] == "benchmark")]
    pool = pool if len(pool) else df[cond]
    gap = pool["ll_ks5"] - pool["ll_ks1"]
    i = gap.idxmax() if how == "max" else (gap - gap.median()).abs().idxmin()
    return df.loc[i]


def plot_study(df, samples, fams, lb, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from MODULES.review29oct_fixed.plotting import PAPER_RC, save_figure

    colors = {"KS1": "#FF7F0E", "KS3": "#1F77B4", "TN": "#D62728"}
    examples = [("Bell in the upper half", pick_example(df, (df["class"] == "interior") & (df["mean_u"] >= 0.5))),
                ("Cut by the bound $z=1$", pick_example(df, (df["class"] == "cut_mode") & (df["density_at_upper"] >= 0.25))),
                ("Piled against $z=1$", pick_example(df, (df["class"] == "pile") & (df["density_at_upper"] >= 0.25), "median")),
                ("At the lower bound", pick_example(df, (df["density_at_lower"] >= 0.25) & (df["class"] != "interior")))]
    z_true = np.load(os.path.join(C.DATA_DIR, "alphas_test.npy"))
    with plt.style.context(["default", PAPER_RC]):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
        panels = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]
        z = np.linspace(lb, 1.0, 500)
        u = np.clip((z - lb) / (1.0 - lb), 1e-6, 1.0 - 1e-6)
        for j, (ax, (title, r)) in enumerate(zip(panels, examples)):
            p, d = int(r["pos"]), int(r["d"])
            ax.hist(samples[p][:, d], bins=40, range=(lb, 1.0), density=True, color="0.85", label="exact posterior (SMC)")
            ks3 = fams["KS3"]
            for k in range(ks3.a.shape[1]):
                a, b, w = ks3.a[p, k, d], ks3.b[p, k, d], ks3.w[p, k, d]
                comp = w * a * b * u ** (a - 1.0) * (1.0 - u ** a) ** (b - 1.0) / (1.0 - lb)
                ax.plot(z, comp, ":", color=colors["KS3"], lw=1.2, label="its 3 weighted components" if k == 0 else None)
            for name, label, lw in (("KS3", "trained, $K=3$", 2.0), ("KS1", "trained, $K=1$", 1.7), ("TN", "trained, truncated Gaussian", 1.4)):
                ax.plot(z, fams[name].pdf_grid(slice(p, p + 1), u)[0, d] / (1.0 - lb), color=colors[name], lw=lw, label=label)
            ax.plot(z, np.exp(ks_logpdf(u, np.array([r["a_ks1"]]), np.array([r["b_ks1"]]), np.array([1.0]))) / (1.0 - lb), "--",
                    color=colors["KS1"], lw=1.2, label="best possible single KS")
            ax.axvline(z_true[p, d], color="k", ls="--", lw=1)
            ax.set(title=f"({'abcd'[j]}) {title}: P{p}, {C.PARAM_NAMES[d]}", xlabel="stiffness factor $z$", ylabel="density", xlim=(lb, 1.0))
            if j == 0:
                ax.legend(fontsize=8, loc="upper left")
        ax = axes[1, 1]
        bell = df[df["class"] == "interior"]
        ax.scatter(bell["mean_u"], bell["skew_exact"], s=10, color="0.35", label="exact marginal")
        ax.scatter(bell["mean_u"], bell["skew_ks1"], s=10, color=colors["KS1"], label="best possible single KS")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set(xlabel=r"mean of $u=(z-0.45)/0.55$", ylabel="skewness", title="(e) Bells: a single KS cannot set its skewness")
        ax.legend(fontsize=9)
        ax.grid(True, ls=":", alpha=0.5)
        ax = axes[1, 2]
        names = [("ll_ks1", "$K=1$"), ("ll_ks2", "$K=2$"), ("ll_ks3", "$K=3$"), ("ll_tn", "trunc. Gauss.")]
        width = 0.26
        for i, (cls, label) in enumerate((("interior", "bell"), ("cut_mode", "cut by a bound"), ("pile", "piled"))):
            g = df[df["class"] == cls]
            ax.bar(np.arange(len(names)) + (i - 1) * width, [(g["ll_ks5"] - g[c]).mean() for c, _ in names], width, label=label)
        ax.set_xticks(np.arange(len(names)), [n for _, n in names])
        ax.set(ylabel=r"KL gap to the best $K=5$ mixture [nats/marginal]", title="(f) What each family can do at best")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", ls=":", alpha=0.5)
        fig.tight_layout()
        save_figure(fig, os.path.join(out_dir, "Fig_marginal_shapes.png"))
        plt.close(fig)
    print("Examples: " + "; ".join(f"{t}: P{int(r['pos'])} {C.PARAM_NAMES[int(r['d'])]}" for t, r in examples))


# =============================================================================
# 4. MAIN
# =============================================================================
def find_run(name):
    for group in ("review29oct_fixed", "baseline_GaussCopula_GaussMarginals", "baseline_GaussMixture_K5",
                  "review29oct_fixed/baseline_GaussCopula_GaussMarginals", "review29oct_fixed/baseline_GaussMixture_K5"):
        path = os.path.join(C.PROJECT_ROOT, "Output", group, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(name)


def main():
    parser = argparse.ArgumentParser(description="Which shapes of the exact marginals need more than one KS component?")
    parser.add_argument("--data-dir", default=C.DATA_DIR)
    parser.add_argument("--ks1", default="KSVAE_fixed_03Nov-FINAL_7D_1KS_obs-fv_Gamma1.0_Ep1500")
    parser.add_argument("--ks3", default="KSVAE_fixed_03Nov-FINAL_7D_3KS_obs-fv_Gamma1.0_Ep1500")
    parser.add_argument("--tn", default="GCTN_03Nov-FINAL_7D_obs-fv_Gamma1.0_Ep1500")
    parser.add_argument("--gmm", default="GMM5full_03Nov-FINAL_7D_obs-fv_Gamma1.0_Ep1500")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--refit", action="store_true", help="repeat the fits even if marginals.csv exists")
    args, _ = parser.parse_known_args()

    data_dir = args.data_dir
    pcfg = load_problem_config(data_dir)
    lb = float(pcfg["lbound"])
    out_dir = os.path.join(C.PROJECT_ROOT, "Output", "model_comparison", os.path.basename(data_dir.rstrip("\\/")),
                           "marginal_shapes")
    os.makedirs(out_dir, exist_ok=True)
    labels = np.load(os.path.join(data_dir, "labels_test.npy"))
    z_true = np.load(os.path.join(data_dir, "alphas_test.npy")).astype(np.float64)
    prior_pos = np.flatnonzero(labels == list(C.SCENARIO_FRACTIONS).index("uniform"))
    positions = list(range(len(C.BENCHMARKS))) + [int(p) for p in prior_pos]
    samples = {p: compute_reference_posterior(data_dir, p, 1.0, None, 4000, split="test")[0].astype(np.float64) for p in positions}
    runs = {"KS1": find_run(args.ks1), "KS3": find_run(args.ks3), "TN": find_run(args.tn), "GMM": find_run(args.gmm)}
    fams = {k: family_from_preds(np.load(os.path.join(r, "Test_predicted_props.npy"), allow_pickle=True).item(), lb)
            for k, r in runs.items()}

    csv_path = os.path.join(out_dir, "marginals.csv")
    if os.path.exists(csv_path) and not args.refit:
        df = pd.read_csv(csv_path)
        print(f"Fits read from {csv_path} (--refit computes them again)")
    else:
        # --- best fit of every family to every exact marginal ---
        jobs = [((p, d), samples[p][:, d], lb, 1000 * p + d) for p in positions for d in range(C.N_DIMS)]
        print(f"{len(jobs)} exact marginals ({len(positions)} scenarios), {args.workers} workers")
        with Pool(args.workers) as pool:
            rows = pool.map(_analyse, jobs, chunksize=8)
        df = pd.DataFrame(rows)
        df["pos"] = [k[0] for k in df["key"]]
        df["d"] = [k[1] for k in df["key"]]
        df["parameter"] = [C.PARAM_NAMES[d] for d in df["d"]]
        df["z_true"] = [z_true[p, d] for p, d in zip(df["pos"], df["d"])]
        df["subset"] = ["benchmark" if p < len(C.BENCHMARKS) else "prior" for p in df["pos"]]
        # --- the trained encoders on the same samples ---
        for name, fam in fams.items():
            ll = np.zeros(len(df))
            for i, (p, d) in enumerate(zip(df["pos"], df["d"])):
                u = np.clip((samples[p][:, d] - lb) / (1.0 - lb), 1e-6, 1.0 - 1e-6)
                ll[i] = np.mean(np.log(fam.pdf_grid(slice(p, p + 1), u)[0, d] + 1e-300))
            df[f"ll_trained_{name}"] = ll
        df = df.drop(columns=["key"])
        df.to_csv(csv_path, index=False, float_format="%.5f")

    df = shape_statistics(df, samples, lb, args.workers)
    df.to_csv(csv_path, index=False, float_format="%.5f")
    report(df, fams, lb, out_dir)
    plot_study(df, samples, fams, lb, out_dir)
    print(f"[OK] {out_dir}")


if __name__ == "__main__":
    main()
