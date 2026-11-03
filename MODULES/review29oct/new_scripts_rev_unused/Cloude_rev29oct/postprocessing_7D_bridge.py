# -*- coding: utf-8 -*-
"""
postprocessing_7D_bridge.py -- STEP (c): metrics and figures of a trained run.

Spyder : set RUN_NAME below, F5.
Console: python postprocessing_7D_bridge.py --run <run folder> [--n_particles 4000] [--skip_reference]

Uses only files written by main_KS_rev29oct.py. Nothing is tuned on test data.

Estimated posterior q(theta | y): samples / exact marginals of the trained encoder.
Reference posterior p(theta | y): adaptive tempered SMC with the same prior,
  likelihood and inference FE model (reference_posterior.py), cached in
  Data/reference_posteriors/ because it does not depend on the network.
q + IS: self-normalized importance sampling with q as proposal,
  w ∝ p(y | theta) p(theta) / q(theta | y); its effective sample size (ESS)
  measures how well q covers the posterior. It is reported separately from q.

Output/<run>/figures, Output/<run>/metrics:
  Fig_training_history          -ELBO, KL, expected log-likelihood, tau, validation NLL
  Fig_calibration               empirical vs nominal coverage of central intervals (R1.3)
  Fig_pred_vs_true_<set>        posterior median and 95 % interval vs truth
  Fig_detection_ROC             damage detection; false alarms on intact parameters (R1.6)
  Fig_location_severity_<set>   single-location damage by location and severity
  Fig_mass_sensitivity          coverage / width / RMSE vs unmodelled mass error (R3.1)
  bench_<variant>/<case>_*      q vs reference: 1-D marginals, active-subspace contours,
                                full 7x7 corner, profile along the beam
  metrics/*.csv, metrics/summary.json
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config as C  # noqa: E402  (sets KERAS_BACKEND before tensorflow is imported)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import gaussian_kde, wasserstein_distance

import GC_KSmix_functions as KS
from GC_KSmix_models import params_to_tf
from preprocessing_and_eigen import load_benchmarks, load_coarse_model
from priors import build_prior
from reference_posterior import LogLikelihood, smc_posterior, importance_refine

DT = tf.float64
RUN_NAME = ""                       # Spyder: folder name in Output/ ("" -> most recent run)
LEVELS = np.array([0.10, 0.25, 0.50, 0.68, 0.80, 0.90, 0.95, 0.99])
IS_SETS = ["test_coarse", "test_fine"]
MASS_SETS = {"test_fine_mass0": 0.0, "test_fine": 0.02, "test_fine_mass5": 0.05}
CALIBRATION_SETS = ["test_coarse", "test_fine", "test_fine_bc", "test_fine_mass0", "test_fine_mass5", "ood_noise2"]
ROC_SETS = ["test_coarse", "test_fine", "test_fine_bc", "test_fine_balanced", "ood_four_sites", "ood_noise2"]
BENCH_TAGS = ("coarse", "fine", "fine_bc")   # benchmark data: inference model, fine model, fine + baseline corr.
COL_Q, COL_REF, COL_IS, COL_TRUE = "#1f77b4", "#222222", "#2ca02c", "#d62728"


# =============================================================================
# Helpers
# =============================================================================
def style():
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "mathtext.fontset": "cm",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "savefig.bbox": "tight", "legend.fontsize": 9})


def save(fig, path, show=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    if show:
        plt.show()
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def latest_run():
    runs = [p for p in C.OUTPUT_DIR.glob("*") if (p / "encoder.weights.h5").exists()]
    if not runs:
        raise FileNotFoundError("No trained run in Output/ - run main_KS_rev29oct.py first.")
    return max(runs, key=lambda p: p.stat().st_mtime)


def bounds_tf():
    lo, hi = C.param_bounds()
    return tf.constant(lo, DT), tf.constant(hi, DT)


def sample_pred(pred, n, idx=None, chunk=200, with_log_q=False):
    """n samples of q for every case of a prediction file -> (N, n, D) [and log q (N, n)]."""
    lo, hi = bounds_tf()
    idx = np.arange(len(pred["a"])) if idx is None else np.asarray(idx)
    th, lq = [], []
    for i in range(0, len(idx), chunk):
        out = KS.sample(params_to_tf(pred, idx[i:i + chunk]), n, lo, hi, return_log_q=with_log_q)
        if with_log_q:
            th.append(out[0].numpy())
            lq.append(out[1].numpy())
        else:
            th.append(out.numpy())
    return (np.concatenate(th), np.concatenate(lq)) if with_log_q else np.concatenate(th)


def exact_marginals(pred, probs):
    """Exact marginal quantiles (len(probs), N, D) and P(theta < DAMAGE_THRESHOLD) (N, D)."""
    lo, hi = bounds_tf()
    p = params_to_tf(pred)
    q = np.stack([KS.marginal_quantile(p, float(pr), lo, hi).numpy() for pr in probs])
    x_thr = (C.DAMAGE_THRESHOLD - C.LBOUND) / (C.UBOUND - C.LBOUND)
    _, cdf, _ = KS.mixture_eval(tf.fill(tf.shape(p["a"])[:2], tf.constant(x_thr, DT)),
                                p["logw"], p["a"], p["b"])
    return q, cdf.numpy()


def weighted_quantile(x, w, probs):
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    cw /= cw[-1]
    return np.interp(probs, cw - 0.5 * w[order] / w.sum(), x[order])


def crps(samples, y):
    """CRPS from samples (N, n, D) and truth (N, D)."""
    n = samples.shape[1]
    t1 = np.mean(np.abs(samples - y[:, None]), axis=1)
    ss = np.sort(samples, axis=1)
    coef = (2 * np.arange(1, n + 1) - n - 1)[None, :, None]
    return t1 - np.sum(coef * ss, axis=1) / n ** 2


def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def roc_curve(score, label, n=201):
    thr = np.linspace(1, 0, n)
    return (np.array([np.mean(score[~label] >= t) for t in thr]),
            np.array([np.mean(score[label] >= t) for t in thr]))


def kde_1d(x, grid, w=None, lo=C.LBOUND, hi=C.UBOUND):
    """Gaussian KDE with reflection at the support bounds (no mass leaks outside [lo, hi])."""
    if np.std(x) < 1e-9:
        return np.zeros_like(grid)
    k = gaussian_kde(x, weights=w)
    return k(grid) + k(2 * lo - grid) + k(2 * hi - grid)


def energy_distance(x, y, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = x[rng.choice(len(x), min(n, len(x)), replace=False)]
    y = y[rng.choice(len(y), min(n, len(y)), replace=False)]
    d = lambda a, b: np.mean(np.linalg.norm(a[:, None] - b[None], axis=-1))  # noqa: E731
    return 2 * d(x, y) - d(x, x) - d(y, y)


# =============================================================================
# Training history
# =============================================================================
def plot_history(run_dir, show=False):
    h = C.load_json(Path(run_dir) / "history.json")
    ep = np.array(h["epoch"])
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))
    axs[0].plot(ep, h["train_neg_elbo"], label="train")
    if "val_neg_elbo" in h:
        axs[0].plot(h["val_elbo_epoch"], h["val_neg_elbo"], ".-", label="validation")
    axs[0].set_ylabel("negative ELBO (tau = 1)")
    axs[0].set_yscale("symlog")
    axs[1].plot(ep, h["train_kl"], label="KL(q || prior)")
    axs[1].plot(ep, -np.array(h["train_exp_log_lik"]), label=r"$-E_q[\log p(y|\theta)]$")
    axs[1].set_yscale("symlog")
    axs[2].plot(ep, h["val_nll"], color="C2")
    axs[2].set_ylabel(r"validation $-\log q(\theta^*|y)$")
    axs[3].semilogy(ep, h["tau"], color="C3", label="likelihood weight tau")
    axs[3].semilogy(ep, h["lr"], color="C4", label="learning rate")
    for ax in axs:
        ax.set_xlabel("epoch")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, Path(run_dir) / "figures" / "Fig_training_history.png", show)


# =============================================================================
# Test-set evaluation
# =============================================================================
def evaluate_set(run_dir, name, n_samples, loglik=None, n_is=0):
    pred = dict(np.load(Path(run_dir) / f"pred_{name}.npz"))
    y = pred["theta"]
    probs = np.concatenate([(1 - LEVELS) / 2, (1 + LEVELS) / 2, [0.5]])
    q, p_dmg = exact_marginals(pred, probs)
    nl = len(LEVELS)
    lo_q, hi_q, med = q[:nl], q[nl:2 * nl], q[-1]
    s = sample_pred(pred, n_samples)
    r = dict(name=name, theta=y, scenario=pred["scenario"], damaged_mask=pred["damaged_mask"],
             nll=-pred["log_q_true"], samples=s, mean=s.mean(1), median=med, sd=s.std(1),
             lower95=lo_q[LEVELS == 0.95][0], upper95=hi_q[LEVELS == 0.95][0],
             coverage=np.array([((y >= lo_q[k]) & (y <= hi_q[k])).mean(0) for k in range(nl)]),
             width=np.array([(hi_q[k] - lo_q[k]).mean(0) for k in range(nl)]),
             crps=crps(s, y), p_damage=p_dmg, damaged=y < C.DAMAGE_THRESHOLD)
    if n_is > 0 and loglik is not None:
        r.update(importance_refined_metrics(pred, loglik, n_is))
    return r


def importance_refined_metrics(pred, loglik, n_is):
    """Coverage, mean and ESS after importance reweighting of q samples with the exact posterior."""
    prior = build_prior()
    th, lq = sample_pred(pred, n_is, with_log_q=True)
    y = pred["theta"]
    cov = np.zeros((len(LEVELS), C.N_DIMS))
    means, ess = np.empty_like(y), np.empty(len(y))
    for i in range(len(y)):
        w, ess[i] = importance_refine(th[i], lq[i], pred["y_logf"][i], pred["y_v"][i], pred["y_r"][i],
                                      loglik, prior)
        means[i] = w @ th[i]
        for d in range(C.N_DIMS):
            qs = weighted_quantile(th[i, :, d], w, np.concatenate([(1 - LEVELS) / 2, (1 + LEVELS) / 2]))
            cov[:, d] += (y[i, d] >= qs[:len(LEVELS)]) & (y[i, d] <= qs[len(LEVELS):])
    return dict(coverage_is=cov / len(y), mean_is=means, ess_is=ess)


def summary_rows(r):
    k95 = int(np.where(LEVELS == 0.95)[0][0])
    rows = []
    for d, pname in enumerate(C.PARAM_NAMES + ["ALL"]):
        sl = slice(None) if pname == "ALL" else slice(d, d + 1)
        intact = r["theta"][:, sl] >= C.INTACT_THRESHOLD
        row = dict(set=r["name"], param=pname,
                   rmse_mean=float(np.sqrt(np.mean((r["mean"][:, sl] - r["theta"][:, sl]) ** 2))),
                   mae_median=float(np.mean(np.abs(r["median"][:, sl] - r["theta"][:, sl]))),
                   crps=float(np.mean(r["crps"][:, sl])), mean_sd=float(np.mean(r["sd"][:, sl])),
                   coverage95=float(np.mean(r["coverage"][k95, sl])),
                   width95=float(np.mean(r["width"][k95, sl])),
                   fpr_interval=float(np.mean(r["upper95"][:, sl][intact] < C.INTACT_THRESHOLD))
                   if intact.any() else np.nan,
                   auc=float(auc(r["p_damage"][:, sl].ravel(), r["damaged"][:, sl].ravel())),
                   nll_joint=float(np.mean(r["nll"])))
        if "coverage_is" in r:
            row.update(coverage95_is=float(np.mean(r["coverage_is"][k95, sl])),
                       rmse_mean_is=float(np.sqrt(np.mean((r["mean_is"][:, sl] - r["theta"][:, sl]) ** 2))),
                       median_ess_fraction_is=float(np.median(r["ess_is"])))
        rows.append(row)
    return rows


def fig_calibration(results, fig_dir, show):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, r in enumerate(results):
        col = f"C{i % 10}"
        for ax, sl in zip(axs, (slice(0, C.N_ZONES), slice(C.N_ZONES, C.N_DIMS))):
            ax.plot(LEVELS, r["coverage"][:, sl].mean(1), "o-", ms=3, color=col, label=r["name"])
            if "coverage_is" in r:
                ax.plot(LEVELS, r["coverage_is"][:, sl].mean(1), "s--", ms=3, color=col,
                        label=f"{r['name']} (q + IS)")
    for ax, t in zip(axs, ("zone factors Z1-Z5", "bearing factors BL, BR")):
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlabel("nominal coverage of central interval")
        ax.set_ylabel("empirical coverage")
        ax.set_title(t)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axs[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, fig_dir / "Fig_calibration.png", show)


def fig_pred_vs_true(r, fig_dir, show, n_max=400):
    idx = np.arange(min(n_max, len(r["theta"])))
    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    for d, ax in enumerate(axs.ravel()[:C.N_DIMS]):
        t, m = r["theta"][idx, d], r["median"][idx, d]
        ax.errorbar(t, m, yerr=[m - r["lower95"][idx, d], r["upper95"][idx, d] - m], fmt="none",
                    ecolor="0.75", lw=0.6, zorder=1)
        ax.scatter(t, m, s=7, c=np.where(r["damaged_mask"][idx, d], COL_TRUE, COL_Q), zorder=2)
        ax.plot([C.LBOUND, C.UBOUND], [C.LBOUND, C.UBOUND], "k--", lw=0.8)
        ax.set_xlim(C.LBOUND, C.UBOUND)
        ax.set_ylim(C.LBOUND, C.UBOUND)
        ax.set_title(C.PARAM_LABELS[d])
        ax.set_xlabel("true value")
        ax.set_ylabel("posterior median")
    ax = axs.ravel()[-1]
    ax.scatter([], [], c=COL_TRUE, label="damaged in the scenario")
    ax.scatter([], [], c=COL_Q, label="healthy in the scenario")
    ax.plot([], [], color="0.75", label="95 % interval")
    ax.legend(frameon=False, loc="center")
    ax.axis("off")
    fig.suptitle(f"{r['name']}: posterior median and exact 95 % marginal interval")
    fig.tight_layout()
    save(fig, fig_dir / f"Fig_pred_vs_true_{r['name']}.png", show)


def detection(results, fig_dir, show):
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, r in enumerate(results):
        healthy = r["scenario"] == "healthy"
        rows.append(dict(
            set=r["name"], n_healthy_cases=int(healthy.sum()),
            false_alarm_rate_healthy_cases=float((r["p_damage"][healthy] > 0.5).any(1).mean())
            if healthy.any() else np.nan,
            auc_zones=float(auc(r["p_damage"][:, :C.N_ZONES].ravel(), r["damaged"][:, :C.N_ZONES].ravel())),
            auc_bearings=float(auc(r["p_damage"][:, C.N_ZONES:].ravel(), r["damaged"][:, C.N_ZONES:].ravel()))))
        for ax, sl in zip(axs, (slice(0, C.N_ZONES), slice(C.N_ZONES, C.N_DIMS))):
            lab = r["damaged"][:, sl].ravel()
            if lab.any() and (~lab).any():
                fpr, tpr = roc_curve(r["p_damage"][:, sl].ravel(), lab)
                ax.plot(fpr, tpr, color=f"C{i % 10}", label=r["name"])
    for ax, t in zip(axs, ("zones", "bearings")):
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlabel("false-positive rate")
        ax.set_ylabel("true-positive rate")
        ax.set_title(fr"Damage detection, {t} ($\theta$ < {C.DAMAGE_THRESHOLD})")
    axs[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, fig_dir / "Fig_detection_ROC.png", show)
    return rows


def location_severity(r, fig_dir, show):
    """Single-location damage cases grouped by damaged parameter and severity."""
    single = r["damaged_mask"].sum(1) == 1
    bins = np.array(C.SEVERITY_BINS)
    nb = len(bins) - 1
    keys = ("n", "P(damage) at location", "localization accuracy", "|median - true|",
            "95% coverage at location", "false alarms elsewhere")
    M = {k: np.full((nb, C.N_DIMS), np.nan) for k in keys}
    rows = []
    for k in range(C.N_DIMS):
        b_idx = np.digitize(r["theta"][:, k], bins) - 1
        for b in range(nb):
            s = single & r["damaged_mask"][:, k] & (b_idx == b)
            if not s.any():
                continue
            pd_ = r["p_damage"][s]
            vals = (s.sum(), pd_[:, k].mean(), np.mean(np.argmax(pd_, 1) == k),
                    np.mean(np.abs(r["median"][s, k] - r["theta"][s, k])),
                    np.mean((r["theta"][s, k] >= r["lower95"][s, k]) & (r["theta"][s, k] <= r["upper95"][s, k])),
                    np.mean(np.delete(pd_, k, axis=1) > 0.5))
            for key, v in zip(keys, vals):
                M[key][b, k] = v
            rows.append(dict(set=r["name"], location=C.PARAM_NAMES[k], severity=f"[{bins[b]:.2f},{bins[b + 1]:.2f})",
                             **{key: float(v) for key, v in zip(keys, vals)}))
    fig, axs = plt.subplots(2, 3, figsize=(15, 6.5))
    for ax, key in zip(axs.ravel(), keys):
        im = ax.imshow(M[key], cmap="viridis", origin="lower", aspect="auto", vmin=0,
                       vmax=None if key in ("n", "|median - true|") else 1)
        for i in range(nb):
            for j in range(C.N_DIMS):
                if np.isfinite(M[key][i, j]):
                    ax.text(j, i, f"{int(M[key][i, j])}" if key == "n" else f"{M[key][i, j]:.2f}",
                            ha="center", va="center", fontsize=7, color="w")
        ax.set_xticks(range(C.N_DIMS))
        ax.set_xticklabels(C.PARAM_NAMES)
        ax.set_yticks(range(nb))
        ax.set_yticklabels([f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(nb)], fontsize=8)
        ax.set_title(key)
        fig.colorbar(im, ax=ax)
    for ax in axs[:, 0]:
        ax.set_ylabel(r"true $\theta$ of the damaged parameter")
    fig.suptitle(f"{r['name']}: single-location damage by location and severity")
    fig.tight_layout()
    save(fig, fig_dir / f"Fig_location_severity_{r['name']}.png", show)
    return rows


def fig_mass_sensitivity(results, fig_dir, show):
    by_name = {r["name"]: r for r in results}
    present = [(n, m) for n, m in MASS_SETS.items() if n in by_name]
    if len(present) < 2:
        return
    k95 = int(np.where(LEVELS == 0.95)[0][0])
    x = [100 * m for _, m in present]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for lab, sl in (("zones", slice(0, C.N_ZONES)), ("bearings", slice(C.N_ZONES, C.N_DIMS))):
        axs[0].plot(x, [by_name[n]["coverage"][k95, sl].mean() for n, _ in present], "o-", label=lab)
        axs[1].plot(x, [by_name[n]["width"][k95, sl].mean() for n, _ in present], "o-", label=lab)
        axs[2].plot(x, [np.sqrt(np.mean((by_name[n]["mean"][:, sl] - by_name[n]["theta"][:, sl]) ** 2))
                        for n, _ in present], "o-", label=lab)
    axs[0].axhline(0.95, color="k", ls=":", lw=0.8)
    for ax, t in zip(axs, ("95 % coverage", "mean 95 % interval width", "RMSE of posterior mean")):
        ax.set_xlabel("element mass perturbation std [%] (not in the inference model)")
        ax.set_title(t)
        ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, fig_dir / "Fig_mass_sensitivity.png", show)


# =============================================================================
# Benchmark cases: q vs reference posterior
# =============================================================================
def reference_for_benchmarks(meta, tag, n_particles, recompute=False):
    tc = meta["train"]
    data_dir = Path(meta["data_dir"])
    variant, bc = tag.replace("_bc", ""), tag.endswith("_bc")
    key = f"ref_{tag}_{tc['obs']}{tc['n_modes']}_N{n_particles}_el{meta['coarse_el_per_zone']}"
    path = data_dir / "reference_posteriors" / f"{key}.npz"
    if path.exists() and not recompute:
        return dict(np.load(path))
    b = load_benchmarks(variant, tc["n_modes"], data_dir, baseline_correction=bc)
    loglik = LogLikelihood(load_coarse_model(data_dir), tc["n_modes"], tc["obs"])
    samples, stages = [], []
    for i, name in enumerate(b["names"]):
        th, info = smc_posterior(b["y_logf"][i], b["y_v"][i], b["y_r"][i], loglik, n_particles=n_particles,
                                 seed=i)
        print(f"  reference {tag}/{name}: {info['n_stages']} stages, acceptance {info['mean_acceptance']:.2f}, "
              f"{info['time_s']:.0f} s")
        samples.append(th)
        stages.append(info["n_stages"])
    out = dict(samples=np.stack(samples), names=np.array(b["names"]), theta=b["theta"], n_stages=np.array(stages))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def active_dims(theta_true):
    """Damaged parameters, their mirror images and the bearings (at least four)."""
    dims = {d for d in range(C.N_DIMS) if theta_true[d] < C.INTACT_THRESHOLD}
    dims |= {C.MIRROR[d] for d in dims} | {5, 6}
    for d in (0, 2, 4, 1, 3):
        if len(dims) >= 4:
            break
        dims.add(d)
    return sorted(dims)


def _hpd_levels(Z, masses=(0.9, 0.5)):
    """Density thresholds of the regions containing the given probability masses."""
    z = np.sort(Z.ravel())[::-1]
    c = np.cumsum(z) / z.sum()
    return sorted(z[min(np.searchsorted(c, m), len(z) - 1)] for m in masses)


def fig_marginals(q_pdf, grid, q_s, ref, w_is, truth, title, path, show):
    fig, axs = plt.subplots(1, C.N_DIMS, figsize=(2.3 * C.N_DIMS, 2.7))
    for d, ax in enumerate(axs):
        f_ref = kde_1d(ref[:, d], grid)
        ax.fill_between(grid, f_ref, color=COL_REF, alpha=0.15, label="reference (SMC)")
        ax.plot(grid, f_ref, color=COL_REF, ls="--", lw=1)
        ax.plot(grid, q_pdf[:, d], color=COL_Q, lw=1.5, label="PI-GC-KSmix-VAE q (exact)")
        if 1.0 / np.sum(w_is ** 2) >= 20:          # enough effective samples for a KDE
            ax.plot(grid, kde_1d(q_s[:, d], grid, w=w_is), color=COL_IS, lw=1, ls="-.", label="q + IS")
        ax.axvline(truth[d], color=COL_TRUE, lw=1, label="true value")
        ax.set_xlabel(C.PARAM_LABELS[d])
        ax.set_yticks([])
    axs[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, path, show)


def fig_active_subspace(q_s, ref, truth, title, path, show, n_kde=2000):
    dims = active_dims(truth)
    n = len(dims)
    rng = np.random.default_rng(0)
    qs = q_s[rng.choice(len(q_s), min(n_kde, len(q_s)), replace=False)]
    rs = ref[rng.choice(len(ref), min(n_kde, len(ref)), replace=False)]
    fig, axs = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n))
    g = np.linspace(C.LBOUND, C.UBOUND, 70)
    for i, di in enumerate(dims):
        for j, dj in enumerate(dims):
            ax = axs[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.plot(g, kde_1d(ref[:, di], g), color=COL_REF, ls="--")
                ax.plot(g, kde_1d(q_s[:, di], g), color=COL_Q)
                ax.axvline(truth[di], color=COL_TRUE, lw=1)
                ax.set_yticks([])
            else:
                X, Y = np.meshgrid(g, g)
                for s, col, ls in ((rs, COL_REF, "--"), (qs, COL_Q, "-")):
                    if np.std(s[:, dj]) < 1e-9 or np.std(s[:, di]) < 1e-9:
                        continue
                    Z = gaussian_kde(np.vstack([s[:, dj], s[:, di]]))(np.vstack([X.ravel(), Y.ravel()]))
                    Z = Z.reshape(X.shape)
                    ax.contour(X, Y, Z, levels=_hpd_levels(Z), colors=col, linestyles=ls, linewidths=1.2)
                ax.plot(truth[dj], truth[di], "*", color=COL_TRUE, ms=9, mec="k", mew=0.5)
                ax.set_ylim(C.LBOUND, C.UBOUND)
            ax.set_xlim(C.LBOUND, C.UBOUND)
            if i == n - 1:
                ax.set_xlabel(C.PARAM_LABELS[dj])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(C.PARAM_LABELS[di])
            elif i != j:
                ax.set_yticklabels([])
    fig.legend(handles=[plt.Line2D([], [], color=COL_REF, ls="--", label="reference (SMC)"),
                        plt.Line2D([], [], color=COL_Q, label="PI-GC-KSmix-VAE q"),
                        plt.Line2D([], [], color=COL_TRUE, marker="*", ls="", label="true value")],
               loc="upper right", frameon=False)
    fig.suptitle(f"{title}\ncontours enclose 50 % and 90 % of the posterior mass")
    fig.tight_layout()
    save(fig, path, show)


def fig_corner(q_s, ref, truth, title, path, show):
    """Lower triangle: q; upper triangle: reference; diagonal: both marginals."""
    D = C.N_DIMS
    fig, axs = plt.subplots(D, D, figsize=(1.8 * D, 1.8 * D))
    rng_ = (C.LBOUND, C.UBOUND)
    g = np.linspace(*rng_, 200)
    for i in range(D):
        for j in range(D):
            ax = axs[i, j]
            ax.tick_params(labelsize=6)
            if i == j:
                ax.plot(g, kde_1d(ref[:, i], g), color=COL_REF, ls="--")
                ax.plot(g, kde_1d(q_s[:, i], g), color=COL_Q)
                ax.axvline(truth[i], color=COL_TRUE, lw=1)
                ax.set_yticks([])
            else:
                s, cmap = (q_s, "Blues") if i > j else (ref, "Greys")
                ax.hist2d(s[:, j], s[:, i], bins=45, range=[rng_, rng_], cmap=cmap, cmin=1)
                ax.plot(truth[j], truth[i], "*", color=COL_TRUE, ms=6)
            ax.set_xlim(rng_)
            if i < D - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(C.PARAM_LABELS[j])
            if j > 0 or i == 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(C.PARAM_LABELS[i])
    fig.suptitle(f"{title}\nlower triangle: PI-GC-KSmix-VAE q; upper triangle: reference (SMC)")
    fig.tight_layout()
    save(fig, path, show)


def fig_profile(q_s, ref, truth, title, path, show):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    g = np.linspace(C.LBOUND, C.UBOUND, 300)
    for d in range(C.N_DIMS):
        pr, pq = kde_1d(ref[:, d], g), kde_1d(q_s[:, d], g)
        sc = 0.42 / max(pr.max(), pq.max(), 1e-12)
        ax.fill_betweenx(g, d - sc * pr, d, color=COL_REF, alpha=0.35, lw=0)
        ax.fill_betweenx(g, d, d + sc * pq, color=COL_Q, alpha=0.6, lw=0)
        ax.plot([d - 0.45, d + 0.45], [truth[d]] * 2, color=COL_TRUE, lw=1.5)
    ax.axvline(C.N_ZONES - 0.5, color="0.6", lw=0.8, ls=":")
    ax.set_xticks(range(C.N_DIMS))
    ax.set_xticklabels(C.PARAM_NAMES)
    ax.set_ylabel(r"stiffness factor $\theta$")
    ax.set_ylim(C.LBOUND - 0.02, C.UBOUND + 0.02)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL_REF, alpha=0.35, label="reference (left)"),
                       plt.Rectangle((0, 0), 1, 1, color=COL_Q, alpha=0.6, label="PI-GC-KSmix-VAE (right)"),
                       plt.Line2D([], [], color=COL_TRUE, label="true value")],
              frameon=False, ncol=3, loc="lower left", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    save(fig, path, show)


def compare_benchmarks(run_dir, meta, tag, n_particles, show, recompute=False):
    bench = dict(np.load(Path(run_dir) / f"bench_samples_{tag}.npz"))
    ref = reference_for_benchmarks(meta, tag, n_particles, recompute)
    tc = meta["train"]
    loglik = LogLikelihood(load_coarse_model(Path(meta["data_dir"])), tc["n_modes"], tc["obs"])
    prior = build_prior()
    lo, hi = bounds_tf()
    grid = C.LBOUND + (np.arange(400) + 0.5) / 400 * (C.UBOUND - C.LBOUND)   # interior points
    rows = []
    fig_dir = Path(run_dir) / "figures" / f"bench_{tag}"
    for i, name in enumerate(bench["names"]):
        p = params_to_tf(bench, slice(i, i + 1))
        q_s, lq = (t.numpy()[0] for t in KS.sample(p, int(bench["samples"].shape[1]), lo, hi, return_log_q=True))
        r_s, truth = ref["samples"][i], bench["theta"][i]
        w, ess = importance_refine(q_s, lq, bench["y_logf"][i], bench["y_v"][i], bench["y_r"][i], loglik, prior)
        q_pdf = KS.marginal_pdf_grid(p, tf.constant(np.repeat(grid[:, None], C.N_DIMS, 1), DT), lo, hi).numpy()[0]
        title = f"{name} ({tag} data)"
        fig_marginals(q_pdf, grid, q_s, r_s, w, truth, title, fig_dir / f"{name}_marginals.png", show)
        fig_active_subspace(q_s, r_s, truth, title, fig_dir / f"{name}_active_subspace.png", show)
        fig_corner(q_s, r_s, truth, title, fig_dir / f"{name}_corner.png", show)
        fig_profile(q_s, r_s, truth, title, fig_dir / f"{name}_profile.png", show)
        row = dict(data=tag, case=str(name), energy_distance_q=energy_distance(q_s, r_s), ess_fraction_is=ess,
                   corr_frobenius_q=float(np.linalg.norm(np.corrcoef(q_s.T) - np.corrcoef(r_s.T))))
        for d, pn in enumerate(C.PARAM_NAMES):
            row[f"W1_q_{pn}"] = wasserstein_distance(q_s[:, d], r_s[:, d])
            row[f"W1_qIS_{pn}"] = wasserstein_distance(q_s[:, d], r_s[:, d], u_weights=w)
            row[f"sd_ratio_q_{pn}"] = q_s[:, d].std() / max(r_s[:, d].std(), 1e-12)
            row[f"dmean_q_{pn}"] = q_s[:, d].mean() - r_s[:, d].mean()
        rows.append(row)
        print(f"  {tag}/{name}: energy distance {row['energy_distance_q']:.4f}, max W1 "
              f"{max(row[f'W1_q_{pn}'] for pn in C.PARAM_NAMES):.4f}, IS ESS {100 * ess:.1f} %")
    return rows


# =============================================================================
def main(run=None, n_samples=500, n_particles=4000, n_is=500, skip_reference=False, show=False,
         recompute_reference=False):
    style()
    run_dir = Path(run) if run and Path(run).is_absolute() else (C.OUTPUT_DIR / run if run else latest_run())
    print(f"Post-processing {run_dir}")
    meta = C.load_json(run_dir / "config.json")
    fig_dir, met_dir = run_dir / "figures", run_dir / "metrics"
    plot_history(run_dir, show)
    tc = meta["train"]
    loglik = LogLikelihood(load_coarse_model(Path(meta["data_dir"])), tc["n_modes"], tc["obs"])

    names = ["val"] + [s + suf for s in C.TEST_SETS for suf in ("", "_bc")]
    results, rows, loc_rows = [], [], []
    for name in names:
        if not (run_dir / f"pred_{name}.npz").exists():
            continue
        r = evaluate_set(run_dir, name, n_samples, loglik, n_is if name.replace("_bc", "") in IS_SETS else 0)
        results.append(r)
        rows += summary_rows(r)
        fig_pred_vs_true(r, fig_dir, show)
        if name in ("val", "test_fine_balanced"):
            loc_rows += location_severity(r, fig_dir, show)
        k95 = int(np.where(LEVELS == 0.95)[0][0])
        msg = (f"  {name:22s} RMSE={np.sqrt(np.mean((r['mean'] - r['theta']) ** 2)):.4f} "
               f"cov95 zones={r['coverage'][k95, :5].mean():.3f} bearings={r['coverage'][k95, 5:].mean():.3f}")
        if "coverage_is" in r:
            msg += (f" | q+IS cov95 zones={r['coverage_is'][k95, :5].mean():.3f} "
                    f"bearings={r['coverage_is'][k95, 5:].mean():.3f} median ESS={100 * np.median(r['ess_is']):.1f}%")
        print(msg)
    write_csv(met_dir / "summary_sets.csv", rows)
    write_csv(met_dir / "location_severity.csv", loc_rows)
    write_csv(met_dir / "coverage.csv",
              [dict(set=r["name"], level=a, **{p: r["coverage"][k, d] for d, p in enumerate(C.PARAM_NAMES)})
               for r in results for k, a in enumerate(LEVELS)])
    # Coverage is only expected to be nominal for cases drawn from the prior
    fig_calibration([r for r in results if r["name"] in CALIBRATION_SETS], fig_dir, show)
    det = detection([r for r in results if r["name"] in ROC_SETS], fig_dir, show)
    write_csv(met_dir / "detection.csv", det)
    fig_mass_sensitivity(results, fig_dir, show)

    bench_rows = []
    if not skip_reference:
        for tag in BENCH_TAGS:
            if (run_dir / f"bench_samples_{tag}.npz").exists():
                bench_rows += compare_benchmarks(run_dir, meta, tag, n_particles, show, recompute_reference)
        write_csv(met_dir / "benchmark_vs_reference.csv", bench_rows)
    C.save_json(dict(sets=[r for r in rows if r["param"] == "ALL"], detection=det,
                     benchmarks=bench_rows), met_dir / "summary.json")
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN_NAME)
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--n_particles", type=int, default=4000)
    ap.add_argument("--n_is", type=int, default=500)
    ap.add_argument("--skip_reference", action="store_true")
    ap.add_argument("--recompute_reference", action="store_true")
    ap.add_argument("--show", action="store_true")
    a, _ = ap.parse_known_args()
    main(a.run or None, a.n_samples, a.n_particles, a.n_is, a.skip_reference, a.show, a.recompute_reference)
