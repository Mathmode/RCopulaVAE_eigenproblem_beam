#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed_v2/postprocessing.py

Figures and metrics of a trained run (Output/review29oct_fixed/<run>):
  1. Training_Loss_Curves.png: total loss, -ELBO, NLL and its frequency / vertical / rotational parts,
     entropy and its copula / marginal parts, likelihood temperature and learning rate (train and validation);
  2. Global_Test_Metrics.json + Fig_calibration.png: RMSE of the posterior mean, coverage of z* by the central
     credible intervals of the exact KS-mixture marginals (zones and bearings), interval width, false alarms;
  3. GT_Comp/ for every benchmark (reference posterior by SMC, same likelihood and gamma as training):
       P{pos}_GT.png, P{pos}_VAE.png                       single corner plots
       P{pos}_Overlay.png, P{pos}_ActiveSubspace_Overlay.png reference (red) vs VAE (blue)
       P{pos}_1D.png                                       1-D marginals (+ exact KS-mixture density)
       Physical_PDF_Profiles/Sample_{pos}_UQ_Violin.png
     GT_vs_Posterior_Metrics.csv/.json and Fig_benchmark_summary.png.

The VAE posterior is plotted as it is: samples of q(z|m). Nothing is re-weighted with the likelihood,
filtered or mirrored (review29oct re-weighted q by the likelihood, i.e. plotted q x L instead of q, kept the
heaviest 95 % of the weight and mirrored half of the samples for the "bimodal" cases).

    python MODULES/review29oct_fixed_v2/postprocessing.py --run-folder KSVAE_fixed_7D_3KS_obs-fvr_Gamma1.0_Ep1500
"""

import os
import sys
import json
import argparse

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from MODULES.review29oct_fixed_v2 import config as C, tf_keras

tf_keras()  # Keras on TensorFlow, float32 (reloads it if this console already imported it with jax)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import wasserstein_distance
from scipy.special import betaln

from MODULES.review29oct_fixed_v2.GC_KSmix_functions import (
    build_correlation_matrices_from_cholesky, gaussian_copula_samples, ks_mixture_quantile, ks_mixture_pdf_np,
)
from MODULES.review29oct_fixed_v2.reference_posterior import compute_many, compute_reference_posterior, load_problem_config
from MODULES.review29oct_fixed_v2 import plotting as P


# =============================================================================
# 1. LOSS HISTORY
# =============================================================================
@P.fixed_style
def plot_training_history(folder_path, warmup=None):
    hist_path = os.path.join(folder_path, "model_history.npy")
    if not os.path.exists(hist_path):
        print("  no model_history.npy, skipping the loss curves")
        return
    h = np.load(hist_path, allow_pickle=True).item()
    ep = np.arange(1, len(h["loss"]) + 1)

    def curve(ax, key, color, label, ls="-", val=True):
        if key in h:
            ax.plot(ep, h[key], color=color, ls=ls, lw=1.6, label=f"{label} (train)")
        if val and f"val_{key}" in h:
            ax.plot(ep, h[f"val_{key}"], color=color, ls="--", lw=1.6, alpha=0.8, label=f"{label} (val)")

    fig, ax = plt.subplots(2, 3, figsize=(20, 10.5), facecolor="white")
    curve(ax[0, 0], "loss", "#1F77B4", r"$\tau\,E_q[\mathrm{NLL}] - \gamma^2 H[q]$")
    ax[0, 0].set_title("Training loss (with the warm-up temperature)", fontweight="bold")
    curve(ax[0, 1], "neg_elbo", "#D62728", r"$-$ELBO ($\tau$ = 1)")
    ax[0, 1].set_title(r"Negative ELBO $E_q[\mathrm{NLL}] - \gamma^2 H[q]$ (model selection)", fontweight="bold")
    curve(ax[0, 2], "NLL", "k", r"$E_q[\mathrm{NLL}]$")
    curve(ax[0, 2], "NLL_freq", "#1F77B4", "frequencies")
    curve(ax[0, 2], "NLL_vert", "#2CA02C", "vertical shapes")
    curve(ax[0, 2], "NLL_rot", "#9467BD", "rotational shapes")
    ax[0, 2].set_yscale("log")
    ax[0, 2].set_title("Expected negative log-likelihood and its terms", fontweight="bold")
    curve(ax[1, 0], "Entropy", "k", r"$H[q]$")
    curve(ax[1, 0], "Entropy_copula", "#FF7F0E", r"copula $\frac{1}{2}\log\det R$")
    curve(ax[1, 0], "Entropy_marginals", "#17BECF", r"marginals $\sum_d H[q_d]$")
    ax[1, 0].set_title("Entropy of q and its parts", fontweight="bold")
    if "val_neg_elbo" in h:
        start = int(warmup or 0)
        v = np.asarray(h["val_neg_elbo"])
        ax[1, 1].plot(ep[start:], v[start:], color="#D62728", lw=1.6, label=r"$-$ELBO (val)")
        ax[1, 1].plot(ep[start:], np.asarray(h["neg_elbo"])[start:], color="#D62728", lw=1.0, alpha=0.5, label=r"$-$ELBO (train)")
        ax[1, 1].set_title("Negative ELBO after the warm-up (zoom)", fontweight="bold")
    if "tau" in h:
        ax[1, 2].plot(ep, h["tau"], color="#8C564B", lw=2, label=r"likelihood temperature $\tau$")
        ax[1, 2].set_yscale("log")
    if "lr" in h:
        a2 = ax[1, 2].twinx()
        a2.plot(ep, h["lr"], color="#7F7F7F", lw=2, ls=":", label="learning rate")
        a2.set_yscale("log")
        a2.set_ylabel("learning rate")
        a2.legend(loc="lower right")
    ax[1, 2].set_title("Warm-up and learning-rate schedules", fontweight="bold")
    for a in ax.ravel():
        a.set_xlabel("epoch")
        a.grid(True, ls=":", alpha=0.5)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=9)
    fig.suptitle(f"Training history: {os.path.basename(folder_path)}", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(folder_path, "Training_Loss_Curves.png"), dpi=170)
    plt.close(fig)


# =============================================================================
# 2. EXACT KS-MIXTURE MARGINALS: MEANS, INTERVALS, COVERAGE
# =============================================================================
def marginal_quantiles(a, b, w, probs, lbound, n_grid=4001, chunk=100):
    """Quantiles (N, D, len(probs)) and means (N, D) in z units of every KS-mixture marginal."""
    N, K, D = a.shape
    x = np.linspace(0.0, 1.0, n_grid)
    q = np.zeros((N, D, len(probs)))
    for i0 in range(0, N, chunk):
        aa, bb, ww = (v[i0:i0 + chunk].transpose(0, 2, 1)[:, :, None, :].astype(np.float64) for v in (a, b, w))
        cdf = np.sum(ww * (1.0 - (1.0 - x[None, None, :, None] ** aa) ** bb), axis=-1) / np.sum(ww, axis=-1)
        for i in range(cdf.shape[0]):
            for d in range(D):
                q[i0 + i, d] = np.interp(probs, cdf[i, d], x)
    log_m1 = np.log(b) + betaln(1.0 + 1.0 / a, b)
    mean = np.sum(w * np.exp(log_m1), axis=1) / np.sum(w, axis=1)
    return lbound + (1.0 - lbound) * q, lbound + (1.0 - lbound) * mean


@P.fixed_style
def global_test_metrics(preds, z_true, lbound, folder_path, threshold=C.DAMAGE_THRESHOLD, suffix="", label="test set"):
    levels = np.array([0.5, 0.68, 0.8, 0.9, 0.95, 0.99])
    probs = np.concatenate([(1 - levels) / 2, (1 + levels) / 2])
    qs, mean = marginal_quantiles(preds["test_a"], preds["test_b"], preds["test_weight_vals"], probs, lbound)
    lo, hi = qs[..., :len(levels)], qs[..., len(levels):]
    inside = (z_true[..., None] >= lo) & (z_true[..., None] <= hi)
    zones, bearings = slice(0, C.N_ZONES), slice(C.N_ZONES, C.N_DIMS)
    i95 = int(np.where(levels == 0.95)[0][0])
    healthy = z_true >= threshold
    metrics = {
        "RMSE_posterior_mean": float(np.sqrt(np.mean((mean - z_true) ** 2))),
        "MAE_posterior_mean": float(np.mean(np.abs(mean - z_true))),
        "RMSE_damaged_parameters": float(np.sqrt(np.mean((mean - z_true)[~healthy] ** 2))),
        "coverage95_zones": float(inside[:, zones, i95].mean()),
        "coverage95_bearings": float(inside[:, bearings, i95].mean()),
        "mean_width95_zones": float((hi - lo)[:, zones, i95].mean()),
        "mean_width95_bearings": float((hi - lo)[:, bearings, i95].mean()),
        "healthy_false_alarm_rate (upper95 < %.2f)" % threshold: float(np.mean(hi[..., i95][healthy] < threshold)),
        "damage_detection_rate (upper95 < %.2f | z* < 0.90)" % threshold:
            float(np.mean(hi[..., i95][z_true < 0.90] < threshold)),
        "coverage_by_level_zones": dict(zip(levels.tolist(), inside[:, zones].mean(axis=(0, 1)).tolist())),
        "coverage_by_level_bearings": dict(zip(levels.tolist(), inside[:, bearings].mean(axis=(0, 1)).tolist())),
        "note": "coverage of z* is only expected to be nominal on average over the prior; the test set follows the DoE",
    }
    with open(os.path.join(folder_path, f"Global_Test_Metrics{suffix}.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].plot(levels, inside[:, zones].mean(axis=(0, 1)), "o-", label="deck zones")
    ax[0].plot(levels, inside[:, bearings].mean(axis=(0, 1)), "s-", label="bearings")
    ax[0].set_xlabel("nominal level of the central credible interval")
    ax[0].set_ylabel("fraction of test parameters with z* inside")
    ax[0].set_title(f"Calibration of the VAE marginals ({label})", fontweight="bold")
    ax[0].legend()
    ax[0].grid(alpha=0.4, ls=":")
    ax[1].scatter(z_true[:, zones].ravel(), mean[:, zones].ravel(), s=5, alpha=0.4, label="deck zones")
    ax[1].scatter(z_true[:, bearings].ravel(), mean[:, bearings].ravel(), s=5, alpha=0.4, label="bearings")
    ax[1].plot([lbound, 1], [lbound, 1], "k--", lw=1)
    ax[1].set_xlabel("true z*")
    ax[1].set_ylabel(r"posterior mean $E_q[z]$")
    ax[1].set_title(f"Posterior mean vs truth ({label})", fontweight="bold")
    ax[1].legend()
    ax[1].grid(alpha=0.4, ls=":")
    plt.tight_layout()
    fig.savefig(os.path.join(folder_path, f"Fig_calibration{suffix}.png"), dpi=170)
    plt.close(fig)
    return metrics


# =============================================================================
# 3. SAMPLES OF q(z | m)
# =============================================================================
def sample_vae_posterior(preds, pos, n_samples, lbound, n_dims=C.N_DIMS, seed=0):
    tf.random.set_seed(seed + pos)
    a = tf.constant(preds["test_a"][pos:pos + 1], tf.float32)
    b = tf.constant(preds["test_b"][pos:pos + 1], tf.float32)
    w = tf.constant(preds["test_weight_vals"][pos:pos + 1], tf.float32)
    LT = build_correlation_matrices_from_cholesky(tf.constant(preds["test_offdiag_elems"][pos:pos + 1], tf.float32),
                                                  tf.constant(preds["test_diag_elems"][pos:pos + 1], tf.float32), n_dims)
    x = ks_mixture_quantile(gaussian_copula_samples(LT, n_samples), a, b, w)
    return (lbound + (1.0 - lbound) * x[0].numpy()).astype(np.float64), (LT[0] @ tf.transpose(LT[0])).numpy()


def energy_distance(x, y, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = x[rng.choice(len(x), min(n, len(x)), replace=False)]
    y = y[rng.choice(len(y), min(n, len(y)), replace=False)]

    def mean_dist(u, v):
        return np.mean(np.sqrt(np.sum((u[:, None, :] - v[None, :, :]) ** 2, axis=-1)))
    return float(2 * mean_dist(x, y) - mean_dist(x, x) - mean_dist(y, y))


def active_subspace(z_true, threshold=C.DAMAGE_THRESHOLD):
    mirror = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 6, 6: 5}
    active = {d for d in range(len(z_true)) if z_true[d] < threshold}
    active |= {mirror[d] for d in list(active)}
    active |= {5, 6}
    for cand in [0, 2, 4, 1, 3]:
        if len(active) < 4:
            active.add(cand)
    return sorted(active)


# =============================================================================
# 4. DRIVER
# =============================================================================
def run_postprocessing(folder_path, positions=None, n_q=4000, n_particles=4000, workers=None, recompute_gt=False):
    if not os.path.isabs(folder_path) and not os.path.exists(folder_path):
        folder_path = os.path.join(C.OUTPUT_DIR, folder_path)
    print(f"\n=== Post-processing {folder_path} ===")
    info = np.load(os.path.join(folder_path, "Problem_info.npy"), allow_pickle=True).item()
    plot_training_history(folder_path, warmup=info.get("warmup"))
    out = {}
    for split in info.get("test_sets", ["test"]):
        out[split] = postprocess_test_set(folder_path, info, split, positions, n_q, n_particles, workers, recompute_gt)
    return out


def postprocess_test_set(folder_path, info, split, positions, n_q, n_particles, workers, recompute_gt):
    """'test' (same model as training) -> GT_Comp/; any other test set -> GT_Comp_<name>/."""
    suffix = "" if split == "test" else f"_{split}"
    preds_file = os.path.join(folder_path, f"Test_predicted_props{suffix}.npy")
    if not os.path.exists(preds_file):
        print(f"  no {os.path.basename(preds_file)}, skipping '{split}'")
        return None
    preds = np.load(preds_file, allow_pickle=True).item()
    data_dir = info["data_dir"]
    pcfg = load_problem_config(data_dir)
    lbound, gamma2, obs_types = float(info["lbound"]), float(info["gamma"]) ** 2, info["obs_types"]
    n_el = pcfg.get("test_sets", {}).get(split, pcfg.get("el_per_zone_decoder", 1)) * int(pcfg["n_zones"])
    label = f"{n_el}-element data" + (", same model as training" if split == "test" else "")

    z_test = np.load(os.path.join(data_dir, f"alphas_{split}.npy")).astype(np.float64)
    gm = global_test_metrics(preds, z_test, lbound, folder_path, suffix=suffix, label=label)
    print(f"--- Test set '{split}' ({label}), exact KS-mixture marginals ---")
    for k, v in gm.items():
        if not isinstance(v, (dict, str)):
            print(f"  {k:52s}: {v:.4f}")

    descs = {b["pos"]: b["desc"] for b in pcfg["benchmarks"]}
    positions = positions if positions is not None else sorted(descs)
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    print(f"--- Reference posteriors (SMC, gamma^2 = {gamma2:.4g}, obs = {obs_types}) ---")
    compute_many(data_dir, positions, gamma2, obs_types, n_particles, workers, recompute_gt, split)

    out_dir = os.path.join(folder_path, f"GT_Comp{suffix}")
    prof_dir = os.path.join(folder_path, f"Physical_PDF_Profiles{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prof_dir, exist_ok=True)
    rows = []
    for pos in positions:
        z_true = z_test[pos]
        desc = descs.get(pos, f"test case {pos}")
        x_gt, _ = compute_reference_posterior(data_dir, pos, gamma2, obs_types, n_particles, split=split)
        x_gt = x_gt.astype(np.float64)
        x_q, R_q = sample_vae_posterior(preds, pos, n_q, lbound)

        m_gt, s_gt = x_gt.mean(0), x_gt.std(0)
        m_q, s_q = x_q.mean(0), x_q.std(0)
        lo_gt, hi_gt = np.quantile(x_gt, [0.025, 0.975], axis=0)
        lo_q, hi_q = np.quantile(x_q, [0.025, 0.975], axis=0)
        row = {"pos": pos, "scenario": desc,
               "mean_abs_diff_mean": float(np.mean(np.abs(m_q - m_gt))),
               "mean_std_ratio_VAE_over_ref": float(np.mean(s_q / s_gt)),
               "min_std_ratio": float(np.min(s_q / s_gt)), "max_std_ratio": float(np.max(s_q / s_gt)),
               "mean_W1": float(np.mean([wasserstein_distance(x_gt[:, d], x_q[:, d]) for d in range(C.N_DIMS)])),
               "energy_distance": energy_distance(x_gt, x_q),
               "corr_frobenius_error": float(np.linalg.norm(np.corrcoef(x_q.T) - np.corrcoef(x_gt.T))),
               "zstar_in_ref95": int(np.sum((z_true >= lo_gt) & (z_true <= hi_gt))),
               "zstar_in_VAE95": int(np.sum((z_true >= lo_q) & (z_true <= hi_q)))}
        for d, name in enumerate(C.PARAM_NAMES):
            row.update({f"{name}_true": z_true[d], f"{name}_ref_mean": m_gt[d], f"{name}_ref_std": s_gt[d],
                        f"{name}_VAE_mean": m_q[d], f"{name}_VAE_std": s_q[d]})
        rows.append(row)
        print(f"  P{pos:<2d} {desc[:48]:48s} |dmean| = {row['mean_abs_diff_mean']:.4f}  std ratio = "
              f"{row['mean_std_ratio_VAE_over_ref']:.2f} [{row['min_std_ratio']:.2f}, {row['max_std_ratio']:.2f}]  "
              f"W1 = {row['mean_W1']:.4f}  z* in 95%: ref {row['zstar_in_ref95']}/7, VAE {row['zstar_in_VAE95']}/7")

        tag = f"Sample {pos} ({desc}), {label}"
        P.corner_single(x_gt, z_true, lbound, f"{tag}: reference posterior $p(\\mathbf{{z}}|\\mathbf{{m}})$",
                        "relative posterior density (reference)", os.path.join(out_dir, f"P{pos}_GT.png"))
        P.corner_single(x_q, z_true, lbound, f"{tag}: estimated KS-Copula posterior ($K={info['num_KS']}$)",
                        "relative posterior density (KS-Copula VAE)", os.path.join(out_dir, f"P{pos}_VAE.png"))
        P.corner_overlay(x_gt, x_q, z_true, list(range(C.N_DIMS)), lbound, tag,
                         os.path.join(out_dir, f"P{pos}_Overlay.png"), "7D bridge deck & bearings")
        P.corner_overlay(x_gt, x_q, z_true, active_subspace(z_true), lbound, tag,
                         os.path.join(out_dir, f"P{pos}_ActiveSubspace_Overlay.png"), "Active-damage subspace")
        a, b, w = (preds[k][pos] for k in ("test_a", "test_b", "test_weight_vals"))

        def analytic_pdf(grid, d):
            return ks_mixture_pdf_np((grid - lbound) / (1 - lbound), a[:, d], b[:, d], w[:, d]) / (1 - lbound)
        P.strip_1d(x_gt, x_q, z_true, lbound, os.path.join(out_dir, f"P{pos}_1D.png"), tag,
                   analytic_pdf=analytic_pdf)
        P.physical_profile(x_gt, x_q, z_true, lbound, os.path.join(prof_dir, f"Sample_{pos}_UQ_Violin.png"),
                           tag)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "GT_vs_Posterior_Metrics.csv"), index=False)
    with open(os.path.join(out_dir, "GT_vs_Posterior_Metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"run_folder": folder_path, "test_set": split, "gamma2": gamma2, "obs_types": obs_types,
                   "global_test_set_metrics": gm, "per_sample": rows}, fh, indent=2, default=float)
    plot_benchmark_summary(df, os.path.join(out_dir, "Fig_benchmark_summary.png"))
    print(f"[OK] figures and metrics saved to {out_dir}")
    return df


@P.fixed_style
def plot_benchmark_summary(df, save_path):
    names = C.PARAM_NAMES
    ratio = np.array([[r[f"{n}_VAE_std"] / r[f"{n}_ref_std"] for n in names] for _, r in df.iterrows()])
    dmean = np.array([[r[f"{n}_VAE_mean"] - r[f"{n}_ref_mean"] for n in names] for _, r in df.iterrows()])
    fig, ax = plt.subplots(1, 2, figsize=(16, 0.5 * len(df) + 2.5), facecolor="white")
    im = ax[0].imshow(np.log2(ratio), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    for (i, j), v in np.ndenumerate(ratio):
        ax[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    ax[0].set_title("std(VAE) / std(reference)   (1 = same width)", fontweight="bold")
    fig.colorbar(im, ax=ax[0], label="log2 ratio")
    im = ax[1].imshow(dmean, cmap="RdBu_r", vmin=-0.05, vmax=0.05, aspect="auto")
    for (i, j), v in np.ndenumerate(dmean):
        ax[1].text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=9)
    ax[1].set_title("mean(VAE) - mean(reference)", fontweight="bold")
    fig.colorbar(im, ax=ax[1], label="difference in z")
    for a in ax:
        a.set_xticks(range(len(names)), names)
        a.set_yticks(range(len(df)), [f"P{p}" for p in df["pos"]])
    plt.tight_layout()
    fig.savefig(save_path, dpi=170)
    plt.close(fig)


# Spyder: set the run to analyse here and press F5 (None = the most recently trained run in Output/review29oct_fixed)
RUN_FOLDER = None


def latest_run_folder():
    runs = [os.path.join(C.OUTPUT_DIR, d) for d in os.listdir(C.OUTPUT_DIR)
            if os.path.exists(os.path.join(C.OUTPUT_DIR, d, "Test_predicted_props.npy"))]
    if not runs:
        raise FileNotFoundError(f"No trained run found in {C.OUTPUT_DIR}")
    return max(runs, key=lambda d: os.path.getmtime(os.path.join(d, "Test_predicted_props.npy")))


def main():
    parser = argparse.ArgumentParser(description="Post-processing of a PI-GC-KSmix-VAE run (fixed pipeline).")
    parser.add_argument("--run-folder", type=str, default=RUN_FOLDER,
                        help="name in Output/review29oct_fixed/ or full path (default: RUN_FOLDER, else the latest run)")
    parser.add_argument("--positions", type=int, nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=4000, help="samples of q per benchmark")
    parser.add_argument("--n-particles", type=int, default=4000, help="SMC particles of the reference posterior")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--recompute-gt", action="store_true")
    args, _ = parser.parse_known_args()
    run_folder = args.run_folder or latest_run_folder()
    run_postprocessing(run_folder, args.positions, args.n_samples, args.n_particles, args.workers, args.recompute_gt)


if __name__ == "__main__":
    main()
