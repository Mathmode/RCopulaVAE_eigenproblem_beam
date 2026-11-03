#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed/diagnose_old_pipeline.py

Reproducible diagnosis of the review29oct run (old data, old model, old post-processing):
  1. the copula "KL" term flattens at -gamma^2 * 50 = the clip of log q: the loss minimised the entropy of q;
  2. the copulas of the trained encoder are singular (|corr| -> 1) and the posterior means are biased;
  3. for benchmarks 6 and 7, four estimates of the posterior on the SAME old observation:
       - old "ground truth" (Sobol + points around z*, IS weights^0.55, beta_eff = 0.55, top-95 % filter)
       - old "estimated KS-Copula posterior" as plotted (q re-weighted by the old likelihood, top-95 % filter)
       - raw q(z|m) of the old trained encoder (what the VAE actually predicts)
       - reference posterior by SMC with the calibrated likelihood of fe_model.py (gamma = 1)
  4. the "bimodal" benchmarks 5, 8, 9 of the old test set are not physical observations.

    python MODULES/review29oct_fixed/diagnose_old_pipeline.py
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

from MODULES.review29oct_fixed import config as C, tf_keras

tf_keras()  # Keras on TensorFlow, float32 (reloads it if this console already imported it with jax)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from MODULES.review29oct_fixed.fe_model import BridgeModel, mac
from MODULES.review29oct_fixed.reference_posterior import smc_sample, load_problem_config
from MODULES.review29oct_fixed.postprocessing import sample_vae_posterior
from MODULES.review29oct_fixed.plotting import kde_1d, TITLES_7D, fixed_style

OLD_RUN = os.path.join(PROJECT_ROOT, "Output", "review29oct", "30Oct_v4_BridgeBearings_7D_3KSmix_Gamma0.35_Ep2500")
OLD_DATA = os.path.join(PROJECT_ROOT, "Data", "BridgeBearings_7D_DoE")


def resample(x, w, n, seed=0):
    idx = np.random.default_rng(seed).choice(len(x), size=n, p=w / w.sum())
    return x[idx]


def old_posteriors(pos, old_run, old_data, n=4000):
    """Old "ground truth" and old plotted estimate, computed with the old functions, as equally weighted samples."""
    from MODULES.review29oct import postprocessing_7D_bridge as OLD
    from MODULES.review29oct.preprocessing_and_eigen import load_bridge_data, load_baseline_matrices
    # same eigensolver as the old one minus its sign flip, which the old (squared-MAC) weights ignore;
    # built on the Keras returned by tf_keras(), so it also works in a console that had Keras on jax
    from MODULES.review29oct_fixed.preprocessing_and_eigen import Solve_Bridge_Eigenproblem

    info = np.load(os.path.join(old_run, "Problem_info.npy"), allow_pickle=True).item()
    preds = np.load(os.path.join(old_run, "Test_predicted_props.npy"), allow_pickle=True).item()
    data = load_bridge_data(old_data)
    f_te, r_te, v_te, a_te = data[8], data[9], data[10], data[11]
    _, K_basis, L_inv = load_baseline_matrices(old_data)
    eigen = Solve_Bridge_Eigenproblem(L_inv.shape[0], f_te.shape[1], L_inv)
    K_tf = tf.constant(K_basis, tf.float32)
    lb, beta = float(info["lbound"]), float(info["gamma"])
    is_bimodal = pos in (5, 8, 9)

    z_b, f_b, r_b, v_b = OLD.build_or_load_base_qmc_cloud_7d(
        os.path.join(PROJECT_ROOT, "Output", "review29oct", "GT_Posteriors_Cache_7D"), K_basis, L_inv, lbound=lb)
    z_all, f_all, r_all, v_all, prior_w = OLD.augment_gt_cloud_7d_for_sample(
        z_b, f_b, r_b, v_b, a_te[pos], K_tf, eigen, lbound=lb, n_local=4096, is_bimodal=is_bimodal, seed=1000 + pos)
    w_gt = OLD.compute_bayesian_weights_7d(f_all, r_all, v_all, f_te[pos], r_te[pos], v_te[pos],
                                           info["mean_f"], info["std_f"], beta, is_bimodal, prior_w)
    x_gt, w_gt_n = OLD.filter_top_weighted_samples(z_all, w_gt, quantile_threshold=0.95)

    z_q, _ = sample_vae_posterior(preds, pos, n, lb)
    f_q, r_q, v_q = OLD.forward_physics_batched_7d(z_q.astype(np.float32), K_tf, eigen)
    w_q = OLD.compute_bayesian_weights_7d(f_q, r_q, v_q, f_te[pos], r_te[pos], v_te[pos],
                                          info["mean_f"], info["std_f"], beta, is_bimodal)
    x_qf, w_qf = OLD.filter_top_weighted_samples(z_q, w_q, quantile_threshold=0.95)
    return resample(x_gt, w_gt_n, n), resample(x_qf, w_qf, n, seed=1), z_q, a_te[pos], lb


@fixed_style
def main():
    parser = argparse.ArgumentParser(description="Diagnosis of the review29oct run.")
    parser.add_argument("--old-run", type=str, default=OLD_RUN)
    parser.add_argument("--old-data", type=str, default=OLD_DATA)
    parser.add_argument("--positions", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--out-dir", type=str, default=os.path.join(C.OUTPUT_DIR, "diagnosis_old_review29oct"))
    args, _ = parser.parse_known_args()
    os.makedirs(args.out_dir, exist_ok=True)
    report = {}

    # --- 1-2. training history, copula degeneracy, bias ---
    h = np.load(os.path.join(args.old_run, "model_history.npy"), allow_pickle=True).item()
    info = np.load(os.path.join(args.old_run, "Problem_info.npy"), allow_pickle=True).item()
    preds = np.load(os.path.join(args.old_run, "Test_predicted_props.npy"), allow_pickle=True).item()
    g2 = float(info["gamma"]) ** 2
    od, dg = preds["test_offdiag_elems"], preds["test_diag_elems"]
    n = dg.shape[1]
    L = np.zeros((len(dg), n, n))
    L[:, np.tril_indices(n, -1)[0], np.tril_indices(n, -1)[1]] = od
    L[:, np.arange(n), np.arange(n)] = dg
    L /= np.linalg.norm(L, axis=-1, keepdims=True)
    R = L @ np.transpose(L, (0, 2, 1))
    eig_min = np.linalg.eigvalsh(R)[:, 0]
    max_corr = np.abs(R - np.eye(n)).max(axis=(1, 2))
    z_old = np.load(os.path.join(args.old_data, "alphas_test.npy"))

    from MODULES.review29oct_fixed.postprocessing import marginal_quantiles
    _, mean_q = marginal_quantiles(preds["test_a"], preds["test_b"], preds["test_weight_vals"], np.array([0.5]),
                                   float(info["lbound"]))
    healthy = z_old >= 0.96
    report.update({
        "KL_term_final_train": float(h["KL_Divergence_term"][-1]), "clip_value_-gamma2*50": -g2 * 50.0,
        "median_min_eigenvalue_R": float(np.median(eig_min)), "median_max_abs_corr": float(np.median(max_corr)),
        "bias_healthy_params": float(np.mean((mean_q - z_old)[healthy])),
        "bias_damaged_params(z*<0.9)": float(np.mean((mean_q - z_old)[z_old < 0.9])),
    })

    fig, ax = plt.subplots(1, 3, figsize=(19, 5), facecolor="white")
    ep = np.arange(1, len(h["loss"]) + 1)
    ax[0].plot(ep, h["KL_Divergence_term"], label=r"logged term $\gamma^2(-E_q[\log q])$ (train)")
    ax[0].plot(ep, h["val_KL_Divergence_term"], "--", label="(val)")
    ax[0].axhline(-g2 * 50, color="k", ls=":", label=r"$-\gamma^2 \cdot 50$ = clip of $\log q$ at +50")
    ax[0].set_xlabel("epoch")
    ax[0].set_title("Old loss: the entropy of q is MINIMISED until the clip", fontweight="bold")
    ax[0].legend()
    ax[1].hist(np.log10(eig_min), bins=50, color="#D62728")
    ax[1].set_xlabel(r"$\log_{10}$ smallest eigenvalue of the copula correlation R (old test set)")
    ax[1].set_title(f"Old copulas are singular (median max|corr| = {np.median(max_corr):.4f})", fontweight="bold")
    ax[2].scatter(z_old.ravel(), mean_q.ravel(), s=4, alpha=0.3)
    ax[2].plot([0.45, 1], [0.45, 1], "k--")
    ax[2].set_xlabel("true z*")
    ax[2].set_ylabel(r"old $E_q[z]$")
    ax[2].set_title("Old posterior means: healthy biased low, damaged biased high", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "Fig_old_training_and_copula.png"), dpi=160)
    plt.close(fig)

    # --- 3. four posteriors on the same old observation ---
    # the old data used the 5-element model with rotations: use the first corrected data set's configuration
    pcfg = load_problem_config(os.path.join(PROJECT_ROOT, "Data", "BridgeBearings_7D_fixed"))
    bm = BridgeModel(pcfg)
    f_hz = np.load(os.path.join(args.old_data, "freqs_test.npy")).astype(np.float64)
    v_all = np.load(os.path.join(args.old_data, "vertmodes_test.npy")).astype(np.float64)
    r_all = np.load(os.path.join(args.old_data, "rotmodes_test.npy")).astype(np.float64)
    for pos in args.positions:
        x_old_gt, x_old_est, x_q, z_true, lb = old_posteriors(pos, args.old_run, args.old_data)
        x_ref, _ = smc_sample(bm.log_likelihood_fn(f_hz[pos], v_all[pos], r_all[pos]), C.N_DIMS, lb,
                              n_particles=4000, rng=np.random.default_rng(pos))
        rows = {}
        fig, axes = plt.subplots(1, C.N_DIMS, figsize=(24, 4.6), facecolor="white")
        grid = np.linspace(lb, 1.0, 300)
        for d in range(C.N_DIMS):
            a = axes[d]
            a.plot(grid, kde_1d(x_ref[:, d], grid, lb), "k-", lw=2.4, label="Reference posterior (SMC, calibrated likelihood)")
            a.plot(grid, kde_1d(x_old_gt[:, d], grid, lb), color="#D62728", ls="--", lw=2, label='Old "ground truth"')
            a.plot(grid, kde_1d(x_q[:, d], grid, lb), color="#1F77B4", lw=2, label="Old VAE: raw q(z|m)")
            a.plot(grid, kde_1d(x_old_est[:, d], grid, lb), color="#1F77B4", ls=":", lw=2,
                   label="Old VAE as plotted (q x L, top-95 %)")
            a.axvline(z_true[d], color="#555555", ls="-.", lw=1.5, label="z*")
            a.set_title(f"{TITLES_7D[d]}  (z* = {z_true[d]:.2f})", fontsize=11, fontweight="bold")
            a.set_yticks([])
            a.set_xlim(lb, 1.0)
            rows[C.PARAM_NAMES[d]] = {k: [float(np.mean(x[:, d])), float(np.std(x[:, d]))] for k, x in
                                      (("reference", x_ref), ("old_GT", x_old_gt), ("old_VAE_raw", x_q),
                                       ("old_VAE_plotted", x_old_est))}
        hnd, lab = axes[0].get_legend_handles_labels()
        fig.legend(hnd, lab, loc="upper center", ncol=5, fontsize=11.5, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(f"Old benchmark {pos}: four 'posteriors' of the same observation (mean, std in the JSON)",
                     fontsize=13, fontweight="bold", y=1.08)
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        fig.savefig(os.path.join(args.out_dir, f"Fig_old_vs_reference_P{pos}.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)
        report[f"P{pos}_mean_std"] = rows
        print(f"P{pos}: std  reference {np.round(x_ref.std(0), 3)}\n     old GT    {np.round(x_old_gt.std(0), 3)}"
              f"\n     old q raw {np.round(x_q.std(0), 3)}\n     old plot  {np.round(x_old_est.std(0), 3)}")

    # --- 4. "bimodal" observations of the old test set ---
    mirror = [4, 3, 2, 1, 0, 6, 5]
    for pos in (5, 8, 9):
        _, v1, r1 = bm.solve(z_old[pos][None])
        _, v2, r2 = bm.solve(z_old[pos][mirror][None])
        report[f"old_P{pos}_1-MAC_vert_vs_zstar"] = (1 - mac(v_all[pos], v1[0])).round(3).tolist()
        report[f"old_P{pos}_1-MAC_vert_vs_mirror"] = (1 - mac(v_all[pos], v2[0])).round(3).tolist()

    with open(os.path.join(args.out_dir, "diagnosis_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    for k, v in report.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
