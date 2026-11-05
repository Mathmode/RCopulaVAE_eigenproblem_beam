#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed_v3/check_consistency.py

Numerical self-checks (run after dataset_generation.py):
  1. TF likelihood of the VAE loss == NumPy likelihood of the reference posterior (same z, same data);
  2. Kumaraswamy-mixture quantile: F(Q(u)) = u, and the implicit gradients match finite differences;
  3. entropy estimator of q (exact copula part + MC marginals) == brute-force -E[log q] with the full copula density;
  4. likelihood calibration (from problem_config.json): E[NLL term at z*] / expected ~ 1.

    python MODULES/review29oct_fixed_v3/check_consistency.py
"""

import os
import sys

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from MODULES.review29oct_fixed_v3 import config as C, tf_keras

tf_keras()  # Keras on TensorFlow, float32 (reloads it if this console already imported it with jax)

import numpy as np
import tensorflow as tf
from scipy.stats import norm
from MODULES.review29oct_fixed_v3.fe_model import BridgeModel, neg_log_likelihood
from MODULES.review29oct_fixed_v3.preprocessing_and_eigen import load_bridge_data, load_baseline_matrices, load_problem_config
from MODULES.review29oct_fixed_v3.GC_KSmix_models import My_CopulaKSVAE_withEigen
from MODULES.review29oct_fixed_v3.GC_KSmix_functions import (
    ks_mixture_quantile, ks_mixture_cdf, ks_mixture_logpdf, build_correlation_matrices_from_cholesky,
    gaussian_copula_samples, copula_entropy,
)

RESULTS = []


def report(name, ok, detail):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def check_likelihood(data_dir):
    pcfg = load_problem_config(data_dir)
    lik = pcfg["likelihood"]
    data = load_bridge_data(data_dir)
    f_te, r_te, v_te, a_te, mean_f, std_f = data[8], data[9], data[10], data[11], data[12], data[13]
    _, K_basis, L_inv = load_baseline_matrices(data_dir)
    obs = pcfg["obs_types"]
    sizes = {"f": f_te.shape[1], "v": v_te.shape[1] * v_te.shape[2], "r": r_te.shape[1] * r_te.shape[2]}
    model = My_CopulaKSVAE_withEigen(
        input_dim=sum(sizes[k] for k in obs), num_dofs=K_basis.shape[-1], n_modes=f_te.shape[1], Ke_matrices=K_basis,
        L_inv=L_inv, n_dims=7, num_KS=3, num_samples=4, gamma=1.0, mean_f=mean_f, std_f=std_f, lbound=pcfg["lbound"],
        likelihood=lik, obs_types=obs, sensor_nodes=pcfg.get("sensor_nodes"))
    n, S, M, ns = 64, 4, f_te.shape[1], v_te.shape[2]
    batch = (f_te[:n], r_te[:n], v_te[:n], a_te[:n])
    _ = model(batch)
    z, _ = model.sample_posterior(batch)
    tf_nll = model.nll_terms(batch, z)[0].numpy()
    z = z.numpy()

    bm = BridgeModel(pcfg)
    f_hz = np.load(os.path.join(data_dir, "freqs_test.npy"))[:n].astype(np.float64)
    f, v, r = bm.solve(z.reshape(-1, 7))
    np_nll, _ = neg_log_likelihood(f_hz[:, None], v_te[:n, None].astype(np.float64), r_te[:n, None].astype(np.float64),
                                   f.reshape(n, S, M), v.reshape(n, S, M, ns), r.reshape(n, S, M, ns), bm.lik, obs)
    rel = np.abs(tf_nll - np_nll) / np.maximum(np.abs(np_nll), 1.0)
    report("TF (VAE loss) vs NumPy (reference) likelihood", np.median(rel) < 2e-3 and np.max(rel) < 2e-2,
           f"median rel. diff {np.median(rel):.1e}, max {np.max(rel):.1e} (float32 vs float64), NLL range "
           f"[{np_nll.min():.1f}, {np_nll.max():.1f}]")

    # loss terms are finite and the gradient flows to every encoder variable
    with tf.GradientTape() as tape:
        terms = model.compute_terms(batch)
    grads = tape.gradient(terms["loss"], model.trainable_variables)
    finite = all(g is not None and np.all(np.isfinite(g.numpy())) for g in grads)
    report("finite loss and gradients for every trainable variable", finite and np.isfinite(terms["loss"].numpy()),
           f"loss = {terms['loss'].numpy():.2f}, H = {terms['Entropy'].numpy():.2f}")


def check_quantile():
    rng = np.random.default_rng(0)
    B, K, D, S = 3, 3, 7, 5
    a = tf.constant(1.0 + rng.gamma(2.0, 20.0, (B, K, D)), tf.float32)
    b = tf.constant(1.0 + rng.gamma(2.0, 10.0, (B, K, D)), tf.float32)
    w = tf.nn.softmax(tf.constant(rng.normal(size=(B, K, D)), tf.float32), axis=1)
    u = tf.constant(rng.uniform(0.02, 0.98, (B, S, D)), tf.float32)
    x = ks_mixture_quantile(u, a, b, w)
    err = np.max(np.abs(ks_mixture_cdf(x, a, b, w).numpy() - u.numpy()))
    report("KS-mixture quantile inverts the CDF", err < 1e-4, f"max |F(Q(u)) - u| = {err:.1e}")

    # implicit gradient vs central finite differences (float64 bisection for the reference)
    a64, b64, w64, u64 = (tf.cast(t, tf.float64) for t in (a, b, w, u))
    with tf.GradientTape() as tape:
        tape.watch([a, b, w, u])
        loss = tf.reduce_sum(tf.sin(7.0 * ks_mixture_quantile(u, a, b, w)))
    g = tape.gradient(loss, [a, b, w, u])

    from MODULES.review29oct_fixed_v3.GC_KSmix_functions import _bisection_quantile

    def f64(a_, b_, w_, u_):
        lo, hi = tf.zeros_like(u_), tf.ones_like(u_)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            less = ks_mixture_cdf(mid, a_, b_, w_) < u_
            lo, hi = tf.where(less, mid, lo), tf.where(less, hi, mid)
        return tf.reduce_sum(tf.sin(7.0 * 0.5 * (lo + hi))).numpy()

    worst = 0.0
    for i, base in enumerate((a64, b64, w64, u64)):
        idx = (1, 2, 3) if i < 3 else (1, 2, 3)
        h = 1e-6 * max(1.0, float(base.numpy()[idx]))
        bump = np.zeros(base.shape)
        bump[idx] = h
        args_p = [a64, b64, w64, u64]
        args_m = [a64, b64, w64, u64]
        args_p[i] = base + bump
        args_m[i] = base - bump
        fd = (f64(*args_p) - f64(*args_m)) / (2 * h)
        an = g[i].numpy()[idx]
        worst = max(worst, abs(fd - an) / max(abs(fd), 1e-3))
    report("implicit quantile gradients vs finite differences", worst < 2e-2, f"worst relative error {worst:.1e}")


def check_entropy():
    rng = np.random.default_rng(1)
    K, D, N = 3, 7, 60000
    a = tf.constant(1.0 + rng.gamma(2.0, 8.0, (1, K, D)), tf.float32)
    b = tf.constant(1.0 + rng.gamma(2.0, 4.0, (1, K, D)), tf.float32)
    w = tf.nn.softmax(tf.constant(rng.normal(size=(1, K, D)), tf.float32), axis=1)
    off = tf.constant(rng.normal(0, 0.6, (1, D * (D - 1) // 2)), tf.float32)
    diag = tf.constant(np.full((1, D), 1.0), tf.float32)
    L = build_correlation_matrices_from_cholesky(off, diag, D)
    u = gaussian_copula_samples(L, N)
    x = ks_mixture_quantile(u, a, b, w)
    log_marg = ks_mixture_logpdf(x, a, b, w).numpy()[0].sum(-1)
    R = (L[0] @ tf.transpose(L[0])).numpy()
    xs = norm.ppf(np.clip(u.numpy()[0].astype(np.float64), 1e-12, 1 - 1e-12))
    log_cop = -0.5 * np.linalg.slogdet(R)[1] - 0.5 * np.sum(xs @ (np.linalg.inv(R) - np.eye(D)) * xs, axis=1)
    H_brute = -np.mean(log_cop + log_marg)
    H_est = copula_entropy(L).numpy()[0] - np.mean(log_marg)
    se = np.std(log_cop + log_marg) / np.sqrt(N)
    report("entropy: exact copula part + MC marginals vs brute-force -E[log q]", abs(H_est - H_brute) < 4 * se + 1e-3,
           f"{H_est:.4f} vs {H_brute:.4f} (MC s.e. {se:.4f})")


def check_calibration(data_dir):
    cc = load_problem_config(data_dir)["calibration_check"]["test"]
    vals = np.concatenate([[cc["ratio_f_total"]], cc["ratio_v"], cc["ratio_r"]])
    report("likelihood calibrated to the same-model test data", np.all(np.abs(vals - 1) < 0.1),
           f"E[NLL term at z*] / expected in [{vals.min():.3f}, {vals.max():.3f}]")


if __name__ == "__main__":
    data_dir = C.DATA_DIR
    check_calibration(data_dir)
    check_quantile()
    check_entropy()
    check_likelihood(data_dir)
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed.")
