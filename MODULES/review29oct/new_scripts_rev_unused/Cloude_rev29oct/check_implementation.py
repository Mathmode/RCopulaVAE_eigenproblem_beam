# -*- coding: utf-8 -*-
"""
check_implementation.py -- Numerical self-checks (run after dataset_generation.py).

 1. Affine inference model == direct FE assembly + generalized eigensolver.
 2. TF decoder == NumPy model; TF likelihood == NumPy likelihood; TF prior == NumPy prior.
 3. Likelihood of noise-free data is maximal at the true parameters (sanity of signs/scales).
 4. Sampler and density agree: log q from the sampling path == log q(theta).
 5. Marginal samples follow the Kumaraswamy-mixture CDF (Kolmogorov-Smirnov test).
 6. Implicit reparameterization gradients == finite differences (a, and the copula factor L).
 7. Prior is continuous inside the support (required by the pathwise gradient).
 8. Importance weights p/rho of the optional forward-KL term are bounded.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import numpy as np
import tensorflow as tf
from scipy.stats import kstest

import GC_KSmix_functions as KS
from dataset_generation import BeamFE
from preprocessing_and_eigen import (load_coarse_model, ModalModelNP, ModalDecoderTF, align_to_reference,
                                     log_likelihood_np, log_likelihood_tf)
from priors import build_prior, build_doe

DT = tf.float64


def random_params(B, D, K, rng):
    off = tf.constant(rng.normal(0, 0.8, (B, D * (D - 1) // 2)), DT)
    diag = tf.constant(rng.normal(0, 0.5, (B, D)), DT)
    return {"logw": tf.nn.log_softmax(tf.constant(rng.normal(0, 1, (B, D, K)), DT), -1),
            "a": tf.constant(1 + rng.gamma(2, 3, (B, D, K)), DT),
            "b": tf.constant(1 + rng.gamma(2, 3, (B, D, K)), DT),
            "L": KS.copula_cholesky(off, diag, D)}


def main():
    rng = np.random.default_rng(0)
    coarse = load_coarse_model()
    beam = C.BeamConfig()
    M = beam.n_modes_max
    prior, doe = build_prior(), build_doe()
    th = prior.sample(64, rng)[0]
    sig = (float(coarse["sig_f"]), coarse["sig_v"], coarse["sig_r"])

    # 1. affine model vs direct assembly
    fe = BeamFE(beam, C.COARSE_EL_PER_ZONE)
    logf_np, v_np, r_np = ModalModelNP(coarse, M)(th)
    err = 0.0
    for i in range(5):
        ek = np.repeat(th[i, :C.N_ZONES], C.COARSE_EL_PER_ZONE)
        lf, v, r = fe.modes(*fe.assemble(ek, np.ones(fe.n_el), th[i, C.N_ZONES:]), M)
        err = max(err, np.abs(lf - logf_np[i]).max(), (1 - np.sum(v * v_np[i], -1) ** 2).max())
    print(f"1. affine model vs direct FE: max |dlogf| or 1-MAC = {err:.2e}")

    # 2. TF vs NumPy
    logf_tf, v_tf, r_tf = ModalDecoderTF(coarse, M)(tf.constant(th, DT))
    mac = np.sum(v_np * v_tf.numpy(), -1) ** 2
    y = (logf_np[0] + 0.01, v_np[0] + 0.01, r_np[0] - 0.02)
    ll_np = log_likelihood_np(*y, logf_np, v_np, r_np, *sig, "fvr")
    rep = lambda a: tf.constant(np.repeat(a[None], 64, 0))  # noqa: E731
    ll_tf = log_likelihood_tf(rep(y[0]), rep(y[1]), rep(y[2]), logf_tf, v_tf, r_tf, sig[0],
                              tf.constant(sig[1], DT), tf.constant(sig[2], DT), "fvr").numpy()
    lp_tf = prior.tf_log_prob_fn()(tf.constant(th, DT)).numpy()
    print(f"2. decoder max|dlogf| = {np.abs(logf_np - logf_tf.numpy()).max():.2e}, min MAC = {mac.min():.10f}; "
          f"max|dloglik| = {np.abs(ll_np - ll_tf).max():.2e}; max|dlogprior| = "
          f"{np.abs(prior.log_prob(th) - lp_tf).max():.2e}")

    # 3. noise-free data: likelihood maximal at the truth among perturbed parameters
    t0 = th[:1]
    lf0, v0, r0 = ModalModelNP(coarse, M)(t0)
    v0, r0 = align_to_reference(v0, r0, coarse["v_ref"], coarse["r_ref"], coarse["sig_v"], coarse["sig_r"])
    pert = np.clip(t0 + 0.02 * rng.standard_normal((200, C.N_DIMS)), C.LBOUND, C.UBOUND)
    lfp, vp, rp = ModalModelNP(coarse, M)(pert)
    ll_true = log_likelihood_np(lf0[0], v0[0], r0[0], lf0, ModalModelNP(coarse, M)(t0)[1],
                                ModalModelNP(coarse, M)(t0)[2], *sig, "fvr")[0]
    ll_pert = log_likelihood_np(lf0[0], v0[0], r0[0], lfp, vp, rp, *sig, "fvr")
    print(f"3. noise-free likelihood at truth {ll_true:.1f} >= max over 200 perturbations "
          f"{ll_pert.max():.1f}: {ll_true >= ll_pert.max()}")

    # 4.-6. variational family
    lo, hi = (tf.constant(v, DT) for v in C.param_bounds())
    p = random_params(4, C.N_DIMS, 3, rng)
    s, lq_path = KS.sample(p, 4000, lo, hi, return_log_q=True)
    print(f"4. max|log q(sampling path) - log q(theta)| = "
          f"{np.abs((lq_path - KS.log_q(p, s, lo, hi)).numpy()).max():.2e}")
    x = (s.numpy()[0, :, 2] - C.LBOUND) / (C.UBOUND - C.LBOUND)
    cdf = lambda u: KS.mixture_eval(tf.constant(u[:, None], DT), p["logw"][0, 2], p["a"][0, 2],  # noqa: E731
                                    p["b"][0, 2])[1].numpy().ravel()
    print(f"5. KS test of marginal samples: p-value = {kstest(x, cdf).pvalue:.3f}")
    g = tf.constant(rng.normal(0, 1, (4, 1, C.N_DIMS)), DT)
    a_var = tf.Variable(p["a"])
    with tf.GradientTape() as tape:
        obj = tf.reduce_sum(KS.reparam_quantile(g, p["logw"][:, None], a_var[:, None], p["b"][:, None]))
    grad = tape.gradient(obj, a_var).numpy()
    h = 1e-6
    a2 = p["a"].numpy()
    a2[0, 3, 1] += h
    fd = (tf.reduce_sum(KS.reparam_quantile(g, p["logw"][:, None], tf.constant(a2)[:, None],
                                            p["b"][:, None])) - obj).numpy() / h
    off = tf.Variable(rng.normal(0, 0.8, (2, C.N_DIMS * (C.N_DIMS - 1) // 2)))
    diag = tf.constant(rng.normal(0, 0.5, (2, C.N_DIMS)), DT)
    pL = {k: v[:2] for k, v in p.items() if k != "L"}

    def theta_sum(off_t):
        pL["L"] = KS.copula_cholesky(off_t, diag, C.N_DIMS)
        return tf.reduce_sum(KS.sample(pL, 50, lo, hi, seed=7))

    with tf.GradientTape() as tape:
        obj_L = theta_sum(off)
    gL = tape.gradient(obj_L, off).numpy()
    off2 = off.numpy()
    off2[1, 4] += h
    fdL = (theta_sum(tf.constant(off2)) - obj_L).numpy() / h
    print(f"6. d sum(x)/d a: implicit {grad[0, 3, 1]:.6e} vs finite difference {fd:.6e}; "
          f"d sum(theta)/d L_offdiag: implicit {gL[1, 4]:.6e} vs finite difference {fdL:.6e}")

    # 7. prior continuity inside the support: a scaled Beta component vanishes at an
    #    edge lying inside [LBOUND, UBOUND] only if its exponent there is > 1
    #    (alpha at the lower edge, beta at the upper edge).
    inner_lo = prior.lo > C.LBOUND + 1e-12
    inner_hi = prior.hi < C.UBOUND - 1e-12
    ok = np.all(prior.a[inner_lo] > 1) and np.all(prior.b[inner_hi] > 1)
    print(f"7. prior components vanish at every interior edge (density continuous inside the support): {ok}")

    # 8. importance weights
    th_doe = doe.sample(20000, rng, stratified=True)[0]
    lw = prior.log_prob(th_doe) - doe.log_prob(th_doe)
    w = np.exp(lw - lw.max())
    w /= w.mean()
    print(f"8. p/rho on DoE samples (normalized): max {w.max():.2f}, ESS/N {w.sum() ** 2 / np.sum(w ** 2) / len(w):.3f}")


if __name__ == "__main__":
    main()
