# -*- coding: utf-8 -*-
"""
ks28_smoke_test.py -- Run this FIRST on every machine (laptop and GPU server).

    python -m MODULES.KS_review28oct.ks28_smoke_test --data Data/<dataset_folder>

It checks, in about one minute, that the TensorFlow implementation is consistent with
the NumPy reference implementation that generated the data and computes the ground truth:

  1. TF decoder  == NumPy FE model         (frequencies and MAC of the mode shapes)
  2. TF prior    == NumPy prior            (log-density of the scenario prior)
  3. TF log q    == NumPy log q            (mixture of Gaussian copulas, KS marginals)
  4. TF sampler  ~  NumPy sampler          (moments)
  5. gradients are finite and reach EVERY head of the encoder, including the copula
     correlation (it received no physics gradient in the old code)
  6. timing of one training step, eigensolver on the default device vs on the CPU
Prints PASS/FAIL per check and the timing to choose --eigh_cpu.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--C", type=int, default=3)
    ap.add_argument("--K", type=int, default=2, help="K>1 also tests the bisection/implicit path")
    args = ap.parse_args(argv)

    from . import ks28_tf_setup  # noqa: F401  (forces the TensorFlow backend of Keras)
    import tensorflow as tf
    from .ks28_dataset import load_dataset_folder, load_npz, InputScaler
    from .ks28_fe_model import build_basis, forward
    from .ks28_likelihood import mac_matrix_diag
    from .ks28_priors import sample_prior, log_prior
    from .ks28_q_numpy import log_q as log_q_np, sample_q
    from .ks28_tf_model import KS28CopulaVAE, ModelConfig
    from .ks28_tf_distributions import log_q as log_q_tf, sample_stratified, raw_to_params

    ok_all = True

    def report(name, ok, msg=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}")

    print(f"TensorFlow {tf.__version__}, Keras backend: {tf.keras.backend.backend()}, GPUs: {tf.config.list_physical_devices('GPU')}")
    cfg, noise, prior, extra, basis = load_dataset_folder(args.data)
    rng = np.random.default_rng(0)
    th, _ = sample_prior(64, cfg, prior, rng)

    # 1 decoder ---------------------------------------------------------------------------
    mc = ModelConfig(n_components=args.C, n_ks=args.K, n_samples=2)
    tr = load_npz(os.path.join(args.data, "val.npz"))
    scaler = InputScaler().fit(tr["f_obs"])
    x = scaler.transform(tr["f_obs"][:16], tr["phi_obs"][:16])
    model = KS28CopulaVAE(cfg, noise, prior, basis, x.shape[1], mc)
    f_tf, p_tf = model.decoder(tf.constant(th, tf.float32))
    f_np, p_np = forward(th.astype(np.float32).astype(np.float64), build_basis(cfg), cfg)
    rel = np.max(np.abs(f_tf.numpy() / f_np - 1))
    mac = np.min(mac_matrix_diag(p_tf.numpy().astype(np.float64), p_np))
    report("1 decoder", rel < 1e-5 and mac > 1 - 1e-6, f"max rel freq diff {rel:.2e}, min MAC {mac:.8f}")

    # 2 prior -----------------------------------------------------------------------------
    lp_tf = model.log_prior(tf.constant(th, tf.float32)).numpy()
    lp_np = log_prior(th.astype(np.float32).astype(np.float64), cfg, prior)
    d = np.max(np.abs(lp_tf - lp_np))
    report("2 prior", d < 1e-2, f"max |diff| {d:.2e}")

    # 3 log q -----------------------------------------------------------------------------
    p = model.q_params(tf.constant(x))
    pn = {k: v.numpy().astype(np.float64) for k, v in p.items()}
    t_eval = rng.uniform(cfg.latent_lower, cfg.latent_upper, size=(16, 5, cfg.n_latent)).astype(np.float32)
    lq_tf = log_q_tf(tf.constant(t_eval)[:, None], p).numpy()[:, 0]
    lq_np = log_q_np(t_eval.astype(np.float64), pn)
    d = np.max(np.abs(lq_tf - lq_np))
    report("3 log q", d < 5e-2, f"max |diff| {d:.2e} (float32 vs float64)")

    # 4 sampler ---------------------------------------------------------------------------
    th_tf, _ = sample_stratified(p, 2000)                                # (B, C, H, D)
    w = np.exp(pn["log_pi"])[:, :, None, None]
    m_tf = np.sum(w * th_tf.numpy().mean(2, keepdims=True), axis=1)[:, 0]
    s_np, _ = sample_q(pn, 6000, rng)
    m_np = s_np.mean(1)
    d = np.max(np.abs(m_tf - m_np) / (cfg.latent_upper - cfg.latent_lower))
    report("4 sampler means", d < 0.03, f"max |diff| / range {d:.3f}")

    # 5 gradients -------------------------------------------------------------------------
    f_o = tf.constant(tr["f_obs"][:16], tf.float32)
    p_o = tf.constant(tr["phi_obs"][:16], tf.float32)
    with tf.GradientTape() as tape:
        loss, mets = model.loss(tf.constant(x), f_o, p_o, 2, 1.0, training=True)
    grads = tape.gradient(loss, model.encoder.trainable_variables)
    finite = all(g is not None and np.all(np.isfinite(g.numpy())) for g in grads)
    heads = {}
    for v, g in zip(model.encoder.trainable_variables, grads):
        name = getattr(v, "path", None) or v.name          # Keras 3: .path = "layer/kernel"
        heads[name] = 0.0 if g is None else float(tf.norm(g))
    zero = [n for n, gn in heads.items() if gn == 0.0 and "logits_pi" not in n]
    report("5 gradients finite", finite, f"loss {float(loss):.3f}")
    report("5 gradients reach all heads (incl. copula)", len(zero) == 0, f"zero-grad variables: {zero}")
    print("    gradient norms:", {k.split('/')[0]: round(v, 4) for k, v in heads.items() if 'kernel' in k})

    # 6 timing ----------------------------------------------------------------------------
    for dev in ["", "/CPU:0"]:
        mc_t = ModelConfig(n_components=args.C, n_ks=1, n_samples=4, eigh_device=dev)
        m_t = KS28CopulaVAE(cfg, noise, prior, basis, x.shape[1], mc_t)
        opt = tf.keras.optimizers.Adam(1e-3)
        try:
            opt.build(m_t.encoder.trainable_variables)
        except Exception:
            pass
        xb = scaler.transform(tr["f_obs"][:256], tr["phi_obs"][:256])
        fb, pb = tr["f_obs"][:256].astype(np.float32), tr["phi_obs"][:256].astype(np.float32)

        @tf.function
        def step(xx, ff, pp):
            with tf.GradientTape() as tape:
                l, _ = m_t.loss(xx, ff, pp, 4, 1.0, True)
            gr = tape.gradient(l, m_t.encoder.trainable_variables)
            opt.apply_gradients(zip(gr, m_t.encoder.trainable_variables))
            return l
        step(xb, fb, pb)
        t0 = time.time()
        for _ in range(5):
            step(xb, fb, pb)
        dt = (time.time() - t0) / 5
        print(f"    timing: batch 256 x C={args.C} x H=4 eigensolves, eigh on '{dev or 'default'}': {dt * 1000:.0f} ms/step")

    print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED -- please send me the output")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
