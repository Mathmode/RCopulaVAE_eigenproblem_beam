# -*- coding: utf-8 -*-
"""
ks28_train.py -- Training loop, model-discrepancy calibration and posterior export.

Plain custom loop (tf.function + GradientTape) instead of Keras .fit with losses that
read model attributes: easier to follow, independent of the Keras version (2.15 on the
server, 3.x locally) and it lets us report exactly the objective of the paper.

Run folder written by `train_model` (Output/KS28/<run_name>/):
    run_config.json            dataset, ModelConfig, TrainConfig, noise used, scaler stats, hardware
    encoder_best.weights.h5    weights with the best VALIDATION objective (beta = 1)
    history.csv                per-epoch train / validation metrics and wall-clock time
    posteriors/<split>.npz     q parameters (log_pi, a, b, log_w, L, lo, hi) for val, test_id,
                               val_shift and every test family + the truth labels
                               -> everything after this point is NumPy only (ks28_postprocess)
"""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from dataclasses import dataclass, asdict

import numpy as np
from . import ks28_tf_setup  # noqa: F401  (forces the TensorFlow backend of Keras)
import tensorflow as tf

from .ks28_config import NoiseConfig
from .ks28_dataset import (load_dataset_folder, load_npz, list_test_families, InputScaler)
from .ks28_likelihood import estimate_model_discrepancy
from .ks28_tf_model import KS28CopulaVAE, ModelConfig


# =============================================================================
@dataclass
class TrainConfig:
    epochs: int = 400
    batch_size: int = 256
    lr: float = 1e-3
    lr_final: float = 1e-5            # exponential decay from lr to lr_final over `epochs`
    clipnorm: float = 1.0
    beta_start: float = 0.1           # entropy/prior weight during warm-up
    warmup_epochs: int = 30           # beta: beta_start -> 1 linearly, then 1 forever
    patience: int = 60                # early stopping on the validation objective (after warm-up)
    n_samples_val: int = 8            # H used for the validation objective
    seed: int = 1234
    max_train: int = 0                # 0 -> use all; >0 -> subsample (quick tests)
    calibrate_model_error: bool = False   # inflate the likelihood noise with the
                                          # discrepancy estimated on val_shift (validation only)


# =============================================================================
def hardware_info():
    gpus = tf.config.list_physical_devices("GPU")
    names = []
    for g in gpus:
        try:
            names.append(tf.config.experimental.get_device_details(g).get("device_name", g.name))
        except Exception:
            names.append(g.name)
    return dict(platform=platform.platform(), processor=platform.processor(),
                tensorflow=tf.__version__, gpus=names, n_cpu=os.cpu_count())


def _arrays(d, scaler, mc):
    x = scaler.transform(d["f_obs"], d["phi_obs"], mc.n_modes_used, mc.use_mode_shapes)
    return (x, d["f_obs"].astype(np.float32), d["phi_obs"].astype(np.float32),
            d["theta"].astype(np.float32))


# =============================================================================
def train_model(dataset_folder: str, out_root: str, run_name: str,
                mc: ModelConfig = None, tc: TrainConfig = None, verbose=True):
    mc = ModelConfig() if mc is None else mc
    tc = TrainConfig() if tc is None else tc
    tf.keras.utils.set_random_seed(tc.seed)
    run = os.path.join(out_root, run_name)
    os.makedirs(os.path.join(run, "posteriors"), exist_ok=True)

    cfg, noise, prior, extra, basis = load_dataset_folder(dataset_folder)
    disc = None
    if tc.calibrate_model_error:
        disc = estimate_model_discrepancy(dataset_folder, cfg, noise, mc.n_modes_used)
        noise = NoiseConfig(**{**asdict(noise), "sigma_f_model": disc["sigma_f_model"],
                               "sigma_phi_model": disc["sigma_phi_model"]})
        print(f"[KS28] model discrepancy from val_shift: {disc}")

    tr = load_npz(os.path.join(dataset_folder, "train.npz"))
    va = load_npz(os.path.join(dataset_folder, "val.npz"))
    if tc.max_train > 0:
        tr = {k: (v[:tc.max_train] if v.ndim > 0 else v) for k, v in tr.items()}
    scaler = InputScaler().fit(tr["f_obs"][:, :mc.n_modes_used])
    Xtr, Ftr, Ptr, _ = _arrays(tr, scaler, mc)
    Xva, Fva, Pva, _ = _arrays(va, scaler, mc)

    model = KS28CopulaVAE(cfg, noise, prior, basis, Xtr.shape[1], mc, seed=tc.seed)
    steps_per_epoch = int(np.ceil(len(Xtr) / tc.batch_size))
    decay = (tc.lr_final / tc.lr) ** (1.0 / max(tc.epochs * steps_per_epoch, 1))
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(tc.lr, decay_steps=1, decay_rate=decay)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_sched, clipnorm=tc.clipnorm)
    variables = model.encoder.trainable_variables
    try:                     # create the optimizer slots outside tf.function (Keras 2.11+ / 3)
        opt.build(variables)
    except Exception:
        pass

    H, Hv = mc.n_samples, tc.n_samples_val

    @tf.function(reduce_retracing=True)
    def train_step(x, f, p, beta):
        with tf.GradientTape() as tape:
            loss, mets = model.loss(x, f, p, H, beta, training=True)
        grads = tape.gradient(loss, variables)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        opt.apply_gradients(zip(grads, variables))
        return mets

    @tf.function(reduce_retracing=True)
    def val_step(x, f, p):
        _, mets = model.loss(x, f, p, Hv, 1.0, training=False)
        return mets

    ds_tr = (tf.data.Dataset.from_tensor_slices((Xtr, Ftr, Ptr))
             .shuffle(len(Xtr), seed=tc.seed, reshuffle_each_iteration=True)
             .batch(tc.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
    ds_va = tf.data.Dataset.from_tensor_slices((Xva, Fva, Pva)).batch(1024)

    cfg_dump = dict(dataset_folder=os.path.abspath(dataset_folder), model=mc.to_dict(), train=asdict(tc),
                    noise_used=asdict(noise), model_discrepancy=disc, scaler=scaler.to_dict(),
                    hardware=hardware_info(), n_train=len(Xtr), n_val=len(Xva),
                    latent_names=cfg.latent_names)
    with open(os.path.join(run, "run_config.json"), "w") as fh:
        json.dump(cfg_dump, fh, indent=2)

    hist_path = os.path.join(run, "history.csv")
    fields = ["epoch", "beta", "time_s", "train_loss", "train_nll", "train_entropy", "train_log_prior",
              "train_pi_max", "val_loss", "val_nll", "val_entropy", "val_log_prior", "val_pi_max"]
    with open(hist_path, "w", newline="") as fh:
        csv.writer(fh).writerow(fields)

    best, wait, t0 = np.inf, 0, time.time()
    best_path = os.path.join(run, "encoder_best.weights.h5")
    for ep in range(tc.epochs):
        beta = min(1.0, tc.beta_start + (1.0 - tc.beta_start) * ep / max(tc.warmup_epochs, 1))
        te = time.time()
        acc = {}
        for x, f, p in ds_tr:
            m = train_step(x, f, p, tf.constant(beta, tf.float32))
            for k, v in m.items():
                acc.setdefault(k, []).append(float(v))
        vacc = {}
        for x, f, p in ds_va:
            m = val_step(x, f, p)
            for k, v in m.items():
                vacc.setdefault(k, []).append(float(v))
        tr_m = {k: np.mean(v) for k, v in acc.items()}
        va_m = {k: np.mean(v) for k, v in vacc.items()}
        dt = time.time() - te
        with open(hist_path, "a", newline="") as fh:
            csv.writer(fh).writerow([ep, beta, dt] + [tr_m[k] for k in ("loss", "nll", "entropy", "log_prior", "pi_max")]
                                    + [va_m[k] for k in ("loss", "nll", "entropy", "log_prior", "pi_max")])
        if verbose and (ep % 5 == 0 or ep == tc.epochs - 1):
            print(f"ep {ep:4d} beta {beta:.2f} | train {tr_m['loss']:.4f} (nll {tr_m['nll']:.1f}, H {tr_m['entropy']:.2f}) "
                  f"| val {va_m['loss']:.4f} (nll {va_m['nll']:.1f}) | pi_max {va_m['pi_max']:.2f} | {dt:.1f}s")
        if beta >= 1.0:                       # model selection only on the true objective
            if va_m["loss"] < best:
                best, wait = va_m["loss"], 0
                model.encoder.save_weights(best_path)
            else:
                wait += 1
                if wait >= tc.patience:
                    print(f"[KS28] early stopping at epoch {ep} (best val loss {best:.4f})")
                    break
    total = time.time() - t0
    if not os.path.exists(best_path):
        model.encoder.save_weights(best_path)
    model.encoder.load_weights(best_path)
    cfg_dump["training_time_s"] = total
    cfg_dump["best_val_loss"] = float(best)
    with open(os.path.join(run, "run_config.json"), "w") as fh:
        json.dump(cfg_dump, fh, indent=2)
    print(f"[KS28] training finished in {total / 60:.1f} min")

    export_all_posteriors(model, scaler, dataset_folder, run, mc)
    return model, run


# =============================================================================
def export_all_posteriors(model, scaler, dataset_folder, run, mc):
    """Save q parameters + truth for val, test_id, val_shift and all test families."""
    splits = {"val": "val.npz", "test_id": "test_id.npz", "val_shift": "val_shift.npz"}
    for fam in list_test_families(dataset_folder):
        splits[fam] = os.path.join("tests", f"{fam}.npz")
    for name, rel in splits.items():
        d = load_npz(os.path.join(dataset_folder, rel))
        x = scaler.transform(d["f_obs"], d["phi_obs"], mc.n_modes_used, mc.use_mode_shapes)
        prm = model.export_params(x)
        keep = {k: d[k] for k in ("theta", "f_obs", "phi_obs", "n_damaged", "damaged_mask",
                                  "support_anomaly", "label_exact") if k in d}
        np.savez_compressed(os.path.join(run, "posteriors", f"{name}.npz"), **prm, **keep)
    print(f"[KS28] posterior parameters exported to {os.path.join(run, 'posteriors')}")


def load_trained_model(run):
    """Rebuild the model of a run folder (TF needed) -> (model, scaler, cfg, noise, prior)."""
    with open(os.path.join(run, "run_config.json")) as fh:
        rc = json.load(fh)
    cfg, noise, prior, _, basis = load_dataset_folder(rc["dataset_folder"])
    noise = NoiseConfig(**rc["noise_used"])
    md = rc["model"]
    md["hidden"] = tuple(md["hidden"])
    mc = ModelConfig(**md)
    scaler = InputScaler.from_dict(rc["scaler"])
    input_dim = mc.n_modes_used * (1 + (cfg.n_sensors if mc.use_mode_shapes else 0))
    model = KS28CopulaVAE(cfg, noise, prior, basis, input_dim, mc)
    model.encoder.load_weights(os.path.join(run, "encoder_best.weights.h5"))
    return model, scaler, cfg, noise, prior
