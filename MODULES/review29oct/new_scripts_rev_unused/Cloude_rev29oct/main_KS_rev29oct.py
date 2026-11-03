# -*- coding: utf-8 -*-
"""
main_KS_rev29oct.py -- STEP (b): train the PI-GC-KSmix-VAE and save everything
needed by the post-processing.

Spyder : edit OVERRIDES below, F5.
Console: python main_KS_rev29oct.py [--gpu 2] [--epochs 400] [--obs fvr] [--n_modes 5] ...
         python main_KS_rev29oct.py --quick        (few-minute smoke test)

Schedule
  epochs < anneal_epochs : likelihood weight tau grows geometrically tau_min -> 1
  afterwards             : exact negative ELBO (tau = 1)
Model selection uses the VALIDATION set only (R3.2): the best epoch is the one
with the lowest validation negative ELBO (label-free) after annealing.

Output/<run_name>/
  config.json, feature_scaler.npz, encoder.weights.h5, history.json,
  timing.json (wall-clock and hardware, R3.8),
  pred_<set>.npz            q-parameters + truth + log q(truth) for val and every test set
  pred_<set>_bc.npz         same, with the healthy-baseline frequency correction (fine-model sets)
  bench_samples_<variant>.npz  posterior samples of the 10 benchmark cases
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# --- GPU selection must happen before tensorflow is imported ------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=str, default=None)
_gpu_arg, _ = _pre.parse_known_args()
if _gpu_arg.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_arg.gpu

import config as C  # noqa: E402  (sets KERAS_BACKEND before tensorflow is imported)
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from GC_KSmix_models import PIGCVAE  # noqa: E402
from preprocessing_and_eigen import (load_set, load_benchmarks, load_coarse_model,  # noqa: E402
                                     features, FeatureScaler)

DT = tf.float64

# Spyder: put any TrainConfig field here, e.g. dict(epochs=400, obs="fvr", n_modes=5)
OVERRIDES = dict()


def setup_devices():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    devices = [d.name for d in tf.config.list_physical_devices()]
    print("TensorFlow", tf.__version__, "| devices:", devices,
          "| CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    return devices


def default_run_name(cfg: C.TrainConfig):
    name = (f"PIGCKS_{cfg.obs}{cfg.n_modes}_K{cfg.num_KS}_beta{cfg.beta:g}_H{cfg.n_samples}"
            f"_ep{cfg.epochs}")
    return name + (f"_wfwd{cfg.w_fwd:g}" if cfg.w_fwd > 0 else "")


def importance_weights(d):
    """p(theta)/rho(theta) normalized to mean 1 (used only by the optional forward-KL term)."""
    lw = d["log_prior"] - d["log_doe"]
    w = np.exp(lw - lw.max())
    return w / w.mean()


def make_dataset(d, scaler, cfg, shuffle):
    feats = scaler(features(d["y_logf"], d["y_v"], d["y_r"], cfg.obs))
    tensors = (feats, d["theta"], d["y_logf"], d["y_v"], d["y_r"], importance_weights(d))
    ds = tf.data.Dataset.from_tensor_slices(tuple(np.asarray(t, np.float64) for t in tensors))
    if shuffle:
        ds = ds.shuffle(len(feats), seed=cfg.seed, reshuffle_each_iteration=True)
    return ds.batch(cfg.batch_size, drop_remainder=shuffle).prefetch(tf.data.AUTOTUNE)


def tau_schedule(epoch, cfg):
    if epoch >= cfg.anneal_epochs:
        return 1.0
    return float(cfg.tau_min ** (1.0 - epoch / cfg.anneal_epochs))


def train(cfg: C.TrainConfig, data_dir: Path = C.DATA_DIR, out_dir: Path = C.OUTPUT_DIR):
    devices = setup_devices()
    tf.keras.utils.set_random_seed(cfg.seed)
    cfg = replace(cfg, run_name=cfg.run_name or default_run_name(cfg))
    run_dir = Path(out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {run_dir}")

    # ------------------------------------------------------------------ data
    train_d = load_set("train", cfg.n_modes, data_dir, cfg.n_train)
    val_d = load_set("val", cfg.n_modes, data_dir)
    scaler = FeatureScaler().fit(features(train_d["y_logf"], train_d["y_v"], train_d["y_r"], cfg.obs))
    scaler.save(run_dir / "feature_scaler.npz")
    ds_train = make_dataset(train_d, scaler, cfg, shuffle=True)
    ds_val = make_dataset(val_d, scaler, cfg, shuffle=False)
    ds_val_elbo = make_dataset({k: (v[:cfg.n_val_elbo] if np.ndim(v) > 0 and len(v) == len(val_d["theta"])
                                    else v) for k, v in val_d.items()}, scaler, cfg, shuffle=False)
    coarse = load_coarse_model(data_dir)
    C.save_json(dict(train=cfg, data_dir=str(data_dir), coarse_el_per_zone=C.COARSE_EL_PER_ZONE,
                     prior_weights=C.PRIOR_SCENARIO_WEIGHTS, doe_weights=C.DOE_SCENARIO_WEIGHTS,
                     data_metadata=C.load_json(Path(data_dir) / "metadata.json")),
                run_dir / "config.json")

    # ----------------------------------------------------------------- model
    model = PIGCVAE(cfg, len(scaler.mean), coarse)
    steps_per_epoch = len(train_d["theta"]) // cfg.batch_size
    sched = tf.keras.optimizers.schedules.CosineDecay(cfg.lr, cfg.epochs * steps_per_epoch,
                                                      alpha=cfg.lr_final / cfg.lr)
    opt = tf.keras.optimizers.Adam(learning_rate=sched)
    tau = tf.Variable(1.0, dtype=DT, trainable=False)
    variables = model.encoder.trainable_variables

    @tf.function(jit_compile=cfg.jit_compile, reduce_retracing=True)
    def train_step(batch):
        with tf.GradientTape() as tape:
            out = model.losses(batch, tau, training=True)
        grads = tape.gradient(out["loss"], variables)
        n_bad = tf.add_n([tf.reduce_sum(tf.cast(~tf.math.is_finite(g), tf.int32)) for g in grads])
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        grads, gnorm = tf.clip_by_global_norm(grads, cfg.clipnorm)
        opt.apply_gradients(zip(grads, variables))
        out["grad_norm"] = gnorm
        out["n_nonfinite_grads"] = tf.cast(n_bad, DT)
        return out

    @tf.function(reduce_retracing=True)
    def val_nll_step(batch):
        feats, theta_true = batch[0], batch[1]
        return tf.reduce_mean(-model.log_q(model.params(feats), theta_true))

    @tf.function(reduce_retracing=True)
    def val_elbo_step(batch, seed):
        return model.losses(batch, tf.constant(1.0, DT), training=False,
                            n_samples=cfg.n_samples_val, seed=seed)

    def validate(epoch):
        rec = {"val_nll": float(np.mean([float(val_nll_step(b)) for b in ds_val]))}
        if epoch % cfg.val_every == 0 or epoch == cfg.epochs - 1:
            outs = [val_elbo_step(b, tf.constant(k, tf.int32)) for k, b in enumerate(ds_val_elbo)]
            for key in ("neg_elbo", "exp_log_lik", "kl"):
                rec[f"val_{key}"] = float(np.mean([float(o[key]) for o in outs]))
            rec["val_elbo_epoch"] = epoch
        return rec

    # -------------------------------------------------------------- training
    history = {}
    best, best_epoch, wait = np.inf, -1, 0
    best_weights = model.encoder.get_weights()
    t_start, epoch_times = time.time(), []
    for epoch in range(cfg.epochs):
        t0 = time.time()
        tau.assign(tau_schedule(epoch, cfg))
        acc = {}
        for batch in ds_train:
            for k, v in train_step(batch).items():
                acc.setdefault(k, []).append(float(v))
        rec = {f"train_{k}": float(np.mean(v)) for k, v in acc.items()}
        rec.update(validate(epoch))
        rec.update(tau=float(tau.numpy()), lr=float(sched(opt.iterations)), epoch_time=time.time() - t0)
        epoch_times.append(rec["epoch_time"])
        for k, v in rec.items():
            history.setdefault(k, []).append(v)
        history.setdefault("epoch", []).append(epoch)

        if "val_neg_elbo" in rec and epoch >= cfg.anneal_epochs:
            monitor = rec["val_neg_elbo"] + cfg.w_fwd * rec["val_nll"]
            if monitor < best - 1e-3:
                best, best_epoch, wait = monitor, epoch, 0
                best_weights = model.encoder.get_weights()
            else:
                wait += cfg.val_every
        shown = ("train_neg_elbo", "train_kl", "val_neg_elbo", "val_nll")
        msg = " ".join(f"{k}={rec[k]:.3f}" for k in shown if k in rec)
        print(f"ep {epoch:4d} tau={rec['tau']:.3f} {msg} ({rec['epoch_time']:.1f}s)", flush=True)
        if epoch >= cfg.anneal_epochs and wait >= cfg.patience:
            print(f"Early stopping at epoch {epoch} (best validation -ELBO {best:.3f} at epoch {best_epoch})")
            break

    train_time = time.time() - t_start
    if best_epoch >= 0:
        model.encoder.set_weights(best_weights)
    model.save(run_dir)
    C.save_json(history, run_dir / "history.json")
    n_ann = min(cfg.anneal_epochs, len(epoch_times))
    C.save_json(dict(train_time_s=train_time, epochs_run=len(epoch_times), best_epoch=best_epoch,
                     best_val_monitor=best, mean_epoch_time_s=float(np.mean(epoch_times)),
                     n_train=len(train_d["theta"]), steps_per_epoch=steps_per_epoch,
                     devices=devices, platform=platform.platform(), processor=platform.processor(),
                     tensorflow=tf.__version__, anneal_epochs_run=n_ann),
                run_dir / "timing.json")
    print(f"Training time: {train_time / 60:.1f} min ({np.mean(epoch_times):.1f} s/epoch)")
    predict_and_save(model, scaler, cfg, run_dir, data_dir)
    return run_dir


def predict_and_save(model, scaler, cfg, run_dir, data_dir=C.DATA_DIR):
    """q-parameters of every evaluation case (the test sets are used only here)."""
    run_dir = Path(run_dir)
    for name in ["val"] + C.TEST_SETS:
        try:
            raw = load_set(name, cfg.n_modes, data_dir)
        except FileNotFoundError:
            continue
        has_shift = np.any(raw["logf_baseline_shift"] != 0)
        for suffix, bc in (("", False), ("_bc", True)):
            if bc and not has_shift:
                continue
            d = load_set(name, cfg.n_modes, data_dir, baseline_correction=bc)
            feats = scaler(features(d["y_logf"], d["y_v"], d["y_r"], cfg.obs))
            p = model.predict_params(feats)
            lq = np.concatenate([model.log_q({k: tf.constant(v[i:i + 4096], DT) for k, v in p.items()},
                                             tf.constant(d["theta"][i:i + 4096], DT)).numpy()
                                 for i in range(0, len(feats), 4096)])
            np.savez_compressed(run_dir / f"pred_{name}{suffix}.npz", **p, theta=d["theta"],
                                scenario=d["scenario"], damaged_mask=d["damaged_mask"],
                                log_q_true=lq, y_logf=d["y_logf"], y_v=d["y_v"], y_r=d["y_r"])
            print(f"  predictions saved: {name}{suffix} ({len(feats)} cases)")
    for variant, bc in (("coarse", False), ("fine", False), ("fine", True)):
        b = load_benchmarks(variant, cfg.n_modes, data_dir, baseline_correction=bc)
        feats = scaler(features(b["y_logf"], b["y_v"], b["y_r"], cfg.obs))
        p = model.predict_params(feats)
        pt = {k: tf.constant(v, DT) for k, v in p.items()}
        samples = model.sample(pt, cfg.n_post_samples).numpy()
        tag = variant + ("_bc" if bc else "")
        np.savez_compressed(run_dir / f"bench_samples_{tag}.npz", **p, samples=samples, theta=b["theta"],
                            names=np.array(b["names"]), y_logf=b["y_logf"], y_v=b["y_v"], y_r=b["y_r"])
    print("  benchmark posteriors saved")


def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=str, default=None)
    for f, v in asdict(C.TrainConfig()).items():
        if isinstance(v, bool):
            ap.add_argument(f"--{f}", type=lambda s: s.lower() in ("1", "true", "yes"))
        elif isinstance(v, tuple):
            ap.add_argument(f"--{f}", type=int, nargs="+")
        else:
            ap.add_argument(f"--{f}", type=type(v))
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    args, _ = ap.parse_known_args()
    kw = {k: v for k, v in vars(args).items() if v is not None and k in C.TrainConfig.__dataclass_fields__}
    if "hidden" in kw:
        kw["hidden"] = tuple(kw["hidden"])
    if args.quick:
        kw.update(epochs=6, anneal_epochs=3, n_train=2048, batch_size=128, n_val_elbo=256,
                  patience=100, n_post_samples=500, val_every=1, run_name=kw.get("run_name", "QUICK_TEST"))
    return kw


def main(**overrides):
    kw = dict(OVERRIDES)
    kw.update(overrides)
    return train(C.TrainConfig(**kw))


if __name__ == "__main__":
    main(**parse_cli())
