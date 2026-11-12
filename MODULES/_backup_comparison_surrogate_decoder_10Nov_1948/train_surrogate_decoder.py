#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/comparison_surrogate_decoder/train_surrogate_decoder.py

STEP 1 of the comparison with the Gaussian copula VAE of Ref. 41 (R1.4): pre-trains the data-driven decoder (surrogate of the
forward operator, surrogate_decoder.py) at the damage states of the SAME training split as the VAE (validation split for the
convergence rule), and saves it with everything needed for the comparison. Convergence rule (make_plateau_controller): the
learning rate is halved after --lr-patience epochs without an improvement of the validation MSE larger than --rel-tol of its
best value, and the training stops at --lr-min under the same condition, as in the VAE training. Targets (--targets): nominal (default) = noise-free modal
data of the nominal model at those damage states, as when a surrogate is trained on simulations; observed = the noisy
observations of the training split (their noise hides the remaining error of the surrogate from the validation loss, so early
stopping keeps under-trained weights).


  surrogate_decoder.weights.h5, surrogate_decoder.keras   trained network
  surrogate_config.json                                     architecture, scaling, data, training time, accuracy, cost
  surrogate_history.csv / .npy                              training and validation loss per epoch
  surrogate_accuracy.csv                                    error of the surrogate with respect to the eigenvalue solver
  Fig_surrogate_accuracy.png/.pdf                           frequency errors and 1 - MAC per mode

Accuracy is measured against the eigenvalue solver of the decoder (fe_model.BridgeModel, nominal model) at the damage states of
every split and at 20 000 states drawn uniformly over the prior box [lbound, 1]^7. Cost: time of one decoder evaluation for a
training batch (256 observations x 6 samples = 1 536 damage states), forward and forward + gradient, on this machine.

Run from the project root:
    python MODULES/comparison_surrogate_decoder/train_surrogate_decoder.py --gpu 0
"""

import argparse
import json
import os
import sys
import time

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.training_common import select_gpu

OUTPUT_DIR = os.path.join(C.PROJECT_ROOT, "Output", "comparison_surrogate_decoder")

parser = argparse.ArgumentParser(description="Step 1: pre-train the surrogate decoder (Ref. 41) on the VAE training split.")
parser.add_argument("--gpu", type=str, default=None)
parser.add_argument("--data-dir", type=str, default=C.DATA_DIR)
parser.add_argument("--n-modes", type=int, default=None, help="first M modes of the data set (default: all)")
parser.add_argument("--hidden", type=int, nargs="+", default=[10, 30, 50, 70, 80],
                    help="hidden layers of the surrogate (default: the architecture of Ref. 41)")
parser.add_argument("--epochs", type=int, default=20000, help="cap only; the convergence rule below stops the training")
parser.add_argument("--rel-tol", type=float, default=0.01,
                    help="an improvement of the validation loss counts only if it is larger than this fraction of the best value")
parser.add_argument("--lr-patience", type=int, default=100,
                    help="epochs without such an improvement before halving the learning rate (at --lr-min: stop)")
parser.add_argument("--lr-min", type=float, default=1e-5)
parser.add_argument("--patience", type=int, default=None, help="(not used; kept for old command lines)")
parser.add_argument("--threads", type=int, default=None, help="CPU threads of TensorFlow (default: all)")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--out-dir", type=str, default=None, help="folder of the surrogate (default: Output/comparison_surrogate_decoder/surrogate_decoder_<data tag>)")
parser.add_argument("--targets", choices=["observed", "nominal"], default="nominal",
                    help="observed: the noisy modal data of the training split (exactly the VAE training data); nominal: the "
                         "noise-free modal data of the nominal model at the same damage states (best case for the surrogate)")
args, _ = parser.parse_known_args()
select_gpu(args.gpu)


def data_tag(pcfg, n_modes):
    tag = str(pcfg.get("dataset_version", "")).split("_")[0]
    return tag if tag.endswith("modes") else f"{tag}{n_modes}modes"


def load_split(data_dir, split):
    return {k: np.load(os.path.join(data_dir, f"{k}_{split}.npy")) for k in ("alphas", "freqs", "vertmodes")}


def mac(a, b):
    num = np.sum(a * b, axis=-1) ** 2
    return num / (np.sum(a * a, axis=-1) * np.sum(b * b, axis=-1) + 1e-300)


def accuracy_rows(name, z, bridge, sur):
    f_ex, v_ex, _ = bridge.solve(z)
    f_su, v_su, _ = sur.solve(z)
    rel = 100.0 * (f_su / f_ex - 1.0)
    one_m_mac = 1.0 - mac(v_ex, v_su)
    rows = []
    for j in range(rel.shape[1]):
        rows.append({"set": name, "n": len(z), "mode": j + 1,
                     "freq_err_mean_abs_pct": float(np.mean(np.abs(rel[:, j]))),
                     "freq_err_p95_abs_pct": float(np.percentile(np.abs(rel[:, j]), 95)),
                     "freq_err_max_abs_pct": float(np.max(np.abs(rel[:, j]))),
                     "freq_err_mean_pct": float(np.mean(rel[:, j])),
                     "one_minus_MAC_mean": float(np.mean(one_m_mac[:, j])),
                     "one_minus_MAC_p95": float(np.percentile(one_m_mac[:, j], 95)),
                     "one_minus_MAC_max": float(np.max(one_m_mac[:, j]))})
    return rows, rel, one_m_mac


def make_plateau_controller(K, rel_tol, lr_patience, lr_min, factor=0.5):
    """
    Convergence rule of the surrogate, the analogue of the rule of the VAE training (GC_KSmix_models.ConvergenceController)
    for a loss without a natural scale: an improvement of the validation loss counts only if it exceeds rel_tol times the
    best value so far. After lr_patience epochs without such an improvement, the learning rate is halved; at lr_min the
    same condition stops the training. The weights with the lowest validation loss are kept. (Plain early stopping counts
    any improvement, however small: with noise-free targets the validation loss keeps decreasing by negligible amounts
    and the training only ends at the cap of epochs.)
    """
    class PlateauController(K.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best, self.best_epoch, self.best_weights = np.inf, 0, None
            self.ref, self.ref_epoch, self.converged = np.inf, 0, False

        def on_epoch_end(self, epoch, logs=None):
            val, done = float((logs or {})["val_loss"]), epoch + 1
            if val < self.best:
                self.best, self.best_epoch = val, done
                self.best_weights = [w.numpy() for w in self.model.weights]
            if val < self.ref * (1.0 - rel_tol):
                self.ref, self.ref_epoch = val, done
            elif done - self.ref_epoch >= lr_patience:
                lr = float(np.asarray(self.model.optimizer.learning_rate))
                if lr <= lr_min * 1.001:
                    print(f"\nConverged at epoch {done}: no improvement larger than {100 * rel_tol:g} % of the validation loss "
                          f"in {lr_patience} epochs at the lowest learning rate {lr:.3g}.")
                    self.converged, self.model.stop_training = True, True
                else:
                    new_lr = max(factor * lr, lr_min)
                    self.model.optimizer.learning_rate = new_lr
                    self.ref_epoch = done
                    print(f"\nEpoch {done}: val MSE {val:.3e} (best {self.best:.3e}); learning rate {lr:.3g} -> {new_lr:.3g}")
            if logs is not None:
                logs["lr"] = float(np.asarray(self.model.optimizer.learning_rate))

        def on_train_end(self, logs=None):
            if self.best_weights is not None:
                for var, value in zip(self.model.weights, self.best_weights):
                    var.assign(value)
    return PlateauController()


def time_call(fn, reps=30, warmup=3):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def main():
    import tensorflow as tf
    from MODULES.review29oct_fixed import tf_keras
    from MODULES.review29oct_fixed.fe_model import BridgeModel, frequency_precision
    from MODULES.review29oct_fixed.preprocessing_and_eigen import (
        load_problem_config, load_baseline_matrices, assemble_global_Kmatrices_bearings, Solve_Bridge_Eigenproblem,
    )
    from MODULES.comparison_surrogate_decoder.surrogate_decoder import (
        build_surrogate, layer_spec, make_targets, MinMaxScaler, to_unit, save_surrogate, update_config, SurrogateBridge,
    )

    K = tf_keras()
    K.utils.set_random_seed(args.seed)
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, args.threads // 2))
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    device = [tf.config.experimental.get_device_details(g).get("device_name", g.name) for g in gpus] or ["CPU only"]

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(C.PROJECT_ROOT, args.data_dir)
    if args.n_modes is not None:
        from MODULES.review29oct_fixed.derive_mode_subset import derive_mode_subset
        data_dir = derive_mode_subset(data_dir, args.n_modes)
    pcfg = load_problem_config(data_dir)
    lbound, n_modes = float(pcfg["lbound"]), int(pcfg["n_modes"])
    bias, _ = frequency_precision(pcfg["likelihood"])
    out_dir = args.out_dir or os.path.join(OUTPUT_DIR, f"surrogate_decoder_{data_tag(pcfg, n_modes)}")
    os.makedirs(out_dir, exist_ok=True)

    tr, va = load_split(data_dir, "train"), load_split(data_dir, "val")
    n_sens = tr["vertmodes"].shape[-1]
    if args.targets == "observed":       # the noisy observations of the VAE training data, minus the calibrated bias b
        y_tr = make_targets(tr["freqs"], tr["vertmodes"], bias)
        y_va = make_targets(va["freqs"], va["vertmodes"], bias)
    else:                                # noise-free nominal model at the same damage states: targets are log f(z) directly
        nominal = BridgeModel(pcfg)
        f_n, v_n, _ = nominal.solve(tr["alphas"])
        y_tr = make_targets(f_n, v_n, np.zeros(n_modes))
        f_n, v_n, _ = nominal.solve(va["alphas"])
        y_va = make_targets(f_n, v_n, np.zeros(n_modes))
    scaler = MinMaxScaler().fit(y_tr)
    u_tr, u_va = to_unit(tr["alphas"], lbound), to_unit(va["alphas"], lbound)
    s_tr, s_va = scaler.transform(y_tr), scaler.transform(y_va)

    hidden = tuple(args.hidden)
    acts, inits = layer_spec(hidden)
    model = build_surrogate(K, u_tr.shape[1], y_tr.shape[1], hidden, acts, inits)
    model.compile(optimizer=K.optimizers.Adam(learning_rate=args.lr), loss="mse")
    n_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    print(f"Surrogate decoder, hidden {list(hidden)} ({'architecture of Ref. 41' if hidden == (10, 30, 50, 70, 80) else 'custom'}): "
          f"{u_tr.shape[1]} -> {list(model.output_shape)} | {n_params} parameters | train {len(u_tr)}, val {len(u_va)} | "
          f"devices {device} -> {out_dir}")

    controller = make_plateau_controller(K, args.rel_tol, args.lr_patience, args.lr_min)
    callbacks = [controller, K.callbacks.CSVLogger(os.path.join(out_dir, "surrogate_history.csv"))]
    t0 = time.time()
    hist = model.fit(u_tr.astype(np.float32), s_tr.astype(np.float32), validation_data=(u_va.astype(np.float32), s_va.astype(np.float32)),
                     epochs=args.epochs, batch_size=args.batch, verbose=2, callbacks=callbacks, shuffle=True)
    train_time = time.time() - t0
    h = hist.history
    best_epoch = int(controller.best_epoch)
    np.save(os.path.join(out_dir, "surrogate_history.npy"), h, allow_pickle=True)
    print(f"Surrogate trained in {train_time:.1f} s ({len(h['loss'])} epochs, best {best_epoch}: val MSE {controller.best:.3e}; "
          f"converged {controller.converged})")

    meta = {"data_dir": data_dir, "dataset_version": pcfg.get("dataset_version"), "lbound": lbound, "n_in": int(u_tr.shape[1]),
            "n_out": int(y_tr.shape[1]), "n_modes": n_modes, "n_sensors": int(n_sens), "bias_logf": np.asarray(bias).tolist(),
            "targets": ("observed: [log f_obs - b (n_modes), observed vertical shapes (n_modes x n_sensors)] of the training split"
                        if args.targets == "observed" else
                        "nominal: [log f(z), vertical shapes] of the nominal model at the damage states of the training split"),
            "target_type": args.targets,
            "training_split": "train", "early_stopping_split": "val", "n_train": int(len(u_tr)), "n_val": int(len(u_va)),
            "trainable_parameters": n_params, "training_time_sec": train_time, "epochs_run": len(h["loss"]),
            "best_epoch": best_epoch, "best_val_mse_scaled": float(controller.best), "converged": bool(controller.converged),
            "seed": args.seed,
            "optimizer": f"Adam lr {args.lr}, batch {args.batch}",
            "convergence_rule": (f"learning rate halved after {args.lr_patience} epochs without an improvement of the validation "
                                 f"MSE larger than {100 * args.rel_tol:g} % of its best value; stop at lr {args.lr_min:g} under "
                                 f"the same condition; best weights kept; cap {args.epochs} epochs"),
            "time_per_epoch_sec": train_time / max(len(h["loss"]), 1),
            "devices": device}
    save_surrogate(out_dir, model, scaler, meta, hidden, acts, inits)

    # --- accuracy with respect to the eigenvalue solver of the decoder (nominal model) ---
    bridge = BridgeModel(pcfg)
    sur = SurrogateBridge(pcfg, out_dir, K)
    rng = np.random.default_rng(args.seed)
    sets = {"train": tr["alphas"], "val": va["alphas"]}
    for split in pcfg.get("test_sets", {"test": None}):
        sets[split] = np.load(os.path.join(data_dir, f"alphas_{split}.npy"))
    sets["uniform prior box"] = rng.uniform(lbound, 1.0, size=(20000, u_tr.shape[1]))
    rows, keep = [], {}
    for name, z in sets.items():
        r, rel, om = accuracy_rows(name, np.asarray(z, np.float64), bridge, sur)
        rows += r
        keep[name] = (rel, om)
    acc = pd.DataFrame(rows)
    acc.to_csv(os.path.join(out_dir, "surrogate_accuracy.csv"), index=False)
    print(acc[acc["set"].isin(["test", "uniform prior box"])].round(4).to_string(index=False))

    # residuals of the observed test data (test set A) against the surrogate and against the eigenvalue solver
    te = load_split(data_dir, "test")
    res = {}
    for name, b in (("surrogate", sur), ("eigenvalue solver", bridge)):
        f_p, v_p, _ = b.solve(te["alphas"])
        d = np.log(te["freqs"]) - np.asarray(bias) - np.log(f_p)
        res[name] = {"std_log_f_residual": np.std(d, axis=0).tolist(), "mean_log_f_residual": np.mean(d, axis=0).tolist(),
                     "mean_1_minus_MAC_obs": np.mean(1.0 - mac(te["vertmodes"], v_p), axis=0).tolist()}
    res["likelihood_sigma_log_f"] = list(pcfg["likelihood"]["sig_logf"])

    # --- cost of one decoder evaluation for a training batch (256 observations x 6 samples) ---
    _, K_basis, L_inv = load_baseline_matrices(data_dir)
    eig = Solve_Bridge_Eigenproblem(K_basis.shape[-1], n_modes, L_inv, pcfg.get("sensor_nodes"))
    Kb = tf.constant(K_basis, tf.float32)
    zb = tf.constant(rng.uniform(lbound, 1.0, size=(1536, u_tr.shape[1])), tf.float32)

    def eig_fwd(z):
        f, _, v = eig(assemble_global_Kmatrices_bearings(z, Kb))
        return f, v

    def sur_fwd(z):
        return (model((z - lbound) / (1.0 - lbound), training=False),)

    def with_grad(fn):
        def run():
            with tf.GradientTape() as tape:
                tape.watch(zb)
                out = fn(zb)
                s = tf.add_n([tf.reduce_sum(o) for o in out])
            return tape.gradient(s, zb)
        return run

    cost = {"batch_size_states": 1536, "devices": device,
            "eigen_forward_s": time_call(lambda: eig_fwd(zb)), "surrogate_forward_s": time_call(lambda: sur_fwd(zb)),
            "eigen_forward_backward_s": time_call(with_grad(eig_fwd)), "surrogate_forward_backward_s": time_call(with_grad(sur_fwd)),
            "note": "median of 30 calls after 3 warm-up calls, on the machine that trained the surrogate"}
    print("decoder cost per batch of 1536 states [s]:", {k: round(v, 5) for k, v in cost.items() if k.endswith("_s")})

    summ = acc[acc["set"] == "uniform prior box"]
    update_config(out_dir, accuracy_uniform_box={
        "freq_err_mean_abs_pct_per_mode": summ["freq_err_mean_abs_pct"].round(5).tolist(),
        "freq_err_p95_abs_pct_per_mode": summ["freq_err_p95_abs_pct"].round(5).tolist(),
        "freq_err_max_abs_pct_per_mode": summ["freq_err_max_abs_pct"].round(5).tolist(),
        "one_minus_MAC_mean_per_mode": summ["one_minus_MAC_mean"].tolist(),
        "one_minus_MAC_p95_per_mode": summ["one_minus_MAC_p95"].tolist()},
        residuals_test_A=res, decoder_cost=cost)

    # --- figure: errors on the uniform prior box ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rel, om = keep["uniform prior box"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].boxplot([np.abs(rel[:, j]) for j in range(rel.shape[1])], showfliers=False)
    ax[0].set_xlabel("mode"); ax[0].set_ylabel("|frequency error| [%]")
    ax[0].axhline(100 * np.mean(pcfg["likelihood"]["parts"]["noise_logf"]), color="r", ls="--", lw=1, label="identification noise (std)")
    ax[0].legend(fontsize=8); ax[0].set_title("Surrogate vs eigenvalue solver: frequencies")
    ax[1].boxplot([np.maximum(om[:, j], 1e-8) for j in range(om.shape[1])], showfliers=False)
    ax[1].set_yscale("log"); ax[1].set_xlabel("mode"); ax[1].set_ylabel("1 - MAC")
    ax[1].set_title("Surrogate vs eigenvalue solver: mode shapes")
    fig.suptitle(f"20 000 damage states drawn uniformly over [{lbound}, 1]$^{u_tr.shape[1]}$", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"Fig_surrogate_accuracy.{ext}"), dpi=200)
    plt.close(fig)
    print(f"[OK] surrogate decoder saved to {out_dir}")
    return out_dir


import numpy as np      # noqa: E402  (after the GPU selection, as in the other entry scripts)
import pandas as pd     # noqa: E402

if __name__ == "__main__":
    main()
