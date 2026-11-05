#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed_v3/main_train_KSVAE.py

Trains the PI-GC-KSmix-VAE (Gaussian copula + Kumaraswamy-mixture marginals) on the balanced 7D
bridge-on-bearings data set with the negative ELBO of GC_KSmix_models.py.

Run from the project root, e.g.
    python MODULES/review29oct_fixed_v3/main_train_KSVAE.py --epochs 1500 --gpu 2 --postprocess
Outputs go to Output/review29oct_fixed/<run_name>/:
    final_model_weights.weights.h5, model_history.npy (+ .csv), Problem_info.npy, run_config.json,
    Test_predicted_props.npy (same keys as review29oct: test_a, test_b, test_weight_vals, test_offdiag_elems,
    test_diag_elems).
"""

import os
import sys
import json
import time
import argparse

# --- 1. PARSE ARGUMENTS BEFORE IMPORTING TENSORFLOW (GPU selection) ---
try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MODULES.review29oct_fixed_v3 import config as C

parser = argparse.ArgumentParser(description="Train the PI-GC-KSmix-VAE (fixed ELBO).")
parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES (default: leave unchanged)")
parser.add_argument("--data-dir", type=str, default=C.DATA_DIR)
parser.add_argument("--epochs", type=int, default=1500)
parser.add_argument("--warmup", type=int, default=150, help="epochs of likelihood tempering (tau: tau0 -> 1)")
parser.add_argument("--tau0", type=float, default=0.01)
parser.add_argument("--num-ks", type=int, default=3, help="Kumaraswamy components per marginal")
parser.add_argument("--samples", type=int, default=6, help="Monte-Carlo samples of q per observation")
parser.add_argument("--gamma", type=float, default=1.0, help="entropy weight gamma^2 (1 = exact Bayesian posterior)")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--patience", type=int, default=300)
parser.add_argument("--obs", type=str, default=None, help="observed quantities, e.g. fvr or f (default: data set)")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--run-name", type=str, default=None)
parser.add_argument("--postprocess", action="store_true", help="run postprocessing.py at the end")
args, _ = parser.parse_known_args()

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from MODULES.review29oct_fixed_v3 import tf_keras

K = tf_keras()  # Keras on TensorFlow, float32 (even if ~/.keras/keras.json says jax / float64)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU initialisation warning: {e}")
print(f"Devices: {[g.name for g in gpus] or 'CPU only'}")

from MODULES.review29oct_fixed_v3.preprocessing_and_eigen import (
    load_bridge_data, load_baseline_matrices, load_problem_config, load_test_split,
)
from MODULES.review29oct_fixed_v3.GC_KSmix_models import (
    My_CopulaKSVAE_withEigen, LikelihoodTempering, LearningRateLogger, FastInMemoryBestWeights,
)


def main():
    K.utils.set_random_seed(args.seed)
    K.backend.set_floatx("float32")

    # --- 1. DATA ---
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(PROJECT_ROOT, args.data_dir)
    if not os.path.exists(os.path.join(data_dir, "problem_config.json")):
        from MODULES.review29oct_fixed_v3.dataset_generation import generate_full_benchmark_datasets
        generate_full_benchmark_datasets(data_dir)
    pcfg = load_problem_config(data_dir)
    obs_types = args.obs or pcfg["obs_types"]
    lik = pcfg["likelihood"]

    (f_train, r_train, v_train, a_train,
     f_val, r_val, v_val, a_val,
     f_test, r_test, v_test, a_test,
     mean_f, std_f) = load_bridge_data(data_dir)
    _, K_basis, L_inv = load_baseline_matrices(data_dir)

    train_ds = tf.data.Dataset.from_tensor_slices(((f_train, r_train, v_train, a_train), a_train))
    train_ds = train_ds.cache().shuffle(len(f_train), seed=args.seed).batch(args.batch).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices(((f_val, r_val, v_val, a_val), a_val))
    val_ds = val_ds.cache().batch(args.batch).prefetch(tf.data.AUTOTUNE)

    n_dims, n_modes, lbound = int(pcfg["n_dims"]), f_train.shape[1], float(pcfg["lbound"])
    n_dofs = K_basis.shape[-1]
    sizes = {"f": f_train.shape[1], "v": v_train.shape[1] * v_train.shape[2], "r": r_train.shape[1] * r_train.shape[2]}
    input_dim = sum(sizes[k] for k in obs_types)

    run_name = args.run_name or f"KSVAE_fixed_{n_dims}D_{args.num_ks}KS_obs-{obs_types}_Gamma{args.gamma}_Ep{args.epochs}"
    folder_path = os.path.join(C.OUTPUT_DIR, run_name)
    os.makedirs(folder_path, exist_ok=True)

    # --- 2. MODEL ---
    model = My_CopulaKSVAE_withEigen(
        input_dim=input_dim, num_dofs=n_dofs, n_modes=n_modes, Ke_matrices=K_basis, L_inv=L_inv,
        n_dims=n_dims, num_KS=args.num_ks, num_samples=args.samples, gamma=args.gamma,
        mean_f=mean_f, std_f=std_f, lbound=lbound,
        likelihood=lik, obs_types=obs_types, sensor_nodes=pcfg.get("sensor_nodes"),
    )
    steps_per_epoch = int(np.ceil(len(f_train) / args.batch))
    lr_schedule = K.optimizers.schedules.CosineDecay(args.lr, decay_steps=args.epochs * steps_per_epoch, alpha=0.02)
    model.compile(optimizer=K.optimizers.Adam(learning_rate=lr_schedule, global_clipnorm=10.0))
    _ = model((f_train[:2], r_train[:2], v_train[:2], a_train[:2]))

    run_cfg = {k: v for k, v in vars(args).items()}
    run_cfg.update({"data_dir": data_dir, "obs_types": obs_types, "input_dim": input_dim, "run_name": run_name,
                    "likelihood": lik, "objective": "tau * E_q[NLL] - gamma^2 * H[q]  (uniform prior)"})
    with open(os.path.join(folder_path, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(run_cfg, fh, indent=2)

    # --- 3. TRAINING ---
    print(f"Training PI-GC-KSmix-VAE: {n_dims}D, K={args.num_ks}, S={args.samples}, gamma={args.gamma}, "
          f"obs={obs_types}, {args.epochs} epochs (warm-up {args.warmup}) -> {folder_path}")
    best_cb = FastInMemoryBestWeights(monitor="val_neg_elbo", patience=args.patience, start_epoch=args.warmup)
    csv_cb = K.callbacks.CSVLogger(os.path.join(folder_path, "model_history.csv"))
    start_time = time.time()
    history = model.fit(
        train_ds, epochs=args.epochs, validation_data=val_ds, verbose=2,
        callbacks=[LikelihoodTempering(args.tau0, args.warmup), LearningRateLogger(), best_cb, csv_cb],
    )
    training_time = time.time() - start_time
    n_run = len(history.history["loss"])
    print(f"Total training time: {training_time:.1f} s ({training_time / n_run:.2f} s/epoch)")

    model.save_weights(os.path.join(folder_path, "final_model_weights.weights.h5"))
    np.save(os.path.join(folder_path, "model_history.npy"), history.history, allow_pickle=True)

    # --- 4. POSTERIOR PARAMETERS OF THE TEST SETS ---
    # 'test' (same model as training) -> Test_predicted_props.npy; other test sets -> Test_predicted_props_<name>.npy
    test_sets = {"test": (f_test, r_test, v_test, a_test)}
    for split in pcfg.get("test_sets", {}):
        if split != "test":
            test_sets[split] = load_test_split(data_dir, split, mean_f, std_f)
    for split, inputs in test_sets.items():
        test_a, test_b, test_w, test_offdiag, test_diag = model.Encoder_model.predict(
            inputs, batch_size=args.batch, verbose=0)
        name = "Test_predicted_props.npy" if split == "test" else f"Test_predicted_props_{split}.npy"
        np.save(os.path.join(folder_path, name), {
            "test_a": test_a, "test_b": test_b, "test_weight_vals": test_w,
            "test_offdiag_elems": test_offdiag, "test_diag_elems": test_diag,
        }, allow_pickle=True)

    Problem_info = {
        "input_dim_enc": input_dim, "n_dims": n_dims, "n_zones": int(pcfg["n_zones"]), "num_KS": args.num_ks,
        "n_samples": args.samples, "epochs": args.epochs, "epochs_run": n_run, "warmup": args.warmup,
        "best_epoch": best_cb.best_epoch, "gamma": args.gamma, "std_f": std_f, "mean_f": mean_f, "lbound": lbound,
        "training_time_sec": training_time, "data_dir": data_dir, "obs_types": obs_types, "likelihood": lik,
        "test_sets": list(test_sets),
    }
    np.save(os.path.join(folder_path, "Problem_info.npy"), Problem_info, allow_pickle=True)
    print(f"Saved weights, history, test posterior parameters and problem info to {folder_path}")

    if args.postprocess:
        from MODULES.review29oct_fixed_v3.postprocessing import run_postprocessing
        run_postprocessing(folder_path)


if __name__ == "__main__":
    main()
