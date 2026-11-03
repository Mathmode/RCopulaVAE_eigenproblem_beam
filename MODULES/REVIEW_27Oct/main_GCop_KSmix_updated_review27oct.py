#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_GCop_KSmix_updated_review27oct.py
======================================
Full Updated Version of `MODULES/review_27oct/Last_used_scripts/main_GCop_KSmix.py`
for the MS CPENG-8674 Major Revision.

Features & Supported Fixes:
---------------------------
1. Supports both Unimodal (`--num-ks 1`) and Multimodal (`--num-ks 3` or `5`) Kumaraswamy Mixture
   marginals with Gaussian Copula correlation.
2. Uses the new Realistic Early-Stage Bridge DoE (`TRAIN_E5_realistic_doe` by default, or
   `TRAIN_E5_finemesh20_realistic_doe` where the physics mesh has 20 FE elements while the Copula
   still estimates strictly `n_dims = 5` damage zone factors!).
3. Supports Sensor-Limitation / Multimodality Experiments (Concern 3):
   - `--sensor-regime full`: All 5 frequencies + vertical & rotational mode shapes (reference).
   - `--sensor-regime no_rotations`: Unmeasured rotational mode shapes (`zero_rotations=True`,
     standard accelerometer-only field setup).
   - `--sensor-regime sparse_2modes`: Only first 2 modes (`n_modes=2`) + no rotations.
   - `--sensor-regime freq_only`: Only natural frequencies (mode shapes zeroed out; on a symmetric
     beam e1<->e5, e2<->e4, frequencies alone cannot distinguish left vs. right damage, inducing
     true physical bimodality!).
4. Saves BOTH `Val_predicted_props.npy` and `Test_predicted_props.npy`, plus `metrics_summary.json`
   inside `code/Output/review_27oct/KS_Copula/<run_name>/`.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import tensorflow as tf
import tensorflow.keras as K

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

# --- GPU CONFIGURATION ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[OK] GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(f"[WARN] GPU Initialization Error: {e}")

from MODULES.review_27oct.preprocessing_updated_review27oct import (
    load_data_updated,
    load_known_matrices_updated,
)
from MODULES.review_27oct.GC_KSmix_models_updated_review27oct import (
    My_CopulaKSVAE_withEigen_Updated,
)
from MODULES.review_27oct.GC_GMm_functions_updated_review27oct import (
    build_correlation_matrices_from_cholesky,
)
from MODULES.review_27oct.GC_KSmix_functions_for_results_analysis_updated_review27oct import (
    calculate_ks_mixture_metrics,
)
from MODULES.COPULAS.GC_postprocessing_copulas import (
    plot_trainval_loss,
    plot_JointPDF_loss,
    plot_Freqs_loss,
)


class DelayedEarlyStopping(K.callbacks.Callback):
    def __init__(self, patience=1000, start_epoch=8000):
        super(DelayedEarlyStopping, self).__init__()
        self.patience = patience
        self.start_epoch = start_epoch
        self.best_loss = np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        if current_loss is None:
            return
        if epoch >= self.start_epoch:
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"\n[OK] Early stopping triggered at epoch {epoch}.")
                    self.model.stop_training = True
        elif current_loss < self.best_loss:
            self.best_loss = current_loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Kumaraswamy Mixture Gaussian Copula CVAE (review_27oct)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="review_27oct/TRAIN_E5_realistic_doe",
        help="Dataset folder inside Data/ (e.g., review_27oct/TRAIN_E5_realistic_doe or 06Oct_Data_Noisy_E5_Lvl25_MassPerturb5).",
    )
    parser.add_argument(
        "--num-ks",
        type=int,
        default=3,
        help="Number of Kumaraswamy mixture components per marginal: 1 (unimodal) or 3, 5 (multimodal).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 1000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size (default: 256).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial learning rate for ExponentialDecay (default: 3e-4).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.50,
        help="Copula density weight gamma (default: 0.50).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of Monte Carlo samples per batch item during training (default: 10).",
    )
    parser.add_argument(
        "--sensor-regime",
        type=str,
        default="full",
        choices=["full", "no_rotations", "sparse_2modes", "freq_only"],
        help="Sensor configuration to study uncertainty & multimodality due to sensor limitations.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix tag for output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    K.utils.set_random_seed(1234)
    K.backend.set_floatx('float32')

    # --- 1. SENSOR REGIME CONFIGURATION ---
    if args.sensor_regime == "full":
        n_modes_override = None
        zero_rotations = False
        include_rotations = True
        include_vert_modes = True
    elif args.sensor_regime == "no_rotations":
        n_modes_override = None
        zero_rotations = True
        include_rotations = False
        include_vert_modes = True
    elif args.sensor_regime == "sparse_2modes":
        n_modes_override = 2
        zero_rotations = True
        include_rotations = False
        include_vert_modes = True
    elif args.sensor_regime == "freq_only":
        n_modes_override = None
        zero_rotations = True
        include_rotations = False
        include_vert_modes = False
    else:
        raise ValueError(f"Unknown sensor_regime: {args.sensor_regime}")

    # --- 2. DATA LOADING ---
    data_path = os.path.join(CODE_ROOT, "Data", args.dataset)
    lbound = 0.45
    batch_size = args.batch_size

    print(f"Loading data from {data_path} (sensor_regime={args.sensor_regime})...")
    (
        Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train,
        Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val,
        Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test,
        mean_f, std_f
    ) = load_data_updated(
        data_path,
        batch_size=batch_size,
        n_modes=n_modes_override,
        zero_rotations=zero_rotations,
    )

    if args.sensor_regime == "freq_only":
        Vertmodes_true_train = np.zeros_like(Vertmodes_true_train)
        Vertmodes_true_val = np.zeros_like(Vertmodes_true_val)
        Vertmodes_true_test = np.zeros_like(Vertmodes_true_test)

    # Note on Concern 1 (FE Mesh Size vs. Copula Dimension):
    # - `n_dims` is ALWAYS the number of unknown damage zones (5 in our bridge benchmark).
    # - The physics decoder always uses the 5-zone reference physics matrices (or 20-FE zone matrices
    #   via Kzone_free), so the Copula only ever estimates D = 5 marginals and 5*(5-1)/2 = 10 correlation
    #   parameters!
    n_elements = 5
    n_dofs = 2 * (n_elements + 1)
    fixed_dofs_indices = [0, n_dofs - 2]

    # Always load the 5-element physics matrices for the 5-zone decoder (if dataset is finemesh20,
    # fall back to TRAIN_E5_realistic_doe's 5-element decoder matrices)
    mass_file = os.path.join(data_path, "Mass_matrix.npy")
    use_direct = False
    if os.path.exists(mass_file):
        m_shape = np.load(mass_file).shape[0]
        if m_shape == 2 * n_elements:
            use_direct = True
    if use_direct:
        Mfree, Ke_matrices, L_inv, Kzone_free, Ksupport_free, K0_free = load_known_matrices_updated(
            data_path, n_elements
        )
    else:
        fallback_path = os.path.join(CODE_ROOT, "Data", "review_27oct", "TRAIN_E5_realistic_doe")
        print(f"  Using 5-zone decoder matrices from {fallback_path} for D=5 Copula parameterization.")
        Mfree, Ke_matrices, L_inv, Kzone_free, Ksupport_free, K0_free = load_known_matrices_updated(
            fallback_path, n_elements
        )

    # --- 3. MODEL CONFIGURATION ---
    flat_vert_tr = Vertmodes_true_train.reshape(Vertmodes_true_train.shape[0], -1)
    flat_rot_tr = Rotmodes_true_train.reshape(Rotmodes_true_train.shape[0], -1)
    modal_train_concat = np.concatenate([Freqs_true_train, flat_vert_tr, flat_rot_tr], axis=1)
    modal_mean = np.mean(modal_train_concat, axis=0).astype("float32")
    modal_std = np.std(modal_train_concat, axis=0).astype("float32")
    modal_std[modal_std < 1e-4] = 1.0

    input_dim = modal_train_concat.shape[1]
    n_modes = Freqs_true_train.shape[1]
    n_epochs = args.epochs
    base_lr = args.lr
    num_KS = args.num_ks
    n_dims = alpha_factors_true_train.shape[1]
    num_samples = args.num_samples
    gamma = args.gamma

    dataset_short = os.path.basename(args.dataset.rstrip("/\\")).replace("TRAIN_E5_", "").replace("_doe", "")
    tag_str = f"_{args.tag}" if args.tag else ""
    filename = (
        f"KSCop_{n_dims}Z_{num_KS}KS_{args.sensor_regime}_"
        f"{dataset_short}_G{gamma}_{n_epochs}ep{tag_str}"
    )
    folder_path = os.path.join(CODE_ROOT, "Output", "review_27oct", "KS_Copula", filename)
    os.makedirs(folder_path, exist_ok=True)

    print("==========================================================================")
    print(f"Training GC-KSmix CVAE | num_KS = {num_KS} ({'Unimodal' if num_KS == 1 else 'Multimodal'})")
    print(f"Copula Dimension D = {n_dims} ({n_dims * (n_dims - 1) // 2} off-diag correlation params)")
    print(f"FE Decoder Elements = {n_elements} | Modes = {n_modes} | Sensor Regime = {args.sensor_regime}")
    print(f"Output Folder: {folder_path}")
    print("==========================================================================")

    # --- 4. INITIALIZATION ---
    model = My_CopulaKSVAE_withEigen_Updated(
        input_dim=input_dim,
        num_dofs=n_dofs,
        n_elements=n_elements,
        n_modes=n_modes,
        Ke_matrices=Ke_matrices,
        Mfree=Mfree,
        L_inv=L_inv,
        n_dims=n_dims,
        num_KS=num_KS,
        num_samples=num_samples,
        gamma=gamma,
        mean_f=mean_f,
        std_f=std_f,
        lbound=lbound,
        fixed_dofs_indices=fixed_dofs_indices,
        include_rotations=include_rotations,
        include_vert_modes=include_vert_modes,
        modal_mean=modal_mean,
        modal_std=modal_std,
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=base_lr,
        decay_steps=1000,
        decay_rate=0.90,
        staircase=False,
    )
    optim = K.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    # Force build using a 1-sample tuple
    _ = model((
        Freqs_true_train[:1],
        Rotmodes_true_train[:1],
        Vertmodes_true_train[:1],
        alpha_factors_true_train[:1],
    ))

    model.compile(
        optimizer=optim,
        loss=model.ELBO_Copula_loss,
        metrics=[model.Freqs_loss, model.MAC_modes_loss, model.Joint_copula_dens_term],
    )

    # --- 5. TRAINING ---
    start_time = time.time()
    delayed_stop = DelayedEarlyStopping(patience=1000, start_epoch=8000)

    model_history = model.fit(
        x=(Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train),
        y=alpha_factors_true_train,
        batch_size=batch_size,
        epochs=n_epochs,
        verbose=2,
        shuffle=True,
        validation_data=(
            (Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val),
            alpha_factors_true_val,
        ),
        callbacks=[delayed_stop],
    )

    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    # Save weights and training history
    model.save_weights(os.path.join(folder_path, "final_model_weights.weights.h5"))
    np.save(os.path.join(folder_path, "model_history.npy"), model_history.history, allow_pickle=True)

    # --- 6. INFERENCE ON VALIDATION AND TEST SETS ---
    inverse_model = model.Encoder_model

    val_a, val_b, val_weight_vals, val_offdiag_elems, val_diag_elems = inverse_model.predict(
        (Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val)
    )
    test_a, test_b, test_weight_vals, test_offdiag_elems, test_diag_elems = inverse_model.predict(
        (Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test)
    )

    # --- 7. SAVING RESULTS ---
    Problem_info = {
        "input_dim_enc": int(input_dim),
        "n_dims": int(n_dims),
        "n_elements": int(n_elements),
        "num_KS": int(num_KS),
        "n_samples": int(num_samples),
        "epochs": int(n_epochs),
        "gamma": float(gamma),
        "std_f": std_f,
        "mean_f": mean_f,
        "lbound": float(lbound),
        "dataset": args.dataset,
        "sensor_regime": args.sensor_regime,
        "training_time_sec": float(training_time),
    }
    np.save(os.path.join(folder_path, "Problem_info.npy"), Problem_info, allow_pickle=True)

    Val_predicted_properties = {
        "test_a": val_a,
        "test_b": val_b,
        "test_offdiag_elems": val_offdiag_elems,
        "test_diag_elems": val_diag_elems,
        "test_weight_vals": val_weight_vals,
    }
    np.save(os.path.join(folder_path, "Val_predicted_props.npy"), Val_predicted_properties, allow_pickle=True)

    Test_predicted_properties = {
        "test_a": test_a,
        "test_b": test_b,
        "test_offdiag_elems": test_offdiag_elems,
        "test_diag_elems": test_diag_elems,
        "test_weight_vals": test_weight_vals,
    }
    np.save(os.path.join(folder_path, "Test_predicted_props.npy"), Test_predicted_properties, allow_pickle=True)

    # --- 8. COMPUTE SUMMARY METRICS (VALIDATION & TEST) ---
    val_L = build_correlation_matrices_from_cholesky(val_offdiag_elems, val_diag_elems, n_dims)
    test_L = build_correlation_matrices_from_cholesky(test_offdiag_elems, test_diag_elems, n_dims)

    val_metrics, _ = calculate_ks_mixture_metrics(
        {"test_a": val_a, "test_b": val_b, "test_weights": val_weight_vals, "test_L_matrices": val_L},
        {"alpha_factors_true_test": alpha_factors_true_val},
        lbound=lbound,
    )
    test_metrics, _ = calculate_ks_mixture_metrics(
        {"test_a": test_a, "test_b": test_b, "test_weights": test_weight_vals, "test_L_matrices": test_L},
        {"alpha_factors_true_test": alpha_factors_true_test},
        lbound=lbound,
    )

    summary_payload = {
        "run_name": filename,
        "dataset": args.dataset,
        "num_KS": int(num_KS),
        "n_dims": int(n_dims),
        "sensor_regime": args.sensor_regime,
        "epochs": int(n_epochs),
        "training_time_sec": float(training_time),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with open(os.path.join(folder_path, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n--- Validation Metrics ---")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.5f}")
    print("\n--- Test Metrics ---")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.5f}")

    # --- 9. PLOTTING LOSS CURVES ---
    try:
        plot_trainval_loss(model, folder_path)
        plot_JointPDF_loss(model, folder_path)
        plot_Freqs_loss(model, folder_path)
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
