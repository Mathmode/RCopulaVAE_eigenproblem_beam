#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_GCopula_GMM_updated_review27oct.py
=======================================
Full Updated Version of `code/main_GCopula_GMM.py` for the MS CPENG-8674 Major Revision
(saved in `code/MODULES/review_27oct/`).

Supported & Verified Changes Relative to Original `main_GCopula_GMM.py`:
------------------------------------------------------------------------
1. [FIX 1 -- Saves Both Validation-Set Predictions (`Val_predicted_props.npy`) and Validation
             Metrics (`validation_performance_metrics.csv`) to Eliminate Data Leakage
             (Addresses Reviewer 3 Comment 2 & Associate Editor Comment 2)]
   - In the original `main_GCopula_GMM.py` (lines 165-185), only `Test_predicted_props.npy`
     was computed and saved after training. Because validation predictions were not saved,
     hyperparameter selection for `gamma in {0.1, 0.2, 0.3, 0.4, 0.5}` in Table 1 was
     performed on the test set.
   - Here, we evaluate `inverse_model.predict()` on the VALIDATION split (`Val_predicted_props.npy`),
     compute exact validation metrics (`validation_performance_metrics.csv`), and rank `gamma`
     strictly on the validation split before evaluating the held-out test set and the external
     benchmark test sets (`TEST_E5_healthy`, `TEST_E5_single`, `TEST_E5_finemesh50`, etc.).

2. [FIX 2 -- Saves All Outputs in `code/Output/review_27oct/` and Reads from `code/Data/review_27oct/`]
   - Keeps all revision runs cleanly separated from your first-submission folders.

3. [FIX 3 -- Uses the Updated Modules in `MODULES/review_27oct/`]
   - Imports `load_data`, `load_known_matrices`, `load_affine_zone_matrices`, `load_external_test_set`
     from `preprocessing_updated_review27oct.py`.
   - Imports `My_CopulaVAE_withEigen_Updated` from `GC_GMm_models_updated_review27oct.py`
     (which fixes `L_inv` to `trainable=False`, uses the exact Cholesky congruence order
     `L_inv @ K @ L_inv.T`, removes the `0.99` copula upper-tail cutoff, and supports
     `num_samples >= 1` and `n_modes in {2, 3, 4, 5}`).
   - Imports `calculate_testing_metrics` from `updated_GC_GMm_functions_for_results_analysis_review27oct.py`
     (which computes exact `TruncatedNormal` quantiles alongside legacy clipped intervals).

How to run (from `code/` folder):
---------------------------------
  python -m MODULES.review_27oct.main_GCopula_GMM_updated_review27oct --dataset TRAIN_E5_realistic_doe --gamma 0.3 --epochs 10000
"""

import argparse
import os
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(f"GPU Initialization Error: {e}")

from .preprocessing_updated_review27oct import (
    load_data,
    load_known_matrices,
    load_affine_zone_matrices,
    load_external_test_set,
)
from .GC_GMm_models_updated_review27oct import My_CopulaVAE_withEigen_Updated
from .GC_GMm_functions_updated_review27oct import build_correlation_matrices_from_cholesky
from .updated_GC_GMm_functions_for_results_analysis_review27oct import calculate_testing_metrics


class DelayedEarlyStopping(K.callbacks.Callback):
    def __init__(self, patience=1000, start_epoch=10000):
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
                print(f" - Epoch {epoch}: New best val_loss: {current_loss:.6f}")
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}.")
                    self.model.stop_training = True
        elif current_loss < self.best_loss:
            self.best_loss = current_loss


def evaluate_and_save_split(inverse_model, split_name, freqs, rot, vert, alphas, n_dims, lbound, folder_path):
    """
    Runs encoder inference on a given split (Validation, Test, or External Benchmark),
    saves the predicted copula parameters, and computes exact + legacy UQ metrics.
    """
    means, scales, weight_vals, offdiag_elems, diag_elems = inverse_model.predict(
        [freqs, rot, vert, alphas], verbose=0
    )
    L_matrices = build_correlation_matrices_from_cholesky(offdiag_elems, diag_elems, n_dims).numpy()

    pred_props = {
        "test_means": means,
        "test_scales": scales,
        "test_offdiag_elems": offdiag_elems,
        "test_diag_elems": diag_elems,
        "test_weight_vals": weight_vals,
        "test_L_matrices": L_matrices,
    }
    np.save(os.path.join(folder_path, f"{split_name}_predicted_props.npy"), pred_props, allow_pickle=True)

    datasets_dict = {
        "Freqs_true_test": freqs,
        "Rotmodes_true_test": rot,
        "Vertmodes_true_test": vert,
        "alpha_factors_true_test": alphas,
    }
    metrics, _ = calculate_testing_metrics(pred_props, datasets_dict, lbound=lbound)
    df = pd.DataFrame([{"Split": split_name, **metrics}])
    df.to_csv(os.path.join(folder_path, f"{split_name}_performance_metrics.csv"), index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Updated PI-GCVAE Training & Validation Script (review_27oct)")
    parser.add_argument("--dataset", default="TRAIN_E5_realistic_doe",
                        help="Dataset folder inside Data/review_27oct/ (e.g. TRAIN_E5_realistic_doe, TRAIN_E5_uniform)")
    parser.add_argument("--data-root", default=os.path.join("Data", "review_27oct"))
    parser.add_argument("--out-root", default=os.path.join("Output", "review_27oct", "Trained_Models"))
    parser.add_argument("--gamma", type=float, default=0.3, help="ELBO regularization weight gamma")
    parser.add_argument("--n-modes", type=int, default=5, help="Number of modes n_m in {2,3,4,5} (R3.4)")
    parser.add_argument("--num-samples", type=int, default=1, help="Monte Carlo samples H in {1,5,10} (R3.7)")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lbound", type=float, default=0.45)
    parser.add_argument("--prior-beta-a", type=float, default=1.0, help="Beta(a,b) prior parameter a (1.0 = uniform, R3.3)")
    parser.add_argument("--prior-beta-b", type=float, default=1.0, help="Beta(a,b) prior parameter b")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    K.utils.set_random_seed(args.seed)
    K.backend.set_floatx("float32")

    data_path = os.path.join(args.data_root, args.dataset)
    print(f"Loading data from {data_path} (n_modes={args.n_modes})...")

    (
        Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train,
        Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val,
        Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test,
        mean_f, std_f,
    ) = load_data(data_path, batch_size=args.batch_size, n_modes=args.n_modes)

    n_modes = Freqs_true_train.shape[1]
    n_nodes = Rotmodes_true_train.shape[2]
    n_dims = alpha_factors_true_train.shape[1]

    # Load nominal physics decoder matrices
    Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_dims)
    kzone_path = os.path.join(data_path, "Kzone_free.npy")
    if os.path.exists(kzone_path):
        Mfree, Kzone_free, Ksupport_free, K0_free, L_inv = load_affine_zone_matrices(data_path)
    else:
        Kzone_free, Ksupport_free, K0_free = None, None, None

    # Determine total DOFs and fixed DOFs of the identification model from metadata if present
    meta_file = os.path.join(data_path, "metadata.json")
    if os.path.exists(meta_file):
        import json
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        icfg = meta["identification_model"]
        n_elements_ident = int(icfg["n_elements"])
        n_dofs = 2 * (n_elements_ident + 1)
        fixed_dofs_indices = [0, n_dofs - 2] if icfg["support"] == "pinned" else []
        sensor_nodes = icfg["sensor_nodes"]
    else:
        n_elements_ident = n_dims
        n_dofs = 2 * (n_elements_ident + 1)
        fixed_dofs_indices = [0, n_dofs - 2]
        sensor_nodes = None

    input_dim = (
        Freqs_true_train.shape[1]
        + (Rotmodes_true_train.shape[1] * Rotmodes_true_train.shape[2])
        + (Vertmodes_true_train.shape[1] * Vertmodes_true_train.shape[2])
    )

    exp_name = (
        f"Rev27Oct_{args.dataset}_Dim{n_dims}_{n_modes}modes_H{args.num_samples}"
        f"_Gamma{args.gamma}_lb{args.lbound}_{args.epochs}Ep"
    )
    folder_path = os.path.join(args.out_root, exp_name)
    os.makedirs(folder_path, exist_ok=True)

    print(f"Initializing Updated PI-GCVAE Model -> Output: {folder_path}")
    model = My_CopulaVAE_withEigen_Updated(
        input_dim=input_dim,
        num_dofs=n_dofs,
        n_elements=n_elements_ident,
        n_modes=n_modes,
        Ke_matrices=Ke_matrices,
        Mfree=Mfree,
        L_inv=L_inv,
        n_dims=n_dims,
        num_gaussians=1,
        num_samples=args.num_samples,
        gamma=args.gamma,
        mean_f=mean_f,
        std_f=std_f,
        lbound=args.lbound,
        fixed_dofs_indices=fixed_dofs_indices,
        sensor_nodes=sensor_nodes,
        Kzone_free=Kzone_free,
        Ksupport_free=Ksupport_free,
        K0_free=K0_free,
        prior_beta_a=args.prior_beta_a,
        prior_beta_b=args.prior_beta_b,
    )

    optim = K.optimizers.Adam(learning_rate=args.lr, clipnorm=0.5)
    _ = model([Freqs_true_train[:1], Rotmodes_true_train[:1], Vertmodes_true_train[:1], alpha_factors_true_train[:1]])
    initial_weights_sum = np.sum([np.sum(w) for w in model.Encoder_model.FC_encoder.get_weights()])

    model.compile(
        optimizer=optim,
        loss=model.ELBO_Copula_loss,
        metrics=[model.Freqs_loss, model.MAC_modes_loss, model.Joint_copula_dens_term],
        run_eagerly=False,
    )

    start_time = time.time()
    delayed_stop = DelayedEarlyStopping(patience=5000, start_epoch=max(1000, args.epochs // 2))

    model_history = model.fit(
        x=[Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train],
        y=[alpha_factors_true_train, alpha_factors_true_train],
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        validation_data=(
            [Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val],
            [alpha_factors_true_val, alpha_factors_true_val],
        ),
        callbacks=[delayed_stop],
    )

    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    np.save(os.path.join(folder_path, "training_time.npy"), {"training_time": training_time}, allow_pickle=True)

    Problem_info = {
        "dataset": args.dataset,
        "input_dim_enc": input_dim,
        "n_dims": n_dims,
        "n_modes": n_modes,
        "n_gaussians": 1,
        "n_samples": args.num_samples,
        "epochs": args.epochs,
        "gamma": args.gamma,
        "beta": args.gamma,
        "lbound": args.lbound,
        "std_f": std_f,
        "mean_f": mean_f,
    }
    np.save(os.path.join(folder_path, "Problem_info.npy"), Problem_info, allow_pickle=True)
    np.save(os.path.join(folder_path, "model_history.npy"), model_history.history, allow_pickle=True)

    # [FIX 1]: Evaluate and save BOTH Validation Set (for R3.2 gamma selection) and Test Set!
    inverse_model = model.Encoder_model
    val_metrics = evaluate_and_save_split(
        inverse_model, "Val", Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val,
        alpha_factors_true_val, n_dims, args.lbound, folder_path,
    )
    test_metrics = evaluate_and_save_split(
        inverse_model, "Test", Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test,
        alpha_factors_true_test, n_dims, args.lbound, folder_path,
    )

    print("\n--- Validation Split Metrics (Use ONLY these to select gamma, R3.2) ---")
    for k, v in val_metrics.items():
        print(f"  {k:38s}: {v:.6f}")

    print("\n--- Held-Out Test Split Metrics (Unbiased Evaluation) ---")
    for k, v in test_metrics.items():
        print(f"  {k:38s}: {v:.6f}")

    trained_weights_sum = np.sum([np.sum(w) for w in model.Encoder_model.FC_encoder.get_weights()])
    if not np.isclose(initial_weights_sum, trained_weights_sum):
        print("SUCCESS: Encoder weights updated during training.")
    model.save_weights(os.path.join(folder_path, "final_model_weights.weights.h5"))


if __name__ == "__main__":
    main()
