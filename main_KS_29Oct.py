#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_GCop_KSmix.py

Updated Main Training Script for Remote GPU Execution:
  - Preserves GPU Memory Growth initialization, tuple-based Keras inputs,
    DelayedEarlyStopping, and direct Encoder test prediction export.
  - Connects seamlessly to the 7D Bridge-with-Bearings FE dataset and
    postprocessing_7D_bridge.py.
"""

import os
import sys
import time
import numpy as np

# Prevent TF info spam and X11 display crashes on remote SSH GPU servers
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
import tensorflow.keras as K

# --- SAFE PROJECT ROOT PATH RESOLUTION ---
try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# --- GPU CONFIGURATION ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Memory Growth Enabled on {len(gpus)} GPU(s): {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(f"⚠️ GPU Initialization Error: {e}")
else:
    print("⚠️ No GPU detected by TensorFlow. Running on CPU.")

from MODULES.review29oct.dataset_generation_7D_bridge import generate_all_bridge_datasets
from MODULES.review29oct.preprocessing_and_eigen import load_bridge_data, load_baseline_matrices
from MODULES.review29oct.GC_KSmix_models import My_CopulaKSVAE_withEigen
from MODULES.review29oct.postprocessing_7D_bridge import plot_training_history_losses, run_full_postprocessing


class DelayedEarlyStopping(K.callbacks.Callback):
    """
    Waits until `start_epoch` before counting patience, tracks the lowest `val_loss`
    across the entire run, and restores the best validation weights on training end
    (strictly addressing Reviewer #3, Comment 2).
    """
    def __init__(self, patience=400, start_epoch=1000, restore_best_weights=True):
        super(DelayedEarlyStopping, self).__init__()
        self.patience = patience
        self.start_epoch = start_epoch
        self.restore_best_weights = restore_best_weights
        self.best_loss = np.inf
        self.best_epoch = 0
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        if current_loss is None or not np.isfinite(current_loss):
            return

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch = epoch + 1
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            if epoch >= self.start_epoch:
                self.wait += 1
                if self.wait >= self.patience:
                    print(
                        f"\n✅ Early stopping triggered at epoch {epoch + 1}. "
                        f"Best val_loss = {self.best_loss:.5f} at epoch {self.best_epoch}."
                    )
                    self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.restore_best_weights and self.best_weights is not None:
            print(f"✅ Restoring best validation model weights from epoch {self.best_epoch} (val_loss = {self.best_loss:.5f}).")
            self.model.set_weights(self.best_weights)


def main():
    K.utils.set_random_seed(1234)
    K.backend.set_floatx('float32')

    # --- 1. DATA LOADING (Auto-generates if not yet present on remote server) ---
    data_folder = "V01_BridgeBearings_7D_DoE"
    data_path = os.path.join("Data", data_folder)

    n_elements_train = 10   # 10-element deck (5 macro-zones) + 2 elastic bearings = 7D
    n_dofs = 2 * (n_elements_train + 1)  # All 22 DOFs active (elastic bearings at both ends)
    lbound = 0.45
    batch_size = 256

    if not os.path.exists(os.path.join(data_path, "bridge_7d_splits.npz")):
        print(f"Dataset not found at {data_path}. Generating 7D Bridge-with-Bearings dataset now...")
        generate_all_bridge_datasets(output_dir=data_path, lbound=lbound)

    print(f"Loading data from {data_path}...")
    (
        Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train,
        Freqs_true_val,   Rotmodes_true_val,   Vertmodes_true_val,   alpha_factors_true_val,
        Freqs_true_test,  Rotmodes_true_test,  Vertmodes_true_test,  alpha_factors_true_test,
        mean_f, std_f
    ) = load_bridge_data(data_path)

    Mfree, K_basis, L_inv = load_baseline_matrices(data_path)

    # --- 2. MODEL & TRAINING HYPERPARAMETERS ---
    input_dim = (
        Freqs_true_train.shape[1]
        + (Rotmodes_true_train.shape[1] * Rotmodes_true_train.shape[2])
        + (Vertmodes_true_train.shape[1] * Vertmodes_true_train.shape[2])
    )

    n_modes = Freqs_true_train.shape[1]
    n_dims = alpha_factors_true_train.shape[1]  # 7 parameters (5 deck zones + 2 bearings)
    num_KS = 3                                  # Number of Kumaraswamy mixture components per marginal
    num_samples = 6                             # Monte Carlo draws per sample during training
    gamma = 0.35                                # Variational KL regularization weight
    base_lr = 3e-4                              # Initial learning rate for ExponentialDecay
    n_epochs = 2500

    # --- SPEED SETTINGS ---
    # Keras 3 resolves jit_compile="auto" to True when a GPU is present, which compiles the whole
    # train step with XLA. XLA's eigh (Jacobi) is ~10-20x slower than LAPACK for the 1536 small
    # 22x22 matrices per batch, and was the cause of the ~17 s/epoch on the remote GPUs.
    jit_compile = False
    # Without XLA, TF's GPU eigh loops over the batch one matrix at a time; LAPACK on CPU is faster.
    eig_device = "/CPU:0"

    filename = f"30Oct_v04_BridgeBearings_{n_dims}D_{num_KS}KSmix_Gamma{gamma}_Ep{n_epochs}"
    folder_path = os.path.join("Output", "review29oct", filename)
    os.makedirs(folder_path, exist_ok=True)

    # --- 3. MODEL INITIALIZATION ---
    model = My_CopulaKSVAE_withEigen(
        input_dim=input_dim,
        num_dofs=n_dofs,
        n_modes=n_modes,
        K_basis=K_basis,
        L_inv=L_inv,
        n_dims=n_dims,
        num_KS=num_KS,
        num_samples=num_samples,
        gamma=gamma,
        mean_f=mean_f,
        std_f=std_f,
        lbound=lbound,
        eig_device=eig_device,
    )

    steps_per_epoch = max(1, len(Freqs_true_train) // batch_size)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=base_lr,
        decay_steps=steps_per_epoch * 150,
        decay_rate=0.85,
        staircase=False,
    )

    optim = K.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.5)

    # Force build using a Tuple instead of a List (remote GPU Keras compatibility)
    _ = model((
        Freqs_true_train[:1],
        Rotmodes_true_train[:1],
        Vertmodes_true_train[:1],
        alpha_factors_true_train[:1],
    ))

    model.compile(
        optimizer=optim,
        loss=model.ELBO_Copula_loss,
        metrics=[
            model.Freqs_loss,
            model.MAC_modes_loss,
            model.KL_Divergence_term,
        ],
        jit_compile=jit_compile,
    )

    # --- 4. TRAINING ---
    start_time = time.time()
    print(f"Starting 7D Kumaraswamy-Copula VAE Training ({n_epochs} epochs, batch_size={batch_size})...")
    delayed_stop = DelayedEarlyStopping(
        patience=400,
        start_epoch=int(0.4 * n_epochs),
        restore_best_weights=True,
    )

    model_history = model.fit(
        x=(Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train),
        y=alpha_factors_true_train,
        batch_size=batch_size,
        epochs=n_epochs,
        shuffle=True,
        validation_data=(
            (Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val),
            alpha_factors_true_val,
        ),
        callbacks=[delayed_stop],
        verbose=2,
    )

    training_time = time.time() - start_time
    print(f"✅ Total training time: {training_time:.2f} seconds")

    # --- 5. SAVE WEIGHTS & TRAINING HISTORY ---
    model.save_weights(os.path.join(folder_path, "final_model_weights.weights.h5"))
    np.save(os.path.join(folder_path, "model_history.npy"), model_history.history, allow_pickle=True)

    # --- 6. DIRECT ENCODER INFERENCE ON UNSEEN 50-ELEMENT FINE-MESH TEST SET ---
    # Prevents having to reload the custom subclassed model during post-processing
    inverse_model = model.Encoder_model
    test_a, test_b, test_weight_vals, test_offdiag_elems, test_diag_elems = inverse_model.predict(
        (Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test),
        batch_size=batch_size,
        verbose=0,
    )

    # --- 7. SAVE PROBLEM INFO & PREDICTED TEST DISTRIBUTION PARAMETERS ---
    Problem_info = {
        "data_dir": data_path,
        "input_dim_enc": input_dim,
        "n_dims": n_dims,
        "num_KS": num_KS,
        "n_samples": num_samples,
        "epochs": n_epochs,
        "gamma": gamma,
        "lbound": lbound,
        "std_f": std_f,
        "mean_f": mean_f,
        "training_time_sec": training_time,
    }
    np.save(os.path.join(folder_path, "Problem_info.npy"), Problem_info, allow_pickle=True)

    Test_predicted_properties = {
        "test_a": test_a,
        "test_b": test_b,
        "test_offdiag_elems": test_offdiag_elems,
        "test_diag_elems": test_diag_elems,
        "test_weight_vals": test_weight_vals,
    }
    np.save(os.path.join(folder_path, "Test_predicted_props.npy"), Test_predicted_properties, allow_pickle=True)
    print(f"✅ Saved Problem_info.npy and Test_predicted_props.npy to: {folder_path}")

    # --- 8. LOSS CURVES & OPTIONAL POST-PROCESSING ---
    try:
        plot_training_history_losses(folder_path)
        print(f"✅ Saved Training_Loss_Curves.png to: {folder_path}")
    except Exception as e:
        print(f"⚠️ Loss plotting warning: {e}")


if __name__ == "__main__":
    main()