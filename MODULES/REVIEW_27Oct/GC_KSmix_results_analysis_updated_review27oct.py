#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GC_KSmix_results_analysis_updated_review27oct.py
================================================
Updated Results Analysis & Uncertainty Quantification Script for the Kumaraswamy
Mixture Gaussian Copula CVAE (`review_27oct`).

Can analyze any trained run folder in `code/Output/review_27oct/KS_Copula/<run_name>/`
and compute full metrics, joint posterior corner plots, and physical beam PDF profiles.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
import tensorflow.keras as K

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from MODULES.review_27oct.GC_KSmix_functions_for_results_analysis_updated_review27oct import (
    calculate_posterior_PDF_info,
    plot_results_PDF_uncertainty,
    plot_physical_pdf_profile,
    calculate_ks_mixture_metrics,
)
from MODULES.review_27oct.preprocessing_updated_review27oct import (
    load_data_updated,
    load_known_matrices_updated,
)
from MODULES.review_27oct.GC_GMm_functions_updated_review27oct import (
    build_correlation_matrices_from_cholesky,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze trained GC-KSmix CVAE results (review_27oct).")
    parser.add_argument(
        "--run-folder",
        type=str,
        required=True,
        help="Path to the trained run folder inside Output/review_27oct/KS_Copula/ (or relative to code/).",
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=[0, 1, 2, 7, 11, 25],
        help="Test sample indices to plot joint posteriors and physical beam profiles for.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2048,
        help="Number of Monte Carlo samples for posterior importance re-weighting plots (default: 2048).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    folder_path = args.run_folder
    if not os.path.isabs(folder_path):
        cand = os.path.join(CODE_ROOT, folder_path)
        if os.path.exists(cand):
            folder_path = cand
        else:
            folder_path = os.path.join(CODE_ROOT, "Output", "review_27oct", "KS_Copula", args.run_folder)

    info_path = os.path.join(folder_path, "Problem_info.npy")
    info = np.load(info_path, allow_pickle=True).item()

    n_dims = int(info["n_dims"])
    num_KS = int(info["num_KS"])
    mean_freq = info["mean_f"]
    std_freq = info["std_f"]
    beta = float(info["gamma"])
    lbound = float(info.get("lbound", 0.45))
    dataset_rel = info.get("dataset", "review_27oct/TRAIN_E5_realistic_doe")
    sensor_regime = info.get("sensor_regime", "full")

    n_elements = int(info.get("n_elements", 5))
    n_dofs = 2 * (n_elements + 1)
    fixed_dofs = [0, n_dofs - 2]
    all_dofs = np.arange(n_dofs)
    free_dofs = np.delete(all_dofs, fixed_dofs)
    batch_size = 256

    zero_rotations = sensor_regime in ("no_rotations", "sparse_2modes", "freq_only")
    n_modes_override = 2 if sensor_regime == "sparse_2modes" else None

    # Load test predictions
    Test_pred_props = np.load(os.path.join(folder_path, "Test_predicted_props.npy"), allow_pickle=True).item()
    test_a = Test_pred_props["test_a"]
    test_b = Test_pred_props["test_b"]
    test_weights = Test_pred_props["test_weight_vals"]
    test_offdiag = Test_pred_props["test_offdiag_elems"]
    test_diag = Test_pred_props["test_diag_elems"]

    # Load dataset and matrices
    data_path = os.path.join(CODE_ROOT, "Data", dataset_rel)
    (_, _, _, _, _, _, _, _, Freqs_test, Rot_test, Vert_test, Alphas_test, _, _) = load_data_updated(
        data_path,
        batch_size=batch_size,
        n_modes=n_modes_override,
        zero_rotations=zero_rotations,
    )
    if sensor_regime == "freq_only":
        Vert_test = np.zeros_like(Vert_test)

    mass_file = os.path.join(data_path, "Mass_matrix.npy")
    use_direct = False
    if os.path.exists(mass_file) and np.load(mass_file).shape[0] == 2 * n_elements:
        use_direct = True
    if use_direct:
        Mfree, Ke_matrices, L_inv, _, _, _ = load_known_matrices_updated(data_path, n_elements)
    else:
        fallback_path = os.path.join(CODE_ROOT, "Data", "review_27oct", "TRAIN_E5_realistic_doe")
        Mfree, Ke_matrices, L_inv, _, _, _ = load_known_matrices_updated(fallback_path, n_elements)

    n_modes = Freqs_test.shape[1]
    test_datasets = {
        "Freqs_true_test": Freqs_test,
        "Rotmodes_true_test": Rot_test,
        "Vertmodes_true_test": Vert_test,
        "alpha_factors_true_test": Alphas_test,
    }

    L_matrices = build_correlation_matrices_from_cholesky(test_offdiag, test_diag, n_dims)
    predicted_stats = {
        "test_a": test_a,
        "test_b": test_b,
        "test_weights": test_weights,
        "test_L_matrices": L_matrices,
    }

    print(f"\n--- KS-VAE Performance Metrics ({os.path.basename(folder_path)}) ---")
    metrics, _ = calculate_ks_mixture_metrics(predicted_stats, test_datasets, lbound)
    for k, v in metrics.items():
        print(f"  {k}: {v:.5f}")

    for pos in args.positions:
        if pos >= len(Alphas_test):
            continue
        print(f"Processing Sample {pos} (z_true = {np.round(Alphas_test[pos], 3)})...")
        z_true, z_samples, post_weights = calculate_posterior_PDF_info(
            fixed_dofs, n_modes, beta, args.n_samples, pos, n_dofs, free_dofs,
            test_datasets, predicted_stats, L_inv, Ke_matrices, Mfree,
            mean_freq, std_freq, lbound, folder_path,
        )
        plot_results_PDF_uncertainty(
            fixed_dofs, n_modes, beta, args.n_samples, pos, n_dofs, free_dofs,
            test_datasets, predicted_stats, L_inv, Ke_matrices, Mfree,
            mean_freq, std_freq, lbound, folder_path, show_plot=False,
        )
        plot_physical_pdf_profile(
            z_samples, post_weights, z_true, n_dims, pos, folder_path, lbound=lbound, show_plot=False
        )

    print("Analysis Complete.")


if __name__ == "__main__":
    main()
