# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:58:31 2026
UNCERTAINTY QUANTIFICATION METRICS AND PLOTS FOR THE GAUSSIAN COPULA WITH KUMARSWAMY MIXTURE MARGINALS
@author: anafd
"""

import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
# from scipy.stats import norm, chi2, pearsonr
# from scipy.linalg import solve_triangular
from scipy.special import beta as beta_func
import os

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

def configure_academic_plots():
    """Sets matplotlib parameters for high-quality journal figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "grid.alpha": 0.4,
        "figure.dpi": 300,
        "text.usetex": False
    })

def ks_mixture_cdf(z_unit, a, b, weights):
    """
    Calculates the CDF of a Kumaraswamy mixture.
    z_unit: values in [0, 1]
    a, b, weights: params of shape (num_components, n_dims)
    Returns: CDF of shape (n_dims,)
    """
    # comp_cdfs: (num_components, n_dims)
    comp_cdfs = 1.0 - np.power(1.0 - np.power(z_unit, a), b)
    # mixture: sum over components
    return np.sum(weights * comp_cdfs, axis=0)

def calculate_ks_mixture_moments(a, b, weights):
    """Calculates analytical mean and variance of the KS Mixture."""
    def ks_moment(a, b, n):
        return b * beta_func(1 + n/a, b)
    
    # Moments for each component: (num_comp, n_dims)
    m1 = ks_moment(a, b, 1)
    m2 = ks_moment(a, b, 2)
    
    # Mixture moments
    mean_mix = np.sum(weights * m1, axis=0)
    var_mix = np.sum(weights * m2, axis=0) - (mean_mix**2)
    return mean_mix, np.sqrt(np.maximum(var_mix, 1e-10))

def calculate_and_plot_ks_calibration(predicted_stats, test_datasets, lbound, folder_path):
    """Calculates Marginal Calibration for KS Mixtures."""
    configure_academic_plots()
    
    test_a = predicted_stats['test_a']           # [N, K, D]
    test_b = predicted_stats['test_b']           # [N, K, D]
    test_weights = predicted_stats['test_weights'] # [N, K, D]
    z_true = test_datasets['alpha_factors_true_test'] # [N, D]
    
    n_samples, _, n_dims = test_a.shape
    z_true_unit = np.clip((z_true - lbound) / (1.0 - lbound), 1e-6, 1.0 - 1e-6)
    
    # Calculate PIT (Probability Integral Transform) values
    pit_values = np.zeros((n_samples, n_dims))
    for i in range(n_samples):
        pit_values[i] = ks_mixture_cdf(z_true_unit[i], test_a[i], test_b[i], test_weights[i])
    
    quantiles = np.linspace(0.05, 0.95, 19)
    empirical_coverages = []
    
    for q in quantiles:
        lower = (1 - q) / 2
        upper = 1 - (1 - q) / 2
        covered = (pit_values >= lower) & (pit_values <= upper)
        empirical_coverages.append(np.mean(covered))
        
    mace = np.mean(np.abs(np.array(empirical_coverages) - quantiles))
    
    plt.figure(figsize=(7, 7), facecolor='white')
    plt.plot([0, 1], [0, 1], 'k--', label='Ideal Calibration', lw=2)
    plt.plot(quantiles, empirical_coverages, 's-', color='#d62728', lw=2, label='KS-VAE Predicted')
    plt.fill_between(quantiles, quantiles - 0.05, quantiles + 0.05, color='gray', alpha=0.1)
    
    plt.xlabel('Nominal Confidence Level')
    plt.ylabel('Empirical Coverage')
    plt.title('Reliability Diagram (KS-Mixture Marginals)')
    
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.9)
    plt.text(0.05, 0.85, f"MACE: {mace:.4f}", transform=plt.gca().transAxes, fontsize=14, bbox=bbox_props)
    
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--')
    
    save_dir = os.path.join(folder_path, "UQ_Metrics")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'Calibration_Curve_KS.png'), bbox_inches='tight')
    plt.show()


def calculate_energy_distance_vectorized(all_samples, all_true_vals):
    """
    Calculates Energy Distance for a batch of scenarios.
    ED(F, z) = E[||X - z||] - 0.5 * E[||X - X'||]
    This is a theory-agnostic metric that compares the distribution F (represented by samples)
    to a point z (ground truth).
    """
    # 1. Mean distance to ground truth
    diffs_to_true = all_samples - all_true_vals[:, np.newaxis, :]
    dists_to_true = np.sqrt(np.sum(diffs_to_true**2, axis=-1) + 1e-10)
    term1 = np.mean(dists_to_true, axis=1) # [N_scenarios]

    # 2. Internal diversity (Interaction term)
    # Measures the 'spread' of the distribution to normalize the accuracy
    n_sub = min(all_samples.shape[1], 100)
    sub_samples = all_samples[:, :n_sub, :]
    
    diffs_internal = sub_samples[:, :, np.newaxis, :] - sub_samples[:, np.newaxis, :, :]
    dists_internal = np.sqrt(np.sum(diffs_internal**2, axis=-1) + 1e-10)
    term2 = np.mean(dists_internal, axis=(1, 2)) # [N_scenarios]

    return term1 - 0.5 * term2

def optimized_peak_check(all_samples, lbound=0.45):
    """
    Sensitive multimodality check using KDE peak-finding.
    Treats the samples as an empirical distribution to identify multiple local maxima.
    Highly effective for capturing close-range peaks or 'shoulders'.
    """
    N, S, D = all_samples.shape
    multimodal_mask = np.zeros((N, D), dtype=bool)
    
    # Grid for KDE evaluation - fine enough to catch close peaks
    x_grid = np.linspace(lbound - 0.05, 1.05, 150)
    
    for d in range(D):
        for i in range(N):
            data = all_samples[i, :, d]
            try:
                # Use a slightly smaller bandwidth to increase sensitivity to close modes
                kde = gaussian_kde(data, bw_method='scott')
                y = kde(x_grid)
                
                # Identify peaks:
                # - height: at least 5% of the main mode to ignore sampling noise
                # - prominence: ensures the peak is a distinct 'bump' in the shape
                peaks, _ = find_peaks(y, height=np.max(y) * 0.05, prominence=np.max(y) * 0.02)
                
                if len(peaks) > 1:
                    multimodal_mask[i, d] = True
            except:
                continue 
                
    per_latent_rate = np.mean(multimodal_mask, axis=0)
    global_multi_rate = np.mean(np.any(multimodal_mask, axis=1))
    
    return global_multi_rate, per_latent_rate

def calculate_non_gaussianity(all_samples):
    """
    Quantifies deviation from Gaussianity (complexity of the posterior shape).
    Uses a combination of Squared Skewness and Excess Kurtosis.
    A pure Gaussian has a score of 0.
    """
    mean = np.mean(all_samples, axis=1, keepdims=True)
    std = np.std(all_samples, axis=1, keepdims=True) + 1e-6
    z = (all_samples - mean) / std
    
    skew = np.mean(z**3, axis=1)
    kurt = np.mean(z**4, axis=1) - 3 # Excess kurtosis
    
    # Negentropy proxy: higher values indicate complex features like multimodality
    negentropy = (skew**2 / 12) + (kurt**2 / 48)
    return np.mean(negentropy, axis=0) 

def enhanced_metrics_comparison(predicted_stats, test_datasets, lbound=0.45, n_samples_for_ed=500):
    """
    Main entry point for calculating posterior adequacy metrics.
    Treats the predicted stats purely as a generative source for sample-based analysis.
    """
    test_a = predicted_stats['test_a']           
    test_b = predicted_stats['test_b']           
    test_weights = predicted_stats['test_weights'] 
    z_true = test_datasets['alpha_factors_true_test'] 
    
    N, num_KS, n_dims = test_a.shape
    
    # --- Theory-Agnostic Sampling ---
    # We use the KS mixture only as a 'black box' sampler
    flat_weights = test_weights.transpose(0, 2, 1).reshape(-1, num_KS)
    cum_weights = np.cumsum(flat_weights, axis=1)
    r = np.random.rand(flat_weights.shape[0], n_samples_for_ed, 1)
    comp_indices = np.argmax(cum_weights[:, np.newaxis, :] > r, axis=2) 
    
    flat_a = test_a.transpose(0, 2, 1).reshape(-1, num_KS)
    flat_b = test_b.transpose(0, 2, 1).reshape(-1, num_KS)
    
    rows = np.arange(flat_a.shape[0])[:, np.newaxis]
    chosen_a = flat_a[rows, comp_indices]
    chosen_b = flat_b[rows, comp_indices]
    
    u = np.random.uniform(1e-6, 1-1e-6, size=chosen_a.shape)
    s_flat = np.power(1.0 - np.power(1.0 - u, 1.0/chosen_b), 1.0/chosen_a)
    
    s_reshaped = s_flat.reshape(N, n_dims, n_samples_for_ed).transpose(0, 2, 1)
    all_samples = lbound + (1.0 - lbound) * s_reshaped
    
    # --- Sample-Based Metric Calculation ---
    ed_scores = calculate_energy_distance_vectorized(all_samples, z_true)
    global_multi_rate, per_latent_rate = optimized_peak_check(all_samples, lbound)
    non_gaussian_scores = calculate_non_gaussianity(all_samples)
    
    metrics = {
        'Avg_Energy_Distance': np.mean(ed_scores),
        'Global_Multimodality_Rate': global_multi_rate,
        'Per_Latent_Multimodality_Rates': per_latent_rate.tolist(),
        'Avg_Non_Gaussianity_Score': np.mean(non_gaussian_scores),
        'Per_Latent_Non_Gaussianity': non_gaussian_scores.tolist(),
        'ED_Std_Dev': np.std(ed_scores)
    }
    
    return metrics
# def calculate_ks_mahalanobis(predicted_stats, test_datasets, lbound, folder_path):
#     """Dependency validation using Mahalanobis distance in latent space."""
#     configure_academic_plots()
    
#     test_a = predicted_stats['test_a']
#     test_b = predicted_stats['test_b']
#     test_weights = predicted_stats['test_weights']
#     z_true = test_datasets['alpha_factors_true_test']
#     L_matrices = predicted_stats['test_L_matrices'] # [N, D, D]
    
#     n_samples, _, n_dims = test_a.shape
#     z_true_unit = np.clip((z_true - lbound) / (1.0 - lbound), 1e-7, 1.0 - 1e-7)
    
#     mahalanobis_sq = []
#     for i in range(n_samples):
#         # 1. Transform to U ~ Uniform(0,1) via KS-Mixture CDF
#         u = ks_mixture_cdf(z_true_unit[i], test_a[i], test_b[i], test_weights[i])
#         u = np.clip(u, 1e-7, 1.0 - 1e-7)
        
#         # 2. Transform to Y ~ N(0, Σ) via Inverse Normal CDF
#         y_latent = norm.ppf(u)
        
#         # 3. Compute D_M^2 = y^T Σ^-1 y = ||L^-1 y||^2
#         try:
#             x = solve_triangular(L_matrices[i], y_latent, lower=True)
#             mahalanobis_sq.append(np.sum(x**2))
#         except:
#             continue
            
#     mahalanobis_sq = np.array(mahalanobis_sq)
#     emp_mean = np.mean(mahalanobis_sq)
    
#     plt.figure(figsize=(8, 6), facecolor='white')
#     plt.hist(mahalanobis_sq, bins=40, density=True, alpha=0.5, color='#1f77b4', edgecolor='black', label='Empirical $D_M^2$')
    
#     x_vals = np.linspace(0, np.max(mahalanobis_sq), 200)
#     plt.plot(x_vals, chi2.pdf(x_vals, df=n_dims), color='#d62728', lw=2.5, label=f'Theoretical $\chi^2$ ($k={n_dims}$)')
    
#     plt.xlabel('Squared Mahalanobis Distance (Latent Space)')
#     plt.ylabel('Density')
#     plt.title('Copula Dependency Validation (KS-VAE)')
    
#     stats_text = f"Empirical $\mu$: {emp_mean:.2f}\nTheoretical $\mu$: {n_dims:.2f}"
#     plt.text(0.6, 0.5, stats_text, transform=plt.gca().transAxes, fontsize=13, bbox=dict(facecolor='white', alpha=0.8))
    
#     plt.legend()
#     plt.grid(True, linestyle='--')
    
#     save_dir = os.path.join(folder_path, "UQ_Metrics")
#     os.makedirs(save_dir, exist_ok=True)
#     plt.savefig(os.path.join(save_dir, 'Mahalanobis_KS.png'), bbox_inches='tight')
#     plt.show()

# def plot_ks_error_vs_confidence(predicted_stats, test_datasets, lbound, folder_path):
#     """Correlation between predicted KS-Mixture variance and absolute error."""
#     configure_academic_plots()
    
#     test_a = predicted_stats['test_a']
#     test_b = predicted_stats['test_b']
#     test_weights = predicted_stats['test_weights']
#     z_true = test_datasets['alpha_factors_true_test']
    
#     n_samples, _, n_dims = test_a.shape
    
#     all_means = []
#     all_stds = []
    
#     # Calculate analytical mean/std for every sample and dimension
#     for i in range(n_samples):
#         m, s = calculate_ks_mixture_moments(test_a[i], test_b[i], test_weights[i])
#         all_means.append(m)
#         all_stds.append(s)
        
#     # Scale back to physical space
#     scale_factor = (1.0 - lbound)
#     pred_means = lbound + scale_factor * np.array(all_means)
#     pred_stds = scale_factor * np.array(all_stds)
    
#     # Flatten for point-wise analysis
#     flat_errors = np.abs(pred_means - z_true).flatten()
#     flat_stds = pred_stds.flatten()
    
#     corr, _ = pearsonr(flat_stds, flat_errors)
    
#     # Binned analysis
#     n_bins = 15
#     sort_idx = np.argsort(flat_stds)
#     stds_s, errs_s = flat_stds[sort_idx], flat_errors[sort_idx]
    
#     bin_edges = np.percentile(stds_s, np.linspace(0, 100, n_bins + 1))
#     bin_c, bin_e, bin_std_err = [], [], []
    
#     for i in range(n_bins):
#         mask = (stds_s >= bin_edges[i]) & (stds_s <= bin_edges[i+1])
#         if np.any(mask):
#             bin_c.append(np.mean(stds_s[mask]))
#             bin_e.append(np.mean(errs_s[mask]))
#             bin_std_err.append(np.std(errs_s[mask]) / np.sqrt(np.sum(mask)))

#     plt.figure(figsize=(8, 6), facecolor='white')
#     plt.scatter(flat_stds, flat_errors, alpha=0.1, color='gray', s=10, label='Test Points')
#     plt.errorbar(bin_c, bin_e, yerr=bin_std_err, fmt='o-', color='#ff7f0e', lw=3, label='Binned Mean Error')
    
#     plt.xlabel('Predicted Uncertainty (KS Mixture $\sigma$)')
#     plt.ylabel('Absolute Error $|z_{true} - z_{pred}|$')
#     plt.title('Uncertainty Awareness (Error vs. Confidence)')
    
#     plt.text(0.05, 0.85, f"Pearson's $r$: {corr:.2f}", transform=plt.gca().transAxes, 
#              fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
#     plt.legend()
#     plt.grid(True, linestyle='--')
    
#     save_dir = os.path.join(folder_path, "UQ_Metrics")
#     os.makedirs(save_dir, exist_ok=True)
#     plt.savefig(os.path.join(save_dir, 'Error_vs_Confidence_KS.png'), bbox_inches='tight')
#     plt.show()

# def run_full_ks_uq_analysis(predicted_stats, test_datasets, lbound, folder_path):
#     """Executes all KS-based UQ metrics."""
#     print("--- Starting KS-Mixture UQ Analysis ---")
#     calculate_and_plot_ks_calibration(predicted_stats, test_datasets, lbound, folder_path)
#     print("Calibration Curve: Done.")
#     calculate_ks_mahalanobis(predicted_stats, test_datasets, lbound, folder_path)
#     print("Mahalanobis Validation: Done.")
#     plot_ks_error_vs_confidence(predicted_stats, test_datasets, lbound, folder_path)
#     print("Error vs Confidence: Done.")
#     print(f"All plots saved to: {os.path.join(folder_path, 'UQ_Metrics')}")