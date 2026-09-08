# -*- coding: utf-8 -*-
"""
Uncertainty Quantification Metrics for GMM Latent Space.
Includes Reliability Diagrams, Energy Distance, and Multimodality Evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import os

from MODULES.REVIEW.GMM_functions_for_results_analysis import generate_gmm_physical_samples

def configure_academic_plots():
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

def gmm_marginal_cdf(z_unit_batch, logits_batch, locs_batch, scales_batch):
    """
    Computes the Marginal CDF of the GMM for each dimension independently.
    z_unit_batch: [N, D]
    logits_batch: [N, K]
    locs_batch, scales_batch: [N, K, D]
    Returns: CDF values of shape [N, D]
    """
    N, D = z_unit_batch.shape
    K = logits_batch.shape[1]

    z_unit_safe = np.clip(z_unit_batch, 1e-6, 1.0 - 1e-6)
    z_raw = np.log(z_unit_safe / (1.0 - z_unit_safe)) 

    # Softmax on logits to get weights
    logits_shifted = logits_batch - np.max(logits_batch, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True) # [N, K]

    z_raw_exp = np.repeat(z_raw[:, np.newaxis, :], K, axis=1) # [N, K, D]

    # CDF per component
    comp_cdfs = stats.norm.cdf(z_raw_exp, loc=locs_batch, scale=scales_batch) # [N, K, D]

    weights_exp = np.repeat(weights[:, :, np.newaxis], D, axis=2) # [N, K, D]
    mix_cdf = np.sum(weights_exp * comp_cdfs, axis=1) # [N, D]
    
    return mix_cdf

def calculate_and_plot_gmm_calibration(predicted_stats, test_datasets, lbound, folder_path):
    """Calculates Marginal Calibration for GMM VAE Marginals."""
    configure_academic_plots()
    
    test_logits = predicted_stats['test_logits'] 
    test_locs = predicted_stats['test_locs']     
    test_scales = predicted_stats['test_logits'] 
    z_true = test_datasets['alpha_factors_true_test'] 
    
    z_true_unit = np.clip((z_true - lbound) / (1.0 - lbound), 1e-6, 1.0 - 1e-6)
    
    # Calculate PIT values utilizing GMM marginal properties
    pit_values = gmm_marginal_cdf(z_true_unit, test_logits, test_locs, test_scales)
    
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
    plt.plot(quantiles, empirical_coverages, 's-', color='#d62728', lw=2, label='GMM-VAE Predicted')
    plt.fill_between(quantiles, quantiles - 0.05, quantiles + 0.05, color='gray', alpha=0.1)
    
    plt.xlabel('Nominal Confidence Level')
    plt.ylabel('Empirical Coverage')
    plt.title('Reliability Diagram (GMM Marginals)')
    
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.9)
    plt.text(0.05, 0.85, f"MACE: {mace:.4f}", transform=plt.gca().transAxes, fontsize=14, bbox=bbox_props)
    
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--')
    
    save_dir = os.path.join(folder_path, "UQ_Metrics")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'Calibration_Curve_GMM.png'), bbox_inches='tight')
    plt.close()

def calculate_energy_distance_vectorized(all_samples, all_true_vals):
    diffs_to_true = all_samples - all_true_vals[:, np.newaxis, :]
    dists_to_true = np.sqrt(np.sum(diffs_to_true**2, axis=-1) + 1e-10)
    term1 = np.mean(dists_to_true, axis=1) 

    n_sub = min(all_samples.shape[1], 100)
    sub_samples = all_samples[:, :n_sub, :]
    
    diffs_internal = sub_samples[:, :, np.newaxis, :] - sub_samples[:, np.newaxis, :, :]
    dists_internal = np.sqrt(np.sum(diffs_internal**2, axis=-1) + 1e-10)
    term2 = np.mean(dists_internal, axis=(1, 2)) 

    return term1 - 0.5 * term2

def optimized_peak_check(all_samples, lbound=0.45):
    N, S, D = all_samples.shape
    multimodal_mask = np.zeros((N, D), dtype=bool)
    x_grid = np.linspace(lbound - 0.05, 1.05, 150)
    
    for d in range(D):
        for i in range(N):
            data = all_samples[i, :, d]
            try:
                kde = gaussian_kde(data, bw_method='scott')
                y = kde(x_grid)
                peaks, _ = find_peaks(y, height=np.max(y) * 0.05, prominence=np.max(y) * 0.02)
                if len(peaks) > 1:
                    multimodal_mask[i, d] = True
            except:
                continue 
                
    per_latent_rate = np.mean(multimodal_mask, axis=0)
    global_multi_rate = np.mean(np.any(multimodal_mask, axis=1))
    
    return global_multi_rate, per_latent_rate

def calculate_non_gaussianity(all_samples):
    mean = np.mean(all_samples, axis=1, keepdims=True)
    std = np.std(all_samples, axis=1, keepdims=True) + 1e-6
    z = (all_samples - mean) / std
    
    skew = np.mean(z**3, axis=1)
    kurt = np.mean(z**4, axis=1) - 3 
    
    negentropy = (skew**2 / 12) + (kurt**2 / 48)
    return np.mean(negentropy, axis=0) 

def enhanced_metrics_comparison(predicted_stats, test_datasets, lbound=0.45, n_samples_for_ed=500):
    test_logits = predicted_stats['test_logits'] 
    test_locs = predicted_stats['test_locs']     
    test_scales = predicted_stats['test_scales'] 
    z_true = test_datasets['alpha_factors_true_test'] 
    
    N = test_logits.shape[0]
    
    all_samples = []
    for i in range(N):
        # Using the GMM generative sampling model 
        samples = generate_gmm_physical_samples(
            test_logits[i], test_locs[i], test_scales[i], n_samples_for_ed, lbound
        )
        all_samples.append(samples)
        
    all_samples = np.array(all_samples)
    
    ed_scores = calculate_energy_distance_vectorized(all_samples, z_true)
    global_multi_rate, per_latent_rate = optimized_peak_check(all_samples, lbound)
    non_gaussian_scores = calculate_non_gaussianity(all_samples)
    
    metrics = {
        'Avg_Energy_Distance': float(np.mean(ed_scores)),
        'Global_Multimodality_Rate': float(global_multi_rate),
        'Per_Latent_Multimodality_Rates': [float(x) for x in per_latent_rate],
        'Avg_Non_Gaussianity_Score': float(np.mean(non_gaussian_scores)),
        'Per_Latent_Non_Gaussianity': [float(x) for x in non_gaussian_scores],
        'ED_Std_Dev': float(np.std(ed_scores))
    }
    
    return metrics