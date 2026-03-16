import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import norm, chi2, pearsonr
from scipy.linalg import solve_triangular
import os

tfd = tfp.distributions

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
        "figure.dpi": 300
    })

def calculate_and_plot_calibration_curve(predicted_stats, test_datasets, lbound, folder_path):
    """
    Calculates the Marginal Calibration Curve and Mean Absolute Calibration Error (MACE).
    """
    configure_academic_plots()
    
    # Squeeze to fix (N, 1, D) -> (N, D) shape mismatches
    locs = np.squeeze(np.array(predicted_stats['test_means'], dtype=np.float64))
    scales = np.squeeze(np.array(predicted_stats['test_scales'], dtype=np.float64))
    z_true = np.squeeze(np.array(test_datasets['alpha_factors_true_test'], dtype=np.float64))
    
    quantiles = np.linspace(0.05, 0.95, 19)
    empirical_coverages = []
    
    trunc_norm = tfd.TruncatedNormal(loc=locs, scale=scales, low=lbound + 0.0001, high=0.9999)
    cdf_true = trunc_norm.cdf(z_true).numpy()
    
    for q in quantiles:
        lower_bound = (1 - q) / 2
        upper_bound = 1 - (1 - q) / 2
        covered = (cdf_true >= lower_bound) & (cdf_true <= upper_bound)
        empirical_coverages.append(np.mean(covered))
        
    empirical_coverages = np.array(empirical_coverages)
    
    # Calculate MACE (Mean Absolute Calibration Error)
    mace = np.mean(np.abs(empirical_coverages - quantiles))
    
    plt.figure(figsize=(7, 7), facecolor='white')
    plt.plot([0, 1], [0, 1], 'k--', label='Ideal Calibration', lw=2)
    plt.plot(quantiles, empirical_coverages, 's-', color='#d62728', lw=2, markersize=7, label='GC-VAE Predicted')
    
    plt.fill_between(quantiles, quantiles - 0.05, quantiles + 0.05, color='gray', alpha=0.15, label=r'$\pm 5\%$ Bounds')
    
    plt.xlabel('Nominal Confidence Level', fontweight='bold')
    plt.ylabel('Empirical Coverage', fontweight='bold')
    plt.title('Reliability Diagram (Marginal Calibration)', pad=15)
    
    # Add metric text box
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.9)
    plt.text(0.05, 0.85, f"MACE: {mace:.3f}", transform=plt.gca().transAxes, fontsize=14, bbox=bbox_props)
    
    plt.legend(loc='lower right', frameon=True, edgecolor='black')
    plt.grid(True, linestyle='--')
    plt.xlim([0, 1]); plt.ylim([0, 1])
    
    save_dir = os.path.join(folder_path, "UQ_Metrics")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'Calibration_Curve.png'), bbox_inches='tight')
    plt.show()
    plt.close()

def calculate_multivariate_mahalanobis(predicted_stats, test_datasets, lbound, folder_path):
    """
    Calculates the Multivariate Mahalanobis distance in the Copula latent space.
    """
    configure_academic_plots()
    
    # Squeeze to remove arbitrary dimension 1
    locs = np.squeeze(np.array(predicted_stats['test_means'], dtype=np.float64))
    scales = np.squeeze(np.array(predicted_stats['test_scales'], dtype=np.float64))
    z_true = np.squeeze(np.array(test_datasets['alpha_factors_true_test'], dtype=np.float64))
    L_matrices = np.squeeze(np.array(predicted_stats['test_L_matrices'], dtype=np.float64))
    
    n_samples, n_dims = locs.shape
    
    trunc_norm = tfd.TruncatedNormal(loc=locs, scale=scales, low=lbound + 0.0001, high=0.9999)
    u_true = trunc_norm.cdf(z_true).numpy()
    u_true = np.clip(u_true, 1e-6, 1.0 - 1e-6)
    y_latent = norm.ppf(u_true)
    
    mahalanobis_sq = np.zeros(n_samples)
    
    for i in range(n_samples):
        try:
            # y_latent[i] is now strictly (5,) and L_matrices[i] is (5, 5)
            x = solve_triangular(L_matrices[i], y_latent[i], lower=True)
            mahalanobis_sq[i] = np.sum(x**2)
        except np.linalg.LinAlgError:
            mahalanobis_sq[i] = np.nan
            
    valid_idx = ~np.isnan(mahalanobis_sq)
    mahalanobis_sq = mahalanobis_sq[valid_idx]
    
    emp_mean = np.mean(mahalanobis_sq)
    theo_mean = n_dims # Mean of Chi-square is 'k'
    
    plt.figure(figsize=(8, 6), facecolor='white')
    
    plt.hist(mahalanobis_sq, bins=40, density=True, alpha=0.5, color='#1f77b4', edgecolor='black', label='Empirical $D_M^2$')
    
    x_vals = np.linspace(0, np.max(mahalanobis_sq), 200)
    chi2_pdf = chi2.pdf(x_vals, df=n_dims)
    plt.plot(x_vals, chi2_pdf, color='#d62728', lw=2.5, label=f'Theoretical $\chi^2$ ($k={n_dims}$)')
    
    plt.xlabel('Squared Mahalanobis Distance in Latent Space ($D_M^2$)', fontweight='bold')
    plt.ylabel('Probability Density', fontweight='bold')
    plt.title('Gaussian Copula Dependency Validation', pad=15)
    
    # Add metric text box
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.9)
    stats_text = f"Empirical $\mu$: {emp_mean:.2f}\nTheoretical $\mu$: {theo_mean:.2f}"
    plt.text(0.65, 0.55, stats_text, transform=plt.gca().transAxes, fontsize=13, bbox=bbox_props)
    
    plt.legend(loc='upper right', frameon=True, edgecolor='black')
    plt.grid(True, linestyle='--')
    plt.xlim(left=0)
    
    save_dir = os.path.join(folder_path, "UQ_Metrics")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'Mahalanobis_ChiSquared.png'), bbox_inches='tight')
    plt.show()
    plt.close()

def plot_error_vs_confidence(predicted_stats, test_datasets, folder_path):
    """
    Plots the Absolute Error against Predicted Standard Deviation to show uncertainty awareness.
    """
    configure_academic_plots()
    
    locs = np.squeeze(np.array(predicted_stats['test_means'], dtype=np.float64)).flatten()
    scales = np.squeeze(np.array(predicted_stats['test_scales'], dtype=np.float64)).flatten()
    z_true = np.squeeze(np.array(test_datasets['alpha_factors_true_test'], dtype=np.float64)).flatten()
    
    abs_errors = np.abs(locs - z_true)
    
    # Calculate correlation to formally quantify uncertainty awareness
    corr, p_val = pearsonr(scales, abs_errors)
    
    sorted_indices = np.argsort(scales)
    scales_sorted = scales[sorted_indices]
    errors_sorted = abs_errors[sorted_indices]
    
    n_bins = 15
    bin_edges = np.percentile(scales_sorted, np.linspace(0, 100, n_bins + 1))
    
    bin_centers = []
    mean_errors = []
    std_errors = []
    
    for i in range(n_bins):
        mask = (scales_sorted >= bin_edges[i]) & (scales_sorted <= bin_edges[i+1])
        if np.any(mask):
            bin_centers.append(np.mean(scales_sorted[mask]))
            mean_errors.append(np.mean(errors_sorted[mask]))
            std_errors.append(np.std(errors_sorted[mask]) / np.sqrt(np.sum(mask))) 
            
    plt.figure(figsize=(8, 6), facecolor='white')
    
    # Hexbin or scatter for dense data
    plt.scatter(scales, abs_errors, alpha=0.15, color='gray', s=15, edgecolor='none', label='Test Set Estimates')
    
    plt.errorbar(bin_centers, mean_errors, yerr=std_errors, fmt='o-', color='#ff7f0e', 
                 lw=3, capsize=5, capthick=2, markersize=8, label='Binned Absolute Error')
    
    plt.plot([0, max(bin_centers)], [0, max(bin_centers) * 0.8], 'k:', lw=2, label='Ideal Correlation')

    plt.xlabel('Predicted Uncertainty ($\sigma$)', fontweight='bold')
    plt.ylabel('Absolute Error ($|z_{true} - z_{pred}|$)', fontweight='bold')
    plt.title('Uncertainty Awareness (Error vs. Confidence)', pad=15)
    
    # Add statistical text box
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.9)
    plt.text(0.05, 0.85, f"Pearson's $r$: {corr:.2f}", transform=plt.gca().transAxes, fontsize=14, bbox=bbox_props)
    
    plt.legend(loc='upper right', frameon=True, edgecolor='black')
    plt.grid(True, linestyle='--')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    save_dir = os.path.join(folder_path, "UQ_Metrics")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'Error_vs_Confidence.png'), bbox_inches='tight')
    plt.show()
    plt.close()