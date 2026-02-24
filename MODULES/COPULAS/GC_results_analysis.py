#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 2026
@author: afernandez

Visualization of Joint Posterior Distributions (Uncertainty Quantification).
- Loads Trained Model.
- Generates Monte Carlo samples from the predicted Copula distribution.
- Propagates samples through the Physics Engine (Eigen Solver).
- Computes Bayesian Posterior based on Data Misfit.
- Plots 2D KDE of the damage parameters.
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import tensorflow_probability as tfp
from scipy.stats import norm

# -----------------------------------------
# --- CRITICAL CONFIGURATION: FORCE CPU & DISABLE JIT ---
# This bypasses the 'libdevice not found' XLA/GPU error by hiding the GPU.
try:
    tf.config.set_visible_devices([], 'GPU')
    # Disable JIT to prevent XLA from trying to compile ops that miss libdevice
    tf.config.optimizer.set_jit(False) 
    print("🖥️ GPU and JIT disabled for visualization to prevent XLA/libdevice errors.")
except:
    pass

tfd = tfp.distributions

# --- LOCAL IMPORTS (Preserving your structure) ---
from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.COPULAS.GC_GMm_models import My_CopulaVAE_withEigen
from MODULES.COPULAS.GC_GMm_functions import build_correlation_matrices_from_cholesky
from MODULES.COPULAS.GC_plot_posteriors import calculate_posterior_PDF_info, plot_results_PDF_uncertainty, plot_physical_damage_profile


def main():
    K.backend.set_floatx('float32') 

    # --- Config ---
    filename = "22Feb_Bayesian_MACloss_Beta0.3_Samples1_LR1e-05_Epochs30000"
    folder_path = os.path.join('Output', 'Gaussian_Copula', filename)
    
    # Load Problem Info
    info_path = os.path.join(folder_path, 'Problem_info.npy')
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Problem_info.npy not found at {info_path}")
    
    info = np.load(info_path, allow_pickle=True).item()
    
    # Extract parameters
    input_dim = info['input_dim_enc']
    n_dims = info['n_dims']
    beta = info['beta']
    mean_freq = info['mean_f']
    std_freq = info['std_f']
    n_elements = 5
    n_dofs = 2 * (n_elements + 1)
    batch_size = 256
    
    # 2. Identify fixed indices (Simply Supported)
    # Node 0 vertical is index 0; Node N vertical is index 10
    fixed_dofs = [0, n_dofs - 2] 
    # 3. Create the list of free DOFs
    all_dofs = np.arange(n_dofs)
    free_dofs = np.delete(all_dofs, fixed_dofs)

    # %% 1. Initialization and Data Loading
    K.utils.set_random_seed(1234)
    data_path = os.path.join("Data", "11Feb2026_Corrected_Randomdata5elements")
    print(f"Loading data from {data_path}...")
    
    (Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, 
     Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, 
     Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test, 
     mean_f, std_f) = load_data(data_path, batch_size)
    
    # Load Physics Matrices
    Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)
    
    # Fix input dim calculation based on your previous messages
    n_modes = Freqs_true_train.shape[1]
    
    # %% 2. Model Re-instantiation and Loading Weights
    print("Initializing model...")
    # Force CPU context for initialization
    with tf.device('/CPU:0'):    
        model = My_CopulaVAE_withEigen(
            input_dim=input_dim,
            num_dofs=n_dofs,
            n_elements=n_elements,
            n_modes=n_modes,
            Ke_matrices=Ke_matrices,
            Mfree=Mfree,
            L_inv=L_inv,
            epsi=0.0,
            n_dims=n_elements,
            num_gaussians=1,
            num_samples=1,
            beta=beta,
            mean_f=mean_f,
            std_f=std_f,
            fixed_dofs_indices=[0, n_dofs-2]
        )
    
        # MANDATORY: We must build the model before loading HDF5 weights.
        # We do this by passing a dummy batch through the model.
        print("Building model variables...")
        dummy_f = tf.zeros((1, n_modes))
        dummy_r = tf.zeros((1, n_modes, 6))
        dummy_v = tf.zeros((1, n_modes, 6))
        dummy_z = tf.zeros((1, n_dims))
        
        # This call creates the internal weights/variables
        _ = model([dummy_f, dummy_r, dummy_v, dummy_z])

    # Now that variables are created, we load the weights
    weights_path = os.path.join(folder_path, "final_model_weights.weights.h5")
    if os.path.exists(weights_path):
        # FIX: Use by_name=True to resolve layer count mismatch in subclassed models
        model.load_weights(weights_path, by_name=True)
        print("Model weights loaded successfully.")
    else:
        print(f"CRITICAL: Weights file not found at {weights_path}")
    
    
   
    # --- Prediction Step ---
    print("Running Encoder Prediction...")
    inverse_model = model.Encoder_model
    # Note: inputs are passed as a list
    test_means, test_scales, test_weight_vals, test_offdiag_elems, test_diag_elems = inverse_model.predict(
        [Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test]
    )
    
    test_datasets = {
        'Freqs_true_test': Freqs_true_test,
        'Rotmodes_true_test': Rotmodes_true_test,
        'Vertmodes_true_test':Vertmodes_true_test,
        'alpha_factors_true_test': alpha_factors_true_test
    }
    
    
    # Build L matrices
    L_matrices = build_correlation_matrices_from_cholesky(test_offdiag_elems, test_diag_elems, n_dims)
    
    predicted_stats = {
        'test_means': test_means,
        'test_scales': test_scales,
        'test_weights':test_weight_vals,
        'test_L_matrices': L_matrices
    }
    
    # --- Visualization Loop ---
    positions = [0, 1, 7, 9, 11, 17, 25, 34, 45, 100, 138, 219, 234, 343, 456, 555, 612, 690, 761]
    # positions  = [2,20,21,41,43,48,50,63, 64, 65, 77, 78, 91,92,98,99,102,560,576]
    # positions = [300,301,302,303,304,305,310,311,312,313,314,315,321,322,323]
    
    
    n_samples = 1500 
    N_test_samples = len(Freqs_true_test)
    
    for pos in positions:
        if pos < N_test_samples:
            # plot_results_PDF_uncertainty(model, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
            #                                  predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, folder_path)

            z_true, z_samples, posterior_weights = calculate_posterior_PDF_info(model, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                             predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, folder_path)

            plot_physical_damage_profile(z_samples, posterior_weights, z_true, n_elements, pos, folder_path)



    # %% 5. Advanced Visualization & Uncertainty Quantification
    print(f"Analyzing and Plotting Position {pos}...")
    
    # 5.1 Uncertainty Quantification
    expected_z = np.sum(z_samples * posterior_pdf[:, np.newaxis], axis=0) / np.sum(posterior_pdf)
    std_z = np.sqrt(np.sum(np.square(z_samples - expected_z) * posterior_pdf[:, np.newaxis], axis=0) / np.sum(posterior_pdf))
    z_true = alpha_factors_true_test[pos] if 'alpha_factors_true_test' in locals() else None
    
    # 5.2 Main Posterior Plot (Pairwise Matrices)
    fig, axes = plt.subplots(n_elements, n_elements, figsize=(10, 10), facecolor='white')
    labels = [f'$z_{{{k+1}}}$' for k in range(n_elements)]
    mask = posterior_pdf > (np.max(posterior_pdf) * 0.0001)
    
    for r in range(n_elements):
        for c in range(n_elements):
            ax = axes[r, c]
            if r == c: # Diagonal: Physical Labels
                ax.text(0.5, 0.5, labels[r], fontsize=22, ha='center', va='center', fontweight='bold', color='#333333')
                ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
                ax.axis('off')
            elif r > c: # Lower Triangle: Smooth 2D Joint PDF
                xi, yi = np.mgrid[0:1:100j, 0:1:100j]
                kde_coords = np.vstack([z_samples[mask, c], z_samples[mask, r]])
                kde = gaussian_kde(kde_coords, weights=posterior_pdf[mask])
                zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                
                ax.contourf(xi, yi, zi, levels=30, cmap='viridis', alpha=0.9)
                ax.contour(xi, yi, zi, levels=5, colors='white', linewidths=0.3, alpha=0.2)
                
                if z_true is not None:
                    ax.plot(z_true[c], z_true[r], 'ro', markersize=7, markeredgecolor='white', markeredgewidth=1, zorder=10)
                
                if c == 0: ax.set_ylabel(labels[r], fontsize=12)
                if r == n_elements - 1: ax.set_xlabel(labels[c], fontsize=12)
                ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
                ax.tick_params(labelsize=8)
                ax.grid(True, linestyle=':', alpha=0.3)
            else:
                ax.axis('off')
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    save_dir = os.path.join("MODULES", "POSTPROCESSING", "Ground_truth_plots")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'P{pos}_JointPosterior.png'), dpi=500, bbox_inches='tight')
    plt.show()
    
    # 5.3 Enhanced Physical Damage Profile (Fancier Visuals)
    fig, (ax_bar, ax_beam) = plt.subplots(2, 1, figsize=(10, 6.5), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    
    elements = np.arange(1, n_elements + 1)
    color_estimate = '#4575b4' # Sophisticated Blue
    color_true = '#d73027'     # Strong Red
    
    # Top Plot: Bar chart with Uncertainty
    ax_bar.bar(elements, expected_z, yerr=std_z, color=color_estimate, alpha=0.65, 
               label='Estimated Stiffness Reduction ($E[z|\mathbf{m}]$)', 
               capsize=8, error_kw={'elinewidth':2, 'capthick':2, 'ecolor': '#1a1a1a'})
    
    if z_true is not None:
        # Step boundaries for 5 elements
        x_step = np.arange(0.5, n_elements + 1.5, 1)
        y_step = np.concatenate([z_true, [z_true[-1]]])
        ax_bar.step(x_step, y_step, where='post', color=color_true, 
                    label='True Damage State ($\mathbf{z}^*$)', linestyle='--', lw=2.5, zorder=5)
    
    ax_bar.set_ylabel('Stiffness Reduction ($z$)', fontsize=13, fontweight='medium')
    ax_bar.set_ylim([0, 1.2])
    ax_bar.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False, fontsize=11)
    ax_bar.grid(axis='y', alpha=0.2, linestyle='-')
    
    # Add explanatory note for Uncertainty
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='silver')
    ax_bar.text(0.02, 0.95, "Note: Error bars denote the 1$\sigma$ \ncredible interval (posterior std. dev.)", 
                transform=ax_bar.transAxes, fontsize=9, verticalalignment='top', bbox=props)
    
    # Bottom Plot: Physical Beam Heatmap
    beam_viz = expected_z.reshape(1, -1)
    im = ax_beam.imshow(beam_viz, cmap='YlGnBu', aspect='auto', extent=[0.5, n_elements + 0.5, 0, 1], vmin=0, vmax=1)
    
    # Clean up beam visualization
    ax_beam.set_yticks([])
    ax_beam.set_xticks(elements)
    ax_beam.set_xticklabels([f'Element {k}' for k in elements], fontsize=11)
    ax_beam.tick_params(axis='x', length=0)
    
    # Add values and physical frame
    for j, val in enumerate(expected_z):
        text_color = 'white' if val > 0.6 else 'black'
        ax_beam.text(j + 1, 0.5, f'{val:.2f}', ha='center', va='center', 
                     color=text_color, fontweight='bold', fontsize=11)
    
    # Outer beam border
    for spine in ax_beam.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'P{pos}_PhysicalProfile_Enhanced.png'), dpi=500, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    main()