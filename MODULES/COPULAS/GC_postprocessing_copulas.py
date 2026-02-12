#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:11:10 2025

@author: afernandez
"""

import tensorflow as tf
import numpy as np
import os
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy import stats
import seaborn as sns

#Graph configuration function (font sizes)
def plot_configuration():
    plt.rc('font', size = 16)          # controls default text sizes
    plt.rc('axes', titlesize = 16)     # fontsize of the axes title
    plt.rc('axes', labelsize = 16)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize = 16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = 16)    # fontsize of the tick labels
    plt.rc('legend', fontsize = 16)   # legend fontsize
    plt.rc('figure', titlesize= 16)   # fontsize of the figure title
    

sns.set(style="whitegrid", rc={"axes.facecolor": "#f0f0f0", "grid.color": "gray", "grid.linestyle": "--"})
sns.set(style = "dark")
plot_configuration()
    
def plot_trainval_loss(model, folder_path):
    trainvalloss_plot_fig = plt.figure()
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(model.history.history['loss'], color = 'black', linewidth = 1.5, linestyle ='--', label  = 'Train')
    plt.plot(model.history.history['val_loss'], color = 'darkgray', linewidth = 1.5, label = 'Validation')
    plt.legend(loc='lower left', fontsize=14)
    plt.grid(True, linestyle='-.', alpha=1.)
    trainvalloss_plot_fig.savefig(os.path.join(folder_path, 'TrainVal_lossevolution.png'),dpi = 500, bbox_inches='tight')
    # plt.show()
    plt.close()

def plot_regularizer_loss(model, folder_path):
    loss_plot = plt.figure()
    # plt.yscale('log')
    # plt.xscale('log')
    plt.plot(model.history.history['Alpha_regularizer'], color = 'tomato', linewidth = 1.5, label = r'$\mathcal{L}_{Reg}$')
    # plt.plot(model_history.history['Joint_copula_dens_term'], color = 'lime', linewidth = 1.5, label = r'$\mathcal{L}_{PDF}$')
    plt.plot(model.history.history['total_loss_with_regularizer'], color = 'black', linewidth = 1.5, linestyle ='--', label = r'$\mathcal{L}_{total}$')
    plt.legend(loc='lower left', fontsize=14)
    plt.grid(True, linestyle='-.', alpha=1.)
    loss_plot.savefig(os.path.join(folder_path, 'Loss_regularizer.png'),dpi = 500, bbox_inches='tight')
    # plt.show()
    plt.close()

def plot_JointPDF_loss(model, folder_path):
    loss_plot = plt.figure()
    # plt.yscale('log')
    # plt.xscale('log')
    plt.plot(model.history.history['Joint_copula_dens_term'], color = 'tomato', linewidth = 1.5, label = r'$\mathcal{L}_{jointPDF}$')
    # plt.plot(model_history.history['Joint_copula_dens_term'], color = 'lime', linewidth = 1.5, label = r'$\mathcal{L}_{PDF}$')
    # plt.plot(model.history.history['total_loss_with_regularizer'], color = 'black', linewidth = 1.5, linestyle ='--', label = r'$\mathcal{L}_{total}$')
    plt.legend(loc='lower left', fontsize=14)
    plt.grid(True, linestyle='-.', alpha=1.)
    loss_plot.savefig(os.path.join(folder_path, 'Loss_jointPDF.png'),dpi = 500, bbox_inches='tight')
    # plt.show()
    plt.close()

def plot_Freqs_loss(model, folder_path):
    loss_plot = plt.figure()
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(model.history.history['Freqs_loss'], color = 'tomato', linewidth = 1.5, label = r'$\mathcal{L}_{Freqs}$')
    plt.plot(model.history.history['Joint_copula_dens_term'], color = 'lime', linewidth = 1.5, label = r'$\mathcal{L}_{PDF}$')
    # plt.plot(model.history.history['total_loss_with_regularizer'], color = 'black', linewidth = 1.5, linestyle ='--', label = r'$\mathcal{L}_{total}$')
    plt.legend(loc='lower left', fontsize=14)
    plt.grid(True, linestyle='-.', alpha=1.)
    loss_plot.savefig(os.path.join(folder_path, 'Loss_freqs.png'),dpi = 500, bbox_inches='tight')
    # plt.show()
    plt.close()

####################################################################################################


def multimodal_marginal(x, locs, scales, weight_vals):
    """
    Multimodal marginal distribution for one dimension.

    Args:
        x: A tensor of shape [n_points], the input values at which to evaluate the distributions.
        locs: A tensor of shape [n_gaussians], specifying the means of the Gaussian components for each dimension.
        scales: A tensor of shape [n_gaussians], specifying the standard deviations of the Gaussian components.
        weights: A tensor of shape [n_gaussians], specifying the mixture weights for each Gaussian component (should sum to 1 along the second axis).

    Returns:
        A tensor of shape [n_points, n_dims], where each entry is the value of the multimodal marginal distribution at the corresponding input.
    """
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=weight_vals),
        components_distribution=tfd.TruncatedNormal(loc=locs, scale=tf.maximum(scales, 1e-03), low =0.0, high=1.0)

    )
    mixture_probs = gm.prob(x)
    return mixture_probs


class Interpolation1D(tf.keras.layers.Layer):
    def __init__(self):
        super(Interpolation1D, self).__init__()

    def call(self, x, xp, fp):
        """
        Perform linear interpolation in 1D.

        Parameters:
        - x: Query points where we want interpolated values, shape [n_points].
        - xp: Points defining the domain of the interpolation, shape [n_values].
        - fp: Values of the function at the domain points, shape [n_values].

        Returns:
        - Interpolated values at `x`, shape [n_points].
        """
        xp = tf.sort(xp)
        tf.debugging.assert_rank(xp, 1, "xp must be a 1D tensor")
        tf.debugging.assert_rank(fp, 1, "fp must be a 1D tensor")
        tf.debugging.assert_equal(tf.shape(xp), tf.shape(fp), "xp and fp must have the same shape")

        x_min = tf.reduce_min(xp)
        x_max = tf.reduce_max(xp)
        x_range = x_max - x_min
        margin = 0.01 * x_range
        x = tf.clip_by_value(x, x_min - margin, x_max + margin)

        is_rank_1 = tf.equal(tf.rank(x), 1)

        def rank_1_case():
            return tf.expand_dims(x, axis=0)

        def rank_2_case():
            return x

        x_processed = tf.cond(is_rank_1, rank_1_case, rank_2_case)

        indices = tf.reduce_sum(tf.cast(x_processed[:, :, None] >= xp[None, None, :], tf.int32), axis=-1) - 1

        n_values = tf.shape(xp)[0]
        indices = tf.clip_by_value(indices, 0, n_values - 2)

        x0 = tf.gather(xp, indices)
        x1 = tf.gather(xp, indices + 1)
        f0 = tf.gather(fp, indices)
        f1 = tf.gather(fp, indices + 1)

        slope = tf.where(tf.abs(x1 - x0) > 1e-8, (f1 - f0) / (x1 - x0), 0.0)
        interpolated = f0 + slope * (x_processed - x0)

        left_extrapolate = x_processed < x_min
        right_extrapolate = x_processed > x_max

        interpolated = tf.where(left_extrapolate, tf.gather(fp, tf.zeros_like(indices, dtype=tf.int32)), interpolated)
        interpolated = tf.where(right_extrapolate, tf.gather(fp, tf.ones_like(indices, dtype=tf.int32) * (n_values-1)), interpolated)

        return tf.squeeze(interpolated)
    
    
def build_marginal_samples(locs, scales, weight_vals, copula_samples):
    """
    Build marginal samples using TensorFlow with a custom Interpolation1D layer.

    Parameters:
    - locs: Locations of components in the Gaussian mixture, shape [n_modes, n_dims].
    - scales: Scales of components in the Gaussian mixture, shape [n_modes, n_dims].
    - weights: Weights of components in the Gaussian mixture, shape [n_modes, n_dims].
    - copula_samples: Samples in uniform space, shape [n_samples, n_dims].

    Returns:
    - Marginal samples corresponding to the copula samples.
    """
    interpolator = Interpolation1D()
    n_dims = locs.shape[-1]
    n_unif_points = 5000
    x = tf.linspace(0.0, 1.0, n_unif_points) 
    marg_samples = tf.TensorArray(dtype=tf.float32, size=n_dims)

    for i in range(n_dims):
        marginal_pdf_vals = multimodal_marginal(
            x, locs[:, i], scales[:, i], weight_vals[:, i]
        )
        marginal_cdf_vals = tf.cumsum(marginal_pdf_vals, axis=0) / tf.reduce_sum(marginal_pdf_vals)            
        marginal = interpolator(copula_samples[:, i], marginal_cdf_vals, x)
        marg_samples = marg_samples.write(i, marginal)

    marginal_samples = marg_samples.stack()
    marginal_samples = tf.transpose(marginal_samples, perm=[1, 0])
    return marginal_samples


def gaussian_copula(LT_matrix, n_dims, n_samples):
    """
    Generates samples from a batch of Gaussian copulas.
    """
    mvn_model = tfd.MultivariateNormalTriL(loc=tf.zeros(n_dims), scale_tril=LT_matrix)
    mvn_samples = mvn_model.sample(n_samples)

    copula_samples = tfd.Normal(loc=0.0, scale=1.0).cdf(mvn_samples)
    return copula_samples, mvn_samples, mvn_model


def gaussian_marginal_samples(locs, scales, copula_samples):
    """
    Builds marginal samples given copula samples and Gaussian marginal parameters.
    Returns marginal_samples with shape (batch_size, num_samples, n_dims).
    """
    marginal_samples = tfd.TruncatedNormal(loc=locs, scale=scales, low = 0.00005, high = 0.95).quantile(copula_samples)
    # marginal_samples: (num_samples, n_dims)
    return marginal_samples





def plot_Datamisift_KDE(point_cloud, pos, chosen_axis, clouds_path):
    # PLOT FOR ONE SPECIFIC TEST CASE (YOU SHOULD NOT PLOT FOR ALL THE TESTING  SAMPLES, THERE ARE MANY )
    # for i in range(len(positions)):
    #     point_cloud = point_clouds[i,:]
    
    # 1. Extract coordinates and dloss
    full_coords = point_cloud[:, :5] # Shape (Ntest, 5)
    dloss = point_cloud[:, 5]   # Shape (Ntest,)
    
    # 2. Calculate weights (inversely proportional to dloss)
    epsilon = 1e-10# A small constant to prevent division by zero
    weights = 1.0 / (dloss + epsilon)
    weights /= np.sum(weights) # Optional: Normalize weights (good practice, though gaussian_kde might handle it)
    
    
    # 3. Create the weighted KDE object
    # Note: gaussian_kde expects data in shape (ndim, npoints)
    coords = full_coords[:, chosen_axis]
    
    try:
        kde = stats.gaussian_kde(coords.T, weights=weights)
        print("Weighted KDE created successfully.")
    except ValueError as e:
        print(f"Error creating KDE: {e}")
        print("Check for NaN or Inf values in coordinates or weights.")

    
    # 4. Define a grid to evaluate the KDE on
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    grid_margin = 1.0 # Add some margin around the data range
    grid_points = 100j # 
    X, Y = np.mgrid[xmin-grid_margin:xmax+grid_margin:grid_points, ymin-grid_margin:ymax+grid_margin:grid_points]
    positis = np.vstack([X.ravel(), Y.ravel()])
    
    # 5. Evaluate the KDE on the grid
    Z = np.reshape(kde(positis).T, X.shape)
    print("KDE evaluated on grid.")
    
    
    
    # Method A: Percentage of Maximum Density
    peak_density = Z.max()
    # Set percentage (e.g., 0.1 means show densities above 10% of the peak)
    threshold_percentage = 0.6 # *** ADJUST THIS VALUE (e.g., 0.05, 0.1, 0.2) ***
    density_threshold = peak_density * threshold_percentage

    print(f"Maximum density: {peak_density:.4f}")
    print(f"Density threshold (showing values above): {density_threshold:.4f}")
    # Use a reasonable number of levels for the focused view (e.g., 50)
    num_high_levels = 150
    high_density_levels = np.linspace(density_threshold, peak_density, num=num_high_levels)

    # # 6. Visualize the result
    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot the density
    contour = ax.contourf(X, Y, Z, levels=high_density_levels, cmap='nipy_spectral', extend='min') # Filled contour plot
    fig.colorbar(contour, label='Density')

    ax.set_xlabel(r'$z_{1}$')
    ax.set_ylabel(r'$z_{2}$')
    ax.set_aspect('equal', adjustable='box')
    
    # --- Grid Configuration ---
    # Apply grid to the specific axes object 'ax'
    # Set a zorder for the grid. Values > 0 are typically above default plot elements.
    # You can also specify a color to ensure it's visible.
    ax.grid(True, linestyle='--', alpha=0.7, color='white', zorder=5) # Increased alpha slightly, added color, and zorder
    
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.])
    plt.savefig(os.path.join(clouds_path, 'GT_KDE'+ str(pos)+'axis'+str(chosen_axis)+'.png'), dpi=300, bbox_inches='tight')

    # plt.show()

# BEST COLOR MAPS: gnuplot2, nipy_spectral, 

###################### now plot the KDE PDF of the predictions provided by the NN 
def plot_KDE_pdf(locs, scales, weights, LT_matrix, n_dims, n_samples, pos,  folder_path):
    copula_samples, mvn_samples, mvn_model = gaussian_copula(LT_matrix, n_dims, n_samples) 
    marginal_samples = build_marginal_samples(locs, scales, weights, copula_samples)
    # choose the  axis (THIS MUST BE DONE IN A LOOP TO PLOT ALL THE AXIS COMBINATIONS) [(0,1),(0,2),(0,3) (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    
    x = marginal_samples[:, 0].numpy()
    y = marginal_samples[:,1].numpy()      
    # Perform kernel density estimation
    kde = gaussian_kde([x, y])
    
    # Create a grid to evaluate the density
    num_points = 500
    xgrid = np.linspace(0,1, num_points)
    ygrid = np.linspace(0,1, num_points)
    X, Y = np.meshgrid(xgrid, ygrid)
    psts = np.vstack([X.ravel(), Y.ravel()])
    
    # 5. Evaluate the KDE on the grid
    Z = kde(psts).reshape(X.shape)
    print("KDE evaluated on grid.")
    
    # Method A: Percentage of Maximum Density
    peak_density = Z.max()
    # Set percentage (e.g., 0.1 means show densities above 10% of the peak)
    threshold_percentage = 0.75 # *** ADJUST THIS VALUE (e.g., 0.05, 0.1, 0.2) ***
    density_threshold = peak_density * threshold_percentage
    
    print(f"Maximum density: {peak_density:.4f}")
    print(f"Density threshold (showing values above): {density_threshold:.4f}")
    # Use a reasonable number of levels for the focused view (e.g., 50)
    num_high_levels = 150
    high_density_levels = np.linspace(density_threshold, peak_density, num=num_high_levels)
    
     # 6. Visualize the result
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the density
    # Lower the zorder of the contourf plot so the grid can be on top
    contour = ax.contourf(X, Y, Z, levels=high_density_levels, cmap='nipy_spectral', extend='min', zorder=0)
    fig.colorbar(contour, label='Density')
    
    ax.set_xlabel(r'$z_{1}$')
    ax.set_ylabel(r'$z_{2}$')
    ax.set_aspect('equal', adjustable='box')
    
    # --- Grid Configuration ---
    # Apply grid to the specific axes object 'ax'
    # Set a zorder for the grid. Values > 0 are typically above default plot elements.
    # You can also specify a color to ensure it's visible.
    # ax.grid(True, linestyle='--', alpha=0.7, color='white', zorder=5) # Increased alpha slightly, added color, and zorder
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
            
    # plt.savefig(os.path.join(folder_path, 'test_KDE_case_'+ str(pos)+'_Axis_'+str(chosen_axis)+'.png'), dpi=300, bbox_inches='tight')
    
    plt.show()
    # plt.close()
      

from MODULES.COPULAS.GC_GMm_eigen_functions import assemble_global_Kmatrices
# %% 2. Physics Engine & MAC Calculation
def calculate_MAC(phi_true, phi_pred):
    """
    Computes MAC: (phi_t^T * phi_p)^2 / ((phi_t^T * phi_t) * (phi_p^T * phi_p))
    phi_true/pred shapes: (Batch, N_dofs, N_modes)
    """
    # Numerator: dot product squared
    inner_prod = np.sum(phi_true * phi_pred, axis=1) # (Batch, N_modes)
    numerator = np.square(inner_prod)
    
    # Denominator: norms squared
    denom_true = np.sum(np.square(phi_true), axis=1)
    denom_pred = np.sum(np.square(phi_pred), axis=1)
    denominator = denom_true * denom_pred + 1e-10
    
    return numerator / denominator

# =============================================================================
# 2. MAIN VISUALIZATION FUNCTION (Using your provided structure)
# =============================================================================
def physics_engine_step(K_batch, L_inv_tf, n_modes, free_dofs, n_dofs):
    """
    K_batch: (Batch, Free_DOF, Free_DOF)
    L_inv_tf: (Free_DOF, Free_DOF) - L^-1 where M_free = L @ L.T
    n_modes: Number of modes to truncate
    free_dofs: List/Array of indices of free degrees of freedom
    n_dofs: Total number of DOFs (2 * n_nodes)
    """
    with tf.device('/CPU:0'):
        # 1. Solve Eigenproblem in Modal Space: A = L^-1 * K * L^-T
        # Note: Script uses L_inv @ K @ L_inv.T. 
        # Since L_inv_tf is L^-1, we use L_inv on left and transpose on right.
        A_batch = tf.matmul(L_inv_tf, tf.matmul(K_batch, tf.transpose(L_inv_tf)))
        
        vals, vecs = tf.linalg.eigh(A_batch)
        
        # 2. Extract and Truncate
        vals_trunc = tf.clip_by_value(vals[:, :n_modes], 1e-08, 1e+20)
        f_Hz = tf.math.sqrt(vals_trunc) / (2 * np.pi)
        
        # 3. Recover Physical Modes for Free DOFs: phi_free = L^-T * vecs
        phi_free = tf.matmul(tf.transpose(L_inv_tf), vecs[:, :, :n_modes])
        
        # 4. FULL RECONSTRUCTION (Include Boundary Conditions)
        # We need to map (Batch, Free_DOF, Modes) -> (Batch, Total_DOF, Modes)
        batch_size = tf.shape(K_batch)[0]
        
        # Create a scatter mask to put free DOFs back into their global positions
        indices = tf.expand_dims(free_dofs, axis=-1) # Shape (Free_DOF, 1)
        
        # We transpose phi_free to (Free_DOF, Batch, Modes) for easier scattering
        phi_free_t = tf.transpose(phi_free, perm=[1, 0, 2])
        full_modes_t = tf.scatter_nd(indices, phi_free_t, shape=[n_dofs, batch_size, n_modes])
        
        # Transpose back to (Batch, Total_DOF, Modes)
        full_modes = tf.transpose(full_modes_t, perm=[1, 0, 2])
        
        # 5. SLICING (0, 2, 4... for Vertical; 1, 3, 5... for Rotational)
        # Shape: (Batch, n_nodes, n_modes)
        vert_modes = full_modes[:, 0::2, :]
        rot_modes = full_modes[:, 1::2, :]

        # 6. SIGN CONSISTENCY (Matching the peak of vertical displacement)
        # Find the index of the maximum absolute vertical displacement for each mode
        # vert_modes shape: (Batch, Node, Mode)
        abs_vert = tf.abs(vert_modes)
        max_idx = tf.argmax(abs_vert, axis=1) # (Batch, Mode)
        
        # Gather the actual values at those indices to find the sign
        # We use batch_gather logic via gather_nd
        batch_idx = tf.range(batch_size)[:, tf.newaxis]
        mode_idx = tf.range(n_modes)[tf.newaxis, :]
        
        # Create coordinates for gather_nd: [batch, max_idx_for_that_mode, mode]
        gather_coords = tf.stack([
            tf.broadcast_to(batch_idx, [batch_size, n_modes]),
            tf.cast(max_idx, tf.int32),
            tf.broadcast_to(mode_idx, [batch_size, n_modes])
        ], axis=-1)
        
        peak_vals = tf.gather_nd(vert_modes, gather_coords)
        signs = tf.sign(peak_vals) # (Batch, Mode)
        
        # Apply signs: (Batch, Node, Mode) * (Batch, 1, Mode)
        vert_modes = vert_modes * signs[:, tf.newaxis, :]
        rot_modes = rot_modes * signs[:, tf.newaxis, :]

        # 7. Final Transpose to match (Batch, n_modes, n_nodes)
        vert_final = tf.transpose(vert_modes, perm=[0, 2, 1])
        rot_final = tf.transpose(rot_modes, perm=[0, 2, 1])

        return f_Hz.numpy(), rot_final.numpy(), vert_final.numpy()
    
  
def plot_results_PDF_uncertainty(model, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                 predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, folder_path):
    """
    Generates uncertainty plots for a specific test sample 'pos'.
    """
    # Unpack stats for this specific sample
    locs = predicted_stats['test_means'][pos,:]
    scales = predicted_stats['test_scales'][pos,:] 
    weights= predicted_stats['test_weights'][pos,:]
    LT_matrix = predicted_stats['test_L_matrices'][pos,:] 
    
    Freqs_true = test_datasets['Freqs_true_test']
    Rotmodes_true = test_datasets['Rotmodes_true_test']
    Vertmodes_true = test_datasets['Vertmodes_true_test']
    Alphas_true = test_datasets['alpha_factors_true_test']
    
    n_dims = locs.shape[1]
    n_elements = n_dims
    
    
    print(f"\n--- Analyzing Sample #{pos} ---")
    
    # 1. GENERATE SAMPLES
    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
    copula_samples, mvn_samples, mvn_model = gaussian_copula(LT_matrix, n_dims, n_samples) 
    z_samples = build_marginal_samples(locs, scales, weights, copula_samples)
    z_samples = tf.cast(z_samples, dtype=tf.float32)
    z_samples = z_samples.numpy()
    
    # 2. PHYSICS PROPAGATION (Vectorized on GPU/CPU)
    print(f"Propagating {n_samples} samples through Physics Engine...")
    # B. Assemble Global K
    print("Assembling Global Stiffness...")
    Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', z_samples, tf.cast(Ke_matrices, dtype=tf.float32))
    Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_elements, n_samples, model.fixed_dofs_indices)
    
    # C. Solve Eigenproblem (Using the Model's Layer for consistency)
    batch_size = 256
    all_f, all_rot, all_vert = [], [], []
    for i in range(0, n_samples, batch_size):
        f, r, v = physics_engine_step(Kfree_matrices[i : i + batch_size], L_inv_tf, n_modes, free_dofs, n_dofs)
        all_f.append(f)
        all_rot.append(r)
        all_vert.append(v)
        
    f_pred = tf.concat(all_f, axis=0).numpy()      # (n_samples, n_modes)
    rot_pred = tf.concat(all_rot, axis=0).numpy()  # (n_samples, n_modes, n_nodes)
    vert_pred = tf.concat(all_vert, axis=0).numpy() # (n_samples, n_modes, n_nodes)
    
    # 3. CALCULATE POSTERIOR (DATA MISFIT)
    log_obs_f = Freqs_true[pos]     # (n_modes,)
    obs_r = Rotmodes_true[pos]  # (n_modes, n_nodes)
    obs_v = Vertmodes_true[pos] # (n_modes, n_nodes)
    
    # A. Frequency Loss
    pred_log = np.log(f_pred)
    pred_logscaled = (pred_log - mean_freq) / std_freq
    f_err = np.square(log_obs_f -pred_logscaled)
    loss_f = np.mean(f_err, axis=1) # (n_samples,)

    
    # B. MAC Loss
    obs_r_batch = tf.convert_to_tensor(np.repeat(obs_r[np.newaxis, :, :], n_samples, axis=0), dtype=tf.float32)
    obs_v_batch = tf.convert_to_tensor(np.repeat(obs_v[np.newaxis, :, :], n_samples, axis=0), dtype=tf.float32)
    
    mac_mat_r = calculate_MAC(obs_r_batch, tf.convert_to_tensor(rot_pred))
    mac_mat_v = calculate_MAC(obs_v_batch, tf.convert_to_tensor(vert_pred))
    
    best_r = tf.reduce_max(mac_mat_r, axis=2).numpy()
    best_v = tf.reduce_max(mac_mat_v, axis=2).numpy()
    
    loss_mac = np.mean((1.0 - best_r) + (1.0 - best_v), axis=1) # (n_samples,)
    
    # C. Total Loss & Likelihood
    inv_gamma_val = 1.0 / (beta**2)
    total_loss = loss_f + loss_mac 
    
    exponent = -total_loss * inv_gamma_val
    max_exp = np.max(exponent)
    likelihood = np.exp(exponent - max_exp)
    posterior_weights = likelihood / np.sum(likelihood) 
           
    
    
    ### PLOTTING prueba
    print("Generating KDE Plots...")
    z_true = Alphas_true[pos,:]
    std_z = np.std(z_samples, axis=0)
    # Thresholding to ignore low-probability noise for KDE stability
    # We use a more robust mask or just use all samples if weights are well-distributed
    threshold = np.max(posterior_weights) * 1e-6
    mask = posterior_weights > threshold
    
    # If the mask is too restrictive, relax it
    if np.sum(mask) < 10:
        mask = np.ones(len(posterior_weights), dtype=bool)

    x_filtered = z_samples[mask]
    w_filtered = posterior_weights[mask]
    w_filtered /= np.sum(w_filtered) # Re-normalize

    fig, axes = plt.subplots(n_elements, n_elements, figsize=(14, 14), facecolor='white')
    labels = [f'$\\alpha_{{{k+1}}}$' for k in range(n_elements)]
    
    for r in range(n_elements):
        for c in range(n_elements):
            ax = axes[r, c]
            
            if r == c: # Diagonal: 1D Marginal Distribution
                try:
                    vals = x_filtered[:, r]
                    # Generate 1D KDE
                    kde1d = gaussian_kde(vals, weights=w_filtered)
                    x_grid = np.linspace(0, 1, 100)
                    y_grid = kde1d(x_grid)
                    
                    ax.fill_between(x_grid, y_grid, color='steelblue', alpha=0.4)
                    ax.plot(x_grid, y_grid, color='steelblue', lw=2)
                    
                    # # Ground truth line
                    # ax.axvline(z_true[r], color='red', linestyle='--', lw=2, label='True')
                    
                    # Formatting
                    # ax.set_title(f"{labels[r]}\nTrue: {z_true[r]:.2f}", fontsize=10)
                    ax.set_xlim(0, 1)
                    ax.set_yticks([]) # Remove density scale for cleaner look
                except Exception as e:
                    ax.text(0.5, 0.5, "KDE Fail", ha='center')
                
            elif r > c: # Lower Triangle: 2D Joint Distribution
                x_vals = x_filtered[:, c]
                y_vals = x_filtered[:, r]
                
                try:
                    # Stacking for KDE
                    values = np.vstack([x_vals, y_vals])
                    kernel = gaussian_kde(values, weights=w_filtered)
                    
                    # Create grid for evaluation
                    # Use 40-50 for performance, higher for smoothness
                    xi, yi = np.mgrid[0:1:50j, 0:1:50j]
                    coords = np.vstack([xi.flatten(), yi.flatten()])
                    zi = kernel(coords).reshape(xi.shape)
                    
                    # Plot filled contours
                    cf = ax.contourf(xi, yi, zi, levels=15, cmap='viridis')
                    
                    # Add Ground Truth point
                    ax.plot(z_true[c], z_true[r], color='red', marker='*', 
                            markersize=12, markeredgecolor='white', label='Truth')
                    
                except Exception as e:
                    # Fallback to scatter if KDE fails (e.g. singular matrix)
                    ax.scatter(x_vals, y_vals, c=w_filtered, s=2, cmap='viridis', alpha=0.3)
                    ax.plot(z_true[c], z_true[r], 'r*', markersize=10)

                # Axis Labels
                if c == 0: ax.set_ylabel(labels[r], fontsize=12)
                if r == n_elements - 1: ax.set_xlabel(labels[c], fontsize=12)
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.grid(True, linestyle=':', alpha=0.3)
                
            else: # Upper Triangle: Empty or hidden
                ax.axis('off')

    plt.suptitle(f"Sample {pos}: Bayesian Posterior Uncertainty ($\\beta={beta}$)\nGround Truth marked in Red", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save Logic
    save_dir = os.path.join(folder_path, "Predicted_Posteriors")
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f'Sample_{pos}_Posterior.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


    # # 4.2 Main Predicted Posterior Plot (Pairwise Matrices)
    # fig, axes = plt.subplots(n_elements, n_elements, figsize=(10, 10), facecolor='white')
    # labels = [f'$z_{{{k+1}}}$' for k in range(n_elements)]

    # for r in range(n_elements):
    #     for c in range(n_elements):
    #         ax = axes[r, c]
    #         if r == c: # Diagonal: Physical Labels
    #             ax.text(0.5, 0.5, labels[r], fontsize=22, ha='center', va='center', fontweight='bold', color='#333333')
    #             ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    #             ax.axis('off')
    #         elif r > c: # Lower Triangle: Smooth 2D Joint PDF
    #             xi, yi = np.mgrid[0:1:100j, 0:1:100j]
                
    #             # Perform KDE on the predicted samples
    #             kde_coords = np.vstack([z_samples[:, c], z_samples[:, r]])
    #             kde = gaussian_kde(kde_coords)
    #             zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                
    #             # Plot density
    #             ax.contourf(xi, yi, zi, levels=30, cmap='viridis', alpha=0.9)
    #             ax.contour(xi, yi, zi, levels=5, colors='white', linewidths=0.3, alpha=0.2)
                
    #             # Mark Ground Truth (Target)
    #             if z_true is not None:
    #                 ax.plot(z_true[c], z_true[r], 'ro', markersize=7, markeredgecolor='white', markeredgewidth=1, zorder=10)
                
    #             if c == 0: ax.set_ylabel(labels[r], fontsize=12)
    #             if r == n_elements - 1: ax.set_xlabel(labels[c], fontsize=12)
    #             ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    #             ax.tick_params(labelsize=8)
    #             ax.grid(True, linestyle=':', alpha=0.3)
    #         else:
    #             ax.axis('off')

    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # save_dir = os.path.join(folder_path, "Prediction_plots")
    # if not os.path.exists(save_dir): os.makedirs(save_dir)
    # plt.savefig(os.path.join(save_dir, f'P{pos}_Predicted_JointPosterior.png'), dpi=500, bbox_inches='tight')
    # plt.show()
    
    ## comment until you are happy with a 
    # # 4.3 Physical Predicted Damage Profile
    # fig, (ax_bar, ax_beam) = plt.subplots(2, 1, figsize=(10, 6.5), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    
    # elements = np.arange(1, n_elements + 1)
    # color_estimate = '#4575b4' # Sophisticated Blue
    # color_true = '#d73027'     # Strong Red
    
    # # Top Plot: Bar chart with Predicted Uncertainty
    # ax_bar.bar(elements, expected_z, yerr=std_z, color=color_estimate, alpha=0.65, 
    #            label='Predicted Mean Stiffness Reduction ($E[z|\mathbf{m}]$)', 
    #            capsize=8, error_kw={'elinewidth':2, 'capthick':2, 'ecolor': '#1a1a1a'})
    
    # if z_true is not None:
    #     x_step = np.arange(0.5, n_elements + 1.5, 1)
    #     y_step = np.concatenate([z_true, [z_true[-1]]])
    #     ax_bar.step(x_step, y_step, where='post', color=color_true, 
    #                 label='True Damage State ($\mathbf{z}^*$)', linestyle='--', lw=2.5, zorder=5)
    
    # ax_bar.set_ylabel('Stiffness Reduction ($z$)', fontsize=13, fontweight='medium')
    # ax_bar.set_ylim([0, 1.2])
    # ax_bar.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False, fontsize=11)
    # ax_bar.grid(axis='y', alpha=0.2, linestyle='-')
    
    # # Explain Predicted Uncertainty
    # props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='silver')
    # ax_bar.text(0.02, 0.95, "Note: Error bars denote predicted 1$\sigma$ \n(Uncertainty from Inverse Model)", 
    #             transform=ax_bar.transAxes, fontsize=9, verticalalignment='top', bbox=props)
    
    # # Physical Beam Heatmap
    # beam_viz = expected_z.reshape(1, -1)
    # im = ax_beam.imshow(beam_viz, cmap='YlGnBu', aspect='auto', extent=[0.5, n_elements + 0.5, 0, 1], vmin=0, vmax=1)
    
    # ax_beam.set_yticks([])
    # ax_beam.set_xticks(elements)
    # ax_beam.set_xticklabels([f'Element {k}' for k in elements], fontsize=11)
    # ax_beam.tick_params(axis='x', length=0)
    
    # for j, val in enumerate(expected_z):
    #     text_color = 'white' if val > 0.6 else 'black'
    #     ax_beam.text(j + 1, 0.5, f'{val:.2f}', ha='center', va='center', 
    #                  color=text_color, fontweight='bold', fontsize=11)
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f'P{pos}_Predicted_PhysicalProfile.png'), dpi=500, bbox_inches='tight')
    # plt.show()

   
    
    

















def plot_Gaussianmarg_joint_pdf(locs, scales, LT_matrix, n_dims, n_samples, i, z_true, folder_path):
    """Plots the joint PDF of pairs of variables from a Gaussian copula.

    Args:
        locs: Locations (means) of the marginal Gaussian distributions.
        scales: Scales (standard deviations) of the marginal Gaussian distributions.
        LT_matrix: Lower triangular matrix for Cholesky decomposition.
        n_dims: Number of dimensions.
        n_samples: Number of samples to generate.
        j: An index or identifier for the plot (used in the filename).
        z_true: True values of the variables (for plotting).
        folder_path: Path to the folder where the plots will be saved.
    """
    copula_samples = gaussian_copula(LT_matrix, n_dims, n_samples)
    marginal_samples = gaussian_marginal_samples(locs, scales, copula_samples)
    ## can you select the highest probability samples?? ? 

    # Ensure marginal_samples is a NumPy array for easier indexing
    marginal_samples = marginal_samples.numpy()
    for j in range(n_dims):
        for k in range(j + 1, n_dims):  # Avoid redundant plots (j starts from i+1)
            x = marginal_samples[:, j]
            y = marginal_samples[:, k]
            z1_true = z_true[j]
            z2_true = z_true[k]

            # Perform kernel density estimation
            try:  # Handle potential singularity errors in KDE
                kde = gaussian_kde([x, y])
            except np.linalg.LinAlgError as e:
                print(f"Singular matrix error during KDE for pair ({j}, {k}): {e}")
                continue  # Skip this pair if KDE fails
            except ValueError as e:
                print(f"ValueError during KDE for pair ({j}, {k}): {e}") #For example if the variance is 0.
                continue

            # Create a grid to evaluate the density
            xgrid = np.linspace(0.05, 0.95, 100)
            ygrid = np.linspace(0.05, 0.95, 100)
            X, Y = np.meshgrid(xgrid, ygrid)
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(positions).reshape(X.shape)

            # Plot the joint PDF
            plt.figure(figsize=(8, 6))
            plt.contourf(X, Y, Z, levels=50, cmap='viridis')
            plt.xlabel(f'z{j+1}')  # Label with variable index
            plt.ylabel(f'z{k+1}')  # Label with variable index
            # plt.xlim(x.min(), x.max()) #Set plot limits based on the sampled data
            # plt.ylim(y.min(), y.max())
            
            plt.xlim(0.05, 0.95)
            plt.ylim(0.05, 0.95)

            
            plt.colorbar(label="Density")
            plt.scatter(z1_true, z2_true, c='red', marker='*', edgecolor='k', s=200, label='True solution')
            # plt.title(f'Joint PDF of z{j+1} and z{k+1}') #Add title for the specific pair
            plt.legend()
            plt.savefig(os.path.join(folder_path, f'PDF_Contourplot_testcase{i}_pair_{j}_{k}.png'), dpi=500, bbox_inches='tight')
            plt.show()
            plt.close()  # Close the figure to free memory





def multimodal_marginal(x, locs, scales, weight_vals):
    """
    Multimodal marginal distribution for one dimension.

    Args:
        x: A tensor of shape [n_points], the input values at which to evaluate the distributions.
        locs: A tensor of shape [n_gaussians], specifying the means of the Gaussian components for each dimension.
        scales: A tensor of shape [n_gaussians], specifying the standard deviations of the Gaussian components.
        weights: A tensor of shape [n_gaussians], specifying the mixture weights for each Gaussian component (should sum to 1 along the second axis).

    Returns:
        A tensor of shape [n_points, n_dims], where each entry is the value of the multimodal marginal distribution at the corresponding input.
    """
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=weight_vals),
        components_distribution=tfd.TruncatedNormal(loc=locs, scale=tf.maximum(scales, 1e-03), low =0.0, high=1.0)

    )
    mixture_probs = gm.prob(x)
    return mixture_probs


class Interpolation1D(tf.keras.layers.Layer):
    def __init__(self):
        super(Interpolation1D, self).__init__()

    def call(self, x, xp, fp):
        """
        Perform linear interpolation in 1D.

        Parameters:
        - x: Query points where we want interpolated values, shape [n_points].
        - xp: Points defining the domain of the interpolation, shape [n_values].
        - fp: Values of the function at the domain points, shape [n_values].

        Returns:
        - Interpolated values at `x`, shape [n_points].
        """
        xp = tf.sort(xp)
        tf.debugging.assert_rank(xp, 1, "xp must be a 1D tensor")
        tf.debugging.assert_rank(fp, 1, "fp must be a 1D tensor")
        tf.debugging.assert_equal(tf.shape(xp), tf.shape(fp), "xp and fp must have the same shape")

        x_min = tf.reduce_min(xp)
        x_max = tf.reduce_max(xp)
        x_range = x_max - x_min
        margin = 0.01 * x_range
        x = tf.clip_by_value(x, x_min - margin, x_max + margin)

        is_rank_1 = tf.equal(tf.rank(x), 1)

        def rank_1_case():
            return tf.expand_dims(x, axis=0)

        def rank_2_case():
            return x

        x_processed = tf.cond(is_rank_1, rank_1_case, rank_2_case)

        indices = tf.reduce_sum(tf.cast(x_processed[:, :, None] >= xp[None, None, :], tf.int32), axis=-1) - 1

        n_values = tf.shape(xp)[0]
        indices = tf.clip_by_value(indices, 0, n_values - 2)

        x0 = tf.gather(xp, indices)
        x1 = tf.gather(xp, indices + 1)
        f0 = tf.gather(fp, indices)
        f1 = tf.gather(fp, indices + 1)

        slope = tf.where(tf.abs(x1 - x0) > 1e-8, (f1 - f0) / (x1 - x0), 0.0)
        interpolated = f0 + slope * (x_processed - x0)

        left_extrapolate = x_processed < x_min
        right_extrapolate = x_processed > x_max

        interpolated = tf.where(left_extrapolate, tf.gather(fp, tf.zeros_like(indices, dtype=tf.int32)), interpolated)
        interpolated = tf.where(right_extrapolate, tf.gather(fp, tf.ones_like(indices, dtype=tf.int32) * (n_values-1)), interpolated)

        return tf.squeeze(interpolated)


def build_marginal_samples(locs, scales, weight_vals, copula_samples):
    """
    Builds marginal samples given copula samples and marginal parameters.

    Args:
        locs:  Shape [batch_size, n_gaussians, n_dims].
        scales: Shape [batch_size, n_gaussians, n_dims].
        weight_vals: Shape [batch_size, n_gaussians, n_dims].  
        copula_samples: Shape [batch_size, n_samples, n_dims].

    Returns:
        Marginal samples, shape [batch_size, n_samples, n_dims].

    Connection to Equations: This function performs the transformation from uniform
    copula samples (u_i) to the actual marginal samples (z_i) by using the
    inverse CDF (F_i^{-1}(u_i)).  The inverse CDF is approximated using
    the Interpolation1D layer.

    Numerical Stability: Relies on the stability of Interpolation1D and
    multimodal_marginal. The input weight_vals MUST be normalized.
    """
    interpolator = Interpolation1D()  
    n_dims = locs.shape[-1]
    n_unif_points = 200
    x = tf.linspace(0.00001, 0.9999, n_unif_points)   # Shape: [n_points]
    
    marg_samples = tf.TensorArray(dtype=tf.float32, size=n_dims)

    for i in range(n_dims):  # Use tf.range for better XLA support
        marginal_pdf_vals = multimodal_marginal(
            x, locs[:, i], scales[:, i], weight_vals[:, i]
        )
        marginal_cdf_vals = tf.cumsum(marginal_pdf_vals, axis=0) / tf.reduce_sum(marginal_pdf_vals)

        # Use custom Interpolation1D layer for inverse CDF
        marginal = interpolator(copula_samples[:, i], marginal_cdf_vals, x)
        marg_samples = marg_samples.write(i, marginal)  # Replace append with write()

    marginal_samples = marg_samples.stack()  # Convert TensorArray to Tensor
    marginal_samples = tf.transpose(marginal_samples, perm=[1, 0])  # Match expected output shape (num_samples,n_dims)

    return marginal_samples

   
def plot_multimodal_joint_pdf(locs, scales, weights, LT_matrix, n_dims, n_samples, i, z_true,  folder_path):
      copula_samples = gaussian_copula(LT_matrix, n_dims, n_samples)
      marginal_samples = build_marginal_samples(locs, scales, weights, copula_samples)
      # Ensure marginal_samples is a NumPy array for easier indexing
      marginal_samples = marginal_samples.numpy()
      for j in range(n_dims):
          for k in range(j + 1, n_dims):  # Avoid redundant plots (j starts from i+1)
              x = marginal_samples[:, j]
              y = marginal_samples[:, k]
              z1_true = z_true[j]
              z2_true = z_true[k]

              # Perform kernel density estimation
              try:  # Handle potential singularity errors in KDE
                  kde = gaussian_kde([x, y])
              except np.linalg.LinAlgError as e:
                  print(f"Singular matrix error during KDE for pair ({j}, {k}): {e}")
                  continue  # Skip this pair if KDE fails
              except ValueError as e:
                  print(f"ValueError during KDE for pair ({j}, {k}): {e}") #For example if the variance is 0.
                  continue

              # Create a grid to evaluate the density
              xgrid = np.linspace(0.05, 0.95, 100)
              ygrid = np.linspace(0.05, 0.95, 100)
              X, Y = np.meshgrid(xgrid, ygrid)
              positions = np.vstack([X.ravel(), Y.ravel()])
              Z = kde(positions).reshape(X.shape)

              # Plot the joint PDF
              plt.figure(figsize=(8, 6))
              plt.contourf(X, Y, Z, levels=50, cmap='viridis')
              plt.xlabel(f'z{j+1}')  # Label with variable index
              plt.ylabel(f'z{k+1}')  # Label with variable index
              # plt.xlim(x.min(), x.max()) #Set plot limits based on the sampled data
              # plt.ylim(y.min(), y.max())
              
              plt.xlim(0.05, 0.95)
              plt.ylim(0.05, 0.95)

              
              plt.colorbar(label="Density")
              plt.scatter(z1_true, z2_true, c='red', marker='*', edgecolor='k', s=200, label='True solution')
              # plt.title(f'Joint PDF of z{j+1} and z{k+1}') #Add title for the specific pair
              plt.legend()
              plt.savefig(os.path.join(folder_path, f'PDF_Contourplot_testcase{i}_pair_{j}_{k}.png'), dpi=500, bbox_inches='tight')
              plt.show()
              plt.close()  # Close the figure to free memory

      
      


      
# def build_marginal_samples(locs, scales, weights, copula_samples, n_dims):
#     """
#     Build marginal samples using TensorFlow with a custom Interpolation1D layer.

#     Parameters:
#     - locs: Locations of components in the Gaussian mixture, shape [n_modes, n_dims].
#     - scales: Scales of components in the Gaussian mixture, shape [n_modes, n_dims].
#     - weights: Weights of components in the Gaussian mixture, shape [n_modes, n_dims].
#     - copula_samples: Samples in uniform space, shape [n_samples, n_dims].

#     Returns:
#     - Marginal samples corresponding to the copula samples.
#     """
#     marg_samples = []
#     interpolator = Interpolation1D()

#     for i in range(n_dims):
#         n_unif_points = 100
#         x = tf.linspace(0.0, 1.0, n_unif_points) 
#         marginal_pdf_vals = multimodal_marginal(x, locs[:, i], scales[:, i], weights[:, i])
#         marginal_cdf_vals = tf.cumsum(marginal_pdf_vals, axis=0) / tf.reduce_sum(marginal_pdf_vals)

#         # Use custom Interpolation1D layer for inverse CDF
#         marginal = interpolator(copula_samples[:, i], marginal_cdf_vals, x)
#         marg_samples.append(marginal)

#     marginal_samples = tf.transpose(tf.convert_to_tensor(marg_samples, dtype=tf.float32), perm=[1, 0])

#     return marginal_samples












def compare_losses(model_history, folder_path):
    losses = model_history.item()
    names = [ 'loss', 'Joint_copula_dens_term', 'Conditional_likelihood_term','val_loss', 'val_Joint_copula_dens_term', 'val_Conditional_likelihood_term']

    plt.figure(figsize=(10, 6))
    # Plot each loss term with a different style
    plt.plot(losses[names[1]], label=r'$\mathcal{L}_{PDF}$', color='tomato', linestyle='-', linewidth=3.)
    plt.plot(losses[names[2]], label=r'$\mathcal{L}_{Data}$', color='blue', linestyle='-', linewidth=3.)
    plt.plot(losses[names[0]], label=r'$\mathcal{L}_{ELBO}$', color='black', linestyle='-', linewidth=3.)
    print(losses[names[0]][-1])
    plt.xlabel('Epochs', fontsize=22)
    plt.ylabel('Loss value', fontsize=22)
    plt.xscale('log')
    # plt.yscale('log')
    # plt.ylim(0,0.1)
    # Add a legend
    plt.legend(loc='upper right', fontsize=22)
    # Add a grid for better readability
    plt.grid(True, linestyle='-.', alpha=1.)
    plt.savefig(os.path.join(folder_path, 'loss_comparison.png'),dpi = 500, bbox_inches='tight')
    plt.show()


def compare_PDF_losses(model_history, folder_path):
    losses = model_history.item()
    names_Copula = [ 'loss', 'Joint_copula_dens_term', 'Conditional_likelihood_term','val_loss', 'val_Joint_copula_dens_term', 'val_Conditional_likelihood_term']
    names_GMM = [ 'loss', 'Mixture_dens_term', 'Conditional_likelihood_term','val_loss', 'val_Mixture_dens_term', 'val_Conditional_likelihood_term']   
    fullcov_history_path = os.path.join("Output", "P101_Test_Cond_16Dec_diag_Roll_2Props_0.08Beta_InverseForward_9Gaussians_1Samples_0.0001LR_10000epochs_8192batch", "loss_autoencoder.csv")


    import pandas as pd
    fullcov_losses = pd.read_csv(fullcov_history_path).to_numpy()
     
    plt.figure(figsize=(10, 6))
    # Plot each loss term with a different style
    plt.plot(fullcov_losses[:,1], label=r'$\mathcal{L}_{GMM}^{diag}$', color='tomato', linestyle='-', linewidth=3.)
    plt.plot(losses[names_Copula[1]], label=r'$\mathcal{L}_{GMM}^{full}$', color='maroon', linestyle='-', linewidth=3.)
    plt.xlabel('Epochs', fontsize=22)
    plt.ylabel('Loss value', fontsize=22)
    plt.xscale('log')

    plt.legend(loc='lower right', fontsize=22)
    # Add a grid for better readability
    plt.grid(True, linestyle='-.', alpha=1.)
    plt.savefig(os.path.join(folder_path, 'Comparing_PDF_losses.png'),dpi = 500, bbox_inches='tight')
    plt.show()









def plot_test_Dataloss_contourplots(input_dim_decoder, u_test, r_test, p_test, num_mixtures, num_gaussians, num_samples_per_mixture, selected_features, positions, det_solutions, folder_path ):
    foldername = 'Deterministic_Test_Cond_16Dec_diag_Roll_2Props_0.001Beta_InverseForward_1Gaussians_1Samples_0.0001LR_5000epochs_8192batch'
    folder_path = os.path.join('Output', foldername)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    Problem_info = np.load(os.path.join(folder_path, 'Problem_info.npy'), allow_pickle = True).item()
    input_dim_encoder, input_dim_decoder, num_mixtures, num_gaussians, num_samples_per_mixture, LR, selected_features, beta = Problem_info['input_dim_enc'], Problem_info['input_dim_dec'], Problem_info['n_mixtures'], Problem_info['n_gaussians'], Problem_info['n_samples'], Problem_info['LR'], Problem_info['selected_features'], Problem_info['beta']
    Data_info = np.load(os.path.join(folder_path, 'Data_info.npy'), allow_pickle = True).item()
    u_val, r_val, u_test, r_test, p_val, p_test = Data_info['u_val'], Data_info['r_val'], Data_info['u_test'], Data_info['r_test'], Data_info['p_val'], Data_info['p_test']
    ## LOAD AND BUILD THE TRAINED AUTOENCODER MODEL 
    output_dim = u_val.shape[1]
    forward_path = os.path.join("Output","23Apr_2Prop_Forward", "model_forward_2mix")
    model_forward = tf.saved_model.load(forward_path)  
    
    deterministic_solutions_file = "Deterministic_Test_Cond_16Dec_diag_Roll_2Props_0.001Beta_InverseForward_1Gaussians_1Samples_0.0001LR_5000epochs_8192batch"
    det_path = os.path.join("Output", deterministic_solutions_file, "Test_results_info.npy")
    deterministic_test_sols = np.load(os.path.join(det_path), allow_pickle = True).item()
    det_z_test = deterministic_test_sols['test_p_predicted']
    det_solutions = det_z_test[positions,:]


    for i in range(len(positions)):
        pos  = positions [i]
        N = 200000
        u_true   = u_test[pos,:].reshape(1,u_test.shape[1]) #Case that we want to exlpore as input
        utrue =  np.array(tf.repeat(u_true, N, axis =0)) #now we have it 1,000 times to associate it to 1,000 different ps
        r_true = r_test[pos,:].reshape(1,r_test.shape[1])
        rtrue =  np.array(tf.repeat(r_true,N,axis = 0))
                
        import random
        ptest = []
        for j in range(N):
            p1 = random.uniform(0,1)
            p2 = random.uniform(0,1)
            p = [p1,p2]
            ptest.append(p)
        
        ptest = np.array(ptest)
        
        ### BUILD AND LOAD THE TRAININD FORWARD WEIGHTS  AND MODEL
        num_samples_per_mixture = 1
        
        output_dim = u_test.shape[1]
        forward_path = os.path.join("Output", "23Apr_2Prop_Forward")
        model_forward = ForwardModel(input_dim_decoder, output_dim, num_mixtures)
        model_forward.build(input_shape = ())
        weights_path = os.path.join(forward_path, "model_forward_weights.h5" )
        model_forward.load_weights(weights_path)
        
        upred = tf.cast(model_forward.predict(tf.concat([ptest, rtrue],axis = 1)), tf.float32)
        utrue = tf.cast(utrue, tf.float32)
        
        def ColumnSliceLayer(inputs, selected_features):
            selected_features = tf.constant(selected_features
        , dtype=tf.int32)
            outputs = tf.gather(inputs, selected_features, axis=1)
            return outputs
        
        def Data_loss(y_true, y_pred, selected_features, num_samples_per_mixture):
            ### este es el antiguo loss antes de tener en cuenta las mixturas, solo mide el error de reconstrucción (la loss del paper original)
            y_true = ColumnSliceLayer(y_true,selected_features)
            y_pred = ColumnSliceLayer(y_pred, selected_features)
            Ytrue = tf.repeat(y_true[:,:,tf.newaxis], num_samples_per_mixture, axis = 2)
            Ytrue = tf.reshape(tf.transpose(Ytrue, perm = [0,2,1]), [-1,y_true.shape[1]])
            # Ydiff = tf.math.square(Ytrue - y_pred)
            Ydiff = tf.math.square((Ytrue - y_pred)/(y_pred+1e-10))
            Data_Loss = tf.math.reduce_mean(Ydiff, axis = 1)
            return Data_Loss
        
        
        # ## PLOT
        
        z1 = ptest[:,0]
        z2 = ptest[:,1]
        
        dloss = Data_loss(utrue,upred, selected_features, num_samples_per_mixture)
        # Filter data based on loss value
        # lb = 0.0000000001
        # ub = 0.006
        lb = 0.000001
        ub = 0.01
        filtered_indices = (dloss > lb) & (dloss < ub)
        
        z1_filtered = z1[filtered_indices]
        z2_filtered = z2[filtered_indices]
        Loss_data_filtered = dloss[filtered_indices]
        combined_array = np.column_stack((z1_filtered, z2_filtered, Loss_data_filtered))
        # np.savetxt(os.path.join("Output","Example3.csv"), combined_array, delimiter=",", header="Z1,Z2,Data_misfit", comments="", fmt="%g")        
        
        
        
        
        z1_case = p_test[pos,0]
        z2_case = p_test[pos,1]
    
        heading = 'Roll'

        # heading = 'ELBO'+str(num_gaussians)+'g'
        # Create a smooth contour plot
        plt.figure(figsize=(10, 8))
        plt.tricontourf(z1_filtered, z2_filtered, Loss_data_filtered, cmap='viridis_r', levels = 100)
        plt.colorbar(label=r'$\mathcal{L}_{\mathcal{M}}$ value')
        plt.scatter(p_test[pos,0], p_test[pos,1], c='red', marker='*', edgecolor = 'k', s=150, label = 'True solution')
        plt.scatter(det_solutions[i,0], det_solutions[i,1], c='red', edgecolor = 'k', s=50, marker='s', label='Deterministic solution')
        # plt.legend()
        plt.xlabel('z1')
        plt.ylabel('z2')
        # Set x and y limits to [0,1]
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # plt.legend()
        legend = plt.legend(frameon=True, loc='upper right')

        # plt.title(f'Scatter plot of z1 vs z2 ({lb} < Loss Data < {ub})')
        # plt.savefig(os.path.join(folder_path, heading+'DataLoss_ContourPlot_withUB_'+str(ub)+'case_'+ str(i)+'.png'), dpi=300, bbox_inches='tight')
        plt.show()
        # plt.close()
