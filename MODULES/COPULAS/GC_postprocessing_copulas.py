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
def plot_KDE_pdf(locs, scales, weights, LT_matrix, n_dims, n_samples, pos, chosen_axis,  folder_path):
    copula_samples, mvn_samples, mvn_model = gaussian_copula(LT_matrix, n_dims, n_samples) 
    marginal_samples = build_marginal_samples(locs, scales, weights, copula_samples)
    # choose the  axis (THIS MUST BE DONE IN A LOOP TO PLOT ALL THE AXIS COMBINATIONS) [(0,1),(0,2),(0,3) (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    
    x = marginal_samples[:, chosen_axis[0]].numpy()
    y = marginal_samples[:, chosen_axis[1]].numpy()      
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
    threshold_percentage = 0.55 # *** ADJUST THIS VALUE (e.g., 0.05, 0.1, 0.2) ***
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
            
    plt.savefig(os.path.join(folder_path, 'test_KDE_case_'+ str(pos)+'_Axis_'+str(chosen_axis)+'.png'), dpi=300, bbox_inches='tight')
    
    # plt.show()
    # plt.close()
      




















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
