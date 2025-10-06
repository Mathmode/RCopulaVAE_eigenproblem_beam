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

from matplotlib import pyplot as plt
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
    plt.figure()
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(model.history.history['loss'], color = 'black', linewidth = 1.5, linestyle ='--', label  = 'Train')
    plt.plot(model.history.history['val_loss'], color = 'darkgray', linewidth = 1.5, label = 'Validation')
    plt.legend(loc='lower left', fontsize=14)
    plt.grid(True, linestyle='-.', alpha=1.)
    plt.savefig(os.path.join(folder_path, 'TrainVal_lossevolution.png'),dpi = 500, bbox_inches='tight')
    plt.show()
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
    plt.show()
    plt.close()

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
    marginal_samples = tfd.TruncatedNormal(loc=locs, scale=scales, low = -0.00005, high = 1.00005).quantile(copula_samples)
    # marginal_samples: (num_samples, n_dims)
    return marginal_samples




def plot_Gaussianmarg_joint_pdf(locs, scales, correlation_matrix, n_dims, n_samples, i, z_true, folder_path):
      copula_samples, mvn_samples, mvn_model = gaussian_copula(correlation_matrix, n_dims, n_samples) 
      marginal_samples = gaussian_marginal_samples(locs, scales, copula_samples)
      # For KDE, TensorFlow does not have a direct equivalent; consider using scipy for now
      x = marginal_samples[:, 0].numpy()
      y = marginal_samples[:, 1].numpy()
      z1_true = z_true[0]
      z2_true = z_true[1]
      # Perform kernel density estimation
      kde = gaussian_kde([x, y])

      # Create a grid to evaluate the density
      xgrid = np.linspace(0,1, 100)
      ygrid = np.linspace(0,1, 100)
      X, Y = np.meshgrid(xgrid, ygrid)
      positions = np.vstack([X.ravel(), Y.ravel()])
      Z = kde(positions).reshape(X.shape)
    
      # Plot the joint PDF
      plt.figure(figsize=(8, 6))
      plt.contourf(X, Y, Z, levels=50, cmap='viridis')
      plt.xlabel('z1')
      plt.ylabel('z2')
      plt.xlim(0, 1)
      plt.ylim(0, 1)
      plt.scatter(z1_true, z2_true, c='red', marker='*', edgecolor = 'k', s=200, label = 'True solution')
      plt.colorbar(label="Density")
      plt.savefig(os.path.join(folder_path, 'PDF_Contourplot_testcase'+str(i)+'.png'),dpi = 500, bbox_inches='tight')
      plt.show()
      
      
      
def plot_joint_pdf(locs, scales, weights, correlation_matrix, n_dims, n_samples, i, folder_path):
      copula_samples, mvn_samples, mvn_model = gaussian_copula(correlation_matrix, n_dims, n_samples) 
      marginal_samples = build_marginal_samples(locs, scales, weights, copula_samples, n_dims)
      # For KDE, TensorFlow does not have a direct equivalent; consider using scipy for now
      x = marginal_samples[:, 0].numpy()
      y = marginal_samples[:, 1].numpy()
      # Perform kernel density estimation
      kde = gaussian_kde([x, y])

      # Create a grid to evaluate the density
      xgrid = np.linspace(0,1, 100)
      ygrid = np.linspace(0,1, 100)
      X, Y = np.meshgrid(xgrid, ygrid)
      positions = np.vstack([X.ravel(), Y.ravel()])
      Z = kde(positions).reshape(X.shape)
    
      # Plot the joint PDF
      plt.figure(figsize=(8, 6))
      plt.contourf(X, Y, Z, levels=50, cmap='viridis')
      plt.xlabel('z1')
      plt.ylabel('z2')
      plt.xlim(0, 1)
      plt.ylim(0, 1)
      plt.colorbar(label="Density")
      plt.savefig(os.path.join(folder_path, 'PDF_Contourplot_testcase'+str(i)+'.png'),dpi = 500, bbox_inches='tight')
      plt.show()
      
def plot_loss_components(model_history, folder_path):
    losses = model_history.item()
    names = [ 'loss', 'Freqs_loss', 'Joint_copula_dens_term', 'Conditional_likelihood_term','val_loss', 'val_Joint_copula_dens_term', 'val_Conditional_likelihood_term']


def compare_Copula_losses(model_history, folder_path):
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
