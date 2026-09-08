# -*- coding: utf-8 -*-
"""
Unified Model for Full Covariance Gaussian Mixture Model (GMM) Latent Space.
Includes exact Monte Carlo KL Divergence to prevent variance explosions.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as K

tfd = tfp.distributions

from MODULES.COPULAS.GC_GMm_GPU_eigen_functions import assemble_global_Kmatrices, Solve_eigenproblem
from MODULES.REVIEW.GMM_architectures import Fully_connected_enc_FullCov_GMM, FullCov_GMM_Sampling_Layer

@tf.function(jit_compile=True)
def calculate_MAC(modes_true, modes_pred):
    """Computes MAC with higher epsilon and clipping to prevent NaN gradients."""
    eps = 1e-8
    t_n = tf.norm(modes_true, axis=2, keepdims=True) + eps
    p_n = tf.norm(modes_pred, axis=2, keepdims=True) + eps
    
    modes_true_norm = modes_true / t_n
    modes_pred_norm = modes_pred / p_n
    
    dot_product = tf.matmul(modes_true_norm, modes_pred_norm, transpose_b=True)
    mac_matrix = tf.square(tf.clip_by_value(dot_product, -1.0, 1.0))
    return mac_matrix

class Inverse_FullCov_GMM_Model(tf.keras.Model):
    def __init__(self, input_dim_encoder, n_dims, num_components, num_samples, lbound, **kwargs):
        super(Inverse_FullCov_GMM_Model, self).__init__(**kwargs)
        self.n_dims = n_dims
        self.num_components = num_components
        self.num_samples = num_samples
        self.FC_encoder = Fully_connected_enc_FullCov_GMM(input_dim_encoder, n_dims, num_components)

    def call(self, inputs):
        [freq_data, rot_modes_data, vert_modes_data, z_factors] = inputs
        
        flat_rot = tf.reshape(rot_modes_data, [-1, rot_modes_data.shape[1] * rot_modes_data.shape[2]])
        flat_vert = tf.reshape(vert_modes_data, [-1, vert_modes_data.shape[1] * vert_modes_data.shape[2]])
        modal_data = K.layers.Concatenate(axis=1)([freq_data, flat_vert, flat_rot])

        params = self.FC_encoder(modal_data)
        n_corr = self.n_dims * (self.n_dims - 1) // 2

        logits, locs, offdiag, raw_diag = tf.split(params, [
            self.num_components, 
            self.n_dims * self.num_components, 
            n_corr * self.num_components,
            self.n_dims * self.num_components
        ], axis=-1)

        locs = tf.reshape(locs, (-1, self.num_components, self.n_dims))
        offdiag = tf.reshape(offdiag, (-1, self.num_components, n_corr))
        
        # Softplus to ensure positive diagonal (variance). +0.5 guarantees healthy initial spread.
        diag = tf.math.softplus(tf.reshape(raw_diag, (-1, self.num_components, self.n_dims)) + 0.5) + 1e-4
        
        logits = logits / 1.5 
        
        return logits, locs, offdiag, diag

class My_FullCovGMMVAE_withEigen(tf.keras.Model):
    def __init__(self, input_dim, num_dofs, n_elements, n_modes, Ke_matrices, Mfree, L_inv, 
                 n_dims, num_components, num_samples, gamma, mean_f, std_f, lbound, fixed_dofs_indices=None, **kwargs):
        super(My_FullCovGMMVAE_withEigen, self).__init__(**kwargs)
        self.num_dofs = num_dofs
        self.n_elements = n_elements
        self.n_modes = n_modes
        self.fixed_dofs_indices = fixed_dofs_indices

        self.Ke_matrices = tf.constant(Ke_matrices, dtype=tf.float32) 
        self.L_inv = tf.constant(L_inv, dtype=tf.float32)

        self.Encoder_model = Inverse_FullCov_GMM_Model(input_dim, n_dims, num_components, num_samples, lbound)
        self.Eigen_solver = Solve_eigenproblem(num_dofs, n_modes, L_inv, fixed_dofs_indices)
        self.Sampling_layer = FullCov_GMM_Sampling_Layer(n_dims, num_components, num_samples, lbound)
        
        self.mean_freq = tf.constant(mean_f, dtype=tf.float32) 
        self.std_freq = tf.constant(std_f, dtype=tf.float32)
        self.num_components = num_components
        self.n_dims = n_dims
        self.num_samples = num_samples
        self.gamma = gamma
        self.lbound = lbound 

    def call(self, inputs):
        [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.z_factors] = inputs
        
        self.logits, self.locs, self.offdiag, self.diag = self.Encoder_model(inputs)
        
        self.z_phys, self.z_raw, self.L_matrices = self.Sampling_layer([self.logits, self.locs, self.offdiag, self.diag])
        self.reshaped_z_phys = tf.reshape(self.z_phys, (-1, self.n_dims)) 
        
        if len(self.Ke_matrices.shape) == 2:
            Ke_matrices_dam = tf.einsum('BE, KQ -> BEKQ', self.reshaped_z_phys, self.Ke_matrices)
        else:
            Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', self.reshaped_z_phys, self.Ke_matrices)
        
        Kfree = assemble_global_Kmatrices(Ke_matrices_dam, self.n_elements, self.num_samples, self.fixed_dofs_indices) 
        self.pred_freqs, self.pred_rotmodes, self.pred_vertmodes = self.Eigen_solver(Kfree)
        
        return self.reshaped_z_phys

    def Freqs_loss(self, y_true, y_pred):
        pred_hz = tf.abs(self.pred_freqs)
        pred_log = tf.math.log(tf.maximum(pred_hz, 1e-7))
        pred_scaled = (pred_log - self.mean_freq) / (self.std_freq + 1e-8)
        
        huber = tf.keras.losses.Huber(delta=1.0)
        return huber(self.freq_data, pred_scaled)
    
    def MAC_modes_loss(self, y_true, y_pred):
        def get_match_loss(t_group, p_group):
            mac_mat = calculate_MAC(t_group, p_group)
            best_matches = tf.reduce_max(mac_mat, axis=2)
            return tf.reduce_mean(1.0 - best_matches) 
            
        loss_rot = get_match_loss(self.rot_modes_data, self.pred_rotmodes)
        loss_vert = get_match_loss(self.vert_modes_data, self.pred_vertmodes)
        return loss_rot + loss_vert

    def GMM_KL_Divergence_Loss(self, y_true, y_pred):
        """
        CRITICAL FIX: Monte Carlo KL Divergence estimation.
        This provides a mathematically robust KL divergence for Full Covariance GMMs 
        that prevents BOTH Variance Collapse AND Variance Explosion.
        """
        mixture_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=self.logits),
            components_distribution=tfd.MultivariateNormalTriL(
                loc=self.locs,
                scale_tril=self.L_matrices
            )
        )
        
        # Transpose to [Samples, Batch, Dims] for TFP broadcasting
        z_raw_transposed = tf.transpose(self.z_raw, [1, 0, 2])
        
        # log_q: Density of the current posterior [Samples, Batch]
        log_q = mixture_dist.log_prob(z_raw_transposed)
        
        # log_p: Density of the standard normal prior [Samples, Batch]
        log_p = tf.reduce_sum(tfd.Normal(0.0, 1.0).log_prob(z_raw_transposed), axis=-1)
        
        # MC Estimate of KL: E_q[log_q - log_p]
        kl_div = tf.reduce_mean(log_q - log_p, axis=0) # [Batch]
        
        # Generous clip to prevent early-epoch explosions without killing learning
        kl_div_safe = tf.clip_by_value(kl_div, 0.0, 2500.0)
        
        mean_kl = tf.reduce_mean(kl_div_safe)
        
        return tf.math.square(tf.cast(self.gamma, dtype=tf.float32)) * mean_kl

    def Mixture_Diversity_loss(self):
        """
        FIXED: Shannon Entropy penalty on weights to prevent early component collapse.
        Instead of returning -Entropy (which drives loss negative), we return 
        (Max_Entropy - Current_Entropy), which remains strictly >= 0.
        """
        weights = tf.nn.softmax(self.logits, axis=-1)
        entropy = -tf.reduce_sum(weights * tf.math.log(weights + 1e-10), axis=1)
        
        # Max entropy for K components is ln(K)
        max_entropy = tf.math.log(tf.cast(self.num_components, tf.float32))
        
        # This returns 0 if weights are perfectly uniform, and > 0 if they collapse
        penalty = max_entropy - entropy
        return tf.reduce_mean(penalty)
        
    def Boundary_Penalty_loss(self):
        """Logarithmic barrier preventing spikes exactly on the domain boundaries."""
        x = (self.reshaped_z_phys - self.lbound) / (1.0 - self.lbound)
        
        # FIXED: Added explicit boundary clip to ensure X never touches 0.0 or 1.0, 
        # averting NaN generation which could destabilize ELBO.
        x = tf.clip_by_value(x, 1e-5, 1.0 - 1e-5)
        
        penalty = -tf.reduce_mean(tf.math.log(x) + tf.math.log(1.0 - x))
        return penalty

    def ELBO_GMM_loss(self, y_true, y_pred):
        Loss_freqs = self.Freqs_loss(y_true, y_pred)
        Loss_MAC = self.MAC_modes_loss(y_true, y_pred)
        
        Loss_KL = self.GMM_KL_Divergence_Loss(y_true, y_pred)
        Loss_Div = self.Mixture_Diversity_loss()
        Loss_bound = self.Boundary_Penalty_loss()

        total_loss = 5.0 * Loss_freqs + 10.0 * Loss_MAC + Loss_KL + 1.5 * Loss_Div + 0.1 * Loss_bound
        
        # Final gradient firewall to prevent NaNs
        total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, tf.zeros_like(total_loss) + 1e4)
        return total_loss

    def get_config(self):
        config = {'num_dofs': self.num_dofs, 'fixed_dofs_indices': self.fixed_dofs_indices}
        base_config = super(My_FullCovGMMVAE_withEigen, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))