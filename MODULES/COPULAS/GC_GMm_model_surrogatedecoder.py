# -*- coding: utf-8 -*-
"""
Updated: March 2026
Surrogate-based CopulaVAE Model.
Adapts the improved numerical stability and loss logic from the Eigen-solver version.
"""

import tensorflow as tf
import tensorflow.keras as K
import tensorflow_probability as tfp
from MODULES.COPULAS.GC_GMm_models import Inverse_Copula_Model, calculate_MAC
from MODULES.COPULAS.GC_GMm_architectures import Copula_pdf_layer

tfd = tfp.distributions

class FullyConnectedSurrogateDecoder(K.Model):
    """
    Standard Neural Network Surrogate for the Eigenvalue Solver.
    Maps Alpha factors (stiffness) -> Frequencies and Mode Shapes.
    """
    def __init__(self, n_dims, n_modes, n_nodes_per_mode):
        super(FullyConnectedSurrogateDecoder, self).__init__()
        self.n_modes = n_modes
        self.n_nodes = n_nodes_per_mode
        
        self.output_dim = n_modes + 2 * (n_modes * n_nodes_per_mode)
        
        self.net = K.Sequential([
            K.layers.Dense(512, activation='relu', kernel_initializer='he_uniform'),
            K.layers.BatchNormalization(),
            K.layers.Dense(1024, activation='relu', kernel_initializer='he_uniform'),
            K.layers.Dense(1024, activation='relu', kernel_initializer='he_uniform'),
            K.layers.Dense(1024, activation='relu', kernel_initializer='he_uniform'),
            K.layers.Dense(self.output_dim, activation='linear')
        ])

    def call(self, alpha_samples):
        flat_output = self.net(alpha_samples)
        
        # 1. Frequencies: must be positive.
        freqs_raw = flat_output[:, :self.n_modes]
        freqs = tf.nn.softplus(freqs_raw) + 1e-6
        
        # 2. Vertical Mode Shapes
        v_start = self.n_modes
        v_end = v_start + (self.n_modes * self.n_nodes)
        vert_modes = tf.reshape(flat_output[:, v_start:v_end], [-1, self.n_modes, self.n_nodes])
        
        # 3. Rotational Mode Shapes
        r_start = v_end
        rot_modes = tf.reshape(flat_output[:, r_start:], [-1, self.n_modes, self.n_nodes])
        
        return freqs, rot_modes, vert_modes

class My_CopulaVAE_Surrogate(K.Model):
    """
    Surrogate version of the CopulaVAE with improved numerical stability.
    """
    def __init__(self, input_dim, n_modes, n_nodes_per_mode, n_dims, num_gaussians, 
                 num_samples, beta, mean_f, std_f, lbound, **kwargs):
        super(My_CopulaVAE_Surrogate, self).__init__()
        
        self.n_modes = n_modes
        self.n_nodes = n_nodes_per_mode
        self.n_dims = n_dims
        self.num_gaussians = num_gaussians
        self.num_samples = num_samples
        self.beta = beta
        self.lbound = lbound
        
        self.mean_freq = tf.constant(mean_f, dtype=tf.float32) 
        self.std_freq = tf.constant(std_f, dtype=tf.float32)

        # 1. Encoder
        self.Encoder_model = Inverse_Copula_Model(input_dim, n_dims, num_gaussians, num_samples, lbound)
        
        # 2. Sampling Layer
        self.Copula_sampling_layer = Copula_pdf_layer(n_dims, num_gaussians, num_samples, lbound)
        
        # 3. Surrogate Decoder
        self.Surrogate_decoder = FullyConnectedSurrogateDecoder(n_dims, n_modes, n_nodes_per_mode)

    def call(self, inputs):
        [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.alpha_factors] = inputs
        
        # Encode
        self.means, self.scales, self.weight_vals, self.offdiag_elems, self.diag_elems = self.Encoder_model(inputs)
        
        # Sampling
        inputs_to_sampling = [self.means, self.scales, self.weight_vals, self.offdiag_elems, self.diag_elems]
        self.marginal_samples_z, self.copula_samples_u, self.LT_matrices = self.Copula_sampling_layer(inputs_to_sampling)
        
        self.reshaped_alpha_samples = tf.reshape(self.marginal_samples_z, (-1, self.n_dims)) 
        self.reshaped_copula_samples = tf.reshape(self.copula_samples_u, (-1, self.n_dims))
        
        # Prepare components for logprob calculation
        lt_ext = tf.repeat(self.LT_matrices[:, tf.newaxis, :, :], self.num_samples, axis=1)
        self.reshaped_LT_matrices = tf.reshape(lt_ext, [-1, self.n_dims, self.n_dims])

        # Decode using Surrogate
        self.pred_freqs, self.pred_rotmodes, self.pred_vertmodes = self.Surrogate_decoder(self.reshaped_alpha_samples)
        
        return self.reshaped_alpha_samples
    
    # --- STABILIZED LOSS FUNCTIONS ---
    
    def Freqs_loss(self, y_true, y_pred):
        true_scaled = self.freq_data
        
        # Scale predicted frequencies for comparison
        pred_hz = tf.abs(self.pred_freqs)
        pred_log = tf.math.log(tf.maximum(pred_hz, 1e-7))
        pred_scaled = (pred_log - self.mean_freq) / (self.std_freq + 1e-8)
        
        # Use Huber Loss for robustness
        huber = tf.keras.losses.Huber(delta=1.0)
        return huber(true_scaled, pred_scaled)
    
    def MAC_modes_loss(self, y_true, y_pred):
        true_rot = self.rot_modes_data
        true_vert = self.vert_modes_data
        pred_rot = self.pred_rotmodes
        pred_vert = self.pred_vertmodes

        def get_match_loss(t_group, p_group):
            mac_mat = calculate_MAC(t_group, p_group)
            best_matches = tf.reduce_max(mac_mat, axis=2)
            return tf.reduce_mean(1.0 - best_matches) 

        return get_match_loss(true_rot, pred_rot) + get_match_loss(true_vert, pred_vert)
    
    def Copula_pdf_logprob(self, y_true, y_pred):
        # Clip copula samples to avoid infinity in quantile function
        u_clipped = tf.clip_by_value(self.reshaped_copula_samples, 1e-5, 1.0 - 1e-5)
        normal_dist = tfd.Normal(loc=0.0, scale=1.0)
        normal_samples = normal_dist.quantile(u_clipped)
               
        mvn = tfd.MultivariateNormalTriL(loc=tf.zeros(self.n_dims), 
                                         scale_tril=self.reshaped_LT_matrices)
        
        log_prob_joint_normal = mvn.log_prob(normal_samples)
        log_prob_marginals_std = tf.reduce_sum(normal_dist.log_prob(normal_samples), axis=-1)      
        
        return log_prob_joint_normal - log_prob_marginals_std
   
    def Marginal_pdf_logprob(self, y_true, y_pred):
        # High stability TruncatedNormal logprob using means/scales from encoder
        gaussian_marginals = tfd.TruncatedNormal(
            loc=tf.reshape(self.means, [-1, self.n_dims]), 
            scale=tf.reshape(self.scales, [-1, self.n_dims]) + 1e-5, 
            low=self.lbound - 1e-4,  
            high=1.0 + 1e-4)

        log_prob_marginals = gaussian_marginals.log_prob(self.reshaped_alpha_samples)  
        return tf.math.reduce_sum(log_prob_marginals, axis=-1)    
    
    def Joint_copula_dens_term(self, y_true, y_pred):
        c_term = self.Copula_pdf_logprob(y_true, y_pred)
        m_term = self.Marginal_pdf_logprob(y_true, y_pred)  
        
        joint_logprob = tf.math.reduce_mean(c_term + m_term)
        return tf.math.square(tf.cast(self.beta, dtype=tf.float32)) * joint_logprob
    
    def ELBO_Copula_loss(self, y_true, y_pred):
        loss_f = self.Freqs_loss(y_true, y_pred)
        loss_m = self.MAC_modes_loss(y_true, y_pred)
        loss_j = self.Joint_copula_dens_term(y_true, y_pred)
                
        # Balancing: MAC loss is given higher weight as per original improvements
        return loss_f + 15.0 * loss_m + loss_j

    def get_config(self):
        config = {
            'n_modes': self.n_modes,
            'n_nodes_per_mode': self.n_nodes,
            'n_dims': self.n_dims,
            'num_gaussians': self.num_gaussians,
            'num_samples': self.num_samples,
            'beta': self.beta,
            'mean_f': self.mean_freq.numpy(),
            'std_f': self.std_freq.numpy(),
            'lbound': self.lbound
        }
        base_config = super(My_CopulaVAE_Surrogate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
    
    
    
    
    
    
    
# import tensorflow as tf
# import tensorflow.keras as K
# import tensorflow_probability as tfp
# from MODULES.COPULAS.GC_GMm_models import Inverse_Copula_Model, calculate_MAC
# from MODULES.COPULAS.GC_GMm_architectures import Copula_pdf_layer

# tfd = tfp.distributions

# class FullyConnectedSurrogateDecoder(K.Model):
#     """
#     Standard Neural Network Surrogate for the Eigenvalue Solver.
#     Maps Alpha factors (stiffness) -> Frequencies and Mode Shapes.
#     """
#     def __init__(self, n_dims, n_modes, n_nodes_per_mode):
#         super(FullyConnectedSurrogateDecoder, self).__init__()
#         self.n_modes = n_modes
#         self.n_nodes = n_nodes_per_mode
        
#         # Output dim: Freqs (n_modes) + RotModes (n_modes * nodes) + VertModes (n_modes * nodes)
#         self.output_dim = n_modes + 2 * (n_modes * n_nodes_per_mode)
        
#         self.net = K.Sequential([
#             K.layers.Dense(512, activation='relu', kernel_initializer='he_uniform'),
#             K.layers.BatchNormalization(),
#             K.layers.Dense(1024, activation='relu', kernel_initializer='he_uniform'),
#             K.layers.Dense(1024, activation='relu', kernel_initializer='he_uniform'),
#             K.layers.Dense(1024, activation='relu', kernel_initializer='he_uniform'),
#             # The output layer should remain linear, but we will apply structure in call()
#             K.layers.Dense(self.output_dim, activation='linear')
#         ])

#     def call(self, alpha_samples):
#         # alpha_samples: (Batch * num_samples, n_dims)
#         flat_output = self.net(alpha_samples)
        
#         # Split and constrain the outputs
#         # 1. Frequencies: must be positive. Softplus ensures positivity.
#         freqs_raw = flat_output[:, :self.n_modes]
#         freqs = tf.nn.softplus(freqs_raw) + 1e-6
        
#         # 2. Vertical Mode Shapes
#         v_start = self.n_modes
#         v_end = v_start + (self.n_modes * self.n_nodes)
#         vert_modes = tf.reshape(flat_output[:, v_start:v_end], [-1, self.n_modes, self.n_nodes])
        
#         # 3. Rotational Mode Shapes
#         r_start = v_end
#         rot_modes = tf.reshape(flat_output[:, r_start:], [-1, self.n_modes, self.n_nodes])
        
#         return freqs, rot_modes, vert_modes

# class My_CopulaVAE_Surrogate(K.Model):
#     """
#     Surrogate version of the CopulaVAE.
#     The Eigen_solver is replaced by FullyConnectedSurrogateDecoder.
#     """
#     def __init__(self, input_dim, n_modes, n_nodes_per_mode, n_dims, num_gaussians, 
#                  num_samples, beta, mean_f, std_f, lbound, **kwargs):
#         super(My_CopulaVAE_Surrogate, self).__init__()
        
#         self.n_modes = n_modes
#         self.n_nodes = n_nodes_per_mode
#         self.n_dims = n_dims
#         self.num_gaussians = num_gaussians
#         self.num_samples = num_samples
#         self.beta = beta
#         self.lbound = lbound
        
#         # Scaling factors for frequencies
#         self.mean_freq = tf.constant(mean_f, dtype=tf.float32) 
#         self.std_freq = tf.constant(std_f, dtype=tf.float32)

#         # 1. Encoder (Same as your original)
#         self.Encoder_model = Inverse_Copula_Model(input_dim, n_dims, num_gaussians, num_samples, lbound)
        
#         # 2. Sampling Layer (Same as your original)
#         self.Copula_sampling_layer = Copula_pdf_layer(n_dims, num_gaussians, num_samples, lbound)
        
#         # 3. SURROGATE DECODER (Replacing Eigen Solver)
#         self.Surrogate_decoder = FullyConnectedSurrogateDecoder(n_dims, n_modes, n_nodes_per_mode)

#     def call(self, inputs):
#         # Unpack data for internal loss calculations
#         [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.alpha_factors] = inputs
        
#         # 1. Encode
#         self.means, self.scales, self.weight_vals, self.offdiag_elems, self.diag_elems = self.Encoder_model(inputs)
        
#         # Prepare for sampling
#         inputs_to_sampling = [self.means, self.scales, self.weight_vals, self.offdiag_elems, self.diag_elems]
#         self.marginal_samples_z, self.copula_samples_u, self.LT_matrices = self.Copula_sampling_layer(inputs_to_sampling)
        
#         self.reshaped_alpha_samples = tf.reshape(self.marginal_samples_z, (-1, self.n_dims)) 
#         self.reshaped_copula_samples  = tf.reshape(self.copula_samples_u, (-1, self.n_dims))
        
#         # STABILITY GUARD: Extreme boundary clipping
#         self.reshaped_copula_samples = tf.clip_by_value(self.reshaped_copula_samples, 1e-5, 1.0 - 1e-5)
        
#         # STABILITY GUARD: Regularize LT_matrices
#         eye = tf.eye(self.n_dims, batch_shape=[tf.shape(self.LT_matrices)[0]])
#         self.LT_matrices = self.LT_matrices + 1e-5 * eye

#         # Extract variables
#         self.reshaped_means = tf.reshape(tf.repeat(self.means[:,tf.newaxis,:,:], self.num_samples, axis=1), [-1, self.num_gaussians, self.n_dims])
#         self.reshaped_scales = tf.reshape(tf.repeat(self.scales[:,tf.newaxis,:,:], self.num_samples, axis=1), [-1, self.num_gaussians, self.n_dims])
#         self.reshaped_scales = tf.maximum(self.reshaped_scales, 1e-4) # Ensure scales never collapse to zero
        
#         self.reshaped_LT_matrices = tf.reshape(tf.repeat(self.LT_matrices[:,tf.newaxis,:,:], self.num_samples, axis=1), [-1, self.n_dims, self.n_dims])

#         # 2. Decode using Surrogate
#         self.pred_freqs, self.pred_rotmodes, self.pred_vertmodes = self.Surrogate_decoder(self.reshaped_alpha_samples)
        
#         return self.reshaped_alpha_samples
    
#     # --- LOSS FUNCTIONS ---
    
#     def Freqs_loss(self, y_true, y_pred):
#         true_scaled = self.freq_data
#         pred_hz = self.pred_freqs
        
#         # Safe log and standardization
#         pred_log = tf.math.log(tf.maximum(pred_hz, 1e-6))
#         pred_scaled = (pred_log - self.mean_freq) / self.std_freq
        
#         loss = tf.math.reduce_mean(tf.square(true_scaled - pred_scaled))
#         return loss
    
#     def MAC_modes_loss(self, y_true, y_pred):
#         true_rot = self.rot_modes_data
#         true_vert = self.vert_modes_data
#         pred_rot = self.pred_rotmodes
#         pred_vert = self.pred_vertmodes

#         def get_best_match_loss(true_group, pred_group):
#             mac_mat = calculate_MAC(true_group, pred_group)
#             best_matches = tf.reduce_max(mac_mat, axis=2)
#             return tf.reduce_mean(1.0 - best_matches) 

#         loss_rot = get_best_match_loss(true_rot, pred_rot)
#         loss_vert = get_best_match_loss(true_vert, pred_vert)
        
#         return loss_rot + loss_vert
    
#     def Copula_pdf_logprob(self, y_true, ypred):
#         normal_dist = tfd.Normal(loc=0.0, scale=1.0)
#         # Convert uniform samples to standard normal
#         normal_samples = normal_dist.quantile(self.reshaped_copula_samples)
#         # Limit normal samples to prevent mvn divergence
#         normal_samples = tf.clip_by_value(normal_samples, -5.0, 5.0)
        
#         mvn = tfd.MultivariateNormalTriL(loc = tf.zeros(self.n_dims), scale_tril = self.reshaped_LT_matrices)
        
#         log_prob_joint_normal = mvn.log_prob(normal_samples)
#         # Clipping log prob to prevent exploding gradients
#         log_prob_joint_normal = tf.clip_by_value(log_prob_joint_normal, -100.0, 100.0)
        
#         log_prob_marginals_standard = tf.reduce_sum(normal_dist.log_prob(normal_samples), axis=-1)      
        
#         return log_prob_joint_normal - log_prob_marginals_standard
   
#     def Marginal_pdf_logprob(self, y_true, y_pred):
#         gaussian_marginals = tfd.TruncatedNormal(
#             loc=self.reshaped_means[:,0,:], 
#             scale=self.reshaped_scales[:,0,:], 
#             low=self.lbound - 1e-4,  
#             high=1.0 + 1e-4)

#         log_prob_marginals = gaussian_marginals.log_prob(self.reshaped_alpha_samples)  
#         log_prob_marginals = tf.clip_by_value(log_prob_marginals, -100.0, 100.0)
#         log_prob_marginal = tf.math.reduce_sum(log_prob_marginals, axis = -1)
#         return log_prob_marginal
    
#     def Joint_copula_dens_term(self, y_true, y_pred):
#         Copula_density_term = self.Copula_pdf_logprob(y_true, y_pred)
#         Marginal_logprob_term = self.Marginal_pdf_logprob(y_true, y_pred)  
        
#         joint_copula_logprob = tf.math.reduce_mean(Copula_density_term + Marginal_logprob_term, axis = None)
#         # Clip the final term before scaling by beta
#         joint_copula_logprob = tf.clip_by_value(joint_copula_logprob, -1e5, 1e5)
#         joint_copula_logprob = tf.math.square(tf.cast(self.beta, dtype=tf.float32))* (joint_copula_logprob)

#         return joint_copula_logprob
    
#     def ELBO_Copula_loss(self, y_true, y_pred):
#         Loss_freqs = self.Freqs_loss(y_true, y_pred)
#         Loss_MAC = self.MAC_modes_loss(y_true, y_pred)
#         Joint_copula_nll = self.Joint_copula_dens_term(y_true, y_pred)
        
#         ELBO_loss = Loss_freqs + 10*Loss_MAC + Joint_copula_nll
#         return ELBO_loss
    
#     def get_config(self):
#         config = {
#             'n_modes': self.n_modes,
#             'n_nodes_per_mode': self.n_nodes,
#             'n_dims': self.n_dims,
#             'num_gaussians': self.num_gaussians,
#             'num_samples': self.num_samples,
#             'beta': self.beta,
#             'mean_f': self.mean_freq.numpy(),
#             'std_f': self.std_freq.numpy(),
#             'lbound': self.lbound,
#             'epsi': self.epsi
#         }
#         base_config = super(My_CopulaVAE_Surrogate, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
    
    