# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:33:24 2026

@author: anafd
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

@tf.function(jit_compile=True)
def _pure_kumaraswamy_sampling_jit(a_params, b_params, weight_vals, copula_samples, lbound):
    """
    Unconditionally stable Bisection solver for Kumaraswamy Mixtures.
    """
    u = tf.clip_by_value(copula_samples, 1e-7, 1.0 - 1e-7)
    
    # Align parameters to (Batch, 1, Dims, Components)
    a_t = tf.transpose(a_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    b_t = tf.transpose(b_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    # Raw logits used for bisection; softmax applied internally
    weights_t = tf.nn.softmax(tf.transpose(weight_vals, perm=[0, 2, 1]), axis=-1)[:, tf.newaxis, :, :]

    low = tf.zeros_like(u)
    high = tf.ones_like(u)

    for _ in range(35):
        mid = (low + high) / 2.0
        mid_expand = tf.expand_dims(mid, axis=-1)
        
        # Kumaraswamy CDF: 1 - (1 - x^a)^b
        x_a = tf.math.pow(tf.maximum(mid_expand, 1e-10), a_t)
        one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-30, 1.0) 
        cdf_components = 1.0 - tf.math.pow(one_minus_x_a, b_t)
        
        cdf_mid = tf.reduce_sum(weights_t * cdf_components, axis=-1)
        is_less = cdf_mid < u
        
        low = tf.where(is_less, mid, low)
        high = tf.where(is_less, high, mid)

    return (low + high) / 2.0

@tf.custom_gradient
def fast_kumaraswamy_quantile(a_params, b_params, weight_vals, copula_samples, lbound):
    """
    Quantile function with Physics-Prioritized Gradient.
    Boosts the frequency signal while shielding against PDF-driven singularities.
    """
    x = _pure_kumaraswamy_sampling_jit(a_params, b_params, weight_vals, copula_samples, lbound)
    
    def grad(dy):
        # 1. Boost physics signal and clip norm
        # The multiplier helps dy compete with the regularizer in Beta_models.py
        signal_multiplier = 5.0
        dy_safe = tf.clip_by_norm(dy * signal_multiplier, 1.0) 

        with tf.GradientTape() as tape:
            tape.watch([a_params, b_params, weight_vals])
            
            a_t = tf.transpose(a_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            b_t = tf.transpose(b_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            w_soft = tf.nn.softmax(tf.transpose(weight_vals, perm=[0, 2, 1]), axis=-1)[:, tf.newaxis, :, :]
            
            x_stop = tf.stop_gradient(x)
            x_expand = tf.expand_dims(x_stop, axis=-1)
            
            # Surrogate CDF for differentiation
            x_a = tf.math.pow(tf.maximum(x_expand, 1e-10), a_t)
            one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-30, 1.0)
            cdf_comp = 1.0 - tf.math.pow(one_minus_x_a, b_t)
            cdf_x = tf.reduce_sum(w_soft * cdf_comp, axis=-1)
            
        # 2. PDF with adaptive safety floor
        x_pow = tf.math.pow(tf.maximum(x_expand, 1e-10), a_t - 1.0)
        one_minus_x_a_pow = tf.math.pow(one_minus_x_a, b_t - 1.0)
        pdf_comp = a_t * b_t * x_pow * one_minus_x_a_pow
        pdf_x = tf.reduce_sum(w_soft * pdf_comp, axis=-1)
        
        # Prevent 1/PDF from becoming too large (vanishing frequency fit)
        # or too small (exploding gradients)
        pdf_safe = tf.clip_by_value(pdf_x, 5e-4, 50.0) 
        
        # Implicit Gradient: dx/dtheta = -(dCDF/dtheta) / (dCDF/dx)
        grad_common = -dy_safe * (1.0 - lbound) / pdf_safe
        
        # Get raw gradients for parameters
        grads = tape.gradient(
            cdf_x, [a_params, b_params, weight_vals], 
            output_gradients=grad_common
        )
        
        def clean(g, limit=20.0):
            if g is None: return None
            # Replace NaNs/Infs with 0
            g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            # Surgical clipping
            return tf.clip_by_value(g, -limit, limit)

        return clean(grads[0]), clean(grads[1]), clean(grads[2]), None, None

    z = lbound + (1.0 - lbound) * x
    return z, grad

@tf.function
def kumarswamy_marginal_samples(a_params, b_params, weight_vals, copula_samples, lbound):
    return fast_kumaraswamy_quantile(a_params, b_params, weight_vals, copula_samples, lbound)
# import tensorflow as tf
# import tensorflow_probability as tfp
# tfd = tfp.distributions

# @tf.function(jit_compile=True)
# def _pure_kumaraswamy_sampling_jit(a_params, b_params, weight_vals, copula_samples, lbound):
#     """
#     Unconditionally stable Bisection solver for Kumaraswamy Mixtures.
#     """
#     u = tf.clip_by_value(copula_samples, 1e-7, 1.0 - 1e-7)
    
#     # Align parameters to (Batch, 1, Dims, Components)
#     a_t = tf.transpose(a_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
#     b_t = tf.transpose(b_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
#     # Apply softmax to ensure weights sum to 1
#     weights_t = tf.nn.softmax(tf.transpose(weight_vals, perm=[0, 2, 1]), axis=-1)[:, tf.newaxis, :, :]

#     low = tf.zeros_like(u)
#     high = tf.ones_like(u)

#     for _ in range(35):
#         mid = (low + high) / 2.0
#         mid_expand = tf.expand_dims(mid, axis=-1)
        
#         # Kumaraswamy CDF: 1 - (1 - x^a)^b
#         x_a = tf.math.pow(tf.maximum(mid_expand, 1e-10), a_t)
#         one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-30, 1.0) 
#         cdf_components = 1.0 - tf.math.pow(one_minus_x_a, b_t)
        
#         cdf_mid = tf.reduce_sum(weights_t * cdf_components, axis=-1)
#         is_less = cdf_mid < u
        
#         low = tf.where(is_less, mid, low)
#         high = tf.where(is_less, high, mid)

#     return (low + high) / 2.0

# @tf.custom_gradient
# def fast_kumaraswamy_quantile(a_params, b_params, weight_vals, copula_samples, lbound):
#     """
#     Quantile function with Straight-Through Estimators for Weights.
#     This ensures that physics misfit (dy) directly pushes the mixture weights.
#     """
#     x = _pure_kumaraswamy_sampling_jit(a_params, b_params, weight_vals, copula_samples, lbound)
    
#     def grad(dy):
#         # 1. Physics gradient normalization: 
#         # Helps prevent gradient explosion while keeping the 'direction' clear.
#         dy_norm = tf.clip_by_norm(dy, 1.0) 

#         with tf.GradientTape() as tape:
#             tape.watch([a_params, b_params, weight_vals])
            
#             a_t = tf.transpose(a_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
#             b_t = tf.transpose(b_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
#             # Re-calculating softmax inside tape for weights gradient
#             w_soft = tf.nn.softmax(tf.transpose(weight_vals, perm=[0, 2, 1]), axis=-1)[:, tf.newaxis, :, :]
            
#             x_stop = tf.stop_gradient(x)
#             x_expand = tf.expand_dims(x_stop, axis=-1)
            
#             # Cumulative density at current sample
#             x_a = tf.math.pow(tf.maximum(x_expand, 1e-10), a_t)
#             one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-30, 1.0)
#             cdf_comp = 1.0 - tf.math.pow(one_minus_x_a, b_t)
#             cdf_x = tf.reduce_sum(w_soft * cdf_comp, axis=-1)
            
#         # PDF at current sample (The denominator for the Implicit Function Theorem)
#         x_pow = tf.math.pow(tf.maximum(x_expand, 1e-10), a_t - 1.0)
#         one_minus_x_a_pow = tf.math.pow(one_minus_x_a, b_t - 1.0)
#         pdf_comp = a_t * b_t * x_pow * one_minus_x_a_pow
#         pdf_x = tf.reduce_sum(w_soft * pdf_comp, axis=-1)
        
#         # Implicit Gradient: dx/dtheta = -(dCDF/dtheta) / (dCDF/dx)
#         # where dCDF/dx = PDF.
#         pdf_safe = tf.clip_by_value(pdf_x, 1e-4, 15.0) 
        
#         # Scaling factor for the physics error
#         # factor = (1 - lbound) because z = lbound + (1-lbound)*x
#         grad_common = -dy_norm * (1.0 - lbound) / pdf_safe
        
#         # Backprop through the CDF surrogate
#         grad_a, grad_b, grad_weights = tape.gradient(
#             cdf_x, [a_params, b_params, weight_vals], 
#             output_gradients=grad_common
#         )
        
#         def clean(g):
#             if g is None: return None
#             return tf.clip_by_value(tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)), -1e3, 1e3)

#         return clean(grad_a), clean(grad_b), clean(grad_weights), None, None

#     z = lbound + (1.0 - lbound) * x
#     return z, grad

# @tf.function
# def kumarswamy_marginal_samples(a_params, b_params, weight_vals, copula_samples, lbound):
#     return fast_kumaraswamy_quantile(a_params, b_params, weight_vals, copula_samples, lbound)




