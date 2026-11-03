#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct/new_scripts_rev/GC_KSmix_models.py

Physics-informed VAE whose approximate posterior is a mixture of K Gaussian-copula
components with Kumaraswamy marginals, decoded by the differentiable eigensolver.

LOSS (paper Eq. 10, written per sample and made consistent with Bayes' rule)
---------------------------------------------------------------------------
Tempered likelihood and explicit prior:
    p(m | z) ∝ exp(-E(z) / gamma^2),   E(z) = L_freq(z) + L_MAC(z)
    p(z)     = prod_d Beta(x_d; alpha, beta),  x = (z - lbound) / (1 - lbound)

Negative ELBO multiplied by gamma^2 (a positive constant, so the optimum is unchanged):
    L(theta) = E_q[ E(z) ] + gamma^2 * E_q[ log q(z|m) - log p(z) ]
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              = gamma^2 * KL(q || p),  "+ log q" sign

Minimising +E_q[log q] MAXIMISES the entropy of q, which is what keeps q from
collapsing. MODULES/KUMARSWAMY/GC_KSmix_models.py used  gamma^2 * (-E_q[log q])
(an NLL of its own samples), i.e. the opposite sign: it rewards collapse, which is why
that version needed a >= 1.01, b >= 1.01, a boundary penalty and a diversity penalty.

Its exact minimiser is q(z|m) = p_gamma(z|m) ∝ exp(-E(z)/gamma^2) p(z), i.e. the SAME
target as the reference posterior in reference_posterior_smc.py. Any extra term
(repulsion, weight-entropy, KL clipping) moves the optimum away from that target, so
none are used here.

The expectation over the discrete mixture component is computed exactly
(stratified over k), so mixture weights receive exact gradients:
    E_q[g] = sum_k pi_k * E_{q_k}[g(z)],   z ~ q_k reparameterized,
    with log q(z) the FULL mixture density (logsumexp over components).

Optional objective='siwae' uses the stratified importance-weighted bound
(Morningstar et al., 2021), which is mass-covering and helps find secondary modes.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from MODULES.review29oct.preprocessing_and_eigen import (
    assemble_global_Kmatrices_bearings,
    Solve_Bridge_Eigenproblem
)
from MODULES.review29oct.new_scripts_rev.GC_KSmix_architectures import Fully_connected_enc_MixGC_KS
from MODULES.review29oct.new_scripts_rev.GC_KSmix_functions import (
    build_correlation_cholesky,
    sample_components,
    mixture_log_density,
    beta_prior_log_density
)


def paired_MAC(modes_true, modes_pred):
    """MAC between mode j of the observation and mode j of the prediction. [..., M, nodes] -> [..., M]"""
    t = tf.math.l2_normalize(modes_true, axis=-1, epsilon=1e-12)
    p = tf.math.l2_normalize(modes_pred, axis=-1, epsilon=1e-12)
    return tf.square(tf.reduce_sum(t * p, axis=-1))


def _f32(*tensors):
    return [tf.cast(t, tf.float32) for t in tensors]


def flatten_modal_input(freq_data, rot_modes_data, vert_modes_data):
    freq_data, rot_modes_data, vert_modes_data = _f32(freq_data, rot_modes_data, vert_modes_data)
    flat_rot = tf.reshape(rot_modes_data, [-1, rot_modes_data.shape[1] * rot_modes_data.shape[2]])
    flat_vert = tf.reshape(vert_modes_data, [-1, vert_modes_data.shape[1] * vert_modes_data.shape[2]])
    return tf.concat([freq_data, flat_vert, flat_rot], axis=1)


class MixGC_KS_VAE_withEigen(K.Model):
    def __init__(self, input_dim, num_dofs, n_modes, Ke_matrices, L_inv, n_dims, n_comp,
                 num_samples, gamma, mean_f, std_f, lbound, prior_alpha=2.2, prior_beta=1.1,
                 lambda_vert=1.5, lambda_rot=0.5, objective='elbo', a_min=1.0, b_min=0.5, **kwargs):
        super().__init__(**kwargs)
        self.n_modes = int(n_modes)
        self.n_dims = int(n_dims)
        self.n_comp = int(n_comp)
        self.num_samples = int(num_samples)
        self.gamma = float(gamma)
        self.lbound = float(lbound)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.lambda_vert = float(lambda_vert)
        self.lambda_rot = float(lambda_rot)
        if objective not in ('elbo', 'siwae'):
            raise ValueError("objective must be 'elbo' or 'siwae'")
        self.objective = objective

        self.K_basis = tf.constant(Ke_matrices, dtype=tf.float32)
        self.mean_freq = tf.constant(mean_f, dtype=tf.float32)
        self.std_freq = tf.constant(std_f, dtype=tf.float32)

        self.Encoder_model = Fully_connected_enc_MixGC_KS(input_dim, self.n_dims, self.n_comp,
                                                          a_min=a_min, b_min=b_min)
        self.Eigen_solver = Solve_Bridge_Eigenproblem(num_dofs, self.n_modes, L_inv)

        # Multiplies the KL term; 1.0 is the exact objective. A callback may anneal it
        # from > 1 (broad, mode-covering start) down to 1.0 early in training.
        self.kl_weight = self.add_weight(name='kl_weight', shape=(), initializer='ones', trainable=False)

        self.loss_tracker = K.metrics.Mean(name='loss')
        self.freqs_loss_tracker = K.metrics.Mean(name='Freqs_loss')
        self.mac_loss_tracker = K.metrics.Mean(name='MAC_modes_loss')
        self.kl_loss_tracker = K.metrics.Mean(name='KL_Divergence_term')

    @property
    def metrics(self):
        return [self.loss_tracker, self.freqs_loss_tracker, self.mac_loss_tracker, self.kl_loss_tracker]

    # ------------------------------------------------------------------ posterior
    def posterior_params(self, freq_data, rot_modes_data, vert_modes_data):
        a, b, log_pi, off_diag, diag = self.Encoder_model(
            flatten_modal_input(freq_data, rot_modes_data, vert_modes_data))
        L = build_correlation_cholesky(off_diag, diag, self.n_dims)
        return a, b, log_pi, L

    def call(self, inputs):
        freq_data, rot_modes_data, vert_modes_data, _ = inputs
        a, b, log_pi, L = self.posterior_params(freq_data, rot_modes_data, vert_modes_data)
        x, z = sample_components(a, b, L, self.num_samples, self.lbound)       # [B, K, S, D]

        K_global = assemble_global_Kmatrices_bearings(tf.reshape(z, [-1, self.n_dims]), self.K_basis)
        pred_freqs, pred_rot, pred_vert = self.Eigen_solver(K_global)
        return a, b, log_pi, L, x, pred_freqs, pred_rot, pred_vert

    # ------------------------------------------------------------------ loss
    def _loss_terms(self, freq_data, rot_modes_data, vert_modes_data, outputs):
        a, b, log_pi, L, x, pred_freqs, pred_rot, pred_vert = outputs
        freq_data, rot_modes_data, vert_modes_data = _f32(freq_data, rot_modes_data, vert_modes_data)
        Kc, S, D, M = self.n_comp, self.num_samples, self.n_dims, self.n_modes
        sample_shape = [-1, Kc, S]

        # Energy E(z) = L_freq + L_MAC per sample, [B, K, S]
        pred_scaled = (tf.math.log(tf.maximum(tf.abs(pred_freqs), 1e-7)) - self.mean_freq) / self.std_freq
        pred_scaled = tf.reshape(pred_scaled, sample_shape + [M])
        e_freq = tf.reduce_mean(tf.square(freq_data[:, None, None, :] - pred_scaled), axis=-1)

        n_nodes = vert_modes_data.shape[-1]
        pred_vert = tf.reshape(pred_vert, sample_shape + [M, n_nodes])
        pred_rot = tf.reshape(pred_rot, sample_shape + [M, n_nodes])
        mac_v = paired_MAC(vert_modes_data[:, None, None], pred_vert)
        mac_r = paired_MAC(rot_modes_data[:, None, None], pred_rot)
        e_mac = (self.lambda_vert * tf.reduce_mean(1.0 - mac_v, axis=-1)
                 + self.lambda_rot * tf.reduce_mean(1.0 - mac_r, axis=-1))
        energy = e_freq + e_mac

        # log q(z|m) (full mixture) and log p(z), [B, K, S]
        n_rep = Kc * S
        x_flat = tf.reshape(x, [-1, D])
        log_q = mixture_log_density(
            x_flat,
            tf.repeat(log_pi, n_rep, axis=0),
            tf.repeat(a, n_rep, axis=0),
            tf.repeat(b, n_rep, axis=0),
            tf.repeat(L, n_rep, axis=0),
            self.lbound)
        log_p = beta_prior_log_density(x_flat, self.prior_alpha, self.prior_beta, self.lbound)
        log_ratio = tf.reshape(log_q - log_p, sample_shape)            # log q - log p

        g2 = self.gamma ** 2
        pi = tf.exp(log_pi)[:, :, None]                                 # [B, K, 1]

        def pi_mean(t):
            return tf.reduce_mean(tf.reduce_sum(pi * tf.reduce_mean(t, axis=-1, keepdims=True), axis=1))

        if self.objective == 'elbo':
            total = pi_mean(energy + g2 * self.kl_weight * log_ratio)
        else:
            log_w = -energy / g2 - self.kl_weight * log_ratio
            log_bound = (tf.reduce_logsumexp(log_pi[:, :, None] + log_w, axis=[1, 2])
                         - tf.math.log(float(S)))
            total = -g2 * tf.reduce_mean(log_bound)

        return total, pi_mean(e_freq), pi_mean(e_mac), pi_mean(log_ratio)

    def _update_metrics(self, total, l_f, l_mac, kl):
        self.loss_tracker.update_state(total)
        self.freqs_loss_tracker.update_state(l_f)
        self.mac_loss_tracker.update_state(l_mac)
        self.kl_loss_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        x_in, _ = data
        freq_data, rot_modes_data, vert_modes_data, _ = x_in
        with tf.GradientTape() as tape:
            outputs = self(x_in, training=True)
            total, l_f, l_mac, kl = self._loss_terms(freq_data, rot_modes_data, vert_modes_data, outputs)
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return self._update_metrics(total, l_f, l_mac, kl)

    def test_step(self, data):
        x_in, _ = data
        freq_data, rot_modes_data, vert_modes_data, _ = x_in
        outputs = self(x_in, training=False)
        return self._update_metrics(*self._loss_terms(freq_data, rot_modes_data, vert_modes_data, outputs))


class KLWeightAnnealing(K.callbacks.Callback):
    """Linearly anneals model.kl_weight from `start` to 1.0 over `n_epochs`, then keeps it at 1.0."""
    def __init__(self, start=3.0, n_epochs=200):
        super().__init__()
        self.start = float(start)
        self.n_epochs = int(n_epochs)

    def on_epoch_begin(self, epoch, logs=None):
        frac = min(1.0, epoch / max(self.n_epochs, 1))
        self.model.kl_weight.assign(self.start + (1.0 - self.start) * frac)


def sample_posterior_np(model, freq, rot, vert, n_samples, seed=0):
    """
    Ancestral samples z ~ q(z|m) for ONE observation (numpy in/out), plus its parameters.
    freq: [M], rot/vert: [M, nodes]. No reweighting or refinement of any kind.
    """
    a, b, log_pi, L = model.posterior_params(freq[None].astype(np.float32),
                                             rot[None].astype(np.float32),
                                             vert[None].astype(np.float32))
    tf.random.set_seed(seed)
    _, z_all = sample_components(a, b, L, n_samples, model.lbound)          # [1, K, S, D]
    rng = np.random.default_rng(seed)
    pi = np.exp(log_pi.numpy()[0])
    comp = rng.choice(model.n_comp, size=n_samples, p=pi / pi.sum())
    z = z_all.numpy()[0][comp, np.arange(n_samples)]
    return z, {'a': a.numpy()[0], 'b': b.numpy()[0], 'pi': pi, 'L': L.numpy()[0]}
