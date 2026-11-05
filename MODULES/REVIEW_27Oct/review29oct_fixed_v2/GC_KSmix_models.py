#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed_v2/GC_KSmix_models.py

PI-GC-KSmix-VAE: Gaussian-copula posterior with Kumaraswamy-mixture marginals and a physics (eigenproblem)
decoder, trained on the negative ELBO.

    loss = tau * E_q[ NLL(z; m) ]  -  gamma^2 * H[q(z | m)]          (tau = 1 after the warm-up)

  NLL     : the calibrated likelihood of fe_model.neg_log_likelihood (correlated log-frequency errors with the
            mesh bias removed, and 1 - MAC of the observed shapes), with the values of problem_config.json;
  H[q]    : entropy of q = 0.5 log det R (copula, exact) + sum_d H[q_d] (marginals, Monte Carlo);
  prior   : uniform on [lbound, 1]^7, so E_q[log p(z)] is a constant;
  gamma=1 : the minimiser is the Bayesian posterior; reference_posterior.py samples the same density
            (for gamma != 1 both target exp(-NLL / gamma^2)).

What was wrong in MODULES/review29oct/GC_KSmix_models.py and why the posterior collapsed:
  1. Joint_copula_dens_term returned +gamma^2 * (-E_q[log q]) = +gamma^2 * H[q]. Minimising the loss therefore
     MINIMISED the entropy of q. The only brake was clip(log q, -50, 50): the logged "KL" term flattened at
     -6.11 = -gamma^2 * 50 and the copulas became singular (smallest eigenvalue of R ~ 1e-12, corr(z1, z2) =
     -1.000 for test sample 6), hence the razor-thin ridges in the corner plots.
  2. Boundary_Penalty_loss (-log x - log(1 - x)) pushed samples away from z = 1, where the posterior of every
     healthy parameter lives: healthy parameters were estimated 0.02-0.06 too low.
  3. 5 * Huber(standardised log f) + 10 * (1 - best MAC) is not the likelihood of the data, so its
     posterior (temperature) is arbitrary and differs from any "ground truth".
  4. The quantile gradient was clipped (see GC_KSmix_functions.py) and the learning rate decayed by
     0.92 every 1000 steps (x 1e-4 after 1900 epochs: training effectively stopped after ~500 epochs).
"""

import numpy as np
import tensorflow as tf

from MODULES.review29oct_fixed_v2 import tf_keras
from MODULES.review29oct_fixed_v2.fe_model import frequency_precision
from MODULES.review29oct_fixed_v2.preprocessing_and_eigen import (
    assemble_global_Kmatrices_bearings,
    Solve_Bridge_Eigenproblem,
)
from MODULES.review29oct_fixed_v2.GC_KSmix_architectures import Fully_connected_enc_GC_KS, positive_shape_parameter
from MODULES.review29oct_fixed_v2.GC_KSmix_functions import (
    build_correlation_matrices_from_cholesky,
    gaussian_copula_samples,
    ks_mixture_quantile,
    ks_mixture_logpdf,
    copula_entropy,
)

K = tf_keras()


def calculate_MAC(modes_true, modes_pred):
    """MAC of paired modes along the last axis."""
    num = tf.square(tf.reduce_sum(modes_true * modes_pred, axis=-1))
    den = tf.reduce_sum(tf.square(modes_true), axis=-1) * tf.reduce_sum(tf.square(modes_pred), axis=-1)
    return num / (den + 1e-12)


class Inverse_Copula_KS_Model(K.Model):
    """Encoder: modal data -> (a, b, w) of the KS mixtures (B, K, D) and the copula factor parameters."""

    def __init__(self, input_dim_encoder, n_dims, num_KS, obs_types="fvr", **kwargs):
        super(Inverse_Copula_KS_Model, self).__init__(**kwargs)
        self.n_dims = int(n_dims)
        self.num_KS = int(num_KS)
        self.obs_types = obs_types
        self.FC_encoder = Fully_connected_enc_GC_KS(input_dim_encoder, n_dims, num_KS)

    def call(self, inputs):
        [freq_data, rot_modes_data, vert_modes_data, _] = inputs
        parts = []
        if "f" in self.obs_types:
            parts.append(freq_data)
        if "v" in self.obs_types:
            parts.append(tf.reshape(vert_modes_data, [-1, vert_modes_data.shape[1] * vert_modes_data.shape[2]]))
        if "r" in self.obs_types:
            parts.append(tf.reshape(rot_modes_data, [-1, rot_modes_data.shape[1] * rot_modes_data.shape[2]]))
        params = self.FC_encoder(tf.concat(parts, axis=1))

        nkd = self.n_dims * self.num_KS
        a_raw, b_raw, w_logits, offdiag, diag_raw = tf.split(
            params, [nkd, nkd, nkd, self.n_dims * (self.n_dims - 1) // 2, self.n_dims], axis=-1)

        a_ks = tf.reshape(positive_shape_parameter(a_raw), (-1, self.num_KS, self.n_dims))
        b_ks = tf.reshape(positive_shape_parameter(b_raw), (-1, self.num_KS, self.n_dims))
        weights = tf.nn.softmax(tf.reshape(w_logits, (-1, self.num_KS, self.n_dims)), axis=1)
        diag = tf.nn.softplus(diag_raw) + 1e-3
        return a_ks, b_ks, weights, offdiag, diag


class My_CopulaKSVAE_withEigen(K.Model):
    TERMS = ["loss", "neg_elbo", "NLL", "NLL_freq", "NLL_vert", "NLL_rot",
             "Entropy", "Entropy_copula", "Entropy_marginals"]

    def __init__(self, input_dim, num_dofs, n_modes, Ke_matrices, L_inv, n_dims, num_KS, num_samples,
                 gamma, mean_f, std_f, lbound, likelihood, obs_types="fv", sensor_nodes=None, **kwargs):
        super(My_CopulaKSVAE_withEigen, self).__init__(**kwargs)
        self.n_modes = int(n_modes)
        self.n_dims = int(n_dims)
        self.num_KS = int(num_KS)
        self.num_samples = int(num_samples)
        self.gamma2 = float(gamma) ** 2
        self.lbound = float(lbound)
        self.obs_types = obs_types

        self.Ke_matrices = tf.constant(Ke_matrices, dtype=tf.float32)
        self.mean_freq = tf.constant(mean_f, dtype=tf.float32)
        self.std_freq = tf.constant(std_f, dtype=tf.float32)
        bias_f, prec_f = frequency_precision(likelihood)
        self.bias_logf = tf.constant(bias_f, dtype=tf.float32)
        self.prec_logf = tf.constant(prec_f, dtype=tf.float32)
        self.inv2var_v = tf.constant(0.5 / np.square(likelihood["sig_v"]), dtype=tf.float32)
        self.inv2var_r = tf.constant(0.5 / np.square(likelihood["sig_r"]), dtype=tf.float32)

        self.Encoder_model = Inverse_Copula_KS_Model(input_dim, n_dims, num_KS, obs_types)
        self.Eigen_solver = Solve_Bridge_Eigenproblem(num_dofs, n_modes, L_inv, sensor_nodes)
        self.tau = self.add_weight(name="likelihood_temperature", shape=(), initializer="ones", trainable=False,
                                   dtype="float32")
        self.trackers = {name: K.metrics.Mean(name=name) for name in self.TERMS}

    @property
    def metrics(self):
        return list(self.trackers.values())

    def build(self, input_shape):
        # the encoder is a functional model (built at construction) and the eigen layer has no weights
        super(My_CopulaKSVAE_withEigen, self).build(input_shape)

    def sample_posterior(self, inputs):
        """Draws num_samples z ~ q(z | m) per observation; returns z (B, S, D) and the pieces of q."""
        a_ks, b_ks, weights, offdiag, diag = self.Encoder_model(inputs)
        LT = build_correlation_matrices_from_cholesky(offdiag, diag, self.n_dims)
        u = gaussian_copula_samples(LT, self.num_samples)
        x = ks_mixture_quantile(u, a_ks, b_ks, weights)
        z = self.lbound + (1.0 - self.lbound) * x
        return z, dict(a=a_ks, b=b_ks, w=weights, LT=LT, u=u)

    def call(self, inputs):
        z, _ = self.sample_posterior(inputs)
        return z

    def nll_terms(self, inputs, z):
        """
        Likelihood terms (B, S) of the samples z (B, S, D) - the formula of fe_model.neg_log_likelihood:
        1/2 d^T P d with d = log f_obs - bias - log f(z), plus (1 - MAC) / (2 s^2) for every observed shape.
        """
        [freq_data, rot_modes_data, vert_modes_data, _] = inputs
        B, S, M = tf.shape(z)[0], tf.shape(z)[1], self.n_modes
        Kfree = assemble_global_Kmatrices_bearings(tf.reshape(z, (-1, self.n_dims)), self.Ke_matrices)
        pred_f, pred_rot, pred_vert = self.Eigen_solver(Kfree)
        n_nodes = tf.shape(pred_vert)[-1]
        pred_f = tf.reshape(pred_f, (B, S, M))
        pred_rot = tf.reshape(pred_rot, (B, S, M, n_nodes))
        pred_vert = tf.reshape(pred_vert, (B, S, M, n_nodes))

        log_f_obs = freq_data * self.std_freq + self.mean_freq
        d = log_f_obs[:, None, :] - self.bias_logf - tf.math.log(pred_f)
        nll_f = 0.5 * tf.einsum("bsm,mn,bsn->bs", d, self.prec_logf, d)
        nll_v = tf.reduce_sum((1.0 - calculate_MAC(vert_modes_data[:, None], pred_vert)) * self.inv2var_v, axis=-1)
        nll_r = tf.reduce_sum((1.0 - calculate_MAC(rot_modes_data[:, None], pred_rot)) * self.inv2var_r, axis=-1)
        nll = tf.zeros_like(nll_f)
        for key, term in (("f", nll_f), ("v", nll_v), ("r", nll_r)):
            if key in self.obs_types:
                nll += term
        return nll, nll_f, nll_v, nll_r

    def compute_terms(self, inputs):
        z, q = self.sample_posterior(inputs)
        nll, nll_f, nll_v, nll_r = self.nll_terms(inputs, z)

        # --- entropy of q: exact copula part + Monte-Carlo marginal part (copula path blocked: zero mean) ---
        H_cop = copula_entropy(q["LT"])
        x_marg = ks_mixture_quantile(tf.stop_gradient(q["u"]), q["a"], q["b"], q["w"])
        log_q_marg = ks_mixture_logpdf(x_marg, q["a"], q["b"], q["w"]) - np.log(1.0 - self.lbound)
        H_marg = -tf.reduce_mean(tf.reduce_sum(log_q_marg, axis=-1), axis=1)
        H = H_cop + H_marg

        nll_mean = tf.reduce_mean(nll, axis=1)
        return {
            "loss": tf.reduce_mean(self.tau * nll_mean - self.gamma2 * H),
            "neg_elbo": tf.reduce_mean(nll_mean - self.gamma2 * H),
            "NLL": tf.reduce_mean(nll_mean),
            "NLL_freq": tf.reduce_mean(nll_f),
            "NLL_vert": tf.reduce_mean(nll_v),
            "NLL_rot": tf.reduce_mean(nll_r),
            "Entropy": tf.reduce_mean(H),
            "Entropy_copula": tf.reduce_mean(H_cop),
            "Entropy_marginals": tf.reduce_mean(H_marg),
        }

    def _update(self, terms):
        for name, tracker in self.trackers.items():
            tracker.update_state(terms[name])
        out = {name: tracker.result() for name, tracker in self.trackers.items()}
        out["tau"] = tf.convert_to_tensor(self.tau)
        return out

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            terms = self.compute_terms(x)
        grads = tape.gradient(terms["loss"], self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return self._update(terms)

    def test_step(self, data):
        x, _ = data
        return self._update(self.compute_terms(x))


# =============================================================================
# CALLBACKS
# =============================================================================
class LikelihoodTempering(K.callbacks.Callback):
    """tau = tau0^(1 - epoch / warmup) during the warm-up, then 1 (exact ELBO). Starting with a flat
    likelihood lets q cover the plausible region before the sharp likelihood switches on."""

    def __init__(self, tau0=0.01, warmup_epochs=150):
        super(LikelihoodTempering, self).__init__()
        self.tau0 = float(tau0)
        self.warmup_epochs = int(warmup_epochs)

    def on_epoch_begin(self, epoch, logs=None):
        tau = 1.0 if epoch >= self.warmup_epochs else self.tau0 ** (1.0 - epoch / max(self.warmup_epochs, 1))
        self.model.tau.assign(tau)


class LearningRateLogger(K.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            lr = self.model.optimizer.learning_rate
            if callable(lr):  # older Keras returns the schedule itself
                lr = lr(self.model.optimizer.iterations)
            logs["lr"] = float(np.asarray(lr))


class FastInMemoryBestWeights(K.callbacks.Callback):
    """Keeps the weights with the lowest monitored value (only after start_epoch) and restores them at the end."""

    def __init__(self, monitor="val_neg_elbo", patience=300, start_epoch=150):
        super(FastInMemoryBestWeights, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.start_epoch = start_epoch
        self.best_val = np.inf
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        val = (logs or {}).get(self.monitor)
        if val is None or epoch < self.start_epoch:
            return
        if val < self.best_val:
            self.best_val, self.best_epoch, self.wait = val, epoch + 1, 0
            self.best_weights = [w.numpy() for w in self.model.weights]
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nEarly stopping at epoch {epoch + 1} (best {self.monitor}={self.best_val:.4f} at epoch {self.best_epoch}).")
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            for var, val in zip(self.model.weights, self.best_weights):
                var.assign(val)
            print(f"Restored best weights from epoch {self.best_epoch} ({self.monitor}={self.best_val:.4f}).")
