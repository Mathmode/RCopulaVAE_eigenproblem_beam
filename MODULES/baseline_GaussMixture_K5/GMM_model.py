#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/baseline_GaussMixture_K5/GMM_model.py

BASELINE 2 of the comparison requested in R1.4 (Ref. 39, Gaussian mixture autoencoder): no copula; q is a mixture of
K = 5 Gaussians with full covariance matrices (as in the earlier GMM code, MultivariateNormalTriL components),

    q(y | m) = sum_k pi_k N(y; mu_k, L_k L_k^T),      z = lbound + (1 - lbound) sigmoid(y),

written in the logit space of the prior box [lbound, 1]^7 (ADVI transformation, Kucukelbir et al. 2017, JMLR 18:1-45).
q then has exactly the support of the uniform prior and the ELBO is finite: an untruncated mixture in z would put
mass outside the box, where the prior density is zero, and a truncated full-covariance mixture has no closed-form
normaliser. COVARIANCE = "diag" gives the diagonal-covariance variant (GaussMixtureDiagVAE).

Objective: the negative ELBO of the proposed method (likelihood, prior, gamma = 1, warm-up: inherited), estimated by
stratified sampling over the components (Morningstar et al. 2021, AISTATS, PMLR 130:3250-3258):
    E_q[g(z)] = sum_k pi_k E_{N_k}[g(z)],   S samples per component (num_samples), each reparameterised,
so the mixture weights receive exact gradients without sampling the component index. The entropy has no closed
form: H[q] = -E_q[log q(z)] with log q(z) = logsumexp_k(log pi_k + log N_k(y)) - log|dz/dy|.
"""

import numpy as np
import tensorflow as tf

from MODULES.review29oct_fixed import tf_keras
from MODULES.review29oct_fixed.GC_KSmix_models import My_CopulaKSVAE_withEigen

K = tf_keras()
LOG_2PI = float(np.log(2.0 * np.pi))


def Fully_connected_enc_GMM(input_dim, n_dims, n_components, covariance="full"):
    """Same trunk as Fully_connected_enc_GC_KS; heads: means, Cholesky diagonal, Cholesky off-diagonal (full
    covariance only, starts at zero = diagonal covariance) and mixture logits."""
    input1 = K.Input(shape=(input_dim,), name="InputLayer")
    x = K.layers.Dense(128, activation="relu", kernel_initializer="he_uniform")(input1)
    x = K.layers.Dense(256, activation="relu", kernel_initializer="he_uniform")(x)
    x = K.layers.Dense(128, activation="relu", kernel_initializer="he_uniform")(x)

    # component k starts around u = (k + 0.5) / K in every dimension (breaks the symmetry of the components)
    start = np.log((np.arange(n_components) + 0.5) / (n_components - np.arange(n_components) - 0.5))
    means = K.layers.Dense(n_components * n_dims, activation="linear", name="means",
                           bias_initializer=K.initializers.Constant(np.repeat(start, n_dims).tolist()))(x)
    chol_diag = K.layers.Dense(n_components * n_dims, activation="linear", name="chol_diag")(x)
    heads = [means, chol_diag]
    if covariance == "full":
        n_off = n_components * n_dims * (n_dims - 1) // 2
        heads.append(K.layers.Dense(n_off, activation="linear", kernel_initializer="zeros", name="chol_offdiag")(x))
    heads.append(K.layers.Dense(n_components, activation="linear", name="mix_logits")(x))
    return K.Model(inputs=input1, outputs=K.layers.Concatenate(axis=1)(heads))


class Inverse_GMM_Model(K.Model):
    """Encoder: modal data -> means (B, K, D), lower Cholesky factors (B, K, D, D) and weights (B, K) of q(y | m)."""

    def __init__(self, input_dim_encoder, n_dims, n_components, covariance="full", obs_types="fv", **kwargs):
        super(Inverse_GMM_Model, self).__init__(**kwargs)
        self.n_dims, self.n_components = int(n_dims), int(n_components)
        self.covariance = covariance
        self.obs_types = obs_types
        self.FC_encoder = Fully_connected_enc_GMM(input_dim_encoder, n_dims, n_components, covariance)
        rows, cols = np.tril_indices(self.n_dims, k=-1)
        basis = np.zeros((len(rows), self.n_dims, self.n_dims), dtype=np.float32)
        basis[np.arange(len(rows)), rows, cols] = 1.0
        self.basis_off = tf.constant(basis)

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

        Kc, D = self.n_components, self.n_dims
        n_off = Kc * D * (D - 1) // 2 if self.covariance == "full" else 0
        means, diag_raw, off, logits = tf.split(params, [Kc * D, Kc * D, n_off, Kc], axis=-1)
        means = tf.reshape(means, (-1, Kc, D))
        L = tf.linalg.diag(tf.nn.softplus(tf.reshape(diag_raw, (-1, Kc, D))) + 1e-3)
        if n_off:
            L += tf.einsum("bko,oij->bkij", tf.reshape(off, (-1, Kc, D * (D - 1) // 2)), self.basis_off)
        return means, L, tf.nn.softmax(logits, axis=-1)


class GaussMixtureVAE(My_CopulaKSVAE_withEigen):
    """Gaussian mixture (full covariance) in the logit space; likelihood, decoder and training steps inherited.
    num_KS = number of mixture components, num_samples = samples PER COMPONENT."""
    COVARIANCE = "full"
    TERMS = ["loss", "neg_elbo", "NLL", "NLL_freq", "NLL_vert", "NLL_rot", "Entropy"]
    HEAD_ORDER = ["means", "chol_diag", "chol_offdiag", "mix_logits"]
    PRED_KEYS = ["test_means", "test_chol", "test_weights"]

    def make_encoder(self, input_dim, n_dims, n_components, obs_types):
        return Inverse_GMM_Model(input_dim, n_dims, n_components, self.COVARIANCE, obs_types)

    def sample_posterior(self, inputs):
        """Stratified samples: num_samples per component; returns z (B, K*S, D) and the weight of every sample."""
        means, L, weights = self.Encoder_model(inputs)
        B, Kc, S, D = tf.shape(means)[0], self.num_KS, self.num_samples, self.n_dims
        eps = tf.random.normal((B, Kc, S, D))
        y = means[:, :, None, :] + tf.einsum("bkij,bksj->bksi", L, eps)
        y = tf.reshape(y, (B, Kc * S, D))
        z = self.lbound + (1.0 - self.lbound) * tf.sigmoid(y)
        sample_w = tf.reshape(tf.tile(weights[:, :, None], [1, 1, S]), (B, Kc * S)) / float(S)
        return z, dict(y=y, means=means, L=L, weights=weights, sample_w=sample_w)

    def log_q(self, y, means, L, weights):
        """log q(z) (B, N) in z units of the samples y (B, N, D) (logit space)."""
        D = self.n_dims
        eye = tf.eye(D, batch_shape=tf.shape(L)[:2], dtype=L.dtype)
        L_inv = tf.linalg.triangular_solve(L, eye, lower=True)                     # (B, K, D, D)
        diff = y[:, :, None, :] - means[:, None, :, :]                             # (B, N, K, D)
        sol = tf.einsum("bkij,bnkj->bnki", L_inv, diff)
        log_det = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=-1)       # (B, K)
        log_comp = -0.5 * tf.reduce_sum(tf.square(sol), axis=-1) - log_det[:, None, :] - 0.5 * D * LOG_2PI
        log_w = tf.math.log(tf.maximum(weights, 1e-30))[:, None, :]
        log_qy = tf.reduce_logsumexp(log_w + log_comp, axis=-1)
        log_jac = D * np.log(1.0 - self.lbound) + tf.reduce_sum(tf.math.log_sigmoid(y) + tf.math.log_sigmoid(-y), -1)
        return log_qy - log_jac

    def compute_terms(self, inputs):
        z, q = self.sample_posterior(inputs)
        nll, nll_f, nll_v, nll_r = self.nll_terms(inputs, z)
        log_q = self.log_q(q["y"], q["means"], q["L"], q["weights"])
        w = q["sample_w"]

        def expect(x):                    # E_q[x] = sum_k pi_k mean_s x(z_ks), per observation
            return tf.reduce_sum(w * x, axis=1)
        nll_mean = expect(nll)
        H = -expect(log_q)
        return {
            "loss": tf.reduce_mean(self.tau * nll_mean - self.gamma2 * H),
            "neg_elbo": tf.reduce_mean(nll_mean - self.gamma2 * H),
            "NLL": tf.reduce_mean(nll_mean),
            "NLL_freq": tf.reduce_mean(expect(nll_f)),
            "NLL_vert": tf.reduce_mean(expect(nll_v)),
            "NLL_rot": tf.reduce_mean(expect(nll_r)),
            "Entropy": tf.reduce_mean(H),
        }


class GaussMixtureDiagVAE(GaussMixtureVAE):
    """Same with diagonal covariance matrices (each component is a product of independent normals in y)."""
    COVARIANCE = "diag"
    HEAD_ORDER = ["means", "chol_diag", "mix_logits"]
