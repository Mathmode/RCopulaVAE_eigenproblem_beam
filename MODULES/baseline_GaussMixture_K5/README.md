# Baseline 2 (R1.4): Gaussian mixture posterior, K = 5, no copula (Ref. 39)

q(y | m) = sum_k pi_k N(y; mu_k, L_k L_k^T), k = 1..5, full covariance matrices (as the earlier Gaussian-mixture code,
`MultivariateNormalTriL` components), with z = 0.45 + 0.55 sigmoid(y).

* **Why the logit space**: the prior is uniform on the box [0.45, 1]^7. A Gaussian mixture written directly in z puts
  mass outside the box, where the prior density is zero (infinite -ELBO), and a truncated full-covariance mixture has
  no closed-form normaliser. The logit transformation (as in ADVI, Kucukelbir et al. 2017) gives q exactly the
  support of the prior. Consequence, visible in the results: the density of a logit-Gaussian vanishes at z = 1,
  whereas the exact posterior of an undamaged parameter is largest at z = 1.
* **Training**: same ELBO, likelihood, gamma = 1, warm-up, optimizer, schedule and early stopping as the proposed model
  (`review29oct_fixed/training_common.py`). The expectation over q is stratified over the components
  (Morningstar et al. 2021): 2 reparameterised samples per component (10 eigen-solutions per observation and step),
  so the mixture weights get exact gradients.
* `--covariance diag` trains the diagonal-covariance variant; `--components` changes K.

```bash
# server, from the project root (training + post-processing + posterior quality), GPU 2:
nohup bash MODULES/baseline_GaussMixture_K5/run_GMM_gpu.sh --gpu 2 --modes 5 --components 5 > baseline_GMM5.log 2>&1 &
```

Results: `Output/baseline_GaussMixture_K5/GMM5full_03Nov-FINAL5modes_7D_obs-fv_Gamma1.0_Ep1500/`, same layout as the
proposed model. In `Fig_copula_correlations` the dependence is the correlation of the normal scores of q (the copula
correlation of a Gaussian copula fitted to q), comparable with the copula correlations of the other models.
