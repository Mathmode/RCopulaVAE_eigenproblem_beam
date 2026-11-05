# Variant: Gaussian copula + mixtures of truncated Gaussians (GC-GMix)

q(z | m) = c_R(F_1(z_1), ..., F_7(z_7)) prod_d sum_k w_kd TN(z_d; mu_kd, sigma_kd, [0.45, 1]): the marginals of the
proposed model with truncated Gaussians in the place of the Kumaraswamy components. Copula, encoder trunk, eigen-decoder,
calibrated likelihood, gamma = 1, warm-up and training are inherited from `review29oct_fixed` (`GCGMix_model.py` subclasses
`My_CopulaKSVAE_withEigen`). At the same K it has the same number of trainable parameters as the KS model.

**Why it is possible in the same way as the Kumaraswamy mixture.**

* No rejection sampling: the CDF of a truncated Gaussian is closed-form, (Phi(xi) - Phi(alpha)) / Z, with its own Z per component.
* The quantile of a mixture has no closed form for either family when K > 1. Both use bisection on the closed-form CDF, with the
  exact implicit gradient dx/dtheta = -(dF/dtheta)(x) / f(x), dx/du = 1 / f(x). No Monte Carlo approximation of the inverse CDF.
* Entropy: 0.5 log det R + Monte Carlo of the marginals with the copula path blocked, as the KS model.
* mu in (0, 1) and sigma in (1e-3, 5), as GC-TN, so that the normalisers stay >= 0.07 and float32 is safe.

Checks: densities integrate to 1; moments, CDF, density and quantile agree between TensorFlow (float32) and NumPy; the implicit
gradients with respect to mu, sigma, w and u agree with finite differences (relative error about 1e-6).

Best member of each family on 229 exact marginals (a 25 % stratified subset of `model_comparison/marginal_shape_study.py`),
KL(p || q) gap to the best K = 5 Kumaraswamy mixture, in nats per marginal, no encoder involved: Kumaraswamy K = 1, 2, 3: 0.0288,
0.0043, 0.0012; one truncated Gaussian: 0.0075; truncated-Gaussian mixture K = 2, 3, 5: 0.0012, 0.0003, -0.0006. This measures what
each family can represent, not what a trained encoder reaches.

`posterior_families.GMixCopula` makes post-processing, posterior quality and the comparison tools work as for the other models.

## Commands (server, from the project root, environment active)

Train to convergence, the rule of `model_comparison/run_convergence_gpu.sh`. Training, post-processing and posterior quality:

```bash
nohup bash MODULES/variant_GaussCopula_GaussMixMarginals/run_GCGMix_gpu.sh --gmix 3 --obs fv --gpu 0 > GCGMix3_fv.log 2>&1 &
nohup bash MODULES/variant_GaussCopula_GaussMixMarginals/run_GCGMix_gpu.sh --gmix 5 --obs fv --gpu 0 > GCGMix5_fv.log 2>&1 &
```

`--obs f`: frequencies only. `--protocol fixed`: 1500 epochs with the cosine schedule, comparable with the `Ep1500` runs of the
other models. Clean time per epoch (idle GPU):

```bash
bash MODULES/model_comparison/run_convergence_gpu.sh --family gmix --gmix 3 --gpu 0 --timing
```

Results: `Output/variant_GaussCopula_GaussMixMarginals/GCGMix<K>_03Nov-FINAL5modes_7D_obs-<obs>_Gamma1.0_Conv6000/`, with the same
layout as the proposed model. `model_comparison/compare_convergence.py` and `compare_models.py` read them.
