# Baseline 1 (R1.4): Gaussian copula + truncated-Gaussian marginals (Ref. 41 / submitted version)

q(z | m) = c_R(F_1(z_1), ..., F_7(z_7)) prod_d TN(z_d; mu_d, sigma_d, [0.45, 1]): one truncated Gaussian per
parameter, as in the submitted manuscript (`MODULES/COPULAS/GC_GMm_*`: location inside the support, bounded scale,
truncation to the prior support), joined by the same Gaussian copula as the proposed model.

**Only the marginals differ from the proposed method.** The model (`GCTN_model.py`) subclasses
`review29oct_fixed.GC_KSmix_models.My_CopulaKSVAE_withEigen` and inherits its eigen-decoder, calibrated likelihood and
training steps; the training (`main_train_GCTN.py`) uses `review29oct_fixed/training_common.py` (same data, gamma = 1,
warm-up, optimizer, schedule, early stopping on the validation -ELBO, 6 samples per observation). The submitted
version's loss (Huber + MAC terms, gamma = 0.30) is NOT used: it is not a likelihood, so its intervals cannot be
compared with those of a Bayesian posterior.

* Sampling: closed-form truncated-normal quantile of the copula uniforms (exact reparameterisation, no rejection).
* Entropy: 0.5 log det R (copula) + closed-form entropy of each truncated Gaussian.

```bash
# server, from the project root (training + post-processing + posterior quality), GPU 2:
nohup bash MODULES/baseline_GaussCopula_GaussMarginals/run_GCTN_gpu.sh --gpu 2 --modes 5 > baseline_GCTN.log 2>&1 &
```

Results: `Output/baseline_GaussCopula_GaussMarginals/GCTN_03Nov-FINAL5modes_7D_obs-fv_Gamma1.0_Ep1500/` with the same layout
as the proposed model (`TestA_same_FE_model_10el/`, `TestB_refined_FE_model_50el/`, `Posterior_quality/`).
