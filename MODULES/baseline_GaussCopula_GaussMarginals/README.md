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

## Variant GC-LogN: one Gaussian in ln z

`GCLogN_model.py`, `main_train_GCLogN.py`: the same encoder, copula, likelihood and training, with y = ln z ~ N(m, s^2) truncated to
[ln 0.45, 0] (a truncated log-normal stiffness factor). Quantile and entropy remain closed-form (H[z] = H[y] + E[y]); no bisection.
The location m may lie up to 3 s outside the support, so the density can keep rising up to a bound.

Best member of each family on the 917 exact marginals of test set A (maximum likelihood, no encoder), W1 to the exact marginal: truncated
Gaussian, mu inside the support 0.0032, mu free 0.0028; Gaussian in ln z 0.0012, which is at the noise of the 4000-particle reference
(about 0.001). Joint E_p[log q] per scenario with the best copula: 0.049 nats above the truncated Gaussian. These numbers say what each
family can represent, not what a trained encoder reaches.

`--hidden "H ..."` (also for GC-TN) changes the hidden layers of the encoder; the default is the trunk of the other models.

```bash
nohup bash MODULES/model_comparison/run_convergence_gpu.sh --family gctn --obs fv --gpu 0 > conv_GCTN_fv.log 2>&1 &
nohup bash MODULES/model_comparison/run_convergence_gpu.sh --family gclogn --obs fv --gpu 1 > conv_GCLogN_fv.log 2>&1 &
nohup bash MODULES/model_comparison/run_convergence_gpu.sh --family gclogn --hidden "256 512 256" --obs fv --gpu 2 > conv_GCLogN_H256_fv.log 2>&1 &
```

Results: `Output/baseline_GaussCopula_GaussMarginals/GCLogN_03Nov-FINAL5modes_7D_obs-fv_Gamma1.0_Conv6000[_H256-512-256]/`.
