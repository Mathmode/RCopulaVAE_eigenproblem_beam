# Model comparison (R1.4) and number of Kumaraswamy components

All models below use the **same** data set, eigen-decoder, calibrated likelihood, uniform prior, ELBO with gamma = 1,
encoder trunk (128-256-128 ReLU), warm-up, optimizer, learning-rate schedule, early stopping on the validation -ELBO,
and the **same post-processing code** (`review29oct_fixed/postprocessing.py` through `posterior_families.py`). They
differ only in the family of the approximate posterior q(z | m):

| Model | Folder | Family of q(z \| m) |
|---|---|---|
| Proposed (K components) | `MODULES/review29oct_fixed` | Gaussian copula + mixtures of K Kumaraswamy distributions |
| Baseline 1 (Ref. 41, submitted version) | `MODULES/baseline_GaussCopula_GaussMarginals` | Gaussian copula + one truncated Gaussian per marginal |
| Baseline 2 (Ref. 39) | `MODULES/baseline_GaussMixture_K5` | Gaussian mixture, K = 5 full-covariance components (logit space of the box), no copula |

## Files

* `evaluate_posterior_quality.py`: for every observation of the validation set and of test sets A and B, the -ELBO,
  the importance-weighted ELBO, a lower bound of KL(q || exact posterior) and the Pareto-k diagnostic, with the
  likelihood of training and of the SMC reference. Writes `<run>/Posterior_quality/`. Also writes the calibration
  metrics of the validation set (`Posterior_quality/Validation_set/`).
* `compare_models.py`: selection of K (validation set only) and the R1.4 table/figures in
  `Output/model_comparison/<data set>/`.
* `run_KS_components_gpu.sh`: trains K = 1..5 for the given seeds (skips existing runs) and evaluates them.

## How K is selected (answer for the reviewers)

* Because the evidence log p(m) does not depend on q, **-ELBO = KL(q(z|m) || p(z|m)) - log p(m)**. On a fixed set of
  observations, the difference between the mean -ELBO of two models is the difference between their mean KL
  divergences from the **exact** Bayesian posterior (nats per observation). No reference samples are needed.
* K = 1, ..., 5 are trained with identical settings (two random seeds each) and evaluated on the **validation set**
  (2 400 observations, 500 posterior samples each). The test sets are never used for the selection (cf. R3.2).
* **Selected K = smallest K whose validation -ELBO is within one standard error of the best K** (one-standard-error
  rule, Hastie, Tibshirani and Friedman 2009, Sec. 7.10). The standard error is the larger of the paired standard
  error over observations and the seed-to-seed spread. This favours the most parsimonious model among those that are
  statistically indistinguishable.
* Cross-checks: Pareto-k diagnostic of the importance weights (Yao et al. 2018; Vehtari et al. 2024: k < 0.7 means
  that q covers the posterior mass) and calibration (MACE, 95 % coverage) on the validation set.

References: Hastie T, Tibshirani R, Friedman J (2009) The Elements of Statistical Learning, 2nd ed., Springer.
Yao Y, Vehtari A, Simpson D, Gelman A (2018) Yes, but did it work?: Evaluating variational inference. ICML, PMLR 80.
Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J (2024) Pareto smoothed importance sampling. JMLR 25(72).
Kucukelbir A, Tran D, Ranganath R, Gelman A, Blei DM (2017) Automatic differentiation variational inference. JMLR 18(14).
Morningstar W, Vikram S, Ham C, Gallagher A, Dillon J (2021) Automatic differentiation variational inference with
mixtures. AISTATS, PMLR 130.

## Commands (server, from the project root, environment active)

```bash
nohup bash MODULES/baseline_GaussCopula_GaussMarginals/run_GCTN_gpu.sh --gpu 2 --modes 5 > baseline_GCTN.log 2>&1 &
nohup bash MODULES/baseline_GaussMixture_K5/run_GMM_gpu.sh --gpu 2 --modes 5 --components 5 > baseline_GMM5.log 2>&1 &
nohup bash MODULES/model_comparison/run_KS_components_gpu.sh --gpu 2 --seeds 1234 --ks "1 2 4 5" --modes 5 > ks_seed1234.log 2>&1 &
nohup bash MODULES/model_comparison/run_KS_components_gpu.sh --gpu 2 --seeds 2024 --ks "1 2 3 4 5" --modes 5 > ks_seed2024.log 2>&1 &
```

(K = 3 with seed 1234 is the definitive run, trained and evaluated by `review29oct_fixed/run_FINAL_case_study.sh`.)
When all have finished, copy the
run folders to the PC and run:

```bash
python MODULES/model_comparison/compare_models.py --data 03Nov_FINAL_GirderBearings_7D_5sensors_5modes
```
