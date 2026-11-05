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
| Variant (K components) | `MODULES/variant_GaussCopula_GaussMixMarginals` | Gaussian copula + mixtures of K truncated Gaussians |
| Variant | `MODULES/baseline_GaussCopula_GaussMarginals` (`GCLogN_model.py`) | Gaussian copula + one Gaussian in ln z (truncated log-normal) |

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

`compare_models.py` uses by default only the runs with the fixed cosine schedule and frequencies + shapes
(`--schedule`, `--obs` select others), so the runs below do not enter the K selection by accident.

## Train-to-convergence protocol (`run_convergence_gpu.sh`, `compare_convergence.py`)

Why: the runs above have a fixed budget of 1500 epochs, and one budget does not separate quality from cost. Besides, the
-ELBO logged by Keras on the validation set is noisy. At fixed weights its Monte Carlo error is 0.018 nats for GC-TN (its
entropy is analytic, so the error of E_q[NLL] is not compensated) against 0.006 for the KS model and 0.005 for the GMM. In
the 1500-epoch GC-TN run a dip of that noise at epoch 910 (recorded 23.800, true level about 23.85) stopped the training
at epoch 1210 and those weights were restored, at a learning rate of 1.7e-4.

Rule, the same for every family (`--schedule plateau`, `GC_KSmix_models.ConvergenceController`):

* learning rate 5e-4; warm-up and everything else as before;
* every 20 epochs after the warm-up the validation -ELBO is estimated with the numpy estimator of
  `evaluate_posterior_quality.py` (mean log w, 20 samples of q per observation, the same random numbers each time): Monte
  Carlo error 0.001-0.004 nats, 3-5 s per evaluation, not counted as training time. The best weights are kept;
* the learning rate is halved after 200 epochs without an improvement larger than 0.002 nats; at 1e-5 (the end of the
  cosine schedule) the same condition stops the training. `--max-epochs` is only a cap (default 6000; `converged` in
  `Problem_info.npy` says whether the rule was met).

`model_history` gets `epoch_time_s` (all runs from now on) and `monitor_neg_elbo`. Run names: `..._obs-<obs>_Gamma1.0_Conv<cap>`,
so they never overwrite the `Ep1500` runs. `--obs f` trains on the frequencies only, with the same data set.

```bash
nohup bash MODULES/model_comparison/run_convergence_gpu.sh --family ks --ks 3 --obs fv --gpu 0 > conv_KS3_fv.log 2>&1 &
nohup bash MODULES/model_comparison/run_convergence_gpu.sh --family gctn --obs fv --gpu 1 > conv_GCTN_fv.log 2>&1 &
nohup bash MODULES/model_comparison/run_convergence_gpu.sh --family gmm --components 5 --obs fv --gpu 2 > conv_GMM5_fv.log 2>&1 &
```

Clean time per epoch (one at a time, on a GPU with nothing else running; written to `Output/model_comparison/timing/`):

```bash
bash MODULES/model_comparison/run_convergence_gpu.sh --family ks --ks 3 --gpu 0 --timing
bash MODULES/model_comparison/run_convergence_gpu.sh --family gctn --gpu 0 --timing
bash MODULES/model_comparison/run_convergence_gpu.sh --family gmm --components 5 --gpu 0 --timing
```

Then, with the run folders together (`--obs f` for the frequency-only runs):

```bash
python MODULES/model_comparison/compare_convergence.py --data 03Nov_FINAL_GirderBearings_7D_5sensors_5modes --obs fv
```

`marginal_shape_study.py`: why more Kumaraswamy components help on unimodal posteriors (best fit of every family to the exact
marginals, no training).
