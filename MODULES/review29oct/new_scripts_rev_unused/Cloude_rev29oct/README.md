# Cloude_rev29oct: PI-GC-KSmix-VAE for the beam on elastic bearings (CPENG-8674 revision)

Self-contained folder: code, `Data/` and `Output/` live here. It builds on the scripts in
`MODULES/review29oct` (kept unchanged) and keeps their structural model, measurement noise,
DoE proportions, architecture and benchmark scenarios. It removes every step that made the
estimated posterior or the "ground truth" depend on anything other than the data, the prior
and the physics.

## Files

| file | replaces (in `MODULES/review29oct`) | content |
|---|---|---|
| `config.py` | settings spread over the scripts | every setting (structure, noise, prior, DoE, datasets, training) |
| `dataset_generation.py` | `dataset_generation.py` | FE model, DoE, datasets, data figures |
| `priors.py` | Beta prior inside `GC_KSmix_models.py` | scenario-mixture prior p(θ) and DoE ρ(θ) |
| `preprocessing_and_eigen.py` | `preprocessing_and_eigen.py` | affine modal model (NumPy and TF), likelihood, data loading |
| `GC_KSmix_functions.py` | `GC_KSmix_functions.py` | Gaussian copula + Kumaraswamy mixture: density, CDF, quantile, sampling |
| `GC_KSmix_architectures.py` | `GC_KSmix_architectures.py` | encoder |
| `GC_KSmix_models.py` | `GC_KSmix_models.py` | PI-GC-KSmix-VAE and its ELBO |
| `main_KS_rev29oct.py` | `main_KS_29Oct.py` | training, model selection on validation, predictions |
| `reference_posterior.py` | MCMC in `postprocessing_7D_bridge.py` | reference posterior (SMC), importance-sampling refinement |
| `postprocessing_7D_bridge.py` | `postprocessing_7D_bridge.py` | metrics and figures |
| `select_hyperparameters.py` | — | β (γ²) and K selection on the validation set |
| `check_implementation.py` | — | numerical self-checks |

## Model and data

**Kept from your scripts.**
- **Beam.** 10 m span, E = 210 GPa, I = 8.33e-6 m⁴, ρ = 7850 kg/m³, A = 0.005 m². Euler–Bernoulli elements with consistent mass and rotary inertia.
- **Bearings.** k_v0 = 1.5e8 N/m and k_r0 = 5e6 N·m/rad at each end; the bearing factor multiplies both springs.
- **Parameters.** 5 zones plus 2 bearing factors, all in [0.45, 1.0]. Sensors at the 6 zone-boundary nodes measure vertical and rotational components; 5 modes are used.
- **Noise.** 1 % on frequencies; mode shapes 2.5 % (vertical) and 6.5 % (rotational) of the peak amplitude.

**Inference (training) model: 2 elements per zone (22 DOFs).** The paper's 1 element per zone is still available via `COARSE_EL_PER_ZONE = 1`. Reason (healthy beam vs the 50-element model):

| frequency error vs 50 elements | mode 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1 element/zone | 0.04 % | 0.32 % | 1.14 % | 2.36 % | **11.8 %** |
| 2 elements/zone | 0.00 % | 0.02 % | 0.08 % | 0.22 % | 0.47 % |

With 1 element per zone, the remaining error after the healthy-baseline correction is still 0.5–0.7 % on modes 4–5 when Z1 = 0.6, i.e. half the noise level. With 2 elements per zone it is 0.02 %.

**Test models (Reviewer 1, comments 1–2; Reviewer 3, comment 1).** 10 elements per zone (50 elements), the same zones and sensors, element mass factors 1 + N(0, σ_m), and 2.5 % within-zone stiffness variability.
- The variability is applied to the flexibility with zero mean, so the zone label is exactly the equivalent (harmonic-mean) stiffness.
- σ_m = 0 %, 2 % (your value) and 5 % give a mass-error sensitivity study.
- `*_bc` results use the healthy-baseline frequency correction (your calibration). It is reported next to the raw result, never instead of it.

**Prior p(θ) vs DoE ρ(θ) (Reviewer 3, comment 3; Reviewer 1, comment 6).**
- **Structure.** Both are mixtures over damage scenarios: healthy, single zone, single bearing, zone + bearing, two zones, any three parameters, and broad.
- **Damaged-parameter shapes.**

  | | damaged parameter |
  |---|---|
  | prior | Beta(2.2, 1.1) (your prior shape: mild damage favoured) |
  | DoE | uniform severity |

  Healthy parameters use Beta(2, 1) on [0.94, 1.0] in both (mean 0.98, your U(0.96, 1.0) made continuous).
- **Scenario weights.** Prior: 50 % healthy, 25 % single zone, 10 % single bearing. DoE: your proportions (12/33/18/22/15 %) plus 3 % broad.
- **Training DoE.** Stratified (exact counts per scenario and location, Latin hypercube in severity). Each location gets about 6,500 damaged training samples spread evenly over severities.
- **Why decoupling works.** The training loss never uses labels, so the DoE only decides where the encoder is trained. The prior enters only through the ELBO. This is the answer to Reviewer 3, comment 3.

## Objective

    loss = −τ · E_q[log p(y | θ)] + β · KL(q(θ | y) ‖ p(θ))

- **Likelihood.** p(y | θ) is the Gaussian measurement model that generated the data:
  - log-frequencies with σ_f = 0.01;
  - mode shapes with per-mode σ (2.5 % / 6.5 % of the healthy peak);
  - the eigenvector sign is marginalized: log[½N(y; φ) + ½N(y; −φ)].
- **β = 1, τ = 1** gives the exact negative ELBO, whose minimizer is the Bayesian posterior. β is the paper's γ² and is chosen on the validation set.
- **Likelihood tempering.** τ grows from 0.01 to 1 during the first 100 epochs. Early stopping and model selection use τ = 1 and the validation negative ELBO only.
- **Optional amortized term.** `w_fwd > 0` adds the forward-KL term with the labels, importance-weighted by p/ρ (bounded, see check 8). It is off by default.

## Changes to your scripts, and why

| # | in `review29oct` | problem | here |
|---|---|---|---|
| 1 | benchmarks 8–9: observations blended 60/40 with the mirrored structure's data | fabricated data: no physical structure produces it | every observation is a plain noisy simulation |
| 2 | "ground truth" = 38-step MCMC chains started at θ* and its mirror, weights exp(0.25 Δlog p), prior × 0.08, β ≥ 0.22, different rotation weight for "bimodal" cases | it uses the truth and is not the posterior the VAE targets | SMC from the prior with exactly the training prior and likelihood; θ* is never used |
| 3 | "estimated posterior" = VAE samples pooled with ground-truth samples, reweighted with exp(0.65 Δlog p) | circular: the estimate contains the reference | q is plotted as it is; the importance-sampling refinement uses only q and is labelled separately with its ESS |
| 4 | repulsion loss between component modes; weight-entropy bonus | forces multimodality and diversity whether or not the posterior has it | removed |
| 5 | KL clipped to [−40, 60]; dy clipped to norm 2, pdf to [0.01, 100], gradients to ±15 inside the quantile gradient | biased gradients | exact implicit gradients (checked against finite differences) |
| 6 | loss = MSE of log-f standardized by the dataset spread + 1.5·MAC_v + 0.5·MAC_r + γ²·KL | arbitrary likelihood temperature, so coverage cannot be calibrated | Gaussian likelihood with the noise levels that generated the data |
| 7 | prior Beta(2.2, 1.1) in training, Beta(1.8, 1.05) × 0.08 in the "ground truth" | the two posteriors are not the same object | one prior everywhere |
| 8 | mode sign set by the largest \|vertical\| component | for antisymmetric modes the two peaks are equal, so noise flips the sign at random (a left/right mirror in the encoder input) | aligned with the healthy mode shapes; sign marginalized in the likelihood |
| 9 | 1.5 % mass perturbation in the training data, nominal mass in the physics decoder | unmodelled error inside the training likelihood | training data from the inference model; mass error tested at 0/2/5 % |
| 10 | stiffness jitter clipped at 1.02 | biased labels | zero-mean flexibility jitter, exact equivalent label |
| 11 | DoE severities in [0.48, 0.88], healthy in [0.96, 1.0] | mild damage (0.88–0.96) never in the training data | uniform severities over [0.45, 1.0] |
| 12 | a, b ≥ 1.15 | the density must vanish at θ = 1, where healthy posteriors concentrate | a, b ≥ 1 |
| 13 | copula u clipped to [1e-5, 1 − 1e-5]; float32 eigensolver | tail truncation; round-off comparable to noise | tail-accurate evaluation; float64 |
| 14 | `CUDA_VISIBLE_DEVICES = "2"` by default | machine-specific | `--gpu N` only when given |

## Workflow

Run each script from this folder (Spyder: open the file and press F5). From the repository root, use `python MODULES/Cloude_rev29oct/<script>.py`.

**(a) Data** (about 2 min):
```bash
python dataset_generation.py
```
Then run the self-checks:
```bash
python check_implementation.py
```
**(b) Training** (about 11 s/epoch on a 16-thread CPU, i.e. about 75 min for 400 epochs; on a GPU node add `--gpu 2`):
```bash
python main_KS_rev29oct.py
```
Options: `--epochs`, `--beta`, `--num_KS`, `--n_modes 3` (Reviewer 3, comment 4), `--obs fv` or `--obs f` (fewer measured quantities), `--n_samples` (H, Reviewer 3, comment 7), `--w_fwd 1` (amortized term), `--quick` (smoke test). In Spyder, edit `OVERRIDES`.

Validation-only selection of β and K (Reviewer 3, comment 2):
```bash
python select_hyperparameters.py --beta 0.5 1 2
```
**(c) Post-processing.** Reference posteriors take 15–55 s per case (4000 particles) and are cached in `Data/reference_posteriors/`:
```bash
python postprocessing_7D_bridge.py --run PIGCKS_fvr5_K3_beta1_H4_ep400
```
Outputs go to `Output/<run>/figures` and `Output/<run>/metrics`: calibration curves, the false-positive rate on intact parameters and ROC, location × severity maps, mass-error sensitivity, and estimated vs reference posteriors for the 10 benchmarks (1-D marginals with the exact q density, active-subspace contours with 50/90 % mass, full 7×7 corner, profile along the beam). Metrics include W1, energy distance, correlation error and IS ESS.

## Results of the default run (`Output/PIGCKS_fvr5_K3_beta1_H4_ep400`)

**Training.** 400 epochs, 122 min on CPU (Intel, 16 threads; part of the time shared with the reference sampler). The best validation −ELBO was at epoch 398. Train / validation −ELBO: −142.81 / −142.79 (no overfitting).

**Test sets.** Prior-distributed except where noted. "IS" = importance-sampling refinement, 500 samples per case.

| set | RMSE | 95 % coverage zones / bearings | q + IS | false alarms, healthy cases |
|---|---|---|---|---|
| `test_coarse` (inference model) | 0.015 | 0.912 / 0.934 | 0.919 / 0.927 | 0.8 % |
| `test_fine` (50 el., 2 % mass) | 0.018 | 0.884 / 0.921 | 0.884 / 0.920 | 1.6 % |
| `test_fine_bc` | 0.018 | 0.892 / 0.923 | 0.887 / 0.919 | 1.1 % |
| `test_fine_mass0` / `mass5` | 0.016 / 0.025 | all parameters: 0.918 / 0.811 | — | — |
| `ood_four_sites` | 0.034 | 0.897 / 0.871 | — | — (no healthy cases) |
| `ood_noise2` | 0.032 | 0.767 / 0.839 | — | 16.8 % |

**Detection** (θ < 0.90). AUC is 0.999 for zones and 0.98–0.99 for bearings. Only 0.1–0.2 % of intact parameters have a 95 % upper bound below 0.94.

**Location × severity** (`test_fine_balanced`). Damage with θ < 0.85 is detected and localized at every location, bearings included, in essentially 100 % of cases. In the 0.85–0.94 band about 50 % (zones) and 20 % (bearings) are detected: the realistic prior favours healthy states, which is the price of the low false-alarm rate.

**Benchmarks vs the reference SMC posterior.**
- **Agreement.** Energy distance is 0.0004–0.012 and the largest per-parameter Wasserstein-1 distance is 0.002–0.028. The worst case is P7 (five damaged sites), a configuration the prior considers rare.
- **Spread.** q is wider than the reference on damaged parameters (std ratio 1.0–2.6, typically 1.3–1.6) and within 0.7–1.3 on intact ones.

**Limitations to report.**
1. **q is slightly over-confident on intact parameters.** Coverage is 0.92 instead of 0.95, which is typical of reverse-KL variational inference. With 5000 importance samples per case (300 `test_coarse` cases), coverage rises to 0.939 / 0.932. That confirms the exact posterior is calibrated, and the gap is the variational approximation.
2. **Unmodelled errors reduce coverage.** 5 % mass error and doubled noise give overconfident posteriors. They are not in the likelihood, by design of the test.
3. **These posteriors are unimodal.** With frequencies plus vertical and rotational mode shapes, the 10 benchmark posteriors have one mode each, in the reference too. Symmetric (Z1 ↔ Z5, BL ↔ BR) multimodality appears when the data cannot separate mirror-image damage, e.g. `--obs f` (frequencies only). See `Data/figures/Fig_identifiability_CRB`.

## Reviewer comments covered

| comment | where |
|---|---|
| R1.1, R1.7 model-form error, finer mesh | `test_fine*` (50 elements), discretization table above, `Data/figures/Fig_model_form_error` |
| R1.2, R3.1 other uncertainty sources | `test_fine_mass0/mass5`, `ood_noise2`, `Fig_mass_sensitivity` |
| R1.3 coverage | `Fig_calibration`, `metrics/coverage.csv` (exact marginal intervals) |
| R1.5 distribution shift | `ood_four_sites`, `ood_noise2` |
| R1.6 healthy / mild / localized damage, false positives | prior, DoE, `Fig_detection_ROC`, `fpr_interval`, `Fig_location_severity_*` |
| R2.2–R2.4 bearings, support damage | bearing factors BL, BR |
| R3.2 hyperparameters on validation | `select_hyperparameters.py`, early stopping on validation −ELBO |
| R3.3 prior vs DoE | `priors.py`, label-free objective |
| R3.4 number of modes | `--n_modes`, `Data/figures/Fig_identifiability_CRB` |
| R3.7 H, R3.8 cost | `--n_samples`, `Output/<run>/timing.json` |
