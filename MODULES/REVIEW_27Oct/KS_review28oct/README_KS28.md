# KS_review28oct: extended case study, extended dataset and multimodal copula VAE

This is new code for the major revision of MS CPENG-8674. It does not modify anything
in `MODULES/KUMARSWAMY`, `MODULES/REVIEW_27Oct` or the ANTIGRAVITY `review_27oct` folder.
Import it from the project root, for example:
`from MODULES.KS_review28oct.ks28_dataset import ...`.

```
MODULES/KS_review28oct/
  ks28_config.py           BridgeConfig / NoiseConfig / PriorConfig (all numbers of the problem)
  ks28_fe_model.py         NumPy FE engine: beam on elastic bearings, zones, sensors, "truth" models
  ks28_priors.py           scenario prior: sampler + exact log-density
  ks28_likelihood.py       likelihood from the noise model (+ model-discrepancy estimate)
  ks28_dataset.py          design of experiments, dataset folder, loading, input scaling
  ks28_dataset_plots.py    overview figures of a dataset
  ks28_tf_physics.py       TF modal decoder (affine K, float64 eigh, no trainable physics)
  ks28_tf_distributions.py TF mixture of Gaussian copulas with Kumaraswamy marginals + TF prior
  ks28_tf_model.py         encoder, ELBO / SIWAE objective, posterior export
  ks28_train.py            training loop (local and GPU), early stopping on validation
  ks28_q_numpy.py          NumPy version of q (used by postprocessing and to check the TF code)
  ks28_ground_truth.py     reference posterior by tempered SMC (same likelihood and prior)
  ks28_postprocess.py      metrics, calibration, detection rates, VAE vs ground-truth figures
  ks28_smoke_test.py       TF vs NumPy consistency checks + timing (run this first)
main_KS28_A_generate_datasets.py   (project root) step A
main_KS28_B_train.py               (project root) step B
main_KS28_C_postprocess.py         (project root) step C
```

---------------------------------------------------------------------------------------
## 1. What was wrong in the current KUMARSWAMY code, and what changes

| # | Current code | Effect | New code |
|---|---|---|---|
| 1 | `Joint_copula_dens_term` returns `gamma^2 * (-E[log q])` and adds it to the loss. The paper's Eq. (loss) has `+gamma^2 * E[log q]`. | The loss rewards **low** entropy, so q is pushed towards a spike. That is consistent with the undercoverage (0.77-0.79 for nominal 95 %, R1.3). The diversity and boundary penalties were patches for this. | Proper objective: `log p(m|θ) + log p(θ) − log q(θ|m)`. No extra penalties. |
| 2 | The custom gradient of the Kumaraswamy quantile returns `None` for the copula samples `u`. | The copula correlation `L` never receives a gradient from the frequency/MAC terms. | K=1: closed-form quantile. K>1: bisection + implicit reparameterisation. Gradients reach `a, b, w, u` and hence `L`. The smoke test checks this. |
| 3 | One Gaussian copula with bimodal marginals. | It cannot represent "zone 2 **or** zone 5 damaged" (mirror solutions of a symmetric bridge). It puts mass on "both damaged" and "neither damaged". | **Mixture of C Gaussian copulas**, each with its own Kumaraswamy marginals and correlation. C=1, K>1 is the paper model. Trained with the stratified ELBO (or SIWAE), so the mixture weights get exact gradients. |
| 4 | `L_inv` (the mass matrix) is a trainable `tf.Variable` in `Solve_eigenproblem`. | The network can change the physics. | Constant mass matrix. |
| 5 | Eigenproblem in float32. | For a realistic stiffness scale (10^9-10^12) the eigenvalue spread is ~10^8, and float32 loses the first eigenvalues. | float64 eigensolver (`physics_dtype`). |
| 6 | Loss weights 5, 10 and γ, with γ chosen on the **test** set (R3.2). | Posterior width is arbitrary, and the test set is not independent. | The likelihood comes from the noise model: `Σ(log f_obs − log f)^2/2σ_f^2 + Σ ||φ_obs||^2 (1−MAC)/2σ_φ^2`. The MAC form is exactly the scale-marginalised Gaussian mode-shape likelihood. β=1 (β<1 only as warm-up). Model selection uses the validation set only. |
| 7 | The "VAE posterior" plots reweight the VAE samples by the likelihood. The ground truth uses the training γ. | The figure shows roughly q × likelihood, not q. The comparison is circular. | The VAE plot shows **q itself**. The ground truth (SMC) uses the same likelihood and prior, and does not depend on any hyper-parameter. |
| 8 | Rejection loop keeps `u < 0.99`. | Truncates the copula, so the samples are not from q. | Not needed: Kumaraswamy marginals live on [0,1]. |

---------------------------------------------------------------------------------------
## 2. Extended case study (point 1)

- **Structure:** 20 m single-span deck (equivalent Euler-Bernoulli beam), E = 34 GPa, I = 0.90 m^4, m = 14 t/m.
  - f1 ≈ 5.8 Hz if pinned, consistent with f1 ≈ 100/L.
- **Elastic bearings at both abutments (R2.2, R2.3):** a vertical spring kv and a rotational spring kr per support. **No DOF is fixed.**
  - kv = 1500 EI/L^3 ≈ 5.7e9 N/m and kr = EI/L.
  - With these bearings the frequencies are 6.68 / 22.8 / 46.4 / 73.2 / 102.5 Hz. A pinned beam would give 5.8 / 23.2 / 52.3 / 92.9 / 145 Hz, so the bearings matter.
  - The bearing condition is **inferred**: `eta_L`, `eta_R` (log2 factors, k = 2^eta k_nom, range ×0.25 to ×2). This covers softening/deterioration and stiffening/seizure.
- **Damage:** 6 zones (even, so there are symmetric pairs 1-6, 2-5, 3-4). The decoder mesh has 2 elements per zone.
- **Sensors:** 7 vertical accelerometers at the zone boundaries, including both bearings. Rotations are not measured.
- **Observations:** 5 modes.
- **Latent vector:** θ = [z1..z6, eta_L, eta_R] (D = 8).
- **Sensitivity:** 10 % damage in one zone changes the frequencies by 0.05-1.5 %. A 30 % bearing softening changes them by 2-5 %. Zone pairs 1/6, 2/5 and 3/4 have identical frequency sensitivities, so with frequencies only the posterior is bimodal (see `figures/fig_reference_posterior_example.png`).
- **Other variants** in `main_KS28_A_generate_datasets.py`:
  - `paper_noise` (σ_f = 2.5 %, σ_φ = 5 %)
  - `dense_sensors` (13 sensors)
  - `fixed_bearings`
  - For scalability (R1.7), set e.g. `BridgeConfig(n_zones=12)`.

## 3. Extended dataset (point 2): design of experiments

**Scenario prior** (`PriorConfig`). This is the training design and also the prior in the ELBO and in the ground truth:

- Number of damaged zones: P = 0.30 / 0.44 / 0.15 / 0.06 / 0.03 / 0.015 / 0.005 for 0…6 zones.
- Healthy zones: z ∈ [0.97, 1] ("as-built" scatter).
- Damaged zones: z = 0.5 + 0.5·Beta(2, 1.2). Mild damage is favoured and severe damage is penalised (R3.3).
- Bearings: nominal ± 7 % (σ_eta = 0.1), anomalous with p = 0.1.
- P(all zones healthy) = 0.33. Under the old U[0.5,1]^6 it was 5e-8.
- The density is evaluated exactly (dynamic programme over the subsets of damaged zones).

**Training and validation designs:** scenario prior plus 10 % uniform box. The uniform share only widens what the encoder has seen; the ELBO prior is still the scenario prior. This answers R3.3: the data density and the Bayesian prior are decoupled.

**Test sets.** Families marked with ★ come from a **different "truth" model** than the decoder: 4× finer mesh, plus the listed perturbation. 1000 cases each.

| family | what | reviewer |
|---|---|---|
| id_prior | scenario prior, training model | baseline |
| healthy | no damage, nominal bearings → false positives | R1.6 |
| single_mild / single_moderate | one zone, z ∈ [0.80,0.95] / [0.60,0.80] | R1.6 |
| double | two damaged zones | R1.6 |
| bearing_anomaly | healthy deck, one bearing softened/stiffened | R2.2, R2.3 |
| uniform_box | design of the submitted paper | comparability |
| ood_severe | z ∈ [0.30, 0.48], below the training bound | R1.5 |
| fine_mesh ★ | discretisation error only | R1.1 |
| mass_perturbed ★ | random element mass field, CoV 5 % | R3.1 |
| temperature ★ | global E × U[0.96, 1.04] | R1.2 |
| kr_mismatch ★ | rotational bearing stiffness × U[0.5, 2] | R2.2 |
| crack_localized ★ | 1-2 fine elements of one zone lose 30-70 %; label = equivalent zone factor | R2.3 |
| combined_shift ★ | mass 3 % + temperature 2 % + kr × U[0.7,1.4] | all |
| val_shift ★ | like combined_shift but a **validation** set: model-discrepancy calibration | R3.1, R3.2 |

What the shifted families do to the data (`figures/fig_model_form_error.png`):

- The fine mesh changes almost nothing (the 2-element decoder is already converged). That is an honest answer to R1.1.
- Mass and temperature perturbations act at about the noise level.
- The rotational restraint of the bearings is the dominant model-form error (up to ±5 % on f1).

---------------------------------------------------------------------------------------
## 4. Steps

### Step 0: check the TF code on each machine (1 min)
```
python -m MODULES.KS_review28oct.ks28_smoke_test --data Data/KS28_Bridge20m_6Z_bearingsLatent_7S_sf1_sp3_main
```
All checks must print PASS. The timing line tells you whether to add `--eigh_cpu` in step B; on GPUs the small batched eigensolver is often faster on the CPU.

### (A) Create the dataset (NumPy only, ~1 min)
```
python main_KS28_A_generate_datasets.py                    # main dataset
python main_KS28_A_generate_datasets.py --variant paper_noise
```
- Output: `Data/<name>/`, containing `config.json`, `decoder_basis.npz`, `train/val/test_id/val_shift.npz`, `tests/*.npz`, `README_dataset.md` and `figures/`.
- Copy the folder to the GPU server as it is.

### (B) Train
```
python main_KS28_B_train.py --data Data/<name> --run test --epochs 2 --max_train 2000     # quick test
python main_KS28_B_train.py --data Data/<name> --run R1_C3K1_elbo --C 3 --K 1 --gpu 0       # main run
```
Suggested runs: see the docstring of `main_KS28_B_train.py`.
- R0 is the paper model (C=1, K=3).
- R1 is the proposed model (C=3).
- R2 uses SIWAE.
- R3 adds the model-discrepancy calibration.
- R4 uses the uniform prior.
- R5/R6 use fewer modes / frequencies only (R3.4).
- R7 uses independent copulas.

Output in `Output/KS28/<run>/`:
- `run_config.json`, including the hardware and the total training time (R3.8)
- `history.csv`
- `encoder_best.weights.h5`
- `posteriors/*.npz`: the parameters of q for every split. Step C needs only NumPy.

### (C) Postprocess (NumPy only, laptop)
```
python main_KS28_C_postprocess.py --runs Output/KS28/R1_C3K1_elbo --gt_cases 4
python main_KS28_C_postprocess.py --runs Output/KS28/R0_C1K3_elbo Output/KS28/R1_C3K1_elbo   # comparison table
```
Output in `<run>/postprocessing/`:
- **Metrics:**
  - `metrics.csv` per family: MAE, RMSE, 95 % coverage, MACE, CRPS, ALL, FPR (structure and zone), TPR, AUC, localisation accuracy, bearing TPR/FPR.
  - `metrics_calibration.png`: calibration curves.
  - `training_history.png`.
- **`gt_comparison/`:** ground truth (SMC) against the VAE posterior.
  - `*_corner.png`: GT as grey filled HPD contours, VAE as blue line contours, true value as a red star.
  - `*_violins.png`: split violins plus the damage probabilities from both.
  - `*_gt_comparison.csv`: energy distance, W1 per coordinate, max |ΔP(damage)|.

The SMC particles are cached in `<run>/ground_truth/`. Each case takes ~10-20 s with 4000 particles.

---------------------------------------------------------------------------------------
## 5. Not covered here (still to do for the response letter)

- **R1.4 baselines:** reference 39 (GMM autoencoder) and reference 41 (data-driven GCVAE) need their own runs on this dataset.
  - R0 (C=1) and R7 (independent copula) are ablations, not those baselines.
- **R2.4 and the editor:** laboratory or field data.
- **R3.5:** a flowchart of the data generation (sections 2-3 of this README can serve as its content).
- **R1.7:** a run with `n_zones=12` for the scaling discussion. The decoder cost is one dense eigh of size 2·(n_el+1) per sample, and the copula has D(D−1)/2 parameters per component.
- **R1.8:** reference cleanup.
- **R3.7:** H is now 4 samples per component (12 per observation), not 1. Justify it with the validation curves.

## 6. What has been verified
- **NumPy parts (FE model, prior, dataset, likelihood, q, SMC, postprocessing):** executed and tested.
  - FE frequencies of the pinned limit match the analytical ones to 1e-6.
  - The prior density matches brute-force enumeration (1e-14) and integrates to 1.
  - The q sampler matches the q density.
  - SMC runs with different seeds agree.
- **TensorFlow parts:** written to mirror the NumPy reference, but **not executed yet**. TensorFlow could not be installed where they were written. `ks28_smoke_test.py` compares them against NumPy; run it before any training.
