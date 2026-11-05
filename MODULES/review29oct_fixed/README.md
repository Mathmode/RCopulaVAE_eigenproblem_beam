# `MODULES/review29oct_fixed` — corrected PI-GC-KSmix-VAE pipeline (29 Oct 2026)

Everything in this folder is new. Nothing in `MODULES/review29oct`, `Data/BridgeBearings_7D_DoE`
or `Output/review29oct` was modified. The model is still **your** model:

- one Gaussian copula, with mixtures of K = 3 Kumaraswamy marginals;
- the same 128-256-128 encoder;
- the same 5-element beam on elastic bearings, the same 7 parameters and z ∈ [0.45, 1];
- the same noise levels;
- the same benchmark states z* (plus one extra).

The fixes concern the loss, the "ground truth", the plots and the data.

## 0l. Update (5 Nov, later): variant with truncated-Gaussian mixture marginals

`MODULES/variant_GaussCopula_GaussMixMarginals` (GC-GMix; `posterior_families.GMixCopula`): the proposed model with truncated
Gaussians in the place of the Kumaraswamy components, same quantile by bisection with implicit gradients. See its README. The
files changed by this are saved in `MODULES/_backup_05Nov_before_GCGMix/`.

## 0k. Update (5 Nov): train-to-convergence protocol and frequency-only runs

`--schedule plateau` in `main_train_KSVAE.py` and in the baselines (`training_common.py`,
`GC_KSmix_models.ConvergenceController`): the same convergence rule for every posterior family, with a low-noise monitor,
wall-clock per epoch in the history, and the learning rate halved on a plateau. The default (`--schedule cosine`) is what all the
runs so far used and does not change. Launcher, commands and reasons: `MODULES/model_comparison/README.md`
(`run_convergence_gpu.sh --family ks|gctn|gmm --obs fv|f`). The code before this change is in
`MODULES/_backup_review29oct_fixed_05Nov_before_convergence_protocol/`.

## 0j. Update (4 Nov): coverage of the exact posterior on the prior scenarios

`prior_scenarios_coverage.py`. The bearings of the 5-mode runs under-cover even on the 120 test structures drawn from
the prior (equal-tailed 95 % coverage 0.900, K = 3, test A), and the exact posteriors of the benchmarks also miss
bearings more often than deck zones (`Latex_docs/Results_analysis.tex`, Sec. 4.3). On prior draws the exact posterior
must cover at 95 % unless the likelihood is misspecified, so the script computes the exact (SMC) posteriors of those
120 structures (cached in `reference_posteriors/`, about 9 s each) and compares exact and VAE posteriors: coverage at
the 19 MACE levels with bootstrap standard errors over structures, PIT tails, agreement of the two posteriors, and the
data terms of the likelihood at z* for the structures whose bearings the exact posterior misses. About 3 min with 8
workers per test set.
```
python MODULES/review29oct_fixed/prior_scenarios_coverage.py --workers 8                     # test set A
python MODULES/review29oct_fixed/prior_scenarios_coverage.py --split test_fine --workers 8   # test set B
```
Results: `Output/review29oct_fixed/prior_scenarios_coverage_5modes/<test set>/` (README.txt lists the files). Default
run: the K = 3 run (with or without `5modes` in its name); `--runs` compares several runs (e.g. K = 1, 3, 5).

**Result (K = 1, 3, 5 runs, 4 Nov).** Test A: the exact posterior covers the deck zones at 0.957 +- 0.008 but the
bearings at 0.904 +- 0.021 (U-shaped PIT: too narrow); the VAE matches it (bearings 0.900, exact - VAE 0.004 +- 0.007).
So the bearing under-coverage comes from the likelihood. `shape_error_structure.py` (seconds) shows why: the shape term
assumes isotropic errors across sensors, but in modes 4-5 the as-built (mass-field) error is concentrated in one
direction (largest eigenvalue 42-44 % of the variance vs 26 % for noise), close to the bearings' shape signature, along
which the misfit variance is 1.6-1.7 x sig_v^2. Candidate fix (not tested): a full covariance of the shape residual per
mode. Test B: exact 0.905 (deck) / 0.842 (bearings), VAE the same within 0.02.

## 0i. FINAL case study and data set (3 Nov 2026) — the only data set in `Data/`

- **Structure**: one girder line of a short-span steel-concrete composite bridge (HEB 600 + 2.0 m x 0.22 m C30/37 slab,
  I = 5.25e-3 m4 in steel units, 1 800 kg/m, L = 10 m; f1 = 12.0 Hz, inside the EN 1991-2 range 8.0-16.9 Hz) on
  laminated elastomeric pads sized for its reaction (200 x 300 mm, 3 x 8 mm rubber, S = 7.5: kv = 6.6e8 N/m,
  kr = 8.0e5 N m/rad). The vertical springs carry 6-56 % of the modal strain energy, so bearing damage is identifiable
  (benchmark P4: z_bL = 0.52 +- 0.03 for z* = 0.55). No intermediate springs (single span, R2.2). The old academic
  beam (39 kg/m, f1 = 3.6 Hz) made the bearings effectively rigid.
- **Data**: `Data/03Nov_FINAL_GirderBearings_7D_5sensors_5modes` (version `03Nov-FINAL_...`): 5 sensors, 5 modes,
  noise 2.5 % / 5 %, mass scatter 5 % per 1 m, stiffness scatter 2.5 % per 0.2 m, balanced DoE (every parameter damaged
  in 33.2 % of the samples). Train/val/test A: 10-element model. **Test B: 50-element model with the damage of every
  damaged zone localized in a 0.2-2.0 m sub-region** (same equivalent stiffness z*; `fe_model.localize_zone_damage`):
  a genuine model-form discrepancy (frequency scatter 0.7-1.6 % from the localization alone; E[NLL]/expected = 1.15 for
  the frequencies and 1.0-1.3 for the shapes, vs 1.02 / 0.98-1.01 on test A).
- **Checks**: exact posteriors of the 11 benchmarks: z* inside the 95 % interval for 68/77 (A) and 69/77 (B)
  parameters; 2-epoch training smoke test on the final data.
- **Removed**: the 10 revision data sets (v1-v6, 01Nov) were deleted; the 12 data sets of the original paper were moved
  to `Data_archive_original_paper/`; all previous runs to `Output/_archive_before_03Nov_FINAL/`. The code before this
  change is in `MODULES/_backup_review29oct_fixed_03Nov_before_final_dataset/`.
- **Run**: `run_FINAL_case_study.sh --gpu N --ks K --modes M [--post]` (training of the proposed model, post-processing, posterior quality,
  localized scenarios, showcase); baselines and K study: `MODULES/baseline_*/run_*_gpu.sh`,
  `MODULES/model_comparison/run_KS_components_gpu.sh` (all point to the final data set).

## 0h. Update (2 Nov, later): baselines of R1.4 and selection of the number of KS components

- **Baselines in their own folders**, trained with this folder's data, decoder, likelihood, gamma = 1 and schedule
  (`training_common.py`), so that only the posterior family differs:
  `MODULES/baseline_GaussCopula_GaussMarginals` (Ref. 41 / submitted version: Gaussian copula + truncated Gaussians) and
  `MODULES/baseline_GaussMixture_K5` (Ref. 39: K = 5 full-covariance Gaussian mixture, no copula).
- `posterior_families.py`: the three posterior families in NumPy with one interface. `postprocessing.py` computes
  every metric through it, so all models are evaluated by the same code; the metrics of the 5-mode run are
  reproduced exactly (checked against the saved tables).
- `GC_KSmix_models.py`: hook `make_encoder` and `HEAD_ORDER` for the subclasses; `main_train_KSVAE.py` now also saves
  `Val_predicted_props.npy` (validation set, used for model selection) and the family in `Problem_info.npy`.
- Selection of K and comparison tables: `MODULES/model_comparison` (see its README).

## 0g. Update (2 Nov): definitive 5-mode case study, clear test-set folders, correlations, showcase

**Run folder layout** (also described in `RESULTS_GUIDE.txt` inside every run folder):
- `TestA_same_FE_model_10el/`: test set A, data from the same 10-element FE model as training;
- `TestB_refined_FE_model_50el/`: test set B, the same structures and noise computed with the refined 50-element model.

Each contains `Manuscript_Metrics.csv` (the five metrics of the submitted paper), `Supplementary_Metrics.csv`,
`Fig_reliability`, `Fig_calibration`, `Fig_copula_correlations` (+ `Copula_Correlations.csv`),
`Benchmarks_VAE_vs_reference/` and `Benchmarks_violins/`.

**`showcase_study.py`**: 11 curated scenarios (healthy, localized severe / mid-span / mild, adjacent and symmetric
double damage, bearing alone, bearing + end zone, multiple damage, short defects at mid-span and at the support), each
repeated with 30 independent as-built structures and noise draws (refined 50-element data). VAE posteriors for all
replicas, exact posteriors (SMC) for 5. Figures: posteriors, robustness (intervals of all replicas with detection and
false-alarm rates), correlation matrices and damaged pairs. `--replot` redraws the figures from the cached posteriors.

**Server script for the definitive case:** `run_5modes_case_study.sh <gpu> [post]` (training, post-processing,
localized scenarios, showcase; `post` skips the training). It sets `LD_LIBRARY_PATH` to the environment's libraries,
which fixes the matplotlib `CXXABI_1.3.15` error.

## 0f. Update (1 Nov): number-of-modes study, metrics of the paper, journal-quality figures

**Data sets** (v6 settings, 5 sensors at the zone centres for both the 10- and the 50-element models):
`Data/01Nov_BridgeBearings_7D_v6_5sensors_5modes`, `..._3modes` (and `..._2modes`), generated with
`dataset_generation.py --n-modes N --data-dir ... --version 01Nov-Nmodes_...`.

**Server run:** `run_01Nov_modes_study.sh <gpu>` trains, post-processes and runs the localized scenarios for 5 and 3
modes, one job after the other.

**Metrics of the paper** (`Manuscript_Metrics*.csv`, definitions stored in the JSON): MSE, MAE, ALL (per-parameter, as
the old scripts) and ALL_joint (joint density with the copula), coverage95 and MACE with exact equal-tailed intervals,
`_HPD` with highest-density intervals and `_gauss` with mean ± kσ (earlier draft), for all parameters, deck,
bearings, damaged, undamaged and prior scenarios. `Fig_reliability*.pdf` shows the three interval types.

**Figures:** journal style (Times-like serif, STIX math, embedded fonts) and every figure saved as PNG (300 dpi)
and vector PDF.

## 0e. Update (31 Oct, v6): 5 % mass scatter, no temperature code, paired test sets

The previous version is frozen in `MODULES/review29oct_fixed_v3` (data: `Data/BridgeBearings_7D_v5_5sensors_4modes`,
2 % mass scatter).

**Data.** `Data/BridgeBearings_7D_v6_mass5`. Sensors, modes, bearings, noise and DoE are those of v5.

**Temperature.** It is gone from the code: no `TEMP_E_STD` in `config.py` and no temperature argument in
`fe_model.py`. The v4 and v5 data already had it switched off (`temp_E_std = 0` in their `problem_config.json`); only
v3 (`Data/BridgeBearings_7D_realistic_v3`) included it.

**Modelling error: the measurements come from an as-built structure, the VAE only knows the nominal model.**
- Every sample (train, val, test, test_fine) is an as-built structure with its own random realisation of:
  - mass factors 1 + N(0, **5 %**) per 1 m segment (one decoder element; the five 0.2 m elements of the 50-element
    model inside a segment share the factor);
  - flexibility scatter N(0, 2.5 %) of 0.2 m cells inside every zone, with zero zone mean, so z keeps its meaning.
- The fields are defined on a mesh-independent grid (`fe_model.draw_asbuilt_fields`, `fields_on_mesh`). Before v6 the
  mass factors were drawn per element of each mesh, so the 50-element model had a smaller effective mass error than
  the 10-element model.
- The decoder of the VAE (`Mass_matrix.npy`, `L_inv.npy`, `Ke_matrix.npy`) and the reference posterior use the
  **nominal** model. The realisations are saved in `<data>/asbuilt_fields/` for analysis only and are never read by
  the VAE, the likelihood or the reference posterior.
- Only the statistical size of the error enters the likelihood. It is calibrated by simulation on the training
  generator with fresh random fields, as before.

**Paired test sets.** `test` and `test_fine` now contain the same as-built structures (z, mass and stiffness fields)
and the same measurement-noise draws, so they differ only by the FE model (10 vs 50 elements).

**Model-error budget.** `dataset_generation.model_error_budget` reports the size of every error source, saved in
`problem_config.json` (`model_error_budget`) and `figures/Fig_model_error_budget.png`. It is the mean and std of
log f_data − log f_decoder over the DoE, in %:

| source | mean, modes 1–4 | std, modes 1–4 |
|---|---|---|
| mesh (50 el., nominal) | −0.001, −0.011, −0.054, −0.165 | ≈ 0 |
| mass field 5 % | +0.01 to +0.07 | 0.94, 0.95, 0.92, 0.87 |
| stiffness scatter 2.5 % | ≈ 0 | 0.04–0.09 |
| measurement noise | ≈ 0 | 2.5 |

The finer mesh alone is a negligible modelling error for 4 modes (≤ 0.17 %, and 1 − MAC ≈ 0). The 5 % mass field is
the dominant one. With 2 % mass scatter the std was only 0.36–0.39 %.

**Run folders.** The default run name now includes the data version (`KSVAE_fixed_v6_7D_3KS_...`), so a v6 run does
not overwrite a v5 run.

**Localized-damage scenarios** (`localized_damage_study.py`). 29 extra test scenarios computed like the test-fine set
(as-built structure, 50-element model, noise of the data set): every deck zone damaged alone at 0.55 / 0.70 / 0.85 / 0.90,
short defects inside a zone (0.2 or 0.4 m cells with stiffness x0.30 / x0.50; the reference value is the harmonic-mean
equivalent of the zone), one degraded bearing, and two double-damage cases. It computes the exact posterior of every
scenario and, with `--run-folder`, the VAE posterior of a trained run, and writes `<run>/Localized_scenarios/`:
`Fig_localization_map.png`, `Fig_within_zone_defects.png`, `Fig_other_scenarios.png`, one violin plot per scenario and
`localized_scenarios.csv` (detections, misses and false alarms with the rule "upper 95 % bound < 0.95"). About 2.5 min.
```
python MODULES/review29oct_fixed/localized_damage_study.py --run-folder KSVAE_fixed_v6_7D_3KS_obs-fv_Gamma1.0_Ep1500
```

## 0d. Update (31 Oct, v5): definitive setup — 5 sensors, 4 modes, realistic bearings, K = 3

The previous version is frozen in `MODULES/review29oct_fixed_v2` (data: `Data/BridgeBearings_7D_v4_noTemp`).

**Data.** `Data/BridgeBearings_7D_v5_5sensors_4modes`.

**Sensors and modes.**
- 5 vertical accelerometers at the zone centres (1, 3, 5, 7, 9 m) and 4 modes.
- The 4 modes are distinct at the sensors (largest MAC between modes 0.012).
- They are the same physical modes in the 10- and 50-element models (paired MAC = 1.000, frequency difference
  ≤ 0.18 %).

**Bearings.** Laminated elastomeric pad sized for this beam: kv0 = 3.6e7 N/m, kr0 = 6.8e4 N·m/rad (see
`sensor_mode_study.py`).
- f1 = 3.56 Hz, close to the simply supported value.
- As reported in the literature, bearing stiffness hardly affects the vertical bending modes, so the reference
  posteriors of the bearings are almost equal to the prior (std ≈ 0.15). The deck zones are identified with std
  0.03–0.09.

**Multimodality check** (`multimodality_study.py`). Fewer sensors and modes widen the exact posteriors but do not
make them multimodal: the deck marginals stay unimodal and the mirror location receives little probability.

**Eigensolver in float64.** The decoder now solves the eigenproblem in float64 (`preprocessing_and_eigen.py`). With
soft bearings the condition number is about 3e5, and float32 caused errors of up to 0.6 % on f1. TF and NumPy now
agree to 1e-5.

**K = 3** Kumaraswamy components by default.

## 0c. Update (30 Oct, v4): no temperature, manuscript noise, reference posteriors in the data step, design study

The previous version is frozen in `MODULES/review29oct_fixed_v1`, with its data in
`Data/BridgeBearings_7D_realistic_v3`.

**Uncertainty sources.**
- **Temperature removed** (`TEMP_E_STD = 0` in `config.py`; left for future experimental work).
- **Modelling error kept:** finer mesh for `test_fine`, 2 % element mass and 2.5 % within-zone stiffness variability.
- **Measurement noise as in the submitted manuscript:** 2.5 % on frequencies, 5 % on mode shapes.
- The exact posteriors keep real uncertainty (benchmarks P3, P6, P7, P9): std 0.02–0.13, widest 95 % intervals
  0.22–0.48.

**Data step.** `dataset_generation.py` now also computes and plots the reference posteriors of the benchmarks listed in
`GT_EXAMPLES`, for both test sets: `Data/BridgeBearings_7D_v4_noTemp/figures/Fig_reference_posterior_P*_<set>.png`.

**Sensors are configurable.**
- `SENSOR_POSITIONS` in `config.py`; every position must be a node of the decoder mesh.
- The mode shapes always have one coordinate per sensor, whatever the number of elements (6 by default).

**Design study.** `sensor_mode_study.py` compares
- sensor layouts,
- numbers of modes,
- bearing stiffness options.

For each combination it reports a linearized posterior std per parameter, the aliasing between modes at the sensors
(largest off-diagonal MAC) and the mesh error of the last mode. Results go to
`Output/review29oct_fixed/sensor_mode_study/`. Edit the `LAYOUTS`, `MODES` and `BEARINGS` dictionaries to explore
other options.

## 0b. Update (30 Oct, v3): two test sets, K = 1

Data in `Data/BridgeBearings_7D_realistic_v3`.

**Two test sets with the same 1,200 damage states (benchmarks P0–P10 included).**
- `test`: generated with the same 10-element model as training and validation. This is the clean verification.
- `test_fine`: generated with the 50-element model. This is the model-form robustness test (R1.1, R1.7).
- Both include temperature, mass and within-zone variability.

**Likelihood.** Calibrated on the training generator.
- Check on `test`: 0.98–1.01.
- Check on `test_fine`: 0.72–1.02, i.e. the finer model deviates *less* than the calibrated error.

**Modes.** Five modes in both models. Modes 1–5 of the 50-element model are the same physical modes as those of the
decoder: paired MAC ≥ 0.998 and frequency difference ≤ 0.53 % over 3,000 random damage states. Modes 6 and higher
(127 Hz and up) are discarded in both models.
- At the 6 sensors, mode 5 looks like mode 3 (spatial aliasing), so modes are paired by frequency order.

**Exact posteriors are calibrated.** The true state lies outside the 95 % interval in 5/77 (`test`) and 3/77
(`test_fine`) benchmark parameters, as expected (~4), and never outside the 99 % interval. The overlays now also draw
the 99 % HPD contour (dotted).

**Defaults and outputs.**
- `main_train_KSVAE.py` uses **K = 1** by default and stores predictions for both sets:
  `Test_predicted_props.npy` and `Test_predicted_props_test_fine.npy`.
- `postprocessing.py` writes `GT_Comp/` and `GT_Comp_test_fine/` (plus `Global_Test_Metrics*.json` and
  `Fig_calibration*.png`).
- `reference_posterior.py` computes both sets (`--splits test test_fine`).

## 0. Update (30 Oct): realistic data with genuine posterior uncertainty

With the first corrected data set (`Data/BridgeBearings_7D_fixed`), the exact posteriors had std 0.013–0.03, so a
deterministic estimate would have done as well. That was not a sampler problem; the data were too informative.
- They measured **rotations** at every node with 3 % noise; no accelerometer does that.
- Train and test came from the same 5-element model, with no unmodelled error source.

On benchmarks P3, P5 and P9, the exact posterior std per parameter is:

| setup | posterior std |
|---|---|
| frequencies + vertical + rotational shapes, 1.5 % / 3 % noise | 0.013–0.028 |
| no rotations | 0.016–0.044 |
| no rotations, original noise 2.5 % / 5 % | 0.024–0.073 |
| as above, 3 modes | 0.031–0.153 |

The new default (`Data/BridgeBearings_7D_realistic`, `config.py`) follows the reviewers.

**Models (R1.1, R1.7).**
- The VAE and the reference sampler decode with a **10-element** model (2 per zone; mode-5 error 0.5 %, against
  13 % with 1 element per zone).
- **Test** observations come from a genuine **50-element** model with the same 5 zones and 6 sensors.
- Train and validation observations come from the 10-element model.

**Error sources the decoder does not model (R1.2, R3.1).** All are applied to every simulated structure:
- temperature (deck modulus ±3 %, the same for all deck elements);
- element mass scatter (2 %);
- within-zone stiffness scatter (2.5 %, harmonic mean = z).

**Measurements.**
- Frequencies, with 1 % identification error.
- **Vertical** mode shapes at the 6 sensors, with 5 % of the peak (as in `Noisy_random_dataset_generation.py`).
- No rotations. `--obs fvr` re-enables them.

**Likelihood, calibrated against the 50-element generator.**
- Mean mesh bias of log f: −0.05 … −0.5 %.
- **Full** 5×5 covariance of log f: std 1.6 %, correlation 0.6 between modes, because temperature moves all
  frequencies together.
- Per-mode shape std of 2.8–3.9 %.
- Check on the test data: E[NLL at z*] / expected = 0.98–1.01.

**Result: the reference posteriors of the 11 benchmarks now have real uncertainty.**
- Std per parameter 0.02–0.13; 95 % intervals up to 0.47 wide.
- Correlation ridges, e.g. z1 ↔ left bearing in P9 and z5 ↔ right bearing in P7.
- Skewed marginals.
- z* inside the 95 % interval for 6–7 of 7 parameters.
- Figures: `Data/BridgeBearings_7D_realistic/reference_posteriors/figures/`.

The procedure below is unchanged. Regenerate the data, then train. Runs on the new data get `obs-fv` in their name.
The earlier runs (`KSVAE_fixed_verify_Ep800`, `..._obs-fvr_..._Ep1500`) belong to the first data set and can still
be post-processed.

New locations:

| What | Where |
|---|---|
| Data | `Data/BridgeBearings_7D_realistic/` (figures in `figures/`, reference posteriors in `reference_posteriors/`); first corrected set: `Data/BridgeBearings_7D_fixed/` |
| Runs | `Output/review29oct_fixed/<run_name>/` |
| Diagnosis of the old run | `Output/review29oct_fixed/diagnosis_old_review29oct/` |

---

## 1. What was wrong

**Summary.** See `Output/review29oct_fixed/diagnosis_old_review29oct/Fig_old_vs_reference_P6.png`. It shows four
posteriors of the *same* old observation of benchmark 6:

| | std per parameter | location |
|---|---|---|
| correct reference posterior (SMC, calibrated likelihood) | 0.013–0.037 | close to z* |
| your old "ground truth" | 0.07–0.13 (4–5× too wide) | — |
| your old VAE (raw q) | 0.03–0.05 | badly off: zBR ≈ 0.86 for z* = 0.64; healthy z3, z5, zBL ≈ 0.85–0.89 for z* = 0.98 |

The VAE also had **singular copulas** (thin ridges).

So the VAE did not look narrow because it was too narrow. It looked narrow because the reference was far too wide,
and it was *wrong* (biased and degenerate) for the reasons below. The data really are informative:
- 5 frequencies with 1.5 % noise;
- 60 mode-shape components with 3 % noise;
- 7 parameters to identify.

A correct VAE must therefore produce narrow posteriors, but in the right place and with the right correlations.

### 1.1 Why the estimated posterior was wrong (`review29oct/GC_KSmix_models.py`)

1. **The entropy term had the wrong sign.** `Joint_copula_dens_term` returns `γ²·(−mean log q(z))`, which is
   `+γ²·H[q]`, and the loss *adds* it. Minimising the loss therefore **minimises the entropy of q**: q is rewarded
   for collapsing. The negative ELBO needs `−γ²·H[q]`.
   - The only brake was `clip(log q, −50, 50)`. In the run `30Oct_v4_..._Ep2500`, the logged `KL_Divergence_term`
     flattens at −6.10, which is exactly `−γ²·50` = −0.1225·50 = −6.125.
   - The encoder reached this limit by making the copula singular:
     - median smallest eigenvalue of R over the test set ≈ 4e-12;
     - median log det R ≈ −75;
     - median max|corr| = 0.9988;
     - corr(z1, z2) = −1.000 for test sample 6.
   - That is where the razor-thin ridges in your VAE corner plots come from.
2. **`Boundary_Penalty_loss`** (−log x − log(1−x)) pushes samples away from z = 1, which is exactly where the
   posterior of every healthy parameter lies. Over the test set, healthy parameters were estimated 0.02–0.06 too low.
   Damaged parameters were estimated 0.06–0.15 too high.
3. **The likelihood was arbitrary.** `5·Huber(standardised log f) + 10·(1 − best MAC)` is not the likelihood of the
   data, so the posterior it implies has an arbitrary width.
4. **The optimisation was distorted.**
   - The Kumaraswamy-quantile gradient was clipped: `dy` was clipped to norm 1 over the **whole** batch, the pdf to
     [5e-4, 50] and the gradients to ±20.
   - The learning rate decayed ×0.92 every 1000 steps. That is ×1e-4 by epoch 1900, so training effectively stopped
     after about 500 epochs.
   - `a, b ∈ [1.01, 200]` used a hard `clip_by_value`, which gives zero gradient outside that range.

### 1.2 Why the "ground truth" was not a posterior (`review29oct/postprocessing_7D_bridge.py`)

- **It used the answer.** 4096 of its points were drawn around z* (and its mirror). The importance weights were
  raised to the power 0.55, which does not correct for that proposal, so the cloud stays biased towards z*.
- **Its likelihood was hand-tuned.** It used `beta_eff = 0.55`, hard-coded, which ignores the γ of the run. It was
  built from the MSE of standardised log f plus 8·MAC, which is neither the VAE loss nor the noise of the data. As a
  result it was about 4× too wide. Example: for test sample 6 your plot shows std ≈ 0.1; the actual posterior has std
  0.013–0.037.
- **The lightest 5 % of the weight was discarded** (top-95 % filter).
- **The "bimodal" cases used a hand-made likelihood** (0.55/0.45 mixture with reversed predicted modes) on
  observations that were themselves fabricated (§1.4).

### 1.3 Why the blue curves were not the VAE

The "Estimated KS-Copula posterior" was **q re-weighted by the likelihood** (q·L, so the likelihood counts twice),
then filtered to the top 95 %. For samples 5, 8 and 9 it was also artificially mirrored. The raw q was never plotted
except as the green dash-dot curve in the 1-D strips.

### 1.4 Data set (`review29oct/dataset_generation.py`)

- **The DoE was balanced by location but had gaps.**
  - Each parameter was damaged in 19–22 % of the samples, which is fine.
  - There was no sample with z in (0.90, 0.95) or in [0.45, 0.50). Mild damage was never seen in training.
  - There were never more than 3 damaged parameters. Benchmark 7 has **5**, so it was out of distribution.
  - Benchmark 6 (two severe + one moderate) did not fit the fixed severity bands of the "triple" category.
- **The "bimodal" benchmarks 5, 8 and 9 were not physical.** Their modal data were the *average* of the data of z*
  and of its mirror image. For modes 2 and 4 this cancels the shape: 1 − MAC ≈ 0.99 against both z* and the mirror,
  so those observed "mode shapes" were pure noise.
- **Mode signs were set by the largest |vertical| component.** For the antisymmetric modes 2 and 4 of a
  near-symmetric state, the sign is random: it flipped in 50 % of the noise draws. The likelihood ignores signs, but
  the encoder input does not.
- **The "50-element fine-mesh" test set was not a finer mesh.** It solved the same 5-element model with a
  harmonic-mean jitter of z. A real 50-element model differs from the 5-element one by 0.04 / 0.3 / 1.2 / 2.4 / 13 %
  on modes 1–5.
- **Not a bug, but worth knowing:** with 1.5 % frequency noise, the frequencies alone resolve a bearing only to ±0.3,
  so most of the information on the bearings comes from the rotational mode shapes. The 6 sensors catch only about
  30 % of the peak amplitude of the 5-element model's mode 5, and its nodal pattern aliases onto mode 3.

---

## 2. What the new scripts do

| file | role |
|---|---|
| `config.py` | Every setting in one place: structure, noise, DoE, benchmarks, paths. |
| `fe_model.py` | NumPy float64 FE model (identical matrices, verified against your solver), batched modal solver, noise model, and **the likelihood**. |
| `dataset_generation.py` | Balanced stratified DoE (see below); purely physical observations; signs aligned to the healthy shapes; **likelihood calibration**. Writes `problem_config.json`, which training, the reference sampler and the plots all read. Also writes data figures. |
| `reference_posterior.py` | Reference posterior by **adaptive-tempering SMC**. Starts from the uniform prior, never uses z*, uses the same likelihood and γ as training, keeps every sample. 4000 particles, about 20 s per case; cached. |
| `prior_scenarios_coverage.py` | Coverage of the exact (SMC) and VAE posteriors on the test structures drawn from the prior, per parameter; tells whether an under-coverage comes from the likelihood or from the VAE. |
| `preprocessing_and_eigen.py` | Your data loader and eigensolver layer, without the sign flip. |
| `GC_KSmix_functions.py` | Copula sampling; stable KS-mixture CDF and log-pdf; quantile by bisection with **exact implicit gradients** (no clipping); copula entropy ½ log det R. |
| `GC_KSmix_architectures.py` | Your encoder. `a, b = 1 + exp(7 tanh(raw/7))` (smooth, range up to about 1100); the copula head starts at R = I. |
| `GC_KSmix_models.py` | The VAE with the **correct negative ELBO** (below), a warm-up callback and a best-weights callback. |
| `main_train_KSVAE.py` | Training script; same output files as before (`Test_predicted_props.npy` keeps the same keys). |
| `postprocessing.py` | Loss curves, test-set calibration and metrics, and for every benchmark the GT / VAE / overlay / active-subspace corner plots, 1-D strips, violin profiles and a summary heat map. |
| `plotting.py` | Shared plots. Both posteriors use the same density estimator: reflected KDE on [0.45, 1]; overlays show 50 % / 90 % HPD contours. |
| `check_consistency.py` | Self-checks (all must PASS). |
| `diagnose_old_pipeline.py` | Reproduces the diagnosis of §1 on your old run (figures and `diagnosis_report.json`). |

### The model now

**Likelihood.** Used identically for training and for the reference; NumPy and TF agree to about 2e-4:

    NLL(z; m) = Σ_j (log f_j − log f_j(z))² / 2σ_f,j²  +  Σ_j (1 − MAC(v_j, v_j(z))) / 2σ_v,j²
                                                    +  Σ_j (1 − MAC(r_j, r_j(z))) / 2σ_r,j²

For a unit-norm shape with Gaussian noise, 1 − MAC equals the squared noise component perpendicular to the shape.
So this is the Gaussian log-likelihood of the noise, and it does not depend on mode signs.

**Calibration of σ.** Measurement noise (1.5 %, 3 %) plus the effect of the 1.5 % zone-mass perturbation (which the
decoder does not model), measured by simulation. Check: E[NLL term at z*] / expected = 0.96–1.04 for all 15 terms.

**Prior.** Uniform on [0.45, 1]⁷.

**Loss.** `loss = τ·E_q[NLL] − γ²·H[q]`, where:

- `H[q] = ½ log det R` (exact) `+ Σ_d H[q_d]` (Monte Carlo);
- γ = 1 makes the minimiser the Bayesian posterior;
- τ rises from 0.01 to 1 over the first 100–150 epochs (warm-up), then stays at 1;
- model selection uses the validation −ELBO (τ = 1), after the warm-up.

**Removed:** the boundary penalty, the mixture-diversity loss, the log-q clip and the gradient clipping. They are no
longer needed, because the correct-sign entropy gives no incentive to collapse.

**Balanced DoE** (exact counts, Latin-hypercube severities, no gaps):

| category | share | damaged parameters |
|---|---|---|
| healthy | 10 % | none |
| single | 30 % | 1 (7 locations, equal counts) |
| double | 25 % | 2 (all 21 pairs) |
| triple | 15 % | 3 (all 35 triples) |
| 4–5 sites | 10 % | 4 or 5 (all 56 combinations) |
| uniform box | 10 % | any (whole prior box) |

- Damaged z ~ U[0.45, 0.95); healthy z ~ U[0.95, 1].
- Every parameter, zones and bearings alike, is damaged in **33.2 %** of the training samples.

---

## 3. Procedure

Run everything **from the project root**. In Spyder, set the working directory to the project root and run the file.
Use the conda env with TensorFlow (`tf_prob_gpu`). On the GPU server add `--gpu 2` (or the GPU you use).

**Keras backend.** Your local `~/.keras/keras.json` says `"backend": "jax"` and `"floatx": "float64"`. With that
setting, any `tf.GradientTape` training fails with `AttributeError: 'Variable' object has no attribute '_id'`, and so
would your original scripts.
- All TF scripts here get Keras from `MODULES.review29oct_fixed.tf_keras()`, which returns Keras on TensorFlow in
  float32.
- If the Spyder console already imported Keras on JAX, it reloads Keras on TensorFlow and prints a one-line notice.
  Keras objects created earlier in that console become unusable.
- If anything still looks odd, restart the kernel (Consoles > Restart kernel).
- To make TensorFlow the default only in this env, without touching your JAX env:
  `conda env config vars set KERAS_BACKEND=tensorflow -n tf_prob_gpu`.

```bash
# 1. Data (about 10 s): balanced DoE, observations, likelihood calibration, data figures
python MODULES/review29oct_fixed/dataset_generation.py
#    -> Data/BridgeBearings_7D_realistic/{*.npy, problem_config.json, figures/Fig_DoE_balance.png,
#       Fig_modal_data.png, Fig_likelihood_calibration.png}

# 2. Self-checks (about 30 s): every line must say PASS
python MODULES/review29oct_fixed/check_consistency.py

# 3. Reference ("ground-truth") posteriors of the 11 benchmarks (about 2 min with 8 workers; cached).
#    Optional: step 5 runs it automatically with the run's gamma.
python MODULES/review29oct_fixed/reference_posterior.py --gamma 1.0 --workers 8
#    -> Data/BridgeBearings_7D_realistic/reference_posteriors/{GT_P*.npz, figures/P*_GT_*.png}

# 4. Training (about 2 s/epoch on a 16-thread CPU; GPU similar, the eigensolver runs on the CPU)
python MODULES/review29oct_fixed/main_train_KSVAE.py --epochs 1500 --warmup 150 --gpu 2
#    -> Output/review29oct_fixed/KSVAE_fixed_7D_3KS_obs-fv_Gamma1.0_Ep1500/
#       {final_model_weights.weights.h5, model_history.npy/.csv, Test_predicted_props.npy,
#        Problem_info.npy, run_config.json}
#    add --postprocess to run step 5 at the end

# 5. Figures and metrics
python MODULES/review29oct_fixed/postprocessing.py --run-folder KSVAE_fixed_7D_3KS_obs-fv_Gamma1.0_Ep1500

# 6. (optional) Diagnosis of the old review29oct run
python MODULES/review29oct_fixed/diagnose_old_pipeline.py
```

### Outputs of step 5 (in the run folder)

| file | content |
|---|---|
| `Training_Loss_Curves.png` | total loss; −ELBO (train/val); E_q[NLL] and its frequency / vertical / rotational terms; entropy and its copula / marginal parts; τ and learning rate |
| `Global_Test_Metrics.json`, `Fig_calibration.png` | coverage of z* by the exact KS-mixture intervals (zones / bearings), RMSE, interval width, false alarms, detection rate |
| `GT_Comp/P{pos}_GT.png`, `P{pos}_VAE.png` | single corner plots (your style) |
| `GT_Comp/P{pos}_Overlay.png`, `P{pos}_ActiveSubspace_Overlay.png` | reference (red) vs VAE (blue), with a statistics table that includes the std ratio |
| `GT_Comp/P{pos}_1D.png` | 1-D marginals plus the exact KS-mixture density of q |
| `Physical_PDF_Profiles/Sample_{pos}_UQ_Violin.png` | violin profile along the bridge |
| `GT_Comp/GT_vs_Posterior_Metrics.csv/.json`, `Fig_benchmark_summary.png` | per-parameter mean and std of both posteriors, std ratio, W1, energy distance, correlation error |

### Options

| option | meaning |
|---|---|
| `--gamma` | entropy weight γ². γ = 1 targets the Bayesian posterior; any other γ targets `exp(−NLL/γ²)`, and the post-processing then computes the reference with the same γ. |
| `--num-ks`, `--samples` | K Kumaraswamy components per marginal; Monte Carlo samples of q per observation. |
| `--obs f` / `--obs fv` | Observe only the frequencies, or the frequencies plus vertical shapes (give the same `--obs` to `reference_posterior.py`; the post-processing reads it from the run). See the warning below. |

**Warning about `--obs f`.** With frequencies only, mirror states (z1↔z5, z2↔z4, BL↔BR) cannot be told apart, but
the problem is also under-determined (5 data, 7 unknowns).
- The posterior spreads over many alternative stiffness distributions rather than two clean modes.
- Benchmark 1: P(z1 < 0.75) ≈ P(z5 < 0.75) ≈ 0.06, and most of the mass sits near z = 1. The SMC result was
  verified against 2 million prior importance samples.
- So `--obs f` does **not** reproduce the "either z1 or z5" pictures of the old hand-made cases 5/8/9.

**Physical way to get clean bimodality** (not implemented). Model an observation whose left/right orientation is
unknown:

    p(m | z) = ½ p(m | z) + ½ p(mirror(m) | z)

- It is a mixture of the same likelihood with the sensor order reversed. In the code this is a log-sum-exp of two NLL
  evaluations, in `fe_model.py` and in `GC_KSmix_models.py`.
- The posterior then has two equally sharp modes, at z* and at its mirror.
- A single Gaussian copula with bimodal KS-mixture marginals can represent this through a strong negative
  correlation (for example between z1 and z5). That is a clean test of the multimodality claim.

---

## 4. Results of the verification run

Run folder: `Output/review29oct_fixed/KSVAE_fixed_verify_Ep800`.

**Settings.** 800 epochs on a 16-thread CPU, γ = 1, K = 3, S = 6, warm-up of 100 epochs. That is shorter than the
1500 epochs recommended for the GPU.

**Training.**
- Best validation −ELBO: 45.15, at epoch 744.
- Train and validation agree within 0.1, so there is no overfitting.
- Validation E[NLL] = 27.8, close to the ≈ 27.5 that the calibrated noise implies at the true posterior.

**Test set** (1200 cases, exact KS-mixture marginals; old run `30Oct_v4_..._Ep2500` in brackets):

| metric | deck zones | bearings |
|---|---|---|
| coverage of z* at 50 / 68 / 80 / 90 / 95 / 99 % | 0.51 / 0.69 / 0.81 / 0.89 / 0.94 / 0.98 | 0.47 / 0.65 / 0.78 / 0.88 / 0.93 / 0.97 |
| coverage at 95 % | 0.94 [old: 0.72 for all parameters] | 0.93 |
| mean width of the 95 % interval | 0.088 | 0.112 |

- RMSE of the posterior mean: 0.026 [old: 0.080].
- False alarms on healthy parameters (upper 95 % bound < 0.95): 0.8 % [old: 5.3 %].
- Detection of z* < 0.90: 96 %.

**Benchmarks vs the reference posterior** (11 cases, 4000 samples each):

| | value |
|---|---|
| width, std(VAE) / std(reference) | median **1.06**; 90 % of the 77 parameters between 0.78 and 1.74 |
| mean offset \|mean(VAE) − mean(reference)\| | 0.013 on average, at most 0.041; the reference std is 0.013–0.037 |
| mean Wasserstein-1 distance | 0.013 |
| z* inside the VAE 95 % interval | 5–7 of 7 per benchmark |
| P6 (triple damage) | width ratio 1.20, mean offset 0.007 |
| P7 (five damaged parameters, the hardest) | width ratio 1.33, mean offset 0.022 |

**What is left.**
- A few marginals are 1.5–2× wider than the reference (P2 z3, P9 z2), and the largest mean offsets (≈ 0.04) are in P5
  and P7. This is an amortisation gap: one encoder serves all observations.
- A longer run (1500 epochs on the GPU) should reduce it. A wider encoder or more Monte Carlo samples S would too.

**Incomplete outputs.** The post-processing of this run was stopped before it drew P9 (1-D, overlays) and P10 and
before it wrote `GT_vs_Posterior_Metrics.csv`. To complete it, run step 5 again with
`--run-folder KSVAE_fixed_verify_Ep800`. The reference posteriors are cached, so it takes a few minutes.
