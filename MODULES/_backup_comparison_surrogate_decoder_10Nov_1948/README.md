# Comparison with the Gaussian copula VAE of Ref. 41: eigenvalue solver vs surrogate decoder (R1.4)

Reviewer 1 (comment 4) asks for a comparison with Refs. 39 and 41. The proposed model and the Gaussian copula VAE of Ref. 41
share the posterior model (Gaussian copula); they differ in the **decoder**:

| | Proposed model | Ref. 41 approach (this folder) |
|---|---|---|
| Decoder | differentiable eigenvalue solver (exact forward operator, no trainable parameters) | fully connected NN (architecture of Ref. 41, Table 3: 10-30-50-70-80, Tanh/ReLU, sigmoid output), pre-trained and frozen |
| Training | one step (encoder) | two steps: surrogate on the training split, then encoder with the surrogate frozen |
| Everything else | identical: data and splits, encoder, Gaussian copula + truncated Gaussian marginals, calibrated likelihood, gamma = 1, warm-up, optimizer, train-to-convergence rule | identical |

The surrogate maps the stiffness factors (unit box) to the quantities of the likelihood: log f(z) and the vertical mode shapes
at the five sensors. It is trained at the damage states of the SAME training split as the VAE, with the validation split for
early stopping. Targets (`--targets`): `nominal` (default), the noise-free modal data of the nominal model at those damage
states, as when a surrogate is trained on simulations; or `observed`, the noisy observations of the split (log f minus the
calibrated mean modelling error b). With `observed`, the 2.5 % noise dominates the validation loss and hides the remaining
error of the surrogate, so early stopping keeps the weights of epoch ~90 and the frequencies stay biased by about -0.3 %.
Convergence rule of the surrogate (`train_surrogate_decoder.py`, the analogue of the rule of the VAE training): the learning
rate (Adam, 1e-3, batch 256) is halved after 100 epochs without an improvement of the validation MSE larger than 1 % of its
best value, and the training stops at 1e-5 under the same condition; the best weights are kept. A first version used plain
early stopping, which counts any improvement: with noise-free targets the validation loss keeps decreasing by negligible
amounts, so it ran to the cap of 20 000 epochs (85 min on the server) without a meaningful gain.

Network size (nominal targets, new rule; five networks trained at the same time on a PC CPU, so the times only compare the
networks with each other; validation set, mean over the modes of the mean |frequency error|, and the largest error):

| Hidden layers | Parameters | Epochs | Time | Mean / max frequency error | Mean 1 - MAC |
|---|---|---|---|---|---|
| 10-30-50-70-80 (Ref. 41, default) | 13 640 | 2 321 | 6.2 min | 0.11 % / 1.9 % | 6e-6 |
| 32-32 | 2 302 | 3 231 | 7.2 min | 0.25 % / 4.1 % | 1e-5 |
| 64-64 | 6 622 | 4 287 | 11.3 min | 0.12 % / 1.7 % | 5e-6 |
| 64-64-64 | 10 782 | 3 640 | 9.8 min | 0.11 % / 2.2 % | 5e-6 |
| 128-128 | 21 406 | 3 886 | 11.7 min | 0.07 % / 2.0 % | 3e-6 |

The architecture of Ref. 41 reaches an error of about 1/25 of the identification noise of the frequencies (2.5 %) in the fewest
epochs; a smaller network is clearly less accurate and larger ones gain little for more training time. It is therefore kept.
`run_surrogate_size_study.sh` repeats this study.

During the VAE training, the weights are selected with the validation -ELBO computed with the surrogate (this
approach has no eigenvalue solver). All the evaluations (post-processing, posterior quality) use the exact model, so both
approximations are judged against the same exact posterior.

The Gaussian mixture model of Ref. 39 is not retrained here: Ref. 41 already compared Gaussian mixtures (diagonal and full
covariance) with the Gaussian copula on the same type of problem (Section 5.1 and Table 4 of Ref. 41).

## Files

* `surrogate_decoder.py`: architecture, scaling, saving/loading, and `SurrogateBridge` (NumPy surrogate with the interface of
  `fe_model.BridgeModel`).
* `train_surrogate_decoder.py` (step 1): trains and saves the surrogate; accuracy w.r.t. the eigenvalue solver
  (`surrogate_accuracy.csv`, `Fig_surrogate_accuracy.png`); cost of one decoder call for a training batch.
* `GCTN_surrogate_model.py`: the proposed model with `nll_terms` evaluating the frozen surrogate.
* `main_train_GCTN_surrogate.py` (step 2): trains the encoder with `review29oct_fixed/training_common.train_and_save`.
* `compare_eigen_vs_surrogate.py` (step 3): tables (`comparison_table.csv/.tex`), `comparison_summary.json` and
  `Fig_benchmark_marginals.png`.
* `analyse_eigen_vs_surrogate.py`: per-observation comparison of both runs (paired -ELBO and KL lower bound by DoE
  category, number of damaged parameters and severity; relation with the surrogate error; benchmarks vs the exact posterior).
* `run_surrogate_comparison_gpu.sh`: steps 1-3 and the post-processing (`--hidden` changes the network size).
* `run_surrogate_size_study.sh`: trains the surrogate with several network sizes and writes `size_study/size_study.csv`.

## Outputs (`Output/comparison_surrogate_decoder/`)

* `surrogate_decoder_06Nov-FINAL5modes_targets-nominal_H10-30-50-70-80/`: surrogate (weights, `.keras`, configuration with scaling, training time, accuracy and
  cost), training history, accuracy table and figure.
* `GCTN-SurrogateDecoder-nominal-H10-30-50-70-80_06Nov-FINAL5modes_7D_obs-fv_Gamma1.0_Conv6000/`: the VAE run, with the same files as every run
  (predictions, history, `Problem_info.npy`, post-processing, `Posterior_quality/`), plus `surrogateVAE_full_model.weights.h5`.
  `final_model_weights.weights.h5` holds the encoder inside a GC-TN model, so `load_trained_model` and the studies work unchanged.
* `comparison_06Nov-FINAL5modes_targets-nominal_H10-30-50-70-80/`: the comparison.

## Commands (server, project root, environment active)

```bash
nohup bash MODULES/model_comparison/run_convergence_gpu.sh --family gctn --obs fv --gpu 0 > S10_GCTN_fv.log 2>&1 &
nohup bash MODULES/comparison_surrogate_decoder/run_surrogate_comparison_gpu.sh --gpu 1 > surrogate_comparison.log 2>&1 &
```

When both have finished (the second one compares automatically if the first is done):

```bash
bash MODULES/comparison_surrogate_decoder/run_surrogate_comparison_gpu.sh --compare
```
