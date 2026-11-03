# new_scripts_rev — corrected PI-VAE with copula / Kumaraswamy posterior

The original scripts in `MODULES/review29oct/` are untouched. These scripts reuse
`preprocessing_and_eigen.py` (data loading, assembly and eigensolver) from there.

| File | Role |
|---|---|
| `GC_KSmix_functions.py` | Copula + Kumaraswamy math (sampling, mixture log-density, Beta prior) |
| `GC_KSmix_architectures.py` | Encoder |
| `GC_KSmix_models.py` | VAE, loss, KL annealing callback, posterior sampler |
| `reference_posterior_smc.py` | Reference posterior via tempered SMC (numpy) |
| `evaluate_posteriors.py` | VAE-vs-reference plots, benchmark CSV, global test metrics |
| `main_GC_KSmix_rev.py` | Training script |

```
python MODULES/review29oct/new_scripts_rev/main_GC_KSmix_rev.py --exp Exp1 --epochs 2000 --gpu 2
python MODULES/review29oct/new_scripts_rev/evaluate_posteriors.py Output/Exp1/<run>
```

## Experiment naming (data and results always share the tag)

| Tag | Data folder | Results folder | Notes |
|---|---|---|---|
| Exp1 | `Data/Exp1_BridgeBearings_7D` | `Output/Exp1/` | Copy of `MODULES/review29oct/Data/BridgeBearings_7D_DoE` (Oct 29), whose test scenarios match `dataset_generation.py`. |

Do NOT use `Data/BridgeBearings_7D_DoE` (repo root, Oct 28): it was made by an older generator, so its
test samples 0-9 do not match the scenario names (e.g. sample 4 there is zone 2 + right bearing,
not left bearing z6 = 0.55). New data or model versions get a new tag (Exp2, Exp3, ...).

## 1. Loss function and the sign of the PDF term

Tempered likelihood p(m|z) ∝ exp(−E(z)/γ²), with E = L_freq + λ_v·L_MAC,v + λ_r·L_MAC,r, and an
explicit prior p(z) (independent Beta(α,β) on x = (z−z_lb)/(1−z_lb)). The negative ELBO times γ² is

    L = E_q[E(z)] + γ² · E_q[ log q(z|m) − log p(z) ]          (paper Eq. 10, L_PDF = +E_q[log q])

* The **+ log q** sign is correct: minimising E_q[log q] maximises the entropy of q, which stops it collapsing.
* `MODULES/KUMARSWAMY/GC_KSmix_models.py` used **γ²·(−E_q[log q])**, the NLL of its own samples. That
  has the wrong sign: it rewards collapse onto a point or a singular density. This is why it needed
  a,b ≥ 1.01, a boundary penalty and a diversity penalty to stay stable.
* `review29oct/GC_KSmix_models.py` had the correct sign, but it also:
  - clipped the per-sample KL to [−40, 60], which zeroes the gradient exactly when q is too sharp;
  - added a repulsion loss and a weight-entropy loss, which move the optimum away from the posterior;
  - used a Beta(2.2, 1.1) prior while the "ground truth" used Beta(1.8, 1.05) with weight 0.08,
    plus a different MAC weight (0.15) for cases 5, 8 and 9. Because the target differed, even a
    perfect VAE could not match it.

The exact minimiser of L is q = p_γ(z|m) ∝ exp(−E/γ²)p(z). This is the same density the
reference sampler targets, so the comparison is fair.

## 2. Posterior family: a mixture of Gaussian copulas

The reference posteriors are L-shaped / either-or. In sample 4, for example, either z_bL ≈ 1 with z_bR
spread out, or the reverse. A **single** Gaussian copula with multimodal marginals can only place
modes on the axis-aligned grid formed by all marginal-mode combinations. It cannot produce this shape.

Now q(z|m) = Σ_k π_k · c_k(F_k(z)) · Π_d Kumaraswamy_kd(z_d):
* each component has its own copula and single-Kumaraswamy marginals, and the marginals of q are still
  Kumaraswamy mixtures;
* the Kumaraswamy quantile is closed form, so samples are exact reparameterisations. The bisection
  solver and the hand-written implicit gradient are gone. That gradient clipped dy by the global norm,
  clipped the pdf to [0.01, 100] and the gradients to ±15. This was strongest exactly between modes,
  where the weights need a signal;
* the expectation over k is exact (stratified), so π gets exact gradients;
* `--n_comp 1` recovers the paper's single Gaussian copula with Kumaraswamy marginals, giving the
  ablation/baseline asked for in reviewer comment 1.4.

Other changes:
* b ≥ 0.5 instead of b ≥ 1.15, so a component can put mass at the healthy bound z = 1. With b > 1
  the density at z = 1 is forced to zero, but the reference posteriors of healthy elements peak there.
  With the correct sign there is no collapse incentive, so the floor is safe.
* No hard clip on a and b, and NaN-safe tail evaluations.
* KL weight annealed 3 → 1 over the first epochs (an option). The run starts broad to find all the
  modes, then the exact objective is restored. Best-weight selection only starts after annealing.
* `--objective siwae`: a stratified importance-weighted bound that is more mass-covering.

## 3. Evaluation problems found in `postprocessing_7D_bridge.py`

* **The "Estimated KS-Copula Posterior" (blue) was not the VAE.** `sample_vae_and_refine_7d` pooled
  VAE draws with 1024 jittered ground-truth draws, then re-weighted the whole pool by the true
  posterior. The blue curves were therefore mostly the reference itself. Only the green "raw encoder"
  curve was the model.
* **The "ground truth" started at the answer.** The chains were initialised at z_true, its mirror and
  a hand-made trade-off point, then ran 28 random-walk steps. Their output mostly reflected that
  initialisation. It was then re-weighted by exp(0.25·logp), which is not a valid posterior estimate.
* **The data and the script disagree.** For sample 4 the stored `alphas_test.npy` holds
  z* = [0.98, 0.62, 0.99, 0.98, 0.99, 0.98, 0.58]. That is zone 2 + right bearing, not
  "left bearing z6 = 0.55" as in `dataset_generation.py`. Regenerate the data, or fix the labels.

`evaluate_posteriors.py` plots the raw q(z|m): analytic marginals plus ancestral samples. It compares
them against tempered SMC that starts from the prior, with the same likelihood, prior and decoder as
training.

Still open from the reviews: select γ on validation data only (reviewer 3.2). Also, the "bimodal"
cases 8 and 9 blend two mode-shape sets into one observation, so the result is not a physical
observation.
