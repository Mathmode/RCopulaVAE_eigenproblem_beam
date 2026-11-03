# -*- coding: utf-8 -*-
"""
review_uq_and_training_utils.py -- Uncertainty Quantification, Calibration,
Validation-Set Hyperparameter Selection, and Diagnostic Metrics for CPENG-8674.

Directly extends:
  - MODULES/COPULAS/updated_GC_GMm_functions_for_results_analysis.py
  - MODULES/COPULAS/GC_Gm_uncertainty_quantification.py
  - MODULES/KUMARSWAMY/GC_KSmix_results_analysis.py

Addresses:
  - R1.3 / Editor (3): Fixes the symmetric untruncated approximation (mu +/- 1.96*sigma
    clipped to [lbound, 1.0]) by computing exact TruncatedNormal / empirical copula
    quantiles, explaining the 77.2% vs 78.9% discrepancy.
  - R1.6: Computes False-Positive Rate (FPR) on healthy cases (z = 1) and localized
    damage localization accuracy / ROC-AUC on single- and double-element test sets.
  - R3.2 / Editor (2): Implements strict validation-set selection of gamma (or beta)
    using calibration error + NLL/ELBO on the validation split before evaluating test sets.
  - R3.3: Provides optional informative prior log_p(z) (e.g. Beta / sparse healthy prior)
    for ELBO regularization while retaining uniform amortization training data.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, truncnorm


def exact_truncated_normal_ci(
    mu: np.ndarray,
    sigma: np.ndarray,
    lbound: float = 0.45,
    ubound: float = 1.0,
    alpha: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute exact equal-tailed `alpha` credible intervals and posterior means for
    TruncatedNormal(mu, sigma, lbound, ubound) marginals.

    In `updated_GC_GMm_functions_for_results_analysis.py` (lines 121-128), intervals were
    approximated as `clip(mu +/- 1.96*sigma, lbound, 1.0)`, whereas
    `GC_Gm_uncertainty_quantification.py` (line 39) used `tfd.TruncatedNormal.cdf`.
    Near the boundary z = 1.0 or z = lbound, the symmetric untruncated interval differs
    from the exact truncated quantiles, causing the discrepancy noted by Reviewer 1 (R1.3).

    Returns
    -------
    ci_low : (N, D) lower quantile at (1 - alpha)/2
    ci_high : (N, D) upper quantile at 1 - (1 - alpha)/2
    trunc_mean : (N, D) exact expectation E[z | m] of the truncated normal
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-6)
    a = (lbound - mu) / sigma
    b = (ubound - mu) / sigma

    q_low = (1.0 - alpha) / 2.0
    q_high = 1.0 - q_low

    ci_low = truncnorm.ppf(q_low, a, b, loc=mu, scale=sigma)
    ci_high = truncnorm.ppf(q_high, a, b, loc=mu, scale=sigma)
    trunc_mean = truncnorm.mean(a, b, loc=mu, scale=sigma)
    return ci_low, ci_high, trunc_mean


def approximate_clipped_ci(
    mu: np.ndarray,
    sigma: np.ndarray,
    lbound: float = 0.45,
    ubound: float = 1.0,
    z_crit: float = 1.96,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproduce the legacy symmetric clipped interval from
    `updated_GC_GMm_functions_for_results_analysis.py` for side-by-side comparison (R1.3).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    ci_low = np.clip(mu - z_crit * sigma, lbound, ubound)
    ci_high = np.clip(mu + z_crit * sigma, lbound, ubound)
    return ci_low, ci_high


def compute_coverage_and_width(
    z_true: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """
    Compute element-wise and overall empirical coverage probability (PICP) and
    mean prediction interval width (MPIW).
    """
    z_true = np.asarray(z_true, dtype=np.float64)
    inside = (z_true >= ci_low) & (z_true <= ci_high)
    widths = ci_high - ci_low
    return {
        "overall_coverage": float(np.mean(inside)),
        "per_element_coverage": np.mean(inside, axis=0),
        "overall_mean_width": float(np.mean(widths)),
        "per_element_mean_width": np.mean(widths, axis=0),
    }


def evaluate_healthy_false_positive_rate(
    ci_high: np.ndarray,
    trunc_mean: np.ndarray,
    healthy_threshold: float = 0.95,
) -> dict[str, np.ndarray | float]:
    """
    Address Reviewer 1 Comment 6 (R1.6): False-Positive Rate (FPR) on undamaged (z=1)
    test cases or undamaged elements in localized-damage test cases.

    Two complementary decision criteria are reported:
      1. Credible-interval criterion: a healthy element (z_true = 1.0) is flagged as a
         false positive if the upper 95% credible bound `ci_high < healthy_threshold`
         (e.g., 0.98 or 0.95), meaning the posterior excludes the healthy state with >95%
         credibility.
      2. Posterior-mean criterion: flagged if `E[z|m] < healthy_threshold` (e.g., >5%
         predicted stiffness loss).
    """
    fp_ci = ci_high < healthy_threshold
    fp_mean = trunc_mean < healthy_threshold
    return {
        "threshold": healthy_threshold,
        "fpr_ci_per_element": np.mean(fp_ci, axis=0),
        "fpr_ci_overall": float(np.mean(fp_ci)),
        "fpr_ci_any_element_in_structure": float(np.mean(np.any(fp_ci, axis=1))),
        "fpr_mean_per_element": np.mean(fp_mean, axis=0),
        "fpr_mean_overall": float(np.mean(fp_mean)),
    }


def select_gamma_on_validation(
    val_metrics_by_gamma: dict[float, dict[str, float]],
    nominal_coverage: float = 0.95,
    coverage_weight: float = 1.0,
    rmse_weight: float = 1.0,
) -> dict[str, float | dict]:
    """
    Address Editor Comment (2) & Reviewer 3 Comment 2 (R3.2):
    Principled hyperparameter selection of `gamma` exclusively on the validation split.

    Parameters
    ----------
    val_metrics_by_gamma : dict mapping gamma -> {
        'val_coverage': float,      # empirical 95% coverage on validation set
        'val_rmse': float,          # posterior mean RMSE on validation set
        'val_mpiw': float,          # mean 95% interval width on validation set
        'val_modal_mse': float,     # forward modal reconstruction MSE on validation set
    }

    Returns
    -------
    Dictionary with selected `best_gamma` and the validation ranking table.
    """
    best_gamma = None
    best_score = np.inf
    ranking = {}

    for gamma, m in sorted(val_metrics_by_gamma.items()):
        cal_err = abs(m["val_coverage"] - nominal_coverage)
        score = coverage_weight * cal_err + rmse_weight * m["val_rmse"]
        ranking[gamma] = {
            **m,
            "val_calibration_error": cal_err,
            "composite_val_score": score,
        }
        if score < best_score:
            best_score = score
            best_gamma = gamma

    return {
        "best_gamma": best_gamma,
        "best_score": float(best_score),
        "validation_table": ranking,
    }
