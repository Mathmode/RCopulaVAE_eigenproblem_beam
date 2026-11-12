#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/baseline_GaussCopula_GaussMarginals/gamma_sensitivity_study.py

Sensitivity analysis to the entropy weight gamma in the loss function (Reviewer 3, Comment R3.2):

    L(theta) = (1 / N) sum_i ( E_q[ NLL(z; m_i) ] - gamma^2 H[ q(z | m_i) ] )

For gamma = 1, L(theta) is the exact negative ELBO (up to an additive constant), whose minimizer is the Bayesian
posterior p(z | m) proportional to exp(-NLL(z; m)) p(z). For gamma != 1, the minimizer is the variational
approximation to the tempered (power) posterior p_gamma(z | m) proportional to exp(-NLL(z; m) / gamma^2) p(z):
  * gamma < 1 weakens the entropy term -> narrower (overconfident) credible intervals -> undercoverage;
  * gamma > 1 strengthens the entropy term -> wider (conservative) credible intervals -> overcoverage.

Following R3.2, the sensitivity analysis is evaluated on the VALIDATION set (2 400 observations), so the test set
remains completely unseen during model/hyperparameter analysis. Test-set (A and B) and SMC benchmark comparisons are
also exported in a separate file for reference.

Outputs in Output/baseline_GaussCopula_GaussMarginals/gamma_study/ (or --out-dir):
  Table_gamma_validation.csv / .tex    Validation-set metrics across gamma (MSE, MAE, ALL, MACE, 95% coverage,
                                       95% interval width, false-alarm/detection rates, and untempered KL difference
                                       to the exact Bayesian posterior)
  Table_gamma_test_sets.csv / .tex     Test-set A and B metrics + agreement with the SMC reference posteriors
  gamma_sensitivity_summary.json       All numerical results in JSON format
  Fig_gamma_sensitivity.png / .pdf     4-panel validation-set sensitivity figure: (a) 95% coverage & interval width,
                                       (b) MACE & extra KL to true posterior, (c) MAE & RMSE of posterior mean,
                                       (d) reliability curves across gamma
  Fig_gamma_benchmarks_1D.png / .pdf   1D marginal posteriors of benchmarks P1, P4, P5 across gamma

Run from the project root:
    python MODULES/baseline_GaussCopula_GaussMarginals/gamma_sensitivity_study.py
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed import plotting as P
from MODULES.review29oct_fixed.posterior_families import family_from_preds

DEFAULT_ROOT = os.path.join(C.PROJECT_ROOT, "Output", "baseline_GaussCopula_GaussMarginals")
SUB = {"A": "TestA_same_FE_model_10el", "B": "TestB_refined_FE_model_50el"}
PRED = {"val": "Val_predicted_props.npy", "A": "Test_predicted_props.npy", "B": "Test_predicted_props_test_fine.npy"}
SPLIT = {"val": "val", "A": "test", "B": "test_fine"}
MACE_LEVELS = np.linspace(0.05, 0.95, 19)


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def localize_data_dir(raw_path):
    if os.path.isdir(raw_path):
        return raw_path
    tail = re.split(r"[\\/]", str(raw_path).rstrip("\\/"))[-1]
    local = os.path.join(C.PROJECT_ROOT, "Data", tail)
    if os.path.isdir(local):
        return local
    raise FileNotFoundError(f"Data directory not found: {raw_path} (nor {local})")


def discover_gamma_runs(root_dir, family="GC-TN", modes=5, obs="fv", schedule="plateau", seed=1234, explicit_runs=None):
    """Finds all evaluated runs for the gamma study, sorted by ascending gamma."""
    candidates = []
    if explicit_runs:
        for r in explicit_runs:
            p = r if os.path.isabs(r) else (os.path.abspath(r) if os.path.isdir(r) else os.path.join(root_dir, r))
            if os.path.isdir(p):
                candidates.append(p)
    elif os.path.isdir(root_dir):
        for name in sorted(os.listdir(root_dir)):
            p = os.path.join(root_dir, name)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "Problem_info.npy")):
                candidates.append(p)

    by_gamma = {}
    for run in candidates:
        info = np.load(os.path.join(run, "Problem_info.npy"), allow_pickle=True).item()
        if not explicit_runs:
            if info.get("family", "GC-TN") != family:
                continue
            if info.get("obs_types", "fv") != obs:
                continue
            if info.get("schedule", "cosine") != schedule:
                continue
            if int(info.get("seed", 1234)) != int(seed):
                continue
            data_tail = re.split(r"[\\/]", str(info["data_dir"]).rstrip("\\/"))[-1]
            if modes == 5 and not (data_tail.endswith("5modes") or "5modes" in os.path.basename(run)):
                continue
            if modes != 5 and f"{modes}modes" not in data_tail and f"{modes}modes" not in os.path.basename(run):
                continue
        q_path = os.path.join(run, "Posterior_quality", "ELBO_summary.json")
        if not os.path.exists(q_path):
            print(f"  [skipped, no Posterior_quality/ELBO_summary.json yet]: {os.path.basename(run)}")
            continue
        g = round(float(info["gamma"]), 6)
        # If both a short-named and long-named folder exist for the same gamma, keep the more recent one
        if g not in by_gamma or os.path.getmtime(q_path) > os.path.getmtime(
                os.path.join(by_gamma[g]["path"], "Posterior_quality", "ELBO_summary.json")):
            by_gamma[g] = {"gamma": g, "gamma2": round(g ** 2, 6), "path": run, "name": os.path.basename(run),
                           "info": info}
    return [by_gamma[g] for g in sorted(by_gamma)]


def alarm_rates(run_dir, data_dir, lbound, split_key):
    """False-alarm rate (z* >= 0.95 declared < 0.95) and detection rate (z* < 0.90 declared < 0.95) at 95% level."""
    pred_path = os.path.join(run_dir, PRED[split_key])
    if not os.path.exists(pred_path):
        return np.nan, np.nan
    fam = family_from_preds(np.load(pred_path, allow_pickle=True).item(), float(lbound))
    z_true = np.load(os.path.join(data_dir, f"alphas_{SPLIT[split_key]}.npy")).astype(np.float64)
    q95, _ = fam.quantiles(np.array([0.025, 0.975]))
    alarm = q95[..., 1] < C.DAMAGE_THRESHOLD
    healthy = z_true >= C.DAMAGE_THRESHOLD
    damaged90 = z_true < 0.90
    return float(alarm[healthy].mean()), float(alarm[damaged90].mean())


def untempered_val_elbo_array(run_dir, gamma2):
    """
    Per-observation untempered negative ELBO on the validation set:
        -ELBO_{gamma=1}(m) = E_q[NLL(z; m)] - H_unit[q(z | m)]
    Reconstructed directly from Posterior_quality/ELBO_val.csv, where E_q_NLL = E_q[NLL] / gamma^2
    and entropy_unit = H_unit[q]. Since log p(m) is independent of q, differences of -ELBO_{gamma=1}
    between two models equal the exact differences of KL(q || p_{gamma=1}(z | m)) [nats / observation].
    """
    csv_path = os.path.join(run_dir, "Posterior_quality", "ELBO_val.csv")
    df = pd.read_csv(csv_path)
    e_nll_unscaled = df["E_q_NLL"].to_numpy(dtype=np.float64) * float(gamma2)
    h_unit = df["entropy_unit"].to_numpy(dtype=np.float64)
    return e_nll_unscaled - h_unit, e_nll_unscaled, h_unit


def collect_metrics(runs):
    """Extracts validation, test A, test B, and benchmark reference agreement metrics for every gamma run."""
    ref_run = next((r for r in runs if abs(r["gamma"] - 1.0) < 1e-6), runs[0])
    ref_untemp, _, _ = untempered_val_elbo_array(ref_run["path"], ref_run["gamma2"])

    records = []
    for r in runs:
        run_dir = r["path"]
        info = r["info"]
        data_dir = localize_data_dir(info["data_dir"])
        lbound = float(info["lbound"])
        g, g2 = r["gamma"], r["gamma2"]

        q_sum = read_json(os.path.join(run_dir, "Posterior_quality", "ELBO_summary.json"))
        val_m = read_json(os.path.join(run_dir, "Posterior_quality", "Validation_set", "Manuscript_Metrics.json"))
        untemp, e_nll, h_unit = untempered_val_elbo_array(run_dir, g2)
        diff_kl = untemp - ref_untemp
        fa_val, det_val = alarm_rates(run_dir, data_dir, lbound, "val")

        rec = {
            "gamma": g,
            "gamma2": g2,
            "run": r["name"],
            "epochs_run": int(info.get("epochs_run", 0)),
            "best_epoch": int(info.get("best_epoch", 0)),
            "converged": bool(info.get("converged", False)),
            "training_time_min": float(info.get("training_time_sec", 0.0)) / 60.0,
            # Validation posterior quality vs true Bayesian posterior (gamma_eval = 1)
            "val_E_q_NLL": float(np.mean(e_nll)),
            "val_entropy_unit": float(np.mean(h_unit)),
            "val_entropy_physical": float(np.mean(h_unit) + info["n_dims"] * np.log(1.0 - lbound)),
            "val_untempered_neg_elbo": float(np.mean(untemp)),
            "val_untempered_neg_elbo_se": float(np.std(untemp, ddof=1) / np.sqrt(len(untemp))),
            "val_dKL_vs_gamma1": float(np.mean(diff_kl)),
            "val_dKL_vs_gamma1_se": float(np.std(diff_kl, ddof=1) / np.sqrt(len(diff_kl))),
            # Validation posterior quality vs tempered target p_gamma
            "val_tempered_neg_elbo": float(q_sum["val"]["neg_elbo_mean"]),
            "val_tempered_kl_lower": float(q_sum["val"]["kl_lower_mean"]),
            "val_tempered_khat_median": float(q_sum["val"]["k_hat_median"]),
            "val_tempered_khat_lt_07": float(q_sum["val"]["frac_k_hat_below_0.7"]),
            "val_false_alarm_rate": fa_val,
            "val_detection_rate": det_val,
        }

        if val_m:
            for grp, tag in (("all_7", "all"), ("damaged_params", "damaged"), ("healthy_params", "undamaged"),
                             ("prior_scenarios", "prior"), ("deck_zones", "deck"), ("bearings", "bearings")):
                if grp in val_m:
                    gm = val_m[grp]
                    rec[f"val_{tag}_MSE"] = float(gm["MSE"])
                    rec[f"val_{tag}_RMSE"] = float(np.sqrt(gm["MSE"]))
                    rec[f"val_{tag}_MAE"] = float(gm["MAE"])
                    rec[f"val_{tag}_ALL_phys"] = float(gm["ALL_physical_units"])
                    rec[f"val_{tag}_ALL_unit"] = float(gm["ALL"])
                    rec[f"val_{tag}_MACE"] = float(gm["MACE"])
                    rec[f"val_{tag}_cov95"] = float(gm["coverage95"])
                    rec[f"val_{tag}_MACE_HPD"] = float(gm["MACE_HPD"])
                    rec[f"val_{tag}_cov95_HPD"] = float(gm["coverage95_HPD"])
                    rec[f"val_{tag}_width95"] = float(gm["mean_interval_width95"])
                    rec[f"val_{tag}_emp_cov"] = gm["empirical_coverage"]
                    rec[f"val_{tag}_emp_cov_HPD"] = gm["empirical_coverage_HPD"]

        for s in ("A", "B"):
            tm = read_json(os.path.join(run_dir, SUB[s], "Manuscript_Metrics.json"))
            fa_s, det_s = alarm_rates(run_dir, data_dir, lbound, s)
            rec[f"{s}_false_alarm_rate"] = fa_s
            rec[f"{s}_detection_rate"] = det_s
            if tm:
                for grp, tag in (("all_7", "all"), ("damaged_params", "damaged"), ("healthy_params", "undamaged"),
                                 ("prior_scenarios", "prior")):
                    if grp in tm:
                        gm = tm[grp]
                        rec[f"{s}_{tag}_MSE"] = float(gm["MSE"])
                        rec[f"{s}_{tag}_MAE"] = float(gm["MAE"])
                        rec[f"{s}_{tag}_ALL_phys"] = float(gm["ALL_physical_units"])
                        rec[f"{s}_{tag}_ALL_unit"] = float(gm["ALL"])
                        rec[f"{s}_{tag}_MACE"] = float(gm["MACE"])
                        rec[f"{s}_{tag}_cov95"] = float(gm["coverage95"])
                        rec[f"{s}_{tag}_width95"] = float(gm["mean_interval_width95"])
            gt_csv = os.path.join(run_dir, SUB[s], "Benchmarks_VAE_vs_reference", "GT_vs_Posterior_Metrics.csv")
            if os.path.exists(gt_csv):
                gdf = pd.read_csv(gt_csv)
                rec[f"{s}_bench_abs_dmean"] = float(gdf["mean_abs_diff_mean"].mean())
                rec[f"{s}_bench_std_ratio"] = float(gdf["mean_std_ratio_VAE_over_ref"].mean())
                rec[f"{s}_bench_W1"] = float(gdf["mean_W1"].mean())
                rec[f"{s}_bench_energy"] = float(gdf["energy_distance"].mean())
                rec[f"{s}_bench_zstar_in_VAE95"] = int(gdf["zstar_in_VAE95"].sum())
                rec[f"{s}_bench_zstar_in_ref95"] = int(gdf["zstar_in_ref95"].sum())
        records.append(rec)
    return records


def write_latex_validation_table(records, path):
    """Writes a simplified LaTeX table of the validation-set sensitivity analysis (R3.2) over all parameters."""
    cols = [rf"$\gamma = {r['gamma']:g}$" + (" (selected)" if abs(r["gamma"] - 1.0) < 1e-6 else "") for r in records]
    spec = "l" + "c" * len(records)

    def row_str(label, vals):
        return label + " & " + " & ".join(vals) + r" \\"

    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{Sensitivity of the proposed method to the hyperparameter $\gamma$ on the validation dataset "
        r"(2,400 observations with seven stiffness factors each).}",
        r"\label{tab:gamma_sensitivity}",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        row_str("Metric", cols),
        r"\midrule",
        row_str("MSE", [f"{r.get('val_all_MSE', np.nan):.5f}" for r in records]),
        row_str("MAE", [f"{r.get('val_all_MAE', np.nan):.4f}" for r in records]),
        row_str("ALL", [f"{r.get('val_all_ALL_phys', np.nan):.3f}" for r in records]),
        row_str("MACE", [f"{r.get('val_all_MACE', np.nan):.3f}" for r in records]),
        row_str(r"95\% coverage, $C(0.95)$", [f"{r.get('val_all_cov95', np.nan):.3f}" for r in records]),
        row_str(r"Mean 95\% interval width", [f"{r.get('val_all_width95', np.nan):.3f}" for r in records]),
        r"\midrule",
        row_str(r"False alarms [\%]", [f"{100 * r.get('val_false_alarm_rate', np.nan):.1f}" for r in records]),
        row_str(r"Positive alarms (detection rate) [\%]", [f"{100 * r.get('val_detection_rate', np.nan):.1f}" for r in records]),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_latex_test_table(records, path):
    """Writes a LaTeX table of Test Set A and Test Set B metrics + SMC reference posterior agreement across gamma."""
    cols = [rf"$\gamma = {r['gamma']:g}$" for r in records]
    spec = "l" + "c" * len(records)

    def row_str(label, vals):
        return label + " & " + " & ".join(vals) + r" \\"

    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{Test-set metrics (Test Set A: 10-element FE model; Test Set B: 50-element FE model with localized "
        r"damage) and agreement with the 11 SMC reference posteriors ($p_\gamma$) across $\gamma$.}",
        r"\label{tab:gamma_test_sets}",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        row_str("Metric", cols),
    ]
    for s, title in (("A", "Test Set A (10-element model, same as decoder)"),
                     ("B", "Test Set B (50-element model, localized damage)")):
        lines += [
            r"\midrule",
            rf"\multicolumn{{{len(records) + 1}}}{{l}}{{\emph{{{title}}}}} \\",
            row_str(r"\quad MSE", [f"{r.get(f'{s}_all_MSE', np.nan):.5f}" for r in records]),
            row_str(r"\quad MAE", [f"{r.get(f'{s}_all_MAE', np.nan):.4f}" for r in records]),
            row_str(r"\quad ALL", [f"{r.get(f'{s}_all_ALL_phys', np.nan):.3f}" for r in records]),
            row_str(r"\quad MACE", [f"{r.get(f'{s}_all_MACE', np.nan):.3f}" for r in records]),
            row_str(r"\quad 95\% coverage, $C(0.95)$", [f"{r.get(f'{s}_all_cov95', np.nan):.3f}" for r in records]),
            row_str(r"\quad Mean 95\% interval width", [f"{r.get(f'{s}_all_width95', np.nan):.3f}" for r in records]),
            row_str(r"\quad Benchmarks $|\mathbb{E}_q[z] - \mathbb{E}_{p_\gamma}[z]|$",
                    [f"{r.get(f'{s}_bench_abs_dmean', np.nan):.4f}" if f"{s}_bench_abs_dmean" in r else "--" for r in records]),
            row_str(r"\quad Benchmarks $\mathrm{std}(q_\gamma)/\mathrm{std}(p_\gamma)$",
                    [f"{r.get(f'{s}_bench_std_ratio', np.nan):.2f}" if f"{s}_bench_std_ratio" in r else "--" for r in records]),
            row_str(r"\quad Benchmarks Wasserstein-1 $W_1$",
                    [f"{r.get(f'{s}_bench_W1', np.nan):.4f}" if f"{s}_bench_W1" in r else "--" for r in records]),
        ]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


@P.pub_style
def plot_gamma_sensitivity(records, out_dir):
    """2-panel publication-quality figure of the validation-set sensitivity to gamma (all stiffness factors)."""
    gammas = np.array([r["gamma"] for r in records], dtype=float)
    cov_all = [r.get("val_all_cov95", np.nan) for r in records]
    w95_all = [r.get("val_all_width95", np.nan) for r in records]
    mace_all = [r.get("val_all_MACE", np.nan) for r in records]
    all_phys = [r.get("val_all_ALL_phys", np.nan) for r in records]
    mae_all = [r.get("val_all_MAE", np.nan) for r in records]
    rmse_all = [r.get("val_all_RMSE", np.nan) for r in records]

    # --- 2-panel figure: (a) 95% coverage & MACE; (b) ALL, MAE & RMSE (all stiffness factors) ---
    fig, ax = plt.subplots(1, 2, figsize=(P.FIG_WIDTH, 2.55))
    fig.subplots_adjust(left=0.09, right=0.91, top=0.88, bottom=0.18, wspace=0.42)

    # Panel (a): 95% coverage and calibration error (MACE)
    a0 = ax[0]
    a0.axhline(0.95, color="k", ls="--", lw=0.8, label="nominal 0.95")
    a0.axvline(1.0, color="0.5", ls=":", lw=0.8)
    a0.plot(gammas, cov_all, "-s", color="#000000", ms=4.2, lw=1.3, label=r"95% coverage $C(0.95)$")
    a0.set_xlabel(r"Hyperparameter $\gamma$")
    a0.set_ylabel(r"95% empirical coverage $C(0.95)$")
    a0.set_ylim(max(0.65, np.nanmin(cov_all) - 0.05), 1.00)
    a0.set_title(r"(a) Coverage $C(0.95)$ and calibration error (MACE)")

    a0_r = a0.twinx()
    a0_r.plot(gammas, mace_all, "--o", color="#D55E00", ms=4.0, lw=1.2, label="MACE")
    a0_r.set_ylabel("Calibration error (MACE)", color="#D55E00", fontsize=8.5)
    a0_r.tick_params(axis="y", colors="#D55E00")
    a0_r.set_ylim(0.08, max(mace_all) * 1.18)
    a0_r.spines["top"].set_visible(False)
    h0, l0 = a0.get_legend_handles_labels()
    h0r, l0r = a0_r.get_legend_handles_labels()
    a0.legend(h0 + h0r, l0 + l0r, loc="lower right", fontsize=6.8)

    # Panel (b): Point-estimate errors (MAE, RMSE) and Average Log-Likelihood (ALL)
    a1 = ax[1]
    a1.axvline(1.0, color="0.5", ls=":", lw=0.8)
    a1.plot(gammas, rmse_all, "-s", color="#000000", ms=4.2, lw=1.3, label="RMSE")
    a1.plot(gammas, mae_all, "--o", color="0.35", ms=4.0, lw=1.2, label="MAE")
    a1.set_xlabel(r"Hyperparameter $\gamma$")
    a1.set_ylabel(r"Point-estimate error ($\hat{z}$)")
    a1.set_ylim(0.03, max(rmse_all) * 1.18)
    a1.set_title(r"(b) Point-estimate error and log-likelihood (ALL)")

    a1_r = a1.twinx()
    a1_r.plot(gammas, all_phys, "-.^", color="#0072B2", ms=4.0, lw=1.2, label="ALL")
    a1_r.set_ylabel("Average log-likelihood (ALL)", color="#0072B2", fontsize=8.5)
    a1_r.tick_params(axis="y", colors="#0072B2")
    a1_r.set_ylim(min(all_phys) - 0.12, max(all_phys) + 0.12)
    a1_r.spines["top"].set_visible(False)
    h1, l1 = a1.get_legend_handles_labels()
    h1r, l1r = a1_r.get_legend_handles_labels()
    a1.legend(h1 + h1r, l1 + l1r, loc="lower right", fontsize=6.8)

    for a in ax.ravel():
        a.grid(True)
        a.set_axisbelow(True)
        P.despine(a)

    P.save_figure(fig, os.path.join(out_dir, "Fig_gamma_sensitivity.png"), tight=False)
    plt.close(fig)


@P.pub_style
def plot_gamma_benchmarks_1d(runs, out_dir, positions=(1, 4, 5)):
    """1D marginal posteriors of benchmark scenarios across gamma (Test Set B)."""
    if not runs:
        return
    info0 = runs[0]["info"]
    data_dir = localize_data_dir(info0["data_dir"])
    lbound = float(info0["lbound"])
    z_true = np.load(os.path.join(data_dir, "alphas_test_fine.npy")).astype(float)
    grid = np.linspace(lbound, 1.0, 400)

    fams = []
    for r in runs:
        pred_file = os.path.join(r["path"], PRED["B"])
        if os.path.exists(pred_file):
            fams.append((r["gamma"], family_from_preds(np.load(pred_file, allow_pickle=True).item(), lbound)))
    if not fams:
        return

    fig, axes = plt.subplots(len(positions), C.N_DIMS, figsize=(P.FIG_WIDTH, 1.35 * len(positions) + 0.45), squeeze=False)
    fig.subplots_adjust(left=0.055, right=0.975, top=0.86, bottom=0.14, wspace=0.14, hspace=0.35)
    cmap = plt.get_cmap("coolwarm")

    for i, pos in enumerate(positions):
        for d in range(C.N_DIMS):
            ax = axes[i, d]
            for idx, (g, fam) in enumerate(fams):
                is_one = abs(g - 1.0) < 1e-6
                col = "#000000" if is_one else cmap(idx / max(len(fams) - 1, 1))
                lw = 1.4 if is_one else 0.9
                ls = "-" if is_one else "--"
                pdf = fam.marginal_pdf_z(int(pos))(grid, d)
                ax.plot(grid, pdf, color=col, lw=lw, ls=ls, label=rf"$\gamma={g:g}$" if (i == 0 and d == 0) else None)
            ax.axvline(z_true[pos, d], color="#D55E00", lw=0.9, ls=":", label=r"true $z^*$" if (i == 0 and d == 0) else None)
            ax.set_xlim(lbound, 1.0)
            ax.set_yticks([])
            if i == 0:
                ax.set_title(C.LABELS_7D[d], fontsize=8.5)
            if d == 0:
                ax.set_ylabel(f"P{pos}", fontsize=8.5)
            if i == len(positions) - 1:
                ax.set_xticks([0.5, 0.75, 1.0])
                ax.tick_params(labelsize=7)
            else:
                ax.set_xticklabels([])
            P.despine(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=len(handles), fontsize=7.5)
    P.save_figure(fig, os.path.join(out_dir, "Fig_gamma_benchmarks_1D.png"), tight=False)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="R3.2 sensitivity analysis to the entropy weight gamma.")
    parser.add_argument("--root-dir", default=DEFAULT_ROOT, help="directory containing the GC-TN runs")
    parser.add_argument("--runs", nargs="+", default=None, help="explicit list of run folders (optional)")
    parser.add_argument("--modes", type=int, default=5)
    parser.add_argument("--obs", default="fv")
    parser.add_argument("--schedule", default="plateau")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", default=None, help="output directory (default: <root-dir>/gamma_study)")
    args, _ = parser.parse_known_args()

    out_dir = args.out_dir or os.path.join(args.root_dir, "gamma_study")
    os.makedirs(out_dir, exist_ok=True)

    runs = discover_gamma_runs(args.root_dir, family="GC-TN", modes=args.modes, obs=args.obs,
                               schedule=args.schedule, seed=args.seed, explicit_runs=args.runs)
    if not runs:
        print(f"No evaluated GC-TN runs found in {args.root_dir}.")
        return

    print(f"\n=== Gamma sensitivity study ({len(runs)} runs found) ===")
    for r in runs:
        print(f"  gamma = {r['gamma']:<5g} (gamma^2 = {r['gamma2']:<6g}) -> {r['name']}")

    records = collect_metrics(runs)

    # Save CSVs (exclude array columns from CSV)
    scalar_records = [{k: v for k, v in rec.items() if not isinstance(v, list)} for rec in records]
    df_all = pd.DataFrame(scalar_records)

    val_cols = [c for c in df_all.columns if c in ("gamma", "gamma2", "run", "epochs_run", "best_epoch", "converged",
                                                   "training_time_min") or c.startswith("val_")]
    test_cols = [c for c in df_all.columns if c in ("gamma", "gamma2", "run") or c.startswith("A_") or c.startswith("B_")]

    df_all[val_cols].to_csv(os.path.join(out_dir, "Table_gamma_validation.csv"), index=False, float_format="%.5g")
    df_all[test_cols].to_csv(os.path.join(out_dir, "Table_gamma_test_sets.csv"), index=False, float_format="%.5g")

    write_latex_validation_table(records, os.path.join(out_dir, "Table_gamma_validation.tex"))
    write_latex_test_table(records, os.path.join(out_dir, "Table_gamma_test_sets.tex"))

    with open(os.path.join(out_dir, "gamma_sensitivity_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    plot_gamma_sensitivity(records, out_dir)
    plot_gamma_benchmarks_1d(runs, out_dir)

    # Print concise summary table to console
    summary_cols = ["gamma", "gamma2", "val_all_MSE", "val_all_MAE", "val_all_ALL_phys", "val_all_MACE",
                    "val_all_cov95", "val_all_width95", "val_damaged_MACE", "val_damaged_cov95",
                    "val_prior_MACE", "val_prior_cov95", "val_dKL_vs_gamma1"]
    present_cols = [c for c in summary_cols if c in df_all.columns]
    print("\n--- Validation Set Summary (R3.2) ---")
    print(df_all[present_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[OK] Tables and figures written to: {out_dir}")


if __name__ == "__main__":
    main()
