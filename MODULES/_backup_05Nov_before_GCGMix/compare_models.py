#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/model_comparison/compare_models.py

Collects the evaluated runs of one data set (postprocessing.py + evaluate_posterior_quality.py) and writes
Output/model_comparison/<data set>/:

  Table_KS_components.csv/.tex    number K of Kumaraswamy components of the proposed model, VALIDATION set only:
                                  mean -ELBO difference to the best K (= difference of the mean KL divergence to the
                                  exact posterior, nats/observation) with its paired standard error, KL lower bound,
                                  Pareto-k, calibration (MACE, 95 % coverage), trainable parameters, time per epoch.
                                  K* = smallest K whose -ELBO is within one standard error of the best K
                                  (one-standard-error rule, Hastie et al. 2009, Sec. 7.10).
  Table_R1.4_comparison.csv/.tex  proposed model (K*) vs the baselines (GC-TN: Ref. 41 / submitted version; GMM:
                                  Ref. 39): validation -ELBO difference, paper metrics on test sets A and B, agreement
                                  with the SMC reference posteriors of the benchmarks, cost.
  Fig_KS_components.png/.pdf      selection of K;   Fig_model_comparison.png/.pdf  reliability diagrams (A and B).

    python MODULES/model_comparison/compare_models.py --data 03Nov_FINAL_GirderBearings_7D_5sensors_5modes
"""

import os
import re
import sys
import json
import argparse

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.plotting import PAPER_RC, save_figure

GROUPS = ["review29oct_fixed", "baseline_GaussCopula_GaussMarginals", "baseline_GaussMixture_K5",
          "review29oct_fixed/baseline_GaussCopula_GaussMarginals", "review29oct_fixed/baseline_GaussMixture_K5"]
TEST_DIRS = {"A": "TestA_same_FE_model_10el", "B": "TestB_refined_FE_model_50el"}
FAMILY_NAMES = {"GC-KS": "Gaussian copula + KS-mixture marginals (proposed)",
                "GC-TN": "Gaussian copula + truncated-Gaussian marginals (Ref. 41, submitted version)",
                "GMM": "Gaussian mixture, no copula (Ref. 39)"}
COLORS = {"GC-KS": "#1F77B4", "GC-TN": "#D62728", "GMM": "#2CA02C"}


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def count_parameters(run):
    """Trainable parameters of the encoder from the weights file (Dense kernels and biases)."""
    import h5py
    total = 0
    with h5py.File(os.path.join(run, "final_model_weights.weights.h5"), "r") as fh:
        def visit(name, obj):
            nonlocal total
            if isinstance(obj, h5py.Dataset) and re.search(r"dense(_\d+)?/vars/\d+$", name) \
                    and not name.startswith("optimizer"):
                total += int(np.prod(obj.shape))
        fh.visititems(visit)
    return total


def collect_runs(data_name, runs=None, schedule=None, obs=None):
    """Evaluated runs of a data set; schedule (cosine, plateau) and obs (fv, f) keep only the runs of that protocol."""
    out_root = os.path.join(C.PROJECT_ROOT, "Output")
    folders = []
    if runs:
        for r in runs:
            folders.append(r if os.path.isdir(r) else next(
                os.path.join(out_root, g, r) for g in GROUPS if os.path.isdir(os.path.join(out_root, g, r))))
    else:
        for g in GROUPS:
            gdir = os.path.join(out_root, g)
            if os.path.isdir(gdir):
                folders += [os.path.join(gdir, d) for d in sorted(os.listdir(gdir))]
    rows = []
    for run in folders:
        pi_path = os.path.join(run, "Problem_info.npy")
        if not os.path.exists(pi_path):
            continue
        info = np.load(pi_path, allow_pickle=True).item()
        if re.split(r"[\\/]", str(info["data_dir"]).rstrip("\\/"))[-1] != data_name:
            continue
        if (schedule and info.get("schedule", "cosine") != schedule) or (obs and info.get("obs_types") != obs):
            continue
        q = read_json(os.path.join(run, "Posterior_quality", "ELBO_summary.json"))
        if q is None:
            print(f"  (not evaluated yet, skipped: {os.path.basename(run)} -> run evaluate_posterior_quality.py)")
            continue
        fam = info.get("family", "GC-KS")
        row = {"run": os.path.basename(run), "path": run, "family": fam,
               "K": int(info.get("n_components", info.get("num_KS", 1))), "seed": int(info.get("seed", 1234)),
               "parameters": int(info.get("trainable_parameters") or count_parameters(run)),
               "s_per_epoch": float(info["training_time_sec"]) / max(int(info["epochs_run"]), 1),
               "best_epoch": int(info.get("best_epoch", 0)), "quality": q, "info": info,
               "schedule": info.get("schedule", "cosine"), "obs": info.get("obs_types")}
        for t, sub in TEST_DIRS.items():
            row[f"metrics_{t}"] = read_json(os.path.join(run, sub, "Manuscript_Metrics.json"))
            gt = os.path.join(run, sub, "Benchmarks_VAE_vs_reference", "GT_vs_Posterior_Metrics.csv")
            row[f"gt_{t}"] = pd.read_csv(gt) if os.path.exists(gt) else None
        rows.append(row)
    return rows


def per_obs(run_row, split="val"):
    return pd.read_csv(os.path.join(run_row["path"], "Posterior_quality", f"ELBO_{split}.csv"))["neg_elbo"].to_numpy()


def paired_difference(a, b):
    d = a - b
    return float(np.mean(d)), float(np.std(d, ddof=1) / np.sqrt(len(d)))


def latex_table(df, path, caption, label):
    cols = list(df.columns)
    lines = [r"\begin{table}[H]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "l" * (df.index.nlevels) + "r" * len(cols) + "}", r"\toprule",
             " & ".join([df.index.name or ""] + [str(c).replace("%", r"\%") for c in cols]) + r" \\", r"\midrule"]
    for idx, r in df.iterrows():
        cells = [str(idx).replace("%", r"\%").replace("_", r"\_")]
        for v in r.to_numpy():
            cells.append(v if isinstance(v, str) else (f"{v:.4g}" if np.isfinite(v) else "--"))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# =============================================================================
# 1. NUMBER OF KUMARASWAMY COMPONENTS (validation set only)
# =============================================================================
def select_ks_components(rows, out_dir):
    ks = [r for r in rows if r["family"] == "GC-KS"]
    if not ks:
        print("No evaluated KS runs.")
        return None, None
    by_K = {}
    for r in ks:
        by_K.setdefault(r["K"], []).append(r)
    # per observation, averaged over the seeds of every K (same validation observations for all runs)
    nelbo = {K: np.mean([per_obs(r) for r in rs], axis=0) for K, rs in sorted(by_K.items())}
    best_K = min(nelbo, key=lambda K: nelbo[K].mean())
    table = []
    for K, rs in sorted(by_K.items()):
        diff, se = paired_difference(nelbo[K], nelbo[best_K]) if K != best_K else (0.0, 0.0)
        seed_means = [r["quality"]["val"]["neg_elbo_mean"] for r in rs]
        seed_sd = float(np.std(seed_means, ddof=1)) if len(rs) > 1 else np.nan
        vq = [r["quality"]["val"] for r in rs]
        vc = [r["quality"].get("val_calibration", {}).get("all_7", {}) for r in rs]
        table.append({"K": K, "seeds": len(rs), "dNegELBO_val": diff, "SE_paired": se,
                      "SD_between_seeds": seed_sd, "NegELBO_val": float(np.mean(seed_means)),
                      "KL_lower_val": float(np.mean([v["kl_lower_mean"] for v in vq])),
                      "frac_khat_lt_0.7": float(np.mean([v["frac_k_hat_below_0.7"] for v in vq])),
                      "median_khat": float(np.mean([v["k_hat_median"] for v in vq])),
                      "MACE_val": float(np.mean([c.get("MACE", np.nan) for c in vc])),
                      "coverage95_val": float(np.mean([c.get("coverage95", np.nan) for c in vc])),
                      "parameters": rs[0]["parameters"], "s_per_epoch": float(np.mean([r["s_per_epoch"] for r in rs]))})
    df = pd.DataFrame(table).set_index("K")
    # one-standard-error rule: the uncertainty of a difference is the paired SE (Monte Carlo + observations) or the
    # seed-to-seed spread, whichever is larger
    tol = np.maximum(df["SE_paired"].to_numpy(), np.nan_to_num(df["SD_between_seeds"].to_numpy() / np.sqrt(df["seeds"])))
    best_tol = tol[list(df.index).index(best_K)]
    within = df["dNegELBO_val"].to_numpy() <= np.maximum(tol, best_tol)
    K_star = int(df.index[np.argmax(within)])
    df["within_1SE_of_best"] = within
    df.to_csv(os.path.join(out_dir, "Table_KS_components.csv"), float_format="%.5g")
    show = df[["seeds", "dNegELBO_val", "SE_paired", "KL_lower_val", "frac_khat_lt_0.7", "MACE_val",
               "coverage95_val", "parameters", "s_per_epoch"]].copy()
    show.columns = ["seeds", "d(-ELBO) [nats]", "SE", "KL lower bound", "k<0.7", "MACE (val)", "cov95 (val)",
                    "parameters", "s/epoch"]
    latex_table(show, os.path.join(out_dir, "Table_KS_components.tex"),
                f"Number of Kumaraswamy components (validation set). d(-ELBO): mean difference to the best K "
                f"(K = {best_K}), i.e. difference of the mean KL divergence to the exact posterior; selected "
                f"K = {K_star} (one-standard-error rule).", "tab:ks_components")

    with plt.style.context(["default", PAPER_RC]):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        Ks = df.index.to_numpy()
        ax[0].errorbar(Ks, df["dNegELBO_val"], yerr=np.maximum(tol, 1e-12), fmt="o-", color="#1F77B4", capsize=4, lw=1.6)
        ax[0].axhline(0, color="gray", lw=0.8, ls=":")
        ax[0].plot([K_star], [df.loc[K_star, "dNegELBO_val"]], "s", ms=13, mfc="none", mec="#D62728", mew=2,
                   label=f"selected $K={K_star}$")
        ax[0].set_ylabel(r"$\Delta(-\mathrm{ELBO})$ to the best $K$ [nats/obs.]")
        ax[0].set_title("(a) Validation $-$ELBO (= KL to exact posterior + const.)")
        ax[0].legend()
        ax[1].plot(Ks, 100 * df["frac_khat_lt_0.7"], "o-", color="#2CA02C", lw=1.6)
        ax[1].set_ylabel(r"observations with $\hat{k} < 0.7$ [%]")
        ax[1].set_title(r"(b) Pareto-$\hat{k}$ diagnostic (validation)")
        ax[2].plot(Ks, df["MACE_val"], "o-", color="#9467BD", lw=1.6, label="MACE")
        ax2 = ax[2].twinx()
        ax2.plot(Ks, df["coverage95_val"], "s--", color="#8C564B", lw=1.4, label="95% coverage")
        ax2.set_ylabel("95% coverage")
        ax[2].set_ylabel("MACE")
        ax[2].set_title("(c) Calibration (validation)")
        for a in ax:
            a.set_xlabel("Kumaraswamy components $K$")
            a.set_xticks(Ks)
            a.grid(True, ls=":", alpha=0.5)
        fig.tight_layout()
        save_figure(fig, os.path.join(out_dir, "Fig_KS_components.png"))
        plt.close(fig)
    print("\n=== Number of Kumaraswamy components (validation set) ===")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"Best K = {best_K}; selected K* = {K_star} (smallest K within one standard error of the best).")
    return K_star, df


# =============================================================================
# 2. R1.4: PROPOSED MODEL VS BASELINES
# =============================================================================
def compare_families(rows, K_star, out_dir):
    chosen = [r for r in rows if r["family"] == "GC-KS" and r["K"] == K_star]
    chosen = chosen[:1] + [r for r in rows if r["family"] != "GC-KS"]
    if len(chosen) < 2:
        print("No baseline evaluated yet: R1.4 table skipped.")
        return None
    ref = per_obs(chosen[0])
    table = {}
    for r in chosen:
        name = f"{r['family']} (K={r['K']})"
        d, se = paired_difference(per_obs(r), ref)
        col = {"d(-ELBO) val vs proposed": d, "SE": se,
               "k_hat < 0.7 (val)": r["quality"]["val"]["frac_k_hat_below_0.7"]}
        for t in ("A", "B"):
            m = r[f"metrics_{t}"]
            if m:
                for k, lab in (("MSE", "MSE"), ("MAE", "MAE"), ("ALL", "ALL"), ("MACE", "MACE"),
                               ("coverage95", "cov95")):
                    col[f"{lab} ({t})"] = m["all_7"][k]
                col[f"MACE damaged ({t})"] = m["damaged_params"]["MACE"]
                if "prior_scenarios" in m:
                    col[f"MACE prior scen. ({t})"] = m["prior_scenarios"]["MACE"]
            gt = r[f"gt_{t}"]
            if gt is not None:
                col[f"|dmean| vs SMC ({t})"] = gt["mean_abs_diff_mean"].mean()
                col[f"std ratio vs SMC ({t})"] = gt["mean_std_ratio_VAE_over_ref"].mean()
                col[f"energy dist. vs SMC ({t})"] = gt["energy_distance"].mean()
        col.update({"parameters": r["parameters"], "s/epoch": r["s_per_epoch"]})
        table[name] = col
    df = pd.DataFrame(table)
    df.index.name = "metric"
    df.to_csv(os.path.join(out_dir, "Table_R1.4_comparison.csv"), float_format="%.5g")
    latex_table(df, os.path.join(out_dir, "Table_R1.4_comparison.tex"),
                "Proposed model and baselines (same data, decoder, likelihood, ELBO and training). A: test set from "
                "the 10-element model; B: test set from the 50-element model.", "tab:r14_comparison")
    print("\n=== R1.4: proposed model vs baselines ===")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    with plt.style.context(["default", PAPER_RC]):
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)
        for j, t in enumerate(("A", "B")):
            ax[j].plot([0, 1], [0, 1], "k--", lw=1.2, label="ideal")
            for r in chosen:
                m = r[f"metrics_{t}"]
                if not m:
                    continue
                lv = np.asarray(m["nominal_levels"])
                ax[j].plot(lv, m["all_7"]["empirical_coverage"], "o-", ms=4, lw=1.6, color=COLORS[r["family"]],
                           label=f"{FAMILY_NAMES[r['family']]}, MACE = {m['all_7']['MACE']:.3f}")
                ax[j].plot(lv, m["damaged_params"]["empirical_coverage"], "s:", ms=3, lw=1.1,
                           color=COLORS[r["family"]], alpha=0.8)
            ax[j].set_title(f"Test set {t} ({'10' if t == 'A' else '50'}-element data)")
            ax[j].set_xlabel("Nominal level $p$")
            ax[j].set_aspect("equal")
            ax[j].grid(True, ls=":", alpha=0.5)
        ax[0].set_ylabel(r"Empirical coverage $\hat{p}$ (solid: all, dotted: damaged)")
        ax[0].legend(fontsize=8.5, loc="upper left")
        fig.tight_layout()
        save_figure(fig, os.path.join(out_dir, "Fig_model_comparison.png"))
        plt.close(fig)
    return df


def main():
    parser = argparse.ArgumentParser(description="K selection and comparison with the baselines (R1.4).")
    parser.add_argument("--data", default="03Nov_FINAL_GirderBearings_7D_5sensors_5modes", help="data set folder name")
    parser.add_argument("--runs", nargs="*", default=None, help="run folders (default: every evaluated run)")
    parser.add_argument("--K", type=int, default=None, help="K of the proposed model in the R1.4 table "
                                                            "(default: the selected K*)")
    parser.add_argument("--schedule", default="cosine", help="only runs of this schedule: cosine or plateau "
                                                             "(see compare_convergence.py for the two together)")
    parser.add_argument("--obs", default="fv", help="only runs with these observed quantities: fv or f")
    args, _ = parser.parse_known_args()
    out_dir = os.path.join(C.PROJECT_ROOT, "Output", "model_comparison", args.data)
    if (args.schedule, args.obs) != ("cosine", "fv"):      # the default tables stay where they are
        out_dir = os.path.join(out_dir, f"{args.schedule}_obs-{args.obs}")
    os.makedirs(out_dir, exist_ok=True)
    rows = collect_runs(args.data, args.runs, args.schedule, args.obs)
    print(f"{len(rows)} evaluated runs on {args.data}: " + ", ".join(r["run"] for r in rows))
    K_star, _ = select_ks_components(rows, out_dir)
    compare_families(rows, args.K or K_star or 3, out_dir)
    print(f"\nTables and figures in {out_dir}")


if __name__ == "__main__":
    main()
