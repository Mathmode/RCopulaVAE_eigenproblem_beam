#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/model_comparison/compare_convergence.py

Quality against cost of the posterior families on one data set and one set of observed quantities: collects the
evaluated runs trained to convergence (run_convergence_gpu.sh, --schedule plateau) and those with the fixed cosine
schedule, and writes Output/model_comparison/<data set>/convergence_obs-<obs>/:

  Table_convergence.csv/.tex   per run: epochs, best epoch, whether the convergence rule was met, s/epoch, time to the
                               best weights, decoder solves per step, validation -ELBO and its paired difference to the
                               proposed model (= difference of the mean KL divergence to the exact posterior, nats per
                               observation), KL lower bound, Pareto-k, paper metrics on test sets A and B.
  Table_time_to_level.csv      runs trained to convergence: epochs and hours until the monitor is within delta of the
                               best -ELBO of all the runs (L*), for several delta. "-": never reached.
  Fig_convergence.png/.pdf     distance to L* against epochs and against training time (monitor of the runs trained to
                               convergence; markers: final value of the cosine runs), and learning rates.

s/epoch: the median of the --timing runs in Output/model_comparison/timing/ (idle GPU) if they exist, otherwise the median
of the run itself, which depends on what else the GPU was doing.

    python MODULES/model_comparison/compare_convergence.py --data 03Nov_FINAL_GirderBearings_7D_5sensors_5modes --obs fv
"""

import os
import sys
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
from MODULES.model_comparison.compare_models import COLORS, collect_runs, latex_table, paired_difference, per_obs

DELTAS = [0.20, 0.10, 0.05, 0.02, 0.01]             # distance to the best -ELBO [nats per observation]
LINESTYLES = {1: "--", 3: "-", 5: "-."}


def load_history(row):
    return np.load(os.path.join(row["path"], "model_history.npy"), allow_pickle=True).item()


def timing_runs(obs):
    """Median epoch time {(family, K): s} of the --timing runs of run_convergence_gpu.sh."""
    folder = os.path.join(C.PROJECT_ROOT, "Output", "model_comparison", "timing")
    out = {}
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        run = os.path.join(folder, name)
        if os.path.exists(os.path.join(run, "Problem_info.npy")):
            info = np.load(os.path.join(run, "Problem_info.npy"), allow_pickle=True).item()
            if info.get("obs_types") == obs:
                t = np.load(os.path.join(run, "model_history.npy"), allow_pickle=True).item()["epoch_time_s"]
                out[(info["family"], int(info["n_components"]))] = float(np.median(t[5:]))    # after the graph tracing
    return out


def seconds_per_epoch(row, history, clean):
    if (row["family"], row["K"]) in clean:
        return clean[(row["family"], row["K"])]
    return float(np.median(history["epoch_time_s"])) if "epoch_time_s" in history else row["s_per_epoch"]


def label_of(row):
    seed = "" if row["seed"] == 1234 else f", seed {row['seed']}"
    return f"{row['family']} K={row['K']}, {'to convergence' if row['schedule'] == 'plateau' else 'cosine'}{seed}"


def build_table(rows, ref, clean, out_dir):
    table = {}
    for r in rows:
        h = load_history(r)
        spe = seconds_per_epoch(r, h, clean)
        solves = int(r["info"]["n_samples"]) * (r["K"] if r["family"] == "GMM" else 1)
        d, se = paired_difference(per_obs(r), per_obs(ref))
        val = r["quality"]["val"]
        col = {"epochs run": int(r["info"]["epochs_run"]), "best epoch": r["best_epoch"],
               "converged": {True: "yes", False: "no (cap)", None: "-"}[r["info"].get("converged")],
               "s/epoch": spe, "time to best [h]": r["best_epoch"] * spe / 3600.0, "parameters": r["parameters"],
               "decoder solves per obs. and step": solves, "-ELBO (val)": val["neg_elbo_mean"],
               "d(-ELBO) vs reference": d, "SE": se, "KL lower bound (val)": val["kl_lower_mean"],
               "median k_hat (val)": val["k_hat_median"], "k_hat < 0.5 (val)": val["frac_k_hat_below_0.5"],
               "k_hat < 0.7 (val)": val["frac_k_hat_below_0.7"]}
        for t in ("A", "B"):
            m = r[f"metrics_{t}"]
            if m:
                for k, lab in (("MSE", "MSE"), ("ALL", "ALL"), ("MACE", "MACE"), ("coverage95", "cov95")):
                    col[f"{lab} ({t})"] = m["all_7"][k]
                if "prior_scenarios" in m:
                    col[f"cov95 prior scen. ({t})"] = m["prior_scenarios"]["coverage95"]
        table[label_of(r)] = col
    df = pd.DataFrame(table)
    df.index.name = "metric"
    df.to_csv(os.path.join(out_dir, "Table_convergence.csv"), float_format="%.5g")
    latex_table(df, os.path.join(out_dir, "Table_convergence.tex"),
                f"Posterior families trained to convergence and with the fixed schedule. d(-ELBO): paired difference to "
                f"{label_of(ref)}, i.e. difference of the mean KL divergence to the exact posterior.", "tab:convergence")
    return df


def monitor_curve(history):
    """Epochs and values of the monitor (runs trained to convergence)."""
    values = np.asarray(history.get("monitor_neg_elbo", []), dtype=np.float64)
    epochs = np.flatnonzero(np.isfinite(values))
    return epochs + 1, values[epochs]


def time_to_level(rows, level, clean, out_dir):
    table = []
    for r in rows:
        if r["schedule"] != "plateau":
            continue
        h = load_history(r)
        spe = seconds_per_epoch(r, h, clean)
        epochs, values = monitor_curve(h)
        row = {"run": label_of(r)}
        for delta in DELTAS:
            hit = np.flatnonzero(values <= level + delta)
            row[f"epochs (delta={delta})"] = int(epochs[hit[0]]) if len(hit) else "-"
            row[f"hours (delta={delta})"] = round(epochs[hit[0]] * spe / 3600.0, 3) if len(hit) else "-"
        table.append(row)
    df = pd.DataFrame(table)
    if len(df):
        df.to_csv(os.path.join(out_dir, "Table_time_to_level.csv"), index=False)
    return df


def plot_convergence(rows, level, clean, out_dir, title):
    with plt.style.context(["default", PAPER_RC]):
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
        for r in rows:
            h = load_history(r)
            spe = seconds_per_epoch(r, h, clean)
            color, ls = COLORS.get(r["family"], "k"), LINESTYLES.get(r["K"], ":")
            if r["schedule"] == "plateau":
                epochs, values = monitor_curve(h)
                gap = np.maximum(values - level, 1e-3)
                ax[0].plot(epochs, gap, ls, color=color, lw=1.5, label=label_of(r))
                ax[1].plot(epochs * spe / 3600.0, gap, ls, color=color, lw=1.5)
                ax[2].plot(np.arange(1, len(h["lr"]) + 1), h["lr"], ls, color=color, lw=1.3)
            else:
                gap = max(r["quality"]["val"]["neg_elbo_mean"] - level, 1e-3)
                ax[0].plot(r["best_epoch"], gap, "o", mfc="none", mec=color, ms=8, mew=1.6, label=label_of(r))
                ax[1].plot(r["best_epoch"] * spe / 3600.0, gap, "o", mfc="none", mec=color, ms=8, mew=1.6)
                ax[2].plot(np.arange(1, len(h["lr"]) + 1), h["lr"], ls, color=color, lw=0.8, alpha=0.5)
        ax[0].set(xlabel="epoch", ylabel=r"$-$ELBO(val) minus the best of all runs [nats/obs.]", yscale="log",
                  title="(a) Distance to the best $-$ELBO against epochs")
        ax[1].set(xlabel="training time [h]", yscale="log", title="(b) Against training time")
        ax[2].set(xlabel="epoch", ylabel="learning rate", yscale="log", title="(c) Learning rate")
        for a in ax:
            a.grid(True, ls=":", alpha=0.5)
        ax[0].legend(fontsize=8)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        save_figure(fig, os.path.join(out_dir, "Fig_convergence.png"))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Quality against cost of the posterior families.")
    parser.add_argument("--data", default="03Nov_FINAL_GirderBearings_7D_5sensors_5modes", help="data set folder name")
    parser.add_argument("--obs", default="fv", help="observed quantities of the runs: fv or f")
    parser.add_argument("--runs", nargs="*", default=None, help="run folders (default: every evaluated run)")
    parser.add_argument("--out-dir", default=None)
    args, _ = parser.parse_known_args()
    out_dir = args.out_dir or os.path.join(C.PROJECT_ROOT, "Output", "model_comparison", args.data,
                                           f"convergence_obs-{args.obs}")
    os.makedirs(out_dir, exist_ok=True)
    rows = collect_runs(args.data, args.runs, None, args.obs)
    if not rows:
        raise SystemExit(f"No evaluated runs on {args.data} with obs = {args.obs}.")
    rows.sort(key=lambda r: (r["family"], r["K"], r["schedule"], r["seed"]))
    proposed = [r for r in rows if r["family"] == "GC-KS" and r["K"] == 3]
    ref = next((r for r in proposed if r["schedule"] == "plateau"), proposed[0] if proposed else rows[0])
    level = min(r["quality"]["val"]["neg_elbo_mean"] for r in rows)
    clean = timing_runs(args.obs)
    print(f"{len(rows)} evaluated runs, obs = {args.obs}; reference: {label_of(ref)}; best -ELBO L* = {level:.4f}; "
          f"timing runs for {sorted(clean) or 'none (times from the runs themselves)'}")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(build_table(rows, ref, clean, out_dir).to_string(float_format=lambda x: f"{x:.4f}"))
    ttl = time_to_level(rows, level, clean, out_dir)
    if len(ttl):
        print("\nEpochs and hours to reach L* + delta (runs trained to convergence)")
        print(ttl.to_string(index=False))
    plot_convergence(rows, level, clean, out_dir, f"{args.data}, observed: {args.obs}")
    print(f"\nTables and figure in {out_dir}")


if __name__ == "__main__":
    main()
