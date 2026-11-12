#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/comparison_surrogate_decoder/compare_eigen_vs_surrogate.py

STEP 3: comparison of the proposed model (differentiable eigenvalue solver as decoder) with the approach of Ref. 41 (the same
Gaussian copula VAE with a pre-trained, frozen surrogate decoder), from the saved results of the two runs (after
postprocessing.py and evaluate_posterior_quality.py). Both runs must use the same data set; every metric is computed with the
exact model (eigenvalue solver and calibrated likelihood), so both approximations are judged against the same exact posterior.

Outputs in --out-dir (default Output/comparison_surrogate_decoder/comparison_<data tag>):
  comparison_table.csv / .tex     cost, decoder accuracy, closeness to the exact posterior, paper metrics (test sets A and B),
                                  agreement with the exact posteriors of the benchmarks, detection and false-alarm rates
  comparison_summary.json         the same numbers, plus the paired validation -ELBO difference
  Fig_benchmark_marginals.png/.pdf exact marginals of benchmarks P1, P4 and P5 (test set A): exact posterior (SMC) and both VAEs

    python MODULES/comparison_surrogate_decoder/compare_eigen_vs_surrogate.py \
        --eigen-run Output/baseline_GaussCopula_GaussMarginals/GCTN_06Nov-5modes_7D_Gamma1.0_Conv6000 \
        --surrogate-run Output/comparison_surrogate_decoder/VAE_surrogate
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.posterior_families import family_from_preds

SUB = {"A": "TestA_same_FE_model_10el", "B": "TestB_refined_FE_model_50el"}
PRED = {"A": "Test_predicted_props.npy", "B": "Test_predicted_props_test_fine.npy"}
SPLIT = {"A": "test", "B": "test_fine"}
LABELS = {"eigen": "Proposed (eigenvalue solver)", "surrogate": "Ref. 41 approach (surrogate decoder)"}


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(C.PROJECT_ROOT, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def summarise_run(run):
    info = np.load(os.path.join(run, "Problem_info.npy"), allow_pickle=True).item()
    hist = np.load(os.path.join(run, "model_history.npy"), allow_pickle=True).item()
    out = {"run": os.path.basename(run), "data_dir": info["data_dir"], "decoder": info.get("decoder", "eigenvalue solver"),
           "encoder_parameters": int(info["trainable_parameters"]),
           "surrogate_parameters": int(info.get("surrogate_parameters") or 0),
           "epochs_run": int(info["epochs_run"]), "best_epoch": int(info["best_epoch"]), "converged": info.get("converged"),
           "vae_training_time_min": info["training_time_sec"] / 60.0,
           "vae_time_per_epoch_s": info["training_time_sec"] / info["epochs_run"],
           "surrogate_training_time_min": (info.get("surrogate_training_time_sec") or 0.0) / 60.0,
           "final_lr": float(np.asarray(hist.get("lr", [np.nan]))[-1])}
    out["total_training_time_min"] = out["vae_training_time_min"] + out["surrogate_training_time_min"]
    q = read_json(os.path.join(run, "Posterior_quality", "ELBO_summary.json"))
    for split in ("val", "test", "test_fine"):
        if split in q:
            s = q[split]
            out.update({f"{split}_neg_elbo": s["neg_elbo_mean"], f"{split}_kl_lower": s["kl_lower_mean"],
                        f"{split}_k_hat_median": s["k_hat_median"], f"{split}_frac_k_hat_below_0.7": s["frac_k_hat_below_0.7"]})
    lb = float(info["lbound"])
    for s in "AB":
        t = pd.read_csv(os.path.join(run, SUB[s], "Manuscript_Metrics.csv"), index_col=0)
        for col, tag in (("All parameters", "all"), ("Damaged (z*<0.95)", "damaged"), ("Undamaged (z*>=0.95)", "undamaged"),
                         ("Scenarios drawn from the prior", "prior")):
            for met in ("MSE", "MAE", "ALL", "MACE", "95% coverage"):
                out[f"{s}_{tag}_{met}"] = float(t.loc[met, col])
        g = pd.read_csv(os.path.join(run, SUB[s], "Benchmarks_VAE_vs_reference", "GT_vs_Posterior_Metrics.csv"))
        out.update({f"{s}_bench_abs_dmean": g.mean_abs_diff_mean.mean(), f"{s}_bench_W1": g.mean_W1.mean(),
                    f"{s}_bench_sd_ratio": g.mean_std_ratio_VAE_over_ref.mean(), f"{s}_bench_energy": g.energy_distance.mean(),
                    f"{s}_bench_zstar_in_95": int(g.zstar_in_VAE95.sum()), f"{s}_bench_zstar_in_ref95": int(g.zstar_in_ref95.sum()),
                    f"{s}_bench_n_params": int(7 * len(g))})
        z = np.load(os.path.join(info["data_dir"], f"alphas_{SPLIT[s]}.npy")).astype(float)
        fam = family_from_preds(np.load(os.path.join(run, PRED[s]), allow_pickle=True).item(), lb)
        qv, _ = fam.quantiles(np.array([0.025, 0.975]))
        alarm = qv[..., 1] < 0.95
        out[f"{s}_false_alarm_rate"] = float(alarm[z >= 0.95].mean())
        out[f"{s}_detection_rate"] = float(alarm[z < 0.90].mean())
    return out, info


def paired_val(eig_run, sur_run):
    a = pd.read_csv(os.path.join(eig_run, "Posterior_quality", "ELBO_val.csv"))
    b = pd.read_csv(os.path.join(sur_run, "Posterior_quality", "ELBO_val.csv"))
    if not (a.obs.values == b.obs.values).all():
        raise ValueError("the two runs were evaluated on different validation observations")
    d = b.neg_elbo.values - a.neg_elbo.values
    return {"surrogate_minus_eigen": float(d.mean()), "paired_se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "share_eigen_better": float(np.mean(d > 0)), "n_obs": int(len(d))}


def benchmark_figure(eig_run, sur_run, info, out_dir, positions=(1, 4, 5)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from MODULES.review29oct_fixed.reference_posterior import compute_reference_posterior
    from MODULES.review29oct_fixed.plotting import kde_1d
    lb = float(info["lbound"])
    fams = {k: family_from_preds(np.load(os.path.join(r, PRED["A"]), allow_pickle=True).item(), lb)
            for k, r in (("eigen", eig_run), ("surrogate", sur_run))}
    z_true = np.load(os.path.join(info["data_dir"], "alphas_test.npy")).astype(float)
    names = list(C.PARAM_NAMES)[:z_true.shape[1]]
    grid = np.linspace(lb, 1.0, 400)
    fig, axes = plt.subplots(len(positions), len(names), figsize=(2.0 * len(names), 1.9 * len(positions)), squeeze=False)
    for i, p in enumerate(positions):
        ref = compute_reference_posterior(info["data_dir"], int(p), float(info["gamma"]) ** 2, info["obs_types"], 4000,
                                          split="test")[0]
        for d in range(len(names)):
            ax = axes[i, d]
            ax.plot(grid, kde_1d(ref[:, d], grid, lb), color="0.3", lw=1.6, label="exact posterior (SMC)")
            ax.plot(grid, fams["eigen"].marginal_pdf_z(int(p))(grid, d), color="tab:blue", lw=1.3, label=LABELS["eigen"])
            ax.plot(grid, fams["surrogate"].marginal_pdf_z(int(p))(grid, d), color="tab:red", lw=1.3, ls="--",
                    label=LABELS["surrogate"])
            ax.axvline(z_true[p, d], color="k", lw=0.8, ls=":")
            ax.set_yticks([])
            if i == 0:
                ax.set_title(names[d], fontsize=9)
            if d == 0:
                ax.set_ylabel(f"P{p}", fontsize=9)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(loc="upper left", fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"Fig_benchmark_marginals.{ext}"), dpi=200)
    plt.close(fig)


def latex_table(rows, path):
    lines = [r"\begin{tabular}{p{7.0cm}p{3.6cm}p{3.6cm}}", r"\toprule",
             rf" & {LABELS['eigen']} & {LABELS['surrogate']} \\", r"\midrule"]
    for name, a, b in rows:
        if a is None:
            lines.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{name}}}}} \\")
        else:
            lines.append(rf"{name} & {a} & {b} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


REQUIRED = ["Problem_info.npy", "model_history.npy", PRED["A"], PRED["B"],
            os.path.join("Posterior_quality", "ELBO_summary.json"), os.path.join("Posterior_quality", "ELBO_val.csv")]
for _s in "AB":
    REQUIRED += [os.path.join(SUB[_s], "Manuscript_Metrics.csv"),
                 os.path.join(SUB[_s], "Benchmarks_VAE_vs_reference", "GT_vs_Posterior_Metrics.csv")]


def run_stage(run):
    """Where a run of run_convergence_gpu.sh / run_surrogate_comparison_gpu.sh stands, from the files it has written."""
    if not os.path.isdir(run):
        return "the run folder does not exist (training not started yet, or another run name)"
    if not os.path.exists(os.path.join(run, "Problem_info.npy")):
        return "training in progress (or interrupted): Problem_info.npy is written at the end of the training"
    if os.path.exists(os.path.join(run, "POSTPROCESSING_RUNNING.txt")):
        return "post-processing in progress (or interrupted): POSTPROCESSING_RUNNING.txt is present"
    if not os.path.exists(os.path.join(run, "POSTPROCESSING_DONE.txt")):
        return "trained, post-processing not started"
    if not os.path.exists(os.path.join(run, "Posterior_quality", "ELBO_summary.json")):
        return "post-processing done; posterior quality (evaluate_posterior_quality.py, last step) in progress or interrupted"
    return "complete"


def check_complete(run, label):
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(run, f))]
    if missing:
        listing = sorted(os.listdir(os.path.dirname(run))) if os.path.isdir(os.path.dirname(run)) else []
        raise SystemExit(f"\nThe {label} run is not complete, so the comparison cannot be made yet.\n"
                         f"  run:     {run}\n  stage:   {run_stage(run)}\n  missing: {', '.join(missing)}\n"
                         f"  folders in {os.path.dirname(run)}: {', '.join(listing) or '(none)'}\n"
                         "Check the end of the log of that run; when it has finished, run the comparison again.")


def main():
    parser = argparse.ArgumentParser(description="Proposed model (eigenvalue solver) vs Ref. 41 approach (surrogate decoder).")
    parser.add_argument("--eigen-run", required=True)
    parser.add_argument("--surrogate-run", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    eig_run, sur_run = resolve(args.eigen_run), resolve(args.surrogate_run)
    check_complete(eig_run, "proposed (eigenvalue solver)")
    check_complete(sur_run, "surrogate-decoder")
    E, info_e = summarise_run(eig_run)
    S, info_s = summarise_run(sur_run)
    if os.path.basename(os.path.normpath(info_e["data_dir"])) != os.path.basename(os.path.normpath(info_s["data_dir"])):
        raise ValueError(f"different data sets: {info_e['data_dir']} vs {info_s['data_dir']}")
    sur_cfg = read_json(os.path.join(info_s["surrogate_dir"], "surrogate_config.json"))
    tag = os.path.basename(os.path.normpath(info_e["data_dir"]))
    out_dir = resolve(args.out_dir) if args.out_dir else os.path.join(C.PROJECT_ROOT, "Output", "comparison_surrogate_decoder", "comparison")
    os.makedirs(out_dir, exist_ok=True)
    pv = paired_val(eig_run, sur_run)
    acc, cost = sur_cfg.get("accuracy_uniform_box", {}), sur_cfg.get("decoder_cost", {})

    def f(x, n=4):
        return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{n}f}" if isinstance(x, float) else str(x))

    rows = [("Model and cost", None, None),
            ("Decoder", "eigenvalue solver (exact, no trainable parameters)",
             f"fully connected NN, {S['surrogate_parameters']} parameters, pre-trained and frozen"),
            ("Trainable parameters of the encoder", f(E["encoder_parameters"]), f(S["encoder_parameters"])),
            ("Training steps", "one (encoder)", "two (surrogate, then encoder)"),
            ("Pre-training of the decoder [min]", "--", f(S["surrogate_training_time_min"], 1)),
            ("Training of the VAE [min] (epochs; s/epoch)", f"{E['vae_training_time_min']:.1f} ({E['epochs_run']}; {E['vae_time_per_epoch_s']:.2f})",
             f"{S['vae_training_time_min']:.1f} ({S['epochs_run']}; {S['vae_time_per_epoch_s']:.2f})"),
            ("Decoder call, 1\\,536 states, forward + gradient [ms]", f(1e3 * cost.get("eigen_forward_backward_s", np.nan), 1),
             f(1e3 * cost.get("surrogate_forward_backward_s", np.nan), 1)),
            ("Error of the decoder w.r.t.\\ the eigenvalue solver", None, None),
            ("Frequencies, mean |error| per mode [\\%]", "0", ", ".join(f"{v:.2f}" for v in acc.get("freq_err_mean_abs_pct_per_mode", []))),
            ("Frequencies, maximum |error| per mode [\\%]", "0", ", ".join(f"{v:.2f}" for v in acc.get("freq_err_max_abs_pct_per_mode", []))),
            ("Mode shapes, mean $1-\\mathrm{MAC}$ per mode", "0", ", ".join(f"{v:.1e}" for v in acc.get("one_minus_MAC_mean_per_mode", []))),
            ("Closeness to the exact posterior (validation set)", None, None),
            ("Negative ELBO (exact model)", f(E.get("val_neg_elbo")), f"{f(S.get('val_neg_elbo'))} ({pv['surrogate_minus_eigen']:+.3f} $\\pm$ {pv['paired_se']:.3f})"),
            ("Lower bound of KL(q $\\|$ exact posterior)", f(E.get("val_kl_lower")), f(S.get("val_kl_lower"))),
            ("Pareto $\\hat k$: median; share $<0.7$", f"{E.get('val_k_hat_median', np.nan):.2f}; {E.get('val_frac_k_hat_below_0.7', np.nan):.3f}",
             f"{S.get('val_k_hat_median', np.nan):.2f}; {S.get('val_frac_k_hat_below_0.7', np.nan):.3f}")]
    for s, name in (("A", "Test set A (10-element model)"), ("B", "Test set B (50-element model, localized damage)")):
        rows.append((name, None, None))
        for met in ("MSE", "MAE", "ALL", "MACE", "95% coverage"):
            rows.append((met.replace("%", "\\%"), f(E[f"{s}_all_{met}"]), f(S[f"{s}_all_{met}"])))
        rows += [("Benchmarks: mean $|\\Delta$mean$|$ / W1 vs exact posterior",
                  f"{E[f'{s}_bench_abs_dmean']:.4f} / {E[f'{s}_bench_W1']:.4f}", f"{S[f'{s}_bench_abs_dmean']:.4f} / {S[f'{s}_bench_W1']:.4f}"),
                 ("Benchmarks: std ratio VAE / exact", f(E[f"{s}_bench_sd_ratio"], 3), f(S[f"{s}_bench_sd_ratio"], 3)),
                 ("Detection rate ($z^*<0.90$) / false-alarm rate", f"{E[f'{s}_detection_rate']:.3f} / {E[f'{s}_false_alarm_rate']:.4f}",
                  f"{S[f'{s}_detection_rate']:.3f} / {S[f'{s}_false_alarm_rate']:.4f}")]
    latex_table(rows, os.path.join(out_dir, "comparison_table.tex"))
    pd.DataFrame([{"quantity": n, LABELS["eigen"]: a, LABELS["surrogate"]: b} for n, a, b in rows if a is not None]).to_csv(
        os.path.join(out_dir, "comparison_table.csv"), index=False)
    with open(os.path.join(out_dir, "comparison_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"eigen": E, "surrogate": S, "paired_validation_neg_elbo": pv, "surrogate_accuracy_uniform_box": acc,
                   "decoder_cost": cost, "surrogate_residuals_test_A": sur_cfg.get("residuals_test_A")}, fh, indent=2, default=float)
    if not args.no_figure:
        benchmark_figure(eig_run, sur_run, info_e, out_dir)
    pd.set_option("display.width", 220)
    print(pd.read_csv(os.path.join(out_dir, "comparison_table.csv")).to_string(index=False))
    print(f"[OK] {out_dir}")


if __name__ == "__main__":
    main()
