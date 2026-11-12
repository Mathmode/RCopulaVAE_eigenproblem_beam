#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/comparison_surrogate_decoder/analyse_eigen_vs_surrogate.py

Detailed comparison of the proposed model (eigenvalue solver) and the approach of Ref. 41 (surrogate decoder), from the saved
results of both runs (no training). For every observation of the validation set and of test sets A and B, both runs were
evaluated against the EXACT model (evaluate_posterior_quality.py), so the paired difference of the negative ELBO is the
difference of their KL divergences from the exact posterior (positive = the surrogate approach is farther from it).

  1. paired -ELBO and KL lower bound, by DoE category, number of damaged parameters and severity (smallest z*);
  2. relation with the error of the surrogate at the true damage state (test set A);
  3. benchmark by benchmark: agreement of both VAEs with the exact posterior (SMC).

Writes analysis_by_scenario.csv, analysis_surrogate_error.csv, analysis_benchmarks.csv and analysis_summary.txt in --out-dir.

    python MODULES/comparison_surrogate_decoder/analyse_eigen_vs_surrogate.py \
        --eigen-run Output/baseline_GaussCopula_GaussMarginals/GCTN_06Nov-5modes_7D_Gamma1.0_Conv6000 \
        --surrogate-run Output/comparison_surrogate_decoder/VAE_surrogate \
        --surrogate-dir Output/comparison_surrogate_decoder/surrogate
"""

import argparse
import io
import os
import sys
from contextlib import redirect_stdout

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

SPLITS = {"val": ("ELBO_val.csv", "val"), "A": ("ELBO_test.csv", "test"), "B": ("ELBO_test_fine.csv", "test_fine")}
SUB = {"A": "TestA_same_FE_model_10el", "B": "TestB_refined_FE_model_50el"}


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(C.PROJECT_ROOT, p)


def paired(e_run, s_run, split, data_dir):
    f, name = SPLITS[split]
    a = pd.read_csv(os.path.join(e_run, "Posterior_quality", f))
    b = pd.read_csv(os.path.join(s_run, "Posterior_quality", f))
    assert (a.obs.values == b.obs.values).all()
    z = np.load(os.path.join(data_dir, f"alphas_{name}.npy")).astype(float)[a.obs.values]
    lab = np.load(os.path.join(data_dir, f"labels_{name}.npy"))[a.obs.values]
    cats = list(C.SCENARIO_FRACTIONS)
    return pd.DataFrame({"obs": a.obs.values, "category": [cats[i] for i in lab], "n_damaged": (z < 0.95).sum(1),
                         "min_z": z.min(1), "d_neg_elbo": b.neg_elbo.values - a.neg_elbo.values,
                         "kl_lower_eigen": a.kl_lower.values, "kl_lower_sur": b.kl_lower.values,
                         "khat_eigen": a.k_hat.values, "khat_sur": b.k_hat.values})


def group_table(t, key):
    g = t.groupby(key, sort=False, observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "d_neg_elbo_mean": g.d_neg_elbo.mean(),
        "d_neg_elbo_se": g.d_neg_elbo.std(ddof=1) / np.sqrt(g.size()),
        "eigen_better_share": g.d_neg_elbo.apply(lambda x: np.mean(x > 0)),
        "kl_lower_eigen": g.kl_lower_eigen.mean(),
        "kl_lower_sur": g.kl_lower_sur.mean(),
    })
    out["kl_lower_ratio_sur_over_eigen"] = out.kl_lower_sur / out.kl_lower_eigen
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eigen-run", required=True)
    p.add_argument("--surrogate-run", required=True)
    p.add_argument("--surrogate-dir", required=True)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    e_run, s_run, s_dir = resolve(args.eigen_run), resolve(args.surrogate_run), resolve(args.surrogate_dir)
    out_dir = resolve(args.out_dir) if args.out_dir else os.path.join(os.path.dirname(s_run), "analysis")
    os.makedirs(out_dir, exist_ok=True)
    data_dir = C.DATA_DIR
    pd.set_option("display.width", 220)
    buf = io.StringIO()
    rows = []
    with redirect_stdout(buf):
        print(f"proposed: {os.path.basename(e_run)}\nsurrogate: {os.path.basename(s_run)}\nsurrogate decoder: {os.path.basename(s_dir)}\n")
        print("d_neg_elbo = -ELBO(surrogate approach) - -ELBO(proposed), per observation, exact model: positive = the proposed "
              "model is closer to the exact posterior (nats).\n")
        tabs = {}
        for split in ("val", "A", "B"):
            t = paired(e_run, s_run, split, data_dir)
            tabs[split] = t
            d = t.d_neg_elbo
            print(f"=== {split}: {len(t)} observations | mean {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f} | "
                  f"proposed better in {np.mean(d > 0):.1%} | KL lower bound {t.kl_lower_eigen.mean():.4f} vs {t.kl_lower_sur.mean():.4f}")
            t["min_z_bin"] = pd.cut(t.min_z, [0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0], include_lowest=True)
            for key in ("category", "n_damaged", "min_z_bin"):
                g = group_table(t, key)
                print(f"--- by {key}")
                print(g.round(4).to_string())
                g = g.reset_index().rename(columns={key: "group"})
                g.insert(0, "by", key)
                g.insert(0, "split", split)
                rows.append(g)
            print()
        pd.concat(rows).to_csv(os.path.join(out_dir, "analysis_by_scenario.csv"), index=False)

        # --- relation with the error of the surrogate at the true damage state (test set A) ---
        from MODULES.review29oct_fixed.fe_model import BridgeModel
        from MODULES.review29oct_fixed.preprocessing_and_eigen import load_problem_config
        from MODULES.comparison_surrogate_decoder.surrogate_decoder import SurrogateBridge
        pcfg = load_problem_config(data_dir)
        bridge, sur = BridgeModel(pcfg), SurrogateBridge(pcfg, s_dir)
        t = tabs["A"]
        z = np.load(os.path.join(data_dir, "alphas_test.npy")).astype(float)[t.obs.values]
        f_e, _, _ = bridge.solve(z)
        f_s, _, _ = sur.solve(z)
        t["sur_err_pct"] = np.mean(np.abs(100 * (f_s / f_e - 1)), axis=1)
        q = pd.qcut(t.sur_err_pct, 5)
        g = t.groupby(q, observed=True).agg(n=("obs", "size"), sur_err_pct=("sur_err_pct", "mean"),
                                            d_neg_elbo=("d_neg_elbo", "mean"),
                                            eigen_better=("d_neg_elbo", lambda x: np.mean(x > 0)),
                                            kl_lower_eigen=("kl_lower_eigen", "mean"), kl_lower_sur=("kl_lower_sur", "mean"))
        from scipy.stats import spearmanr
        rho, pval = spearmanr(t.sur_err_pct, t.d_neg_elbo)
        print("=== test set A: paired -ELBO difference vs mean |frequency error| of the surrogate at z* (quintiles)")
        print(g.round(4).to_string())
        print(f"Spearman correlation: {rho:.3f} (p = {pval:.2g})\n")
        g.reset_index(drop=True).to_csv(os.path.join(out_dir, "analysis_surrogate_error.csv"), index=False)

        # --- benchmarks vs the exact posterior: both runs from their predictions and the same SMC samples ---
        from MODULES.review29oct_fixed.posterior_families import family_from_preds
        from MODULES.review29oct_fixed.reference_posterior import compute_reference_posterior
        probs = np.linspace(0.005, 0.995, 199)
        pred = {"A": "Test_predicted_props.npy", "B": "Test_predicted_props_test_fine.npy"}
        brow = []
        for s in "AB":
            name = SPLITS[s][1]
            fams = {k: family_from_preds(np.load(os.path.join(r, pred[s]), allow_pickle=True).item(), float(pcfg["lbound"]))
                    for k, r in (("eigen", e_run), ("sur", s_run))}
            qs = {k: f.quantiles(probs) for k, f in fams.items()}          # (quantiles (N, D, P), means (N, D))
            for pos, (_, label) in enumerate(C.BENCHMARKS):
                ref = compute_reference_posterior(data_dir, pos, 1.0, None, 4000, split=name)[0].astype(float)
                rq, rm = np.quantile(ref, probs, axis=0).T, ref.mean(0)
                row = {"set": s, "pos": pos, "scenario": label}
                for k in ("eigen", "sur"):
                    q, m = qs[k][0][pos], qs[k][1][pos]
                    row[f"W1_{k}"] = float(np.mean(np.abs(q - rq)))
                    row[f"absdmean_{k}"] = float(np.mean(np.abs(m - rm)))
                brow.append(row)
        b = pd.DataFrame(brow)
        b.to_csv(os.path.join(out_dir, "analysis_benchmarks.csv"), index=False)
        print("=== benchmarks: W1 and |difference of means| with respect to the exact posterior (SMC), both runs computed alike")
        print(b.round(4).to_string(index=False))
        for s in "AB":
            bb = b[b.set == s]
            print(f"{s}: proposed closer (W1) in {int((bb.W1_eigen < bb.W1_sur).sum())} of {len(bb)} benchmarks; "
                  f"mean W1 {bb.W1_eigen.mean():.4f} vs {bb.W1_sur.mean():.4f}; mean |dmean| {bb.absdmean_eigen.mean():.4f} vs "
                  f"{bb.absdmean_sur.mean():.4f}")
    text = buf.getvalue()
    with open(os.path.join(out_dir, "analysis_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"[OK] {out_dir}")


if __name__ == "__main__":
    main()
