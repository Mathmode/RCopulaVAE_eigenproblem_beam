#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed/prior_scenarios_coverage.py

Does the under-coverage of the bearings come from the Bayesian model or from the VAE? (Results_analysis, Sec. 4.3)

Bayesian credible intervals have nominal coverage on average over the prior when the likelihood describes the data.
The 120 test structures of the "uniform" DoE category have all seven parameters drawn from the prior, so on them the
EXACT posterior (adaptive SMC of reference_posterior.py, same likelihood as the VAE) must cover z* at the nominal rate,
up to sampling error, unless the likelihood is misspecified for these data. The script
  1. computes the exact posteriors of those structures (cached in <data_dir>/reference_posteriors/, about 9 s each
     with 4000 particles; cached ones are reused),
  2. compares the exact and the VAE posteriors of the same structures: coverage of the central credible intervals at
     the 19 levels of the MACE and at 95 % (bootstrap standard errors over structures), tails of the probability
     integral transform (PIT), and the agreement of the two posteriors (means, std's, 1-Wasserstein, correlations),
  3. evaluates the data terms of the likelihood at z* (observed / expected, as dataset_generation.check_calibration)
     for the structures whose bearings the exact posterior misses and for the others, to help find the cause.
If the exact posterior covers the bearings at the nominal rate and the VAE does not, the VAE is responsible; if both
under-cover, the likelihood (or the data model) is, and step 3 shows which data term departs from its expected value.

    python MODULES/review29oct_fixed/prior_scenarios_coverage.py --workers 8                      # test set A
    python MODULES/review29oct_fixed/prior_scenarios_coverage.py --split test_fine --workers 8    # test set B
    python MODULES/review29oct_fixed/prior_scenarios_coverage.py --runs <run folder> [<run folder> ...]
Default run: the K = 3 run of the data set. VAE samples come from the numpy sampler of its posterior family.
Results: Output/review29oct_fixed/prior_scenarios_coverage_<M>modes/<test set>/ (README.txt lists the files).
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
from scipy.stats import wasserstein_distance

from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.fe_model import BridgeModel, mac, frequency_precision
from MODULES.review29oct_fixed.posterior_families import family_from_preds
from MODULES.review29oct_fixed.reference_posterior import compute_many, compute_reference_posterior, load_problem_config
from MODULES.review29oct_fixed.derive_mode_subset import resolve_data_dir

LEVELS = np.linspace(0.05, 0.95, 19)                       # levels of the MACE (postprocessing.MACE_LEVELS)
I95 = int(np.argmin(np.abs(LEVELS - 0.95)))
EXACT = "exact (SMC)"
TEST_SET_DIRS = {"test": "TestA_same_FE_model_10el", "test_fine": "TestB_refined_FE_model_50el"}   # as postprocessing
LEGACY_K3_RUN = "KSVAE_fixed_03Nov-FINAL_7D_3KS_obs-fv_Gamma1.0_Ep1500"   # 5 modes; trained before the mode count
                                                                           # entered the run names
N_BOOT = 2000


def parameter_groups():
    groups = {"all": list(range(C.N_DIMS)), "deck zones": list(range(C.N_ZONES)),
              "bearings": list(range(C.N_ZONES, C.N_DIMS))}
    groups.update({name: [d] for d, name in enumerate(C.PARAM_NAMES)})
    return groups


# =============================================================================
# 1. PRIOR SCENARIOS, EXACT AND VAE POSTERIORS
# =============================================================================
def prior_positions(data_dir, split):
    """Test positions of the 'uniform' DoE category (all seven parameters drawn from the prior)."""
    labels = np.load(os.path.join(data_dir, f"labels_{split}.npy"))
    return np.flatnonzero(labels == list(C.SCENARIO_FRACTIONS).index("uniform"))


def exact_posteriors(data_dir, positions, split, gamma2, n_particles, workers, recompute=False):
    """Samples (P, N, 7) of the exact posteriors, computed once and cached by reference_posterior.py."""
    compute_many(data_dir, [int(p) for p in positions], gamma2, None, n_particles, workers, recompute, split)
    return np.stack([compute_reference_posterior(data_dir, int(p), gamma2, None, n_particles, split=split)[0]
                     for p in positions]).astype(np.float64)


def default_runs(n_modes):
    """The K = 3 run of the data set, with or without the mode count in its name."""
    names = [f"KSVAE_fixed_03Nov-FINAL{n_modes}modes_7D_3KS_obs-fv_Gamma1.0_Ep1500"]
    if n_modes == 5:
        names.append(LEGACY_K3_RUN)
    return [n for n in names if os.path.isdir(os.path.join(C.OUTPUT_DIR, n))][:1]


def load_vae(run, split, pcfg, data_dir):
    """(label, folder name, posterior family on `split`) of a run, or None if it was trained on another data set."""
    run_dir = run if os.path.isabs(run) else os.path.join(C.OUTPUT_DIR, run)
    name = os.path.basename(os.path.normpath(run_dir))
    cfg_file = os.path.join(run_dir, "run_config.json")
    if os.path.exists(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as fh:
            rc = json.load(fh)
        family = lambda path: re.sub(r"_\d+modes$", "", os.path.basename(os.path.normpath(str(path))))
        run_modes = int(rc.get("n_modes") or pcfg["n_modes"])
        if family(rc.get("data_dir", data_dir)) != family(data_dir) or run_modes != int(pcfg["n_modes"]):
            print(f"  [skip] {name}: trained on {rc.get('data_dir')} with {run_modes} modes, "
                  f"not on {os.path.basename(data_dir)}")
            return None
    pred_file = "Test_predicted_props.npy" if split == "test" else f"Test_predicted_props_{split}.npy"
    fam = family_from_preds(np.load(os.path.join(run_dir, pred_file), allow_pickle=True).item(), float(pcfg["lbound"]))
    return f"VAE {getattr(fam, 'name', '')}, K={fam.n_components}", name, fam


# =============================================================================
# 2. COVERAGE AND AGREEMENT
# =============================================================================
def pit_exact(samples, z_true):
    """PIT (P, D): exact posterior probability of z_d <= z*_d."""
    return np.mean(samples <= z_true[:, None, :], axis=1)


def pit_vae(fam, z_all, positions, lbound):
    """PIT (P, D) from the exact CDF of the VAE marginals, as postprocessing.py."""
    u = np.clip((z_all - lbound) / (1.0 - lbound), 1e-6, 1.0 - 1e-6)
    return fam.cdf(u)[positions]


def inside_levels(pit):
    """(P, D, L): z* inside the central credible interval of level LEVELS[l]."""
    return np.abs(2.0 * pit - 1.0)[..., None] <= LEVELS


def agreement(x_gt, x_q):
    """|mean difference|, std ratio and 1-Wasserstein (P, D) of the VAE against the exact samples, and the Frobenius
    error (P,) of the correlation matrix, as the benchmark comparison of postprocessing.py."""
    dm = np.abs(x_q.mean(1) - x_gt.mean(1))
    sr = x_q.std(1) / x_gt.std(1)
    w1 = np.array([[wasserstein_distance(x_gt[i, :, d], x_q[i, :, d]) for d in range(x_gt.shape[-1])]
                   for i in range(len(x_gt))])
    cf = np.array([np.linalg.norm(np.corrcoef(x_q[i].T) - np.corrcoef(x_gt[i].T)) for i in range(len(x_gt))])
    return dm, sr, w1, cf


# =============================================================================
# 3. DATA TERMS OF THE LIKELIHOOD AT z*
# =============================================================================
def data_terms_at_truth(pcfg, data_dir, split, positions, z_true):
    """
    Data terms of the likelihood at z* divided by their expected values when the likelihood matches the data
    (frequencies: n_modes / 2; every vertical mode shape: (n_sensors - 1) / 2), as dataset_generation.check_calibration,
    and the log-frequency residuals log f_obs - bias - log f(z*) in units of sig_logf, per structure.
    """
    lik = pcfg["likelihood"]
    f_o = np.load(os.path.join(data_dir, f"freqs_{split}.npy"))[positions].astype(np.float64)
    v_o = np.load(os.path.join(data_dir, f"vertmodes_{split}.npy"))[positions].astype(np.float64)
    f_p, v_p, _ = BridgeModel(pcfg).solve(z_true)
    bias, prec = frequency_precision(lik)
    d = np.log(f_o) - bias - np.log(f_p)
    n_modes, n_s = v_o.shape[-2], v_o.shape[-1]
    ratio_f = 0.5 * np.einsum("pm,mn,pn->p", d, prec, d) / (n_modes / 2)
    ratio_v = (1.0 - mac(v_o, v_p)) / (2.0 * np.asarray(lik["sig_v"]) ** 2) / ((n_s - 1) / 2)
    return ratio_f, ratio_v, d / np.asarray(lik["sig_logf"])


# =============================================================================
# 4. FIGURE
# =============================================================================
def plot_summary(curves, pits, n_struct, title, path):
    """(a, b) coverage of the central intervals against their level, deck zones and bearings; (c) PIT histogram of the
    bearings (flat if the posterior is calibrated)."""
    import matplotlib.pyplot as plt
    from MODULES.review29oct_fixed.plotting import fixed_style, save_figure

    colors = ["#D62728", "#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E", "#8C564B"]

    @fixed_style
    def draw():
        fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))
        for k, g in enumerate(("deck zones", "bearings")):
            n = n_struct * len(parameter_groups()[g])
            band = 2.0 * np.sqrt(LEVELS * (1.0 - LEVELS) / n)
            ax[k].fill_between(LEVELS, LEVELS - band, LEVELS + band, color="0.88", label=r"$\pm2$ binomial s.e.")
            ax[k].plot([0, 1], [0, 1], "k--", lw=1)
            for c, (label, cur) in zip(colors, curves.items()):
                ax[k].plot(LEVELS, cur[g], "o-" if label == EXACT else "s--", color=c, ms=3.5, lw=1.4, label=label)
            ax[k].set(xlabel="nominal level of the central credible interval", ylabel="fraction of test parameters with z* inside",
                      title=f"({'ab'[k]}) {g} ({n} parameters)", xlim=(0, 1), ylim=(0, 1))
            ax[k].grid(alpha=0.3)
            ax[k].legend(loc="upper left", fontsize=9)
        bins = np.linspace(0.0, 1.0, 21)
        for c, (label, pit) in zip(colors, pits.items()):      # exact: thick solid; VAE: thin dashed, so both show
            ax[2].hist(pit[:, C.N_ZONES:].ravel(), bins=bins, density=True, histtype="step", color=c, label=label,
                       lw=2.4 if label == EXACT else 1.4, ls="-" if label == EXACT else "--")
        ax[2].axhline(1.0, color="k", ls="--", lw=1)
        ax[2].set(xlabel=r"PIT of z* (posterior probability of $z<z^*$)", ylabel="density",
                  title="(c) PIT of the bearings (flat if calibrated)", xlim=(0, 1))
        ax[2].grid(alpha=0.3)
        ax[2].legend(fontsize=9)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        save_figure(fig, path)
        plt.close(fig)

    draw()


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Coverage of the exact and VAE posteriors on the prior scenarios.")
    parser.add_argument("--data-dir", type=str, default=C.DATA_DIR)
    parser.add_argument("--modes", type=int, default=None, help="use the M-mode subset of the data set")
    parser.add_argument("--split", type=str, default="test", help="test (set A, 10 elements) or test_fine (set B)")
    parser.add_argument("--runs", type=str, nargs="*", default=None,
                        help="run folders (names in Output/review29oct_fixed or paths); default: the K = 3 run")
    parser.add_argument("--gamma", type=float, default=1.0, help="gamma of the runs (likelihood ^ 1/gamma^2)")
    parser.add_argument("--n-particles", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--n-vae-samples", type=int, default=4000)
    parser.add_argument("--max-scenarios", type=int, default=None, help="only the first prior scenarios (smoke test)")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--recompute", action="store_true")
    args, _ = parser.parse_known_args()

    data_dir = resolve_data_dir(args.data_dir, args.modes)
    pcfg = load_problem_config(data_dir)
    n_modes, lb = int(pcfg["n_modes"]), float(pcfg["lbound"])
    positions = prior_positions(data_dir, args.split)[:args.max_scenarios]
    z_all = np.load(os.path.join(data_dir, f"alphas_{args.split}.npy")).astype(np.float64)
    z_true = z_all[positions]
    out_dir = args.out_dir or os.path.join(C.OUTPUT_DIR, f"prior_scenarios_coverage_{n_modes}modes",
                                           TEST_SET_DIRS.get(args.split, args.split))
    os.makedirs(out_dir, exist_ok=True)
    groups = parameter_groups()
    print(f"Prior scenarios of '{args.split}': {len(positions)} structures, data set {os.path.basename(data_dir)} "
          f"({n_modes} modes)")

    # --- posteriors ---
    x_gt = exact_posteriors(data_dir, positions, args.split, args.gamma ** 2, args.n_particles, args.workers,
                            args.recompute)
    posteriors = {EXACT: {"tag": "exact", "run": "", "pit": pit_exact(x_gt, z_true), "mean": x_gt.mean(1),
                          "std": x_gt.std(1)}}
    vae_samples = {}
    for run in (args.runs if args.runs else default_runs(n_modes)):
        loaded = load_vae(run, args.split, pcfg, data_dir)
        if loaded is None:
            continue
        label, name, fam = loaded
        label = label if label not in posteriors else f"{label} ({name})"
        m_u, s_u = fam.moments()
        posteriors[label] = {"tag": f"vae{len(vae_samples) + 1}", "run": name, "pit": pit_vae(fam, z_all, positions, lb),
                             "mean": lb + (1.0 - lb) * m_u[positions], "std": (1.0 - lb) * s_u[positions]}
        vae_samples[label] = fam.sample(args.n_vae_samples, seed=0, idx=positions)
    if not vae_samples:
        print("  no VAE run of this data set found: only the exact posterior is evaluated")

    # --- coverage per group, with bootstrap standard errors over structures (paired between posteriors) ---
    boot = np.random.default_rng(0).integers(0, len(positions), size=(N_BOOT, len(positions)))
    rows, diff_rows, curves, pits = [], [], {}, {}
    in95_exact = inside_levels(posteriors[EXACT]["pit"])[..., I95]
    for label, p in posteriors.items():
        inside = inside_levels(p["pit"])
        curves[label] = {g: inside[:, cols].reshape(-1, len(LEVELS)).mean(0) for g, cols in groups.items()}
        pits[label] = p["pit"]
        for g, cols in groups.items():
            per_structure = inside[:, cols, I95].mean(1)
            n_par = len(positions) * len(cols)
            rows.append({"posterior": label, "run": p["run"], "group": g, "n_parameters": n_par,
                         "coverage95": curves[label][g][I95], "se_bootstrap": per_structure[boot].mean(1).std(),
                         "se_binomial": np.sqrt(0.95 * 0.05 / n_par),
                         "MACE": np.mean(np.abs(curves[label][g] - LEVELS)),
                         "PIT_below_0.025": np.mean(p["pit"][:, cols] < 0.025),
                         "PIT_above_0.975": np.mean(p["pit"][:, cols] > 0.975),
                         "mean_posterior_std": p["std"][:, cols].mean(),
                         "mean_bias_of_posterior_mean": (p["mean"] - z_true)[:, cols].mean()})
            if label != EXACT:
                d = in95_exact[:, cols].mean(1) - inside[:, cols, I95].mean(1)
                diff_rows.append({"posterior": label, "group": g, "coverage95_exact_minus_VAE": d.mean(),
                                  "se_bootstrap": d[boot].mean(1).std()})
    cov = pd.DataFrame(rows)
    cov.to_csv(os.path.join(out_dir, "coverage_by_group.csv"), index=False, float_format="%.4f")
    pd.DataFrame(diff_rows).to_csv(os.path.join(out_dir, "coverage_difference.csv"), index=False, float_format="%.4f")

    # --- agreement of the VAE with the exact posterior ---
    agree_rows = []
    for label, x_q in vae_samples.items():
        dm, sr, w1, cf = agreement(x_gt, x_q)
        for g, cols in groups.items():
            agree_rows.append({"posterior": label, "group": g, "mean_abs_diff_mean": dm[:, cols].mean(),
                               "max_abs_diff_mean": dm[:, cols].max(), "mean_std_ratio": sr[:, cols].mean(),
                               "min_std_ratio": sr[:, cols].min(), "max_std_ratio": sr[:, cols].max(),
                               "mean_W1": w1[:, cols].mean(),
                               **({"corr_frobenius_mean": cf.mean(), "corr_frobenius_min": cf.min(),
                                   "corr_frobenius_max": cf.max()} if g == "all" else {})})
    agree = pd.DataFrame(agree_rows)
    if len(agree):
        agree.to_csv(os.path.join(out_dir, "agreement_VAE_vs_exact.csv"), index=False, float_format="%.4f")

    # --- per structure ---
    per = {"pos": positions, **{f"{n}_true": z_true[:, d] for d, n in enumerate(C.PARAM_NAMES)}}
    for p in posteriors.values():
        for d, n in enumerate(C.PARAM_NAMES):
            per.update({f"{n}_{p['tag']}_mean": p["mean"][:, d], f"{n}_{p['tag']}_std": p["std"][:, d],
                        f"{n}_{p['tag']}_PIT": p["pit"][:, d]})
    pd.DataFrame(per).to_csv(os.path.join(out_dir, "per_structure.csv"), index=False, float_format="%.4f")

    # --- data terms at z*: structures whose bearings the exact posterior misses vs the others ---
    ratio_f, ratio_v, res = data_terms_at_truth(pcfg, data_dir, args.split, positions, z_true)
    missed = np.any(~in95_exact[:, C.N_ZONES:], axis=1)
    terms = pd.DataFrame({"pos": positions, "bearing_outside_exact95": missed, "ratio_f": ratio_f,
                          **{f"ratio_v_mode{m + 1}": ratio_v[:, m] for m in range(ratio_v.shape[1])},
                          **{f"res_logf_mode{m + 1}": res[:, m] for m in range(res.shape[1])}})
    terms.to_csv(os.path.join(out_dir, "likelihood_terms_at_zstar.csv"), index=False, float_format="%.4f")
    term_rows = []
    for name, m in (("all prior scenarios", np.ones_like(missed)), ("a bearing outside the exact 95% interval", missed),
                    ("both bearings inside", ~missed)):
        if not m.any():
            continue
        term_rows.append({"structures": name, "n": int(m.sum()), "ratio_f": ratio_f[m].mean(),
                          **{f"ratio_v_mode{k + 1}": ratio_v[m, k].mean() for k in range(ratio_v.shape[1])},
                          **{f"mean_res_logf_mode{k + 1}": res[m, k].mean() for k in range(res.shape[1])},
                          **{f"std_res_logf_mode{k + 1}": res[m, k].std() for k in range(res.shape[1])}})
    term_summary = pd.DataFrame(term_rows)
    term_summary.to_csv(os.path.join(out_dir, "likelihood_terms_summary.csv"), index=False, float_format="%.4f")

    # --- figure and guide ---
    split_name = {"test": "test set A (same FE model as training)", "test_fine": "test set B (50 elements, localized damage)"}
    plot_summary(curves, pits, len(positions),
                 f"Prior scenarios of {split_name.get(args.split, args.split)}: {len(positions)} structures, {n_modes} modes",
                 os.path.join(out_dir, "Fig_prior_scenarios_coverage.png"))
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            f"Prior scenarios ('uniform' DoE category) of '{args.split}': {len(positions)} structures, data set "
            f"{os.path.basename(data_dir)} ({n_modes} modes).\n"
            f"Exact posterior: adaptive SMC, {args.n_particles} particles, gamma^2 = {args.gamma ** 2:.4g}, cached in "
            f"<data_dir>/reference_posteriors/.\nPosteriors (tags of per_structure.csv): exact = exact posterior; "
            + "; ".join(f"{p['tag']} = {p['run']}" for p in posteriors.values() if p["run"]) + "\n\n"
            "coverage_by_group.csv          coverage of the central intervals (95 % and MACE over 19 levels), PIT tails,\n"
            "                               mean posterior std and mean bias of the posterior mean, per group\n"
            "coverage_difference.csv        exact minus VAE 95 % coverage per group (paired bootstrap over structures)\n"
            "agreement_VAE_vs_exact.csv     means, std's, 1-Wasserstein and correlations of the VAE against the exact posterior\n"
            "per_structure.csv              z*, posterior mean, std and PIT of every structure and parameter\n"
            "likelihood_terms_at_zstar.csv  data terms at z* / expected value, standardized log-frequency residuals\n"
            "likelihood_terms_summary.csv   the same averaged over all structures, over those with a bearing outside the\n"
            "                               exact 95 % interval and over the others\n"
            "Fig_prior_scenarios_coverage.* coverage against the level (deck zones, bearings), PIT histogram of the bearings\n\n"
            "Bootstrap standard errors resample the structures, so they keep the dependence between the parameters of a\n"
            "structure; binomial ones do not. A data-term ratio of 1 means the likelihood matches the data on average.\n")

    # --- console summary ---
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    main_groups = cov[cov.group.isin(["all", "deck zones", "bearings"])]
    print("\nCoverage of the central 95 % interval on the prior scenarios")
    print(main_groups[["posterior", "group", "n_parameters", "coverage95", "se_bootstrap", "MACE", "PIT_below_0.025",
                       "PIT_above_0.975", "mean_posterior_std"]].round(4).to_string(index=False))
    print("\nPer parameter (95 % coverage)")
    print(cov[cov.group.isin(C.PARAM_NAMES)].pivot(index="posterior", columns="group", values="coverage95")
          [C.PARAM_NAMES].round(3).to_string())
    if diff_rows:
        d = pd.DataFrame(diff_rows)
        print("\nExact minus VAE 95 % coverage (paired bootstrap s.e.)")
        print(d[d.group.isin(["all", "deck zones", "bearings"])].round(4).to_string(index=False))
    if len(agree):
        print("\nVAE against the exact posterior")
        print(agree[agree.group.isin(["all", "deck zones", "bearings"])].round(4).to_string(index=False))
    print("\nData terms of the likelihood at z* (observed / expected; 1 = calibrated)")
    print(term_summary[["structures", "n", "ratio_f"] + [c for c in term_summary if c.startswith("ratio_v")]]
          .round(3).to_string(index=False))
    print(f"[OK] {out_dir}")


if __name__ == "__main__":
    main()
