#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed/ood_study.py

R1.5: how does the trained VAE behave when the test data fall OUTSIDE the training distribution?

Six shifted test sets. Each changes ONE factor with respect to test set B (refined 50-element model, assumed noise and
modelling error), which is the in-distribution reference:

  ID_testB            in-distribution reference: test set B of the paper
  OOD1_severity       damage beyond the prior and training support: every damaged deck zone of test set B has its z*
                      mapped from [0.45, 0.95) to [0.30, 0.45) (samples without a damaged zone are left out)
  OOD2_defects        damage not aligned with the zones: 1-2 defects of 0.2-0.6 m with local stiffness x0.25-0.70 at
                      random positions (also across zone boundaries and next to the supports); z* = equivalent
                      (harmonic-mean) zone stiffness, the definition used for the as-built stiffness scatter
  OOD3_noise2x        identification noise twice the assumed one (5 % on frequencies, 10 % on mode shapes)
  OOD4_modelerr2x     random modelling error twice the assumed one (10 % mass scatter, 5 % stiffness scatter)
  OOD5_addedmass      systematic modelling error: +5 % uniform mass that the model does not contain (e.g. surfacing)
  OOD6_stiffbearings  bearings three times stiffer than nominal (aged or partially seized pads): outside the prior

ID, 1, 3, 4, 5 and 6 contain the SAME structures (mass and stiffness fields) and the SAME noise draws as test set B,
so every difference from ID is caused by the shift alone. The VAE and the exact posterior always use the ASSUMED model
(nominal decoder and calibrated likelihood of the data set); nothing is retrained.

Two failure modes are separated:
  * misspecification of the Bayesian model: the EXACT posterior (adaptive SMC, first n_gt cases of every set) misses
    z*, i.e. what any Bayesian method with this model would do;
  * amortisation: the VAE departs from the exact posterior for inputs unlike its training data: Pareto-k of the
    importance weights and KL lower bound (posterior_quality.py, every case), VAE vs SMC (first n_gt cases).

Data: Data/<data set>_OOD/ (problem_config.json copied: same likelihood; arrays per set; ood_sets.json documents the
sets). Results: <run>/OOD_tests/ (OOD_summary.csv, figures, one folder per set with the metrics of the paper).

    python MODULES/review29oct_fixed/ood_study.py --run-folder KSVAE_fixed_03Nov-FINAL5modes_7D_3KS_obs-fv_Gamma1.0_Ep1500
"""

import os
import sys
import json
import time
import shutil
import argparse

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.fe_model import (BeamMesh, draw_asbuilt_fields, simulate_true_structure,
                                                add_measurement_noise, localize_zone_damage)
from MODULES.review29oct_fixed.reference_posterior import (load_problem_config, compute_many,
                                                           compute_reference_posterior, cache_path)
from MODULES.review29oct_fixed.posterior_families import family_from_preds
from MODULES.review29oct_fixed.posterior_quality import evaluate_split

OOD_SETS = {   # name: (label in the figures, description)
    "ID_testB": ("ID\n(test B)", "In-distribution reference: test set B (50-element model, assumed noise and "
                                 "modelling error)"),
    "OOD1_severity": ("1: severity\nbelow 0.45", "Damage beyond the prior and training support: damaged deck zones "
                                                 "of test set B with z* mapped to [0.30, 0.45)"),
    "OOD2_defects": ("2: off-zone\ndefects", "Localized 0.2-0.6 m defects (stiffness x0.25-0.70) at random "
                                             "positions, not aligned with the zones; z* = equivalent zone stiffness"),
    "OOD3_noise2x": ("3: noise\nx2", "Identification noise twice the assumed one (5 % frequencies, 10 % mode shapes)"),
    "OOD4_modelerr2x": ("4: model\nerror x2", "Random modelling error twice the assumed one (10 % mass, 5 % "
                                              "stiffness scatter)"),
    "OOD5_addedmass": ("5: +5 %\nmass", "Systematic modelling error: +5 % uniform mass not in the model "
                                        "(e.g. a new surfacing layer)"),
    "OOD6_stiffbearings": ("6: stiff\nbearings", "Bearings three times stiffer than nominal (aged or partially "
                                                 "seized pads), outside the prior"),
}
SEVERE_RANGE = (0.30, 0.45)        # OOD1: new range of the damaged deck zones
DEFECT_STIFFNESS = (0.25, 0.70)    # OOD2: stiffness factor of the defective 0.2 m cells
BEARING_FACTOR = 3.0               # OOD6
ADDED_MASS = 0.05                  # OOD5
MACE_LEVELS = np.linspace(0.05, 0.95, 19)


# =============================================================================
# 1. SHIFTED DATA SETS
# =============================================================================
def generate_ood_data(data_dir, ood_dir, n_defects=600, seed=2026):
    """Writes the seven sets to ood_dir with the layout of the data set (problem_config.json copied unchanged)."""
    pcfg = load_problem_config(data_dir)
    lb, nz, n_modes = float(pcfg["lbound"]), int(pcfg["n_zones"]), int(pcfg["n_modes"])
    cells, thr = int(pcfg["el_per_zone_test"]), float(pcfg["damage_threshold"])
    mesh = BeamMesh(nz, cells, sensor_positions=pcfg["sensor_positions"], **pcfg["structure"])
    ref = (np.asarray(pcfg["ref_vert"]), np.asarray(pcfg["ref_rot"]))
    fn, mn = float(pcfg["freq_noise"]), float(pcfg["mode_noise"])
    noise_seed = int(pcfg["seeds"]["test"]) + 1000          # noise draws of the test sets (dataset_generation.py)

    def load(key):
        return np.load(os.path.join(data_dir, f"{key}_test_fine.npy")).astype(np.float64)
    zB, labB = load("alphas"), np.load(os.path.join(data_dir, "labels_test_fine.npy"))
    fdir = os.path.join(data_dir, "asbuilt_fields")
    massB = np.load(os.path.join(fdir, "mass_factors_test_fine.npy")).astype(np.float64)
    # test set B = as-built scatter of test set A + damage localized inside the zones (dataset_generation.py); the
    # paired sets rebuild it from the scatter alone and the stored geometry of the localized damage
    localized = bool(pcfg.get("test_fine_localized_damage", False))
    scatterB = np.load(os.path.join(fdir, "flexibility_deviations_test.npy" if localized else
                                    "flexibility_deviations_test_fine.npy")).astype(np.float64)
    recB = (np.load(os.path.join(fdir, "localized_damage_test_fine.npy")) if localized else np.zeros((0, 5)))
    s_min = float(pcfg.get("local_damage_min_stiffness", 0.15))

    def apply_localization(z, scatter, recs):
        """Zone factors for the FE model and flexibility deviations with the localized damage of geometry recs
        ([sample, zone, first cell, n cells, s]); s is recomputed so that the equivalent zone factor is z."""
        z_model = np.array(z, dtype=np.float64, copy=True)
        a = 1.0 + np.array(scatter, dtype=np.float64, copy=True).reshape(len(z_model), nz, cells)
        for i, k, start, n, _ in recs:
            i, k, start, n = int(i), int(k), int(start), int(n)
            a_in = a[i, k, start:start + n].sum()
            a[i, k, start:start + n] /= a_in / (cells / z_model[i, k] - (a[i, k].sum() - a_in))
            z_model[i, k] = 1.0
        return z_model, (a - 1.0).reshape(len(z_model), -1)

    def observe(z, mass, scatter, freq_noise=fn, mode_noise=mn, rng=None, recs=recB):
        z_model, flex = apply_localization(z, scatter, recs)
        f, v, r = simulate_true_structure(z_model, mesh, n_modes, mass, flex, ref=ref)
        rng = np.random.default_rng(noise_seed) if rng is None else rng
        return (*add_measurement_noise(f, v, r, rng, freq_noise, mode_noise, ref=ref), flex)

    # the stored fields, localization and noise seed must reproduce test set B (pairing of the shifted sets)
    fR, vR, _, flexB = observe(zB, massB, scatterB)
    pairing = {"max_rel_diff_freqs": float(np.max(np.abs(fR / load("freqs") - 1.0))),
               "max_abs_diff_vertmodes": float(np.max(np.abs(vR - load("vertmodes"))))}
    print(f"  test set B regenerated from the stored fields: max rel. diff. of the frequencies "
          f"{pairing['max_rel_diff_freqs']:.1e}, of the shapes {pairing['max_abs_diff_vertmodes']:.1e}")
    if pairing["max_rel_diff_freqs"] > 1e-4:
        raise RuntimeError("the stored fields do not reproduce test set B: the shifted sets would not be paired")

    sets, extra = {}, {}
    sets["ID_testB"] = (zB, labB, load("freqs"), load("vertmodes"), load("rotmodes"), massB, flexB)

    z1 = zB.copy()
    zones = z1[:, :nz]
    dam = zones < thr
    zones[dam] = SEVERE_RANGE[0] + (zones[dam] - lb) * (SEVERE_RANGE[1] - SEVERE_RANGE[0]) / (thr - lb)
    keep = np.where(dam.any(axis=1))[0]
    rec1 = (localize_zone_damage(z1, scatterB, np.random.default_rng(seed + 1), nz, cells, thr, s_min)[2]
            if localized else recB)                      # new extents: the severities changed
    f, v, r, flex1 = observe(z1, massB, scatterB, recs=rec1)
    sets["OOD1_severity"] = (z1[keep], labB[keep], f[keep], v[keep], r[keep], massB[keep], flex1[keep])
    extra["OOD1_severity"] = {"rows_of_test_set_B": keep.tolist()}

    rng = np.random.default_rng(seed)
    z2 = rng.uniform(*C.HEALTHY_RANGE, size=(n_defects, int(pcfg["n_dims"])))
    mass2, flex2 = draw_asbuilt_fields(n_defects, rng, nz, int(pcfg["el_per_zone_decoder"]), cells,
                                       float(pcfg["mass_perturb"]), float(pcfg["within_zone_stiff_var"]))
    n_cells = nz * cells
    L_cell = float(pcfg["structure"]["L_total"]) / n_cells
    defects = []
    for i in range(n_defects):
        taken, desc = np.zeros(n_cells, bool), []
        for _ in range(1 + int(rng.random() < 0.3)):
            while True:                                   # defects do not touch each other
                length = int(rng.integers(1, 4))
                start = int(rng.integers(0, n_cells - length + 1))
                if not taken[max(start - 1, 0):start + length + 1].any():
                    break
            s = float(rng.uniform(*DEFECT_STIFFNESS))
            idx = np.arange(start, start + length)
            taken[idx] = True
            flex2[i, idx] = (1.0 + flex2[i, idx]) / s - 1.0           # stiffness of the defective cells x s
            desc.append({"start_m": round(start * L_cell, 3), "length_m": round(length * L_cell, 3),
                         "stiffness_factor": round(s, 4), "zones": sorted({int(c // cells) + 1 for c in idx})})
        defects.append(desc)
    zeq = z2.copy()
    zeq[:, :nz] = z2[:, :nz] / np.mean(1.0 + flex2.reshape(n_defects, nz, cells), axis=2)
    f, v, r, _ = observe(z2, mass2, flex2, rng=rng, recs=np.zeros((0, 5)))
    sets["OOD2_defects"] = (zeq, (zeq[:, :nz] < thr).sum(axis=1), f, v, r, mass2, flex2)
    extra["OOD2_defects"] = {"seed": seed, "defects": defects}

    f, v, r, x3 = observe(zB, massB, scatterB, 2.0 * fn, 2.0 * mn)
    sets["OOD3_noise2x"] = (zB, labB, f, v, r, massB, x3)
    m4 = 1.0 + 2.0 * (massB - 1.0)
    f, v, r, x4 = observe(zB, m4, 2.0 * scatterB)              # twice the scatter, same localized damage
    sets["OOD4_modelerr2x"] = (zB, labB, f, v, r, m4, x4)
    m5 = (1.0 + ADDED_MASS) * massB
    f, v, r, x5 = observe(zB, m5, scatterB)
    sets["OOD5_addedmass"] = (zB, labB, f, v, r, m5, x5)
    z6 = zB.copy()
    z6[:, nz:] = BEARING_FACTOR
    f, v, r, x6 = observe(z6, massB, scatterB)
    sets["OOD6_stiffbearings"] = (z6, labB, f, v, r, massB, x6)

    os.makedirs(os.path.join(ood_dir, "asbuilt_fields"), exist_ok=True)
    shutil.copy(os.path.join(data_dir, "problem_config.json"), os.path.join(ood_dir, "problem_config.json"))
    for name, (z, lab, f, v, r, mass, flex) in sets.items():
        for key, arr in (("freqs", f), ("vertmodes", v), ("rotmodes", r), ("alphas", z)):
            np.save(os.path.join(ood_dir, f"{key}_{name}.npy"), arr.astype(np.float32))
        np.save(os.path.join(ood_dir, f"labels_{name}.npy"), np.asarray(lab))
        np.save(os.path.join(ood_dir, "asbuilt_fields", f"mass_factors_{name}.npy"), mass.astype(np.float32))
        np.save(os.path.join(ood_dir, "asbuilt_fields", f"flexibility_deviations_{name}.npy"), flex.astype(np.float32))
        print(f"  {name:20s} {len(z):5d} cases: {OOD_SETS[name][1]}")
    meta = {"source_data_set": os.path.basename(data_dir.rstrip("\\/")), "pairing_check": pairing,
            "assumed_model": "problem_config.json of the source data set (nominal decoder, calibrated likelihood)",
            "sets": {name: {"description": OOD_SETS[name][1], "n": int(len(sets[name][0])), **extra.get(name, {})}
                     for name in sets}}
    with open(os.path.join(ood_dir, "ood_sets.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    # the exact posteriors of the benchmarks of test set B (computed for the paper) are reused for ID_testB
    for pos in range(len(pcfg.get("benchmarks", []))):
        src = cache_path(data_dir, pos, 1.0, pcfg["obs_types"], 4000, "test_fine")
        if os.path.exists(src):
            dst = cache_path(ood_dir, pos, 1.0, pcfg["obs_types"], 4000, "ID_testB")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
    return meta


# =============================================================================
# 2. METRICS
# =============================================================================
def vae_summary(fam, z, lb, thr):
    """Metrics of the VAE over every case of a set. Parameters whose z* lies outside the prior support [lb, 1]
    (OOD1 zones, OOD6 bearings) cannot be covered by any posterior: they are reported separately."""
    inside = (z >= lb - 1e-6) & (z <= 1.0 + 1e-6)
    pit = fam.cdf(np.clip((z - lb) / (1.0 - lb), 1e-6, 1.0 - 1e-6))
    emp = (np.abs(2.0 * pit - 1.0)[..., None] <= MACE_LEVELS)[inside].mean(axis=0)
    i95 = int(np.argmin(np.abs(MACE_LEVELS - 0.95)))
    q, mean = fam.quantiles(np.array([0.025, 0.5, 0.975]))
    flagged = q[..., 2] < thr                                  # damage declared: upper 95 % limit below threshold
    deck = np.zeros_like(inside)
    deck[:, :C.N_ZONES] = True
    healthy = inside & (z >= thr)
    damaged = z < 0.90
    out = ~inside
    return {
        "n_cases": int(len(z)), "coverage95": float(emp[i95]), "MACE": float(np.mean(np.abs(emp - MACE_LEVELS))),
        "MSE": float(np.mean(((mean - z) ** 2)[inside])), "MAE": float(np.mean(np.abs(mean - z)[inside])),
        "detection_rate": float(flagged[damaged].mean()) if damaged.any() else np.nan,
        "false_alarm_deck": float(flagged[healthy & deck].mean()) if (healthy & deck).any() else np.nan,
        "false_alarm_bearings": float(flagged[healthy & ~deck].mean()) if (healthy & ~deck).any() else np.nan,
        "n_outside_support": int(out.sum()),
        "outside_zstar_mean": float(z[out].mean()) if out.any() else np.nan,
        "outside_posterior_median_mean": float(q[..., 1][out].mean()) if out.any() else np.nan,
        "outside_flagged": float(flagged[out].mean()) if out.any() else np.nan,
        "empirical_coverage": emp.tolist(),
    }


def exact_comparison(fam, z, ood_dir, name, positions, lb, thr, n_particles, workers):
    """Exact posterior (SMC, same likelihood) of the first cases: coverage of z* by the exact and by the VAE 95 %
    intervals (same parameters), mean |difference of the means| and ratio of the standard deviations."""
    compute_many(ood_dir, positions, 1.0, None, n_particles, workers, False, name)
    rows, samples = [], {}
    for pos in positions:
        x_gt, _ = compute_reference_posterior(ood_dir, pos, 1.0, None, n_particles, split=name)
        x_gt = x_gt.astype(np.float64)
        x_q = fam.sample(n_particles, seed=pos, idx=[pos])[0]
        zt = z[pos]
        ins = (zt >= lb - 1e-6) & (zt <= 1.0 + 1e-6)
        lo_g, hi_g = np.quantile(x_gt, [0.025, 0.975], axis=0)
        lo_q, hi_q = np.quantile(x_q, [0.025, 0.975], axis=0)
        rows.append({"pos": pos, "n_inside": int(ins.sum()),
                     "exact_hits": int(((zt >= lo_g) & (zt <= hi_g))[ins].sum()),
                     "vae_hits": int(((zt >= lo_q) & (zt <= hi_q))[ins].sum()),
                     "exact_false_alarms_deck": int(np.sum((hi_g < thr) & (zt >= thr) & ins & (np.arange(len(zt)) < C.N_ZONES))),
                     "vae_false_alarms_deck": int(np.sum((hi_q < thr) & (zt >= thr) & ins & (np.arange(len(zt)) < C.N_ZONES))),
                     "abs_dmean": float(np.mean(np.abs(x_q.mean(0) - x_gt.mean(0)))),
                     "std_ratio": float(np.mean(x_q.std(0) / np.maximum(x_gt.std(0), 1e-9)))})
        samples[pos] = (x_gt, x_q)
    return pd.DataFrame(rows), samples


# =============================================================================
# 3. FIGURES
# =============================================================================
def plot_summary(df, n_gt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from MODULES.review29oct_fixed.plotting import PAPER_RC, save_figure
    x = np.arange(len(df))
    labels = [OOD_SETS[n][0] for n in df["set"]]
    with plt.style.context(["default", PAPER_RC]):
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        a = ax[0, 0]
        a.bar(x - 0.2, df["coverage95"], 0.38, color="steelblue", label="VAE, all cases")
        a.bar(x + 0.2, df["exact_coverage95"], 0.38, color="#D62728", alpha=0.8,
              label=f"exact posterior (SMC), first {n_gt} cases")
        a.plot(x + 0.2, df["vae_coverage95_subset"], "o", color="navy", ms=6, label="VAE, same cases as SMC")
        a.axhline(0.95, color="k", ls="--", lw=1)
        a.set_ylim(0, 1.05)
        a.set_title("(a) Coverage of $z^*$ by the 95 % intervals (parameters inside the prior support)")
        a.legend(loc="lower left", fontsize=9)
        a = ax[0, 1]
        a.bar(x, df["MACE"], 0.6, color="steelblue")
        a.set_title("(b) Calibration error of the VAE (MACE, 19 levels)")
        a = ax[1, 0]
        a.bar(x - 0.2, df["detection_rate"], 0.38, color="#2CA02C", label=r"detection ($z^*<0.90$, any support)")
        a.bar(x + 0.2, df["false_alarm_deck"], 0.38, color="#FF7F0E", label="false alarms (undamaged deck zones)")
        a.set_ylim(0, 1.05)
        a.set_title("(c) Damage declared when the upper 95 % limit is below 0.95")
        a.legend(loc="upper left", fontsize=9)
        a = ax[1, 1]
        a.bar(x, 100 * df["khat_lt_0.7"], 0.6, color="#9467BD", label=r"cases with Pareto-$\hat{k}<0.7$ (all)")
        a.set_ylim(0, 105)
        a.set_ylabel("%")
        a2 = a.twinx()
        a2.plot(x, df["abs_dmean_vs_exact"], "D-", color="#8C564B", ms=6, label="|mean VAE - mean exact| (SMC cases)")
        a2.set_ylabel("|difference of the posterior means|")
        a2.set_ylim(0, max(0.05, 1.2 * np.nanmax(df["abs_dmean_vs_exact"])))
        a.set_title("(d) Amortisation: agreement of the VAE with the exact posterior")
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9)
        for a in ax.ravel():
            a.set_xticks(x, labels, fontsize=9.5)
            a.grid(axis="y", ls=":", alpha=0.6)
        fig.tight_layout()
        save_figure(fig, path)
        plt.close(fig)


def plot_reliability(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from MODULES.review29oct_fixed.plotting import PAPER_RC, save_figure
    with plt.style.context(["default", PAPER_RC]):
        fig, ax = plt.subplots(figsize=(7.2, 6.6))
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="ideal")
        cmap = plt.get_cmap("tab10")
        for k, (_, row) in enumerate(df.iterrows()):
            ax.plot(MACE_LEVELS, row["empirical_coverage"], "o-" if k else "s-", ms=4, lw=2.2 if k == 0 else 1.5,
                    color="k" if k == 0 else cmap(k), label=f"{OOD_SETS[row['set']][0].replace(chr(10), ' ')} "
                                                            f"(MACE {row['MACE']:.3f})")
        ax.set_xlabel("Nominal level $p$")
        ax.set_ylabel(r"Empirical coverage $\hat{p}$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(fontsize=8.5, loc="upper left")
        ax.set_title("Reliability of the VAE intervals under distribution shift")
        fig.tight_layout()
        save_figure(fig, path)
        plt.close(fig)


def plot_examples(examples, lb, path):
    """Rows: sets; columns: two cases. VAE (filled) vs exact posterior (dashed) and z* (star)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from MODULES.review29oct_fixed.plotting import PAPER_RC, LABELS_7D, RED, save_figure
    from MODULES.review29oct_fixed.localized_damage_study import violin_axes
    names = list(examples)
    with plt.style.context(["default", PAPER_RC]):
        fig, axes = plt.subplots(len(names), 2, figsize=(12.5, 2.9 * len(names) + 1.0), squeeze=False)
        for i, name in enumerate(names):
            for j, (title, x_gt, x_q, zt) in enumerate(examples[name][:2]):
                a = axes[i, j]
                violin_axes(a, x_gt, x_q, np.clip(zt, lb, 1.0), lb)
                low = min(lb, float(np.min(zt))) - 0.03
                a.set_ylim(low, 1.06)
                for d in np.where(zt < lb)[0]:                 # z* below the support: true position
                    a.plot(d + 1, zt[d], "*", color="#FFD700", ms=11, mec="black", mew=0.8, zorder=11)
                    a.axhline(lb, color="#555555", lw=0.6, ls="--")
                for d in np.where(zt > 1.0)[0]:                # z* above the support (OOD6 bearings)
                    a.annotate(f"$z^*={zt[d]:.0f}$", (d + 1, 1.0), xytext=(d + 1, 1.045), ha="center", fontsize=8,
                               arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8))
                a.set_xticks(range(1, len(zt) + 1), LABELS_7D, fontsize=8.5)
                a.set_title(f"{OOD_SETS[name][0].replace(chr(10), ' ')}: {title}", fontsize=9.5)
                if j == 0:
                    a.set_ylabel("stiffness factor $z$")
        handles = [mpatches.Patch(color="steelblue", alpha=0.4, label="VAE posterior (deck zones)"),
                   mpatches.Patch(color="#2CA02C", alpha=0.4, label="VAE posterior (bearings)"),
                   mlines.Line2D([], [], color=RED, ls="--", label="exact posterior (SMC, same likelihood)"),
                   mlines.Line2D([], [], color="#FFD700", marker="*", ls="", mec="black", ms=11,
                                 label="true / equivalent value $z^*$")]
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 1 - 0.6 / (2.9 * len(names) + 1.0)])
        save_figure(fig, path)
        plt.close(fig)


# =============================================================================
# 4. DRIVER
# =============================================================================
def example_cases(name, meta, pcfg, z, n_gt):
    """Two cases per set for the example figure (inside the SMC subset)."""
    desc = {b["pos"]: b["desc"] for b in pcfg.get("benchmarks", [])}
    if name == "OOD1_severity":
        rows = meta["sets"][name]["rows_of_test_set_B"]
        picks = [rows.index(p) for p in (1, 2) if p in rows][:2] or [0, 1]
        return [(i, f"{desc.get(rows[i], 'case')} -> zone damage below 0.45") for i in picks]
    if name == "OOD2_defects":
        dfx = meta["sets"][name]["defects"][:n_gt]
        mid = next((i for i, d in enumerate(dfx) if len(d) == 1 and d[0]["zones"] == [3]), 0)
        sup = next((i for i, d in enumerate(dfx) if len(d) == 1 and (d[0]["start_m"] <= 0.4 or
                    d[0]["start_m"] + d[0]["length_m"] >= 9.6)), 1)
        out = []
        for i in (mid, sup):
            d = dfx[i][0]
            out.append((i, f"{d['length_m']:.1f} m defect at x = {d['start_m']:.1f} m, stiffness x{d['stiffness_factor']:.2f}"))
        return out
    return [(0, desc.get(0, "case 0")), (2, desc.get(2, "case 2"))]


def run_ood_study(run_folder, n_gt=40, n_particles=4000, workers=None, n_elbo=300, regenerate=False, sets=None):
    from MODULES.review29oct_fixed.postprocessing import load_run_info, manuscript_metrics
    if not os.path.isabs(run_folder) and not os.path.exists(run_folder):
        run_folder = os.path.join(C.OUTPUT_DIR, run_folder)
    info = load_run_info(run_folder)
    data_dir, lb, gamma2 = info["data_dir"], float(info["lbound"]), float(info["gamma"]) ** 2
    ood_dir = data_dir.rstrip("\\/") + "_OOD"
    if regenerate or not os.path.exists(os.path.join(ood_dir, "ood_sets.json")):
        print(f"--- Generating the shifted test sets in {ood_dir} ---")
        source = load_problem_config(data_dir).get("derived_from")
        if source:   # M-mode subset (derive_mode_subset.py): shifted sets of the source, first M modes kept
            from MODULES.review29oct_fixed.derive_mode_subset import derive_mode_subset
            src = os.path.join(os.path.dirname(data_dir.rstrip("\\/")), source["data_dir"])
            if regenerate or not os.path.exists(os.path.join(src + "_OOD", "ood_sets.json")):
                generate_ood_data(src, src + "_OOD")
            if os.path.isdir(ood_dir):
                shutil.rmtree(ood_dir)
            derive_mode_subset(src + "_OOD", int(load_problem_config(data_dir)["n_modes"]), dst_dir=ood_dir)
        else:
            generate_ood_data(data_dir, ood_dir)
    with open(os.path.join(ood_dir, "ood_sets.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    pcfg = load_problem_config(ood_dir)
    thr = float(pcfg["damage_threshold"])
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    out_dir = os.path.join(run_folder, "OOD_tests")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== R1.5 distribution-shift study: {os.path.basename(run_folder)} ===")

    model, rows, examples = None, [], {}
    for name in (sets or list(OOD_SETS)):
        t0 = time.time()
        sub = os.path.join(out_dir, name)
        os.makedirs(sub, exist_ok=True)
        f, v, r, z = (np.load(os.path.join(ood_dir, f"{k}_{name}.npy")).astype(np.float64)
                      for k in ("freqs", "vertmodes", "rotmodes", "alphas"))
        pred_path = os.path.join(sub, "Predicted_props.npy")
        if regenerate or not os.path.exists(pred_path):
            if model is None:
                from MODULES.review29oct_fixed.training_common import load_trained_model
                model, keys = load_trained_model(run_folder, info)
            f_s = ((np.log(f) - info["mean_f"]) / info["std_f"]).astype(np.float32)
            outs = model.Encoder_model.predict((f_s, r.astype(np.float32), v.astype(np.float32),
                                                np.ones((len(f), int(info["n_dims"])), np.float32)),
                                               batch_size=256, verbose=0)
            np.save(pred_path, dict(zip(keys, outs)), allow_pickle=True)
        fam = family_from_preds(np.load(pred_path, allow_pickle=True).item(), lb)

        # the metrics of the paper for this set (same code as the test sets)
        manuscript_metrics(fam, z, lb, sub, label=OOD_SETS[name][1])
        row = {"set": name, **vae_summary(fam, z, lb, thr)}
        # amortisation quality on every case
        q = evaluate_split(fam, (f, v, r), pcfg, info["obs_types"], gamma2, n_elbo, seed=0)
        q.to_csv(os.path.join(sub, "ELBO_per_case.csv"), index=False, float_format="%.6g")
        row.update({"khat_lt_0.7": float(np.mean(q["k_hat"] < 0.7)), "khat_median": float(q["k_hat"].median()),
                    "KL_lower_mean": float(q["kl_lower"].mean()), "neg_elbo_mean": float(q["neg_elbo"].mean())})
        # exact posterior of the first cases
        positions = list(range(min(n_gt, len(z))))
        ex, samples = exact_comparison(fam, z, ood_dir, name, positions, lb, thr, n_particles, workers)
        ex.to_csv(os.path.join(sub, "Exact_vs_VAE_first_cases.csv"), index=False, float_format="%.5g")
        n_in = ex["n_inside"].sum()
        healthy_deck = sum(int(np.sum((z[p, :C.N_ZONES] >= thr))) for p in positions)
        row.update({"gt_cases": len(positions), "exact_coverage95": ex["exact_hits"].sum() / n_in,
                    "vae_coverage95_subset": ex["vae_hits"].sum() / n_in,
                    "exact_false_alarm_deck": ex["exact_false_alarms_deck"].sum() / max(healthy_deck, 1),
                    "vae_false_alarm_deck_subset": ex["vae_false_alarms_deck"].sum() / max(healthy_deck, 1),
                    "abs_dmean_vs_exact": float(ex["abs_dmean"].mean()), "std_ratio_vs_exact": float(ex["std_ratio"].mean()),
                    "seconds": time.time() - t0})
        rows.append(row)
        examples[name] = [(title, *samples[i], z[i]) for i, title in example_cases(name, meta, pcfg, z, n_gt)
                          if i in samples]
        print(f"  {name:20s} coverage95 VAE {row['coverage95']:.3f} | exact (first {len(positions)}) "
              f"{row['exact_coverage95']:.3f} vs VAE {row['vae_coverage95_subset']:.3f} | MACE {row['MACE']:.3f} | "
              f"detection {row['detection_rate']:.2f} | false alarms deck {row['false_alarm_deck']:.3f} | "
              f"k_hat<0.7 {100 * row['khat_lt_0.7']:.1f} % | |dmean| vs exact {row['abs_dmean_vs_exact']:.4f}")

    df = pd.DataFrame(rows)
    df.drop(columns=["empirical_coverage"]).to_csv(os.path.join(out_dir, "OOD_summary.csv"), index=False,
                                                   float_format="%.4f")
    with open(os.path.join(out_dir, "OOD_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"run": os.path.basename(run_folder), "ood_data": ood_dir, "sets": {k: OOD_SETS[k][1] for k in df["set"]},
                   "rows": rows}, fh, indent=1, default=float)
    plot_summary(df, n_gt, os.path.join(out_dir, "Fig_OOD_summary.png"))
    plot_reliability(df, os.path.join(out_dir, "Fig_OOD_reliability.png"))
    plot_examples(examples, lb, os.path.join(out_dir, "Fig_OOD_examples.png"))
    write_guide(out_dir, n_gt)
    print(f"[OK] {out_dir}")
    return df


def write_guide(out_dir, n_gt):
    lines = ["R1.5: behaviour under distribution shift (ood_study.py)", "",
             "Every set changes ONE factor with respect to test set B (ID); sets 1, 3, 4, 5, 6 contain the same",
             "structures and noise draws as ID. The VAE and the exact posterior use the ASSUMED model (nothing retrained).", ""]
    lines += [f"{k:20s} {v[1]}" for k, v in OOD_SETS.items()]
    lines += ["", "OOD_summary.csv columns:",
              "  coverage95, MACE, MSE, MAE   VAE, every case, parameters with z* inside the prior support [0.45, 1]",
              "  detection_rate               fraction of parameters with z* < 0.90 whose upper 95 % limit is below 0.95",
              "  false_alarm_deck/_bearings   fraction of undamaged parameters (z* >= 0.95) declared damaged",
              "  outside_*                    parameters with z* outside the support (OOD1 zones, OOD6 bearings)",
              "  khat_lt_0.7, KL_lower_mean   amortisation quality (Pareto-k of importance weights, every case)",
              f"  exact_coverage95             coverage of the EXACT posterior (SMC, first {n_gt} cases): misspecification",
              "  vae_coverage95_subset        coverage of the VAE on the same cases and parameters",
              "  abs_dmean_vs_exact, std_ratio_vs_exact   VAE vs exact posterior on the same cases",
              "", "Per set: Manuscript_Metrics.csv (metrics of the paper), ELBO_per_case.csv, Exact_vs_VAE_first_cases.csv",
              "Figures: Fig_OOD_summary, Fig_OOD_reliability, Fig_OOD_examples (PNG + PDF)"]
    with open(os.path.join(out_dir, "README_OOD.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="R1.5: behaviour of a trained run under distribution shift.")
    parser.add_argument("--run-folder", default="KSVAE_fixed_03Nov-FINAL5modes_7D_3KS_obs-fv_Gamma1.0_Ep1500",
                        help="run name in Output/review29oct_fixed or full path (also baseline runs)")
    parser.add_argument("--n-gt", type=int, default=40, help="cases per set with an exact (SMC) posterior")
    parser.add_argument("--n-particles", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--elbo-samples", type=int, default=300, help="posterior samples per case for Pareto-k")
    parser.add_argument("--regenerate", action="store_true", help="regenerate the shifted sets and predictions")
    parser.add_argument("--sets", nargs="+", default=None, choices=list(OOD_SETS))
    args, _ = parser.parse_known_args()
    run_ood_study(args.run_folder, args.n_gt, args.n_particles, args.workers, args.elbo_samples, args.regenerate,
                  args.sets)


if __name__ == "__main__":
    main()
