#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed/dataset_generation.py

Balanced, physically consistent data for the 7-parameter bridge on elastic bearings.

Changes with respect to MODULES/review29oct/dataset_generation.py:
  * Stratified DoE with exact counts: every location (5 zones + 2 bearings) and every combination of
    locations gets the same number of samples in each scenario category; severities are Latin-hypercube
    stratified over a continuous range [0.45, 0.95) (the old DoE had no sample in (0.90, 0.95) nor in
    [0.45, 0.50), and never more than 3 damaged parameters, although benchmark 7 has 5).
  * Every observation is a plain noisy simulation of one structure. The old "bimodal" test cases 5, 8, 9
    averaged the modal data of z* and of its mirror image, which no structure can produce (modes 2 and 4
    of those observations had 1 - MAC ~ 0.99 with every physical shape).
  * Modelling error (reviewers R1.1, R1.2, R1.7, R3.1). Every measurement comes from an as-built structure that
    differs from the NOMINAL model of the decoder: mass factors 1 + N(0, 5 %) per 1 m segment and 2.5 %
    flexibility scatter of 0.2 m cells inside every zone (config.MASS_PERTURB, config.WITHIN_ZONE_STIFF_VAR), a
    new realisation for every sample. The VAE and the reference posterior only know the nominal model.
    Train, validation and "test" observations are computed with the 10-element model, "test_fine" with a genuine
    50-element model; both test sets contain the same as-built structures and the same noise draws, so they differ
    only by the FE model. (The old "50-element fine mesh" test set solved the same 5-element model with a
    harmonic-mean jitter of z, so it was not a finer mesh.) There is no temperature effect.
  * Realistic measurements: frequencies (2.5 % identification error) and vertical mode shapes (5 % of the
    peak) at the sensors of config.SENSOR_POSITIONS. Rotations are not measured by accelerometers; measuring
    them at every node is what made the previous posteriors so narrow.
  * Mode signs are aligned with the healthy shapes. The old rule (sign of the largest |vertical| component)
    flips modes 2 and 4 at random for near-symmetric states.
  * The likelihood is calibrated here on the training generator (as-built 10-element structures + noise): mean
    offset and full covariance of log f (a mass deviation moves several modes together), per-mode std of the
    shapes. Only the statistics of the modelling error are used, never the realisations of the data. It is
    written to problem_config.json, which every other script reads. model_error_budget() reports the size of
    every error source (mesh, mass, stiffness scatter, noise).

Run from the project root:
    python MODULES/review29oct_fixed/dataset_generation.py [--data-dir <folder>]   (default: config.DATA_DIR)
"""

import os
import sys
import json
import argparse
from itertools import combinations

import numpy as np

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.plotting import PAPER_RC, save_figure
from MODULES.review29oct_fixed.fe_model import (
    BeamMesh, build_baseline_matrices, modal_solve, draw_asbuilt_fields, simulate_true_structure,
    add_measurement_noise, mac,
)


# =============================================================================
# 1. BALANCED, STRATIFIED DESIGN OF EXPERIMENTS
# =============================================================================
def lhs_uniform(n, lo, hi, rng):
    """n stratified uniform values on [lo, hi): one per equal-width stratum, in random order."""
    return lo + (hi - lo) * (rng.permutation(n) + rng.uniform(size=n)) / max(n, 1)


def exact_counts(fractions, n):
    """Largest-remainder rounding of n * fractions (sums to n exactly)."""
    raw = np.array(list(fractions.values()), dtype=float) * n / sum(fractions.values())
    counts = np.floor(raw).astype(int)
    counts[np.argsort(raw - counts)[::-1][:n - counts.sum()]] += 1
    return dict(zip(fractions.keys(), counts))


def sample_balanced_doe(n_samples, rng, n_dims=C.N_DIMS, fractions=C.SCENARIO_FRACTIONS,
                        healthy=C.HEALTHY_RANGE, damage=C.DAMAGE_RANGE, lbound=C.LBOUND):
    """
    Returns z (n, 7) and the scenario category of every sample (index in `fractions`).
    Inside a category the damaged-location combinations are cycled in a shuffled order, so every
    location (and every pair / triple ...) appears the same number of times; the severity of every
    damaged parameter is Latin-hypercube stratified over `damage`, the others are uniform on `healthy`.
    """
    combos_by_cat = {
        "healthy": [()],
        "single": list(combinations(range(n_dims), 1)),
        "double": list(combinations(range(n_dims), 2)),
        "triple": list(combinations(range(n_dims), 3)),
        "multi45": list(combinations(range(n_dims), 4)) + list(combinations(range(n_dims), 5)),
    }
    counts = exact_counts(fractions, n_samples)
    z_blocks, cat_blocks = [], []
    for cat_id, (cat, cnt) in enumerate(counts.items()):
        if cnt == 0:
            continue
        if cat == "uniform":
            block = rng.uniform(lbound, 1.0, size=(cnt, n_dims))
        else:
            combos = combos_by_cat[cat]
            order = rng.permutation(len(combos))
            mask = np.zeros((cnt, n_dims), dtype=bool)
            for i in range(cnt):
                mask[i, list(combos[order[i % len(combos)]])] = True
            block = rng.uniform(*healthy, size=(cnt, n_dims))
            for d in range(n_dims):
                rows = np.flatnonzero(mask[:, d])
                block[rows, d] = lhs_uniform(len(rows), *damage, rng)
        z_blocks.append(block)
        cat_blocks.append(np.full(cnt, cat_id, dtype=np.int32))

    z = np.vstack(z_blocks)
    cats = np.concatenate(cat_blocks)
    perm = rng.permutation(len(z))
    return z[perm], cats[perm]


# =============================================================================
# 2. OBSERVATIONS AND LIKELIHOOD CALIBRATION
# =============================================================================
def build_meshes():
    """Decoder mesh (training data, VAE, reference posterior) and test mesh (50 elements)."""
    return (BeamMesh(C.N_ZONES, C.EL_PER_ZONE_DECODER, sensor_positions=C.SENSOR_POSITIONS, **C.STRUCTURE),
            BeamMesh(C.N_ZONES, C.EL_PER_ZONE_TEST, sensor_positions=C.SENSOR_POSITIONS, **C.STRUCTURE))


def draw_fields(n, rng, mass_std=C.MASS_PERTURB, stiff_var=C.WITHIN_ZONE_STIFF_VAR):
    """As-built deviations of n structures: mass per 1 m segment (decoder element), flexibility per 0.2 m cell."""
    return draw_asbuilt_fields(n, rng, C.N_ZONES, C.EL_PER_ZONE_DECODER, C.EL_PER_ZONE_TEST, mass_std, stiff_var)


def true_modal_data(z, mesh, fields, ref):
    """Noise-free modal data of the as-built structures (z + mass and stiffness fields) on `mesh`."""
    return simulate_true_structure(z, mesh, C.N_MODES, *fields, ref=ref)


def simulate_observations(z, mesh, fields, ref, rng):
    """Noisy measurements: as-built structure on `mesh` + identification noise."""
    f, v, r = true_modal_data(z, mesh, fields, ref)
    return add_measurement_noise(f, v, r, rng, C.FREQ_NOISE, C.MODE_NOISE, ref=ref)


def decoder_modal_data(z, dec, ref):
    K_basis, M, sensors = dec
    nd = K_basis.shape[-1]
    K = (z @ K_basis.reshape(len(K_basis), -1)).reshape(len(z), nd, nd)
    return modal_solve(K, M, sensors, C.N_MODES, ref)


def reference_shapes(dec):
    """Healthy decoder shapes; the sign makes the largest |vertical| component positive."""
    _, v, r = decoder_modal_data(np.ones((1, C.N_DIMS)), dec, None)
    v, r = v[0], r[0]
    s = np.sign(v[np.arange(C.N_MODES), np.argmax(np.abs(v), axis=1)])[:, None]
    return v * s, r * s


def calibrate_likelihood(dec, mesh, ref, rng, n_cal=6000):
    """
    Model discrepancy of the nominal decoder with respect to the as-built structures on `mesh`, by simulation on
    the DoE with fresh random fields (statistics only, never the realisations of the data):
      log f: mean offset b and full covariance (a mass deviation moves several modes together) + identification
      noise; shapes: per-mode std of each component of the unit-norm vector (E[1 - MAC] = (n_sensors - 1) s^2).
    """
    z, _ = sample_balanced_doe(n_cal, rng)
    f_t, v_t, r_t = true_modal_data(z, mesh, draw_fields(n_cal, rng), ref)
    f_d, v_d, r_d = decoder_modal_data(z, dec, ref)
    n_s = v_t.shape[-1]
    d = np.log(f_t) - np.log(f_d)
    bias = d.mean(axis=0)
    cov = np.cov(d, rowvar=False) + C.FREQ_NOISE ** 2 * np.eye(C.N_MODES)
    model_v = np.mean(1.0 - mac(v_t, v_d), axis=0) / (n_s - 1)
    model_r = np.mean(1.0 - mac(r_t, r_d), axis=0) / (n_s - 1)
    noise_v = C.MODE_NOISE ** 2 * np.mean(np.max(np.abs(v_d), axis=-1) ** 2, axis=0)
    noise_r = C.MODE_NOISE ** 2 * np.mean(np.max(np.abs(r_d), axis=-1) ** 2, axis=0)
    sd = np.sqrt(np.diag(cov))
    return {
        "bias_logf": bias.tolist(), "cov_logf": cov.tolist(), "prec_logf": np.linalg.inv(cov).tolist(),
        "sig_logf": sd.tolist(), "corr_logf": (cov / np.outer(sd, sd)).tolist(),
        "sig_v": np.sqrt(noise_v + model_v).tolist(), "sig_r": np.sqrt(noise_r + model_r).tolist(),
        "parts": {
            "noise_logf": [C.FREQ_NOISE] * C.N_MODES, "model_logf": np.std(d, axis=0).tolist(),
            "noise_v": np.sqrt(noise_v).tolist(), "model_v": np.sqrt(model_v).tolist(),
            "noise_r": np.sqrt(noise_r).tolist(), "model_r": np.sqrt(model_r).tolist(),
        },
    }


def check_calibration(dec, mesh, ref, lik, rng, n=4000):
    """
    Simulates measurements of random structures on `mesh` and evaluates the NLL terms at the TRUE z with the
    decoder. If the likelihood matches the data, the frequency term averages n_modes / 2 and every shape term
    (n_sensors - 1) / 2. Returns observed / expected.
    """
    from MODULES.review29oct_fixed.fe_model import neg_log_likelihood
    z, _ = sample_balanced_doe(n, rng)
    f_o, v_o, r_o = simulate_observations(z, mesh, draw_fields(n, rng), ref, rng)
    f_p, v_p, r_p = decoder_modal_data(z, dec, ref)
    n_s = v_o.shape[-1]
    _, parts = neg_log_likelihood(f_o, v_o, r_o, f_p, v_p, r_p, lik, "fvr")
    term_v = (1 - mac(v_o, v_p)) / (2 * np.asarray(lik["sig_v"]) ** 2)
    term_r = (1 - mac(r_o, r_p)) / (2 * np.asarray(lik["sig_r"]) ** 2)
    return {
        "ratio_f_total": float(parts["f"].mean() / (C.N_MODES / 2)),
        "ratio_v": (term_v.mean(0) / ((n_s - 1) / 2)).tolist(),
        "ratio_r": (term_r.mean(0) / ((n_s - 1) / 2)).tolist(),
    }


def model_error_budget(dec, meshes, ref, rng, n=3000):
    """
    Size of every source of discrepancy between the measurements and the nominal decoder, per mode, over DoE states:
    mean and std of log f_data - log f_decoder [%] and mean 1 - MAC of the vertical shapes, for each source alone
    (mesh: 50-element nominal model; mass field; within-zone stiffness scatter), for the as-built structures of the
    training / "test" data (10 elements) and of "test_fine" (50 elements), and for the identification noise alone.
    """
    z, _ = sample_balanced_doe(n, rng)
    mass, flex = draw_fields(n, rng)
    f_d, v_d, r_d = decoder_modal_data(z, dec, ref)
    cases = {
        "mesh (50 el., nominal)": (meshes["test_fine"], None, None),
        "mass field": (meshes["test"], mass, None),
        "stiffness scatter": (meshes["test"], None, flex),
        "as-built, 10 el. (train/val/test)": (meshes["test"], mass, flex),
        "as-built, 50 el. (test_fine)": (meshes["test_fine"], mass, flex),
    }
    budget = {}
    for name, (mesh, m, fl) in cases.items():
        f, v, _ = simulate_true_structure(z, mesh, C.N_MODES, m, fl, ref=ref)
        d = 100.0 * (np.log(f) - np.log(f_d))
        budget[name] = {"logf_mean_pct": d.mean(0).tolist(), "logf_std_pct": d.std(0).tolist(),
                        "one_minus_mac_v": np.mean(1.0 - mac(v, v_d), axis=0).tolist()}
    f_n, v_n, _ = add_measurement_noise(f_d, v_d, r_d, rng, C.FREQ_NOISE, C.MODE_NOISE, ref=ref)
    d = 100.0 * (np.log(f_n) - np.log(f_d))
    budget["measurement noise"] = {"logf_mean_pct": d.mean(0).tolist(), "logf_std_pct": d.std(0).tolist(),
                                   "one_minus_mac_v": np.mean(1.0 - mac(v_n, v_d), axis=0).tolist()}
    return budget


def print_error_budget(budget):
    modes = "".join(f"{'mode ' + str(j + 1):>10s}" for j in range(C.N_MODES))
    for key, title in (("logf_mean_pct", "mean of log f_data - log f_decoder [%]"),
                       ("logf_std_pct", "std of log f_data - log f_decoder [%]"),
                       ("one_minus_mac_v", "mean 1 - MAC of the vertical shapes")):
        print(f"  {title}\n  {'':36s}{modes}")
        for name, b in budget.items():
            fmt = "{:10.4f}" if key == "one_minus_mac_v" else "{:10.3f}"
            print(f"  {name:36s}" + "".join(fmt.format(x) for x in b[key]))


# =============================================================================
# 3. DATA SET GENERATION
# =============================================================================
def generate_full_benchmark_datasets(data_dir=C.DATA_DIR, n_train=C.N_TRAIN, n_val=C.N_VAL, n_test=C.N_TEST,
                                     make_figures=True):
    os.makedirs(data_dir, exist_ok=True)
    M, K_basis, sensors, L_inv = build_baseline_matrices(C.N_ZONES, C.EL_PER_ZONE_DECODER, sensor_positions=C.SENSOR_POSITIONS, **C.STRUCTURE)
    np.save(os.path.join(data_dir, "Mass_matrix.npy"), M.astype(np.float32))
    np.save(os.path.join(data_dir, "Ke_matrix.npy"), K_basis.astype(np.float32))
    np.save(os.path.join(data_dir, "L_inv.npy"), L_inv.astype(np.float32))
    dec = (K_basis, M, sensors)
    dec_mesh, test_mesh = build_meshes()

    ref = reference_shapes(dec)
    meshes = {"test": dec_mesh, "test_fine": test_mesh}
    print("--- Model-error budget (as-built structures vs nominal decoder) ---")
    budget = model_error_budget(dec, meshes, ref, np.random.default_rng(C.SEEDS["calibration"] + 10))
    print_error_budget(budget)
    # The likelihood describes the data the VAE is trained on (as-built 10-element structures with the mass and
    # within-zone stiffness fields). The 50-element test set is then a genuine model-form robustness test.
    print(f"--- Calibrating the likelihood on the training generator ({C.EL_PER_ZONE_DECODER * C.N_ZONES} elements) ---")
    lik = calibrate_likelihood(dec, dec_mesh, ref, np.random.default_rng(C.SEEDS["calibration"]))
    calib = {name: check_calibration(dec, meshes[name], ref, lik, np.random.default_rng(C.SEEDS["calibration"] + 1 + k))
             for k, name in enumerate(C.TEST_SETS)}
    print(f"  bias log f = {np.round(lik['bias_logf'], 4)}")
    print(f"  std  log f = {np.round(lik['sig_logf'], 4)}   corr (mode 1 vs others) = {np.round(lik['corr_logf'][0][1:], 2)}")
    print(f"  sig_v = {np.round(lik['sig_v'], 4)}")
    for name, cc in calib.items():
        print(f"  E[NLL at z*] / expected on '{name}' data: f {cc['ratio_f_total']:.3f}  v {np.round(cc['ratio_v'], 2)}")

    fields_dir = os.path.join(data_dir, "asbuilt_fields")
    os.makedirs(fields_dir, exist_ok=True)
    with open(os.path.join(fields_dir, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write("Realisations of the as-built deviations from the nominal model that produced each measurement\n"
                 "(mass factors per 1 m segment, flexibility deviations per 0.2 m cell). Stored for reproducibility\n"
                 "and analysis only: they are never read by the VAE, the likelihood or the reference posterior.\n")

    def save_split(name, z, cats, f, v, r, fields):
        np.save(os.path.join(data_dir, f"freqs_{name}.npy"), f.astype(np.float32))
        np.save(os.path.join(data_dir, f"rotmodes_{name}.npy"), r.astype(np.float32))
        np.save(os.path.join(data_dir, f"vertmodes_{name}.npy"), v.astype(np.float32))
        np.save(os.path.join(data_dir, f"alphas_{name}.npy"), z.astype(np.float32))
        np.save(os.path.join(data_dir, f"labels_{name}.npy"), cats)
        np.save(os.path.join(fields_dir, f"mass_factors_{name}.npy"), fields[0].astype(np.float32))
        np.save(os.path.join(fields_dir, f"flexibility_deviations_{name}.npy"), fields[1].astype(np.float32))

    splits = {}
    for name, n in (("train", n_train), ("val", n_val)):
        rng = np.random.default_rng(C.SEEDS[name])
        z, cats = sample_balanced_doe(n, rng)
        fields = draw_fields(n, rng)
        f, v, r = simulate_observations(z, dec_mesh, fields, ref, rng)
        splits[name] = (z, cats, f, v, r)
        save_split(name, z, cats, f, v, r, fields)
        print(f"--- {name}: {n} as-built structures computed with the {C.EL_PER_ZONE_DECODER * C.N_ZONES}-element model")

    # test sets: the SAME as-built structures (damage state, mass and stiffness fields; benchmarks in positions
    # 0..10) and the SAME noise draws (same seed, same array shapes), computed with each FE model
    rng = np.random.default_rng(C.SEEDS["test"])
    z, cats = sample_balanced_doe(n_test, rng)
    for pos, (z_b, _) in enumerate(C.BENCHMARKS):
        z[pos] = z_b
        n_dam = int(np.sum(np.asarray(z_b) < C.DAMAGE_THRESHOLD))
        cats[pos] = {0: 0, 1: 1, 2: 2, 3: 3}.get(n_dam, 4)
    fields = draw_fields(n_test, rng)
    for name, el_per_zone in C.TEST_SETS.items():
        f, v, r = simulate_observations(z, meshes[name], fields, ref, np.random.default_rng(C.SEEDS["test"] + 1000))
        splits[name] = (z, cats, f, v, r)
        save_split(name, z, cats, f, v, r, fields)
        print(f"--- {name}: the {n_test} test structures computed with the {el_per_zone * C.N_ZONES}-element model")

    dam = splits["train"][0] < C.DAMAGE_THRESHOLD
    balance = {
        "categories": list(C.SCENARIO_FRACTIONS.keys()),
        "train_counts_per_category": np.bincount(splits["train"][1], minlength=len(C.SCENARIO_FRACTIONS)).tolist(),
        "train_damaged_fraction_per_parameter": dam.mean(axis=0).round(4).tolist(),
        "train_n_damaged_histogram": np.bincount(dam.sum(axis=1), minlength=C.N_DIMS + 1).tolist(),
    }

    pcfg = {
        "dataset_version": C.DATASET_VERSION,
        "n_zones": C.N_ZONES, "n_dims": C.N_DIMS, "n_modes": C.N_MODES, "lbound": C.LBOUND,
        "structure": C.STRUCTURE, "el_per_zone_decoder": C.EL_PER_ZONE_DECODER, "el_per_zone_test": C.EL_PER_ZONE_TEST,
        "test_sets": C.TEST_SETS,
        "sensor_positions": C.SENSOR_POSITIONS, "sensor_nodes": sensors.tolist(),
        "freq_noise": C.FREQ_NOISE, "mode_noise": C.MODE_NOISE,
        "mass_perturb": C.MASS_PERTURB, "within_zone_stiff_var": C.WITHIN_ZONE_STIFF_VAR,
        "modelling_error": ("as-built structures: mass factors 1 + N(0, mass_perturb) per 1 m segment and "
                            "flexibility scatter N(0, within_zone_stiff_var) of 0.2 m cells (zero zone mean), a new "
                            "realisation per sample; the decoder uses the nominal model; test and test_fine contain "
                            "the same structures and noise draws"),
        "model_error_budget": budget,
        "obs_types": C.OBS_TYPES,
        "prior": f"uniform on [{C.LBOUND}, 1]^{C.N_DIMS}",
        "healthy_range": C.HEALTHY_RANGE, "damage_range": C.DAMAGE_RANGE, "damage_threshold": C.DAMAGE_THRESHOLD,
        "scenario_fractions": C.SCENARIO_FRACTIONS, "seeds": C.SEEDS,
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "ref_vert": ref[0].tolist(), "ref_rot": ref[1].tolist(),
        "likelihood": lik, "calibration_check": calib, "balance": balance,
        "benchmarks": [{"pos": i, "z": list(zb), "desc": d} for i, (zb, d) in enumerate(C.BENCHMARKS)],
    }
    with open(os.path.join(data_dir, "problem_config.json"), "w", encoding="utf-8") as fh:
        json.dump(pcfg, fh, indent=2)
    with open(os.path.join(data_dir, "dataset_version.json"), "w", encoding="utf-8") as fh:
        json.dump({"version": C.DATASET_VERSION}, fh)

    if make_figures:
        plot_dataset_figures(data_dir, splits, pcfg, None, ref)
        plot_reference_examples(data_dir, pcfg)
    print(f"Dataset '{C.DATASET_VERSION}' saved to {data_dir}")
    return pcfg


# =============================================================================
# 4. DATA FIGURES
# =============================================================================
def plot_dataset_figures(data_dir, splits, pcfg, mats, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.style.context(["default", PAPER_RC]):  # same look whatever style the console has set
        _plot_dataset_figures(plt, data_dir, splits, pcfg, mats, ref)


def _plot_dataset_figures(plt, data_dir, splits, pcfg, mats, ref):
    fig_dir = os.path.join(data_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    cats = list(C.SCENARIO_FRACTIONS.keys())
    z_tr, c_tr = splits["train"][0], splits["train"][1]
    names = C.PARAM_NAMES

    # --- DoE balance ---
    fig, ax = plt.subplots(2, 2, figsize=(16, 11), facecolor="white")
    bottom = np.zeros(C.N_DIMS)
    colors = plt.get_cmap("tab10").colors
    for k, cat in enumerate(cats):
        cnt = (z_tr[c_tr == k] < C.DAMAGE_THRESHOLD).sum(axis=0)
        ax[0, 0].bar(names, cnt, bottom=bottom, color=colors[k], label=cat)
        bottom += cnt
    ax[0, 0].set_title("Training samples in which each parameter is damaged (z < %.2f)" % C.DAMAGE_THRESHOLD,
                       fontweight="bold")
    ax[0, 0].set_ylim(0, 1.45 * bottom.max())
    ax[0, 0].legend(title="scenario category", ncol=3, loc="upper center")
    ax[0, 0].set_ylabel("count")

    bins = np.arange(C.LBOUND, 1.0 + 1e-9, 0.025)
    H = np.stack([np.histogram(z_tr[:, d], bins=bins)[0] for d in range(C.N_DIMS)])
    im = ax[0, 1].imshow(np.log10(H + 1), aspect="auto", cmap="viridis",
                         extent=[bins[0], bins[-1], C.N_DIMS - 0.5, -0.5])
    ax[0, 1].set_yticks(range(C.N_DIMS), names)
    ax[0, 1].set_xlabel("stiffness factor z")
    ax[0, 1].set_title("Training z histogram per parameter (log10 count): no gaps", fontweight="bold")
    fig.colorbar(im, ax=ax[0, 1], label="log10(count + 1)")

    width = 0.27
    for k, name in enumerate(("train", "val", "test")):
        z = splits[name][0]
        h = np.bincount((z < C.DAMAGE_THRESHOLD).sum(axis=1), minlength=C.N_DIMS + 1) / len(z)
        ax[1, 0].bar(np.arange(C.N_DIMS + 1) + (k - 1) * width, h, width, label=name)
    ax[1, 0].set_xlabel("number of damaged parameters per sample")
    ax[1, 0].set_ylabel("fraction of samples")
    ax[1, 0].set_title("Damage multiplicity", fontweight="bold")
    ax[1, 0].legend()

    zb = np.array([b[0] for b in C.BENCHMARKS])
    im = ax[1, 1].imshow(zb, cmap="YlOrRd_r", vmin=C.LBOUND, vmax=1.0, aspect="auto")
    for i in range(zb.shape[0]):
        for j in range(zb.shape[1]):
            ax[1, 1].text(j, i, f"{zb[i, j]:.2f}", ha="center", va="center", fontsize=9,
                          color="white" if zb[i, j] < 0.65 else "black")
    ax[1, 1].set_xticks(range(C.N_DIMS), names)
    ax[1, 1].set_yticks(range(len(zb)), [f"P{i}" for i in range(len(zb))])
    ax[1, 1].set_title("Benchmark scenarios (test positions 0..%d)" % (len(zb) - 1), fontweight="bold")
    fig.colorbar(im, ax=ax[1, 1], label="z*")
    plt.tight_layout()
    save_figure(plt.gcf(), os.path.join(fig_dir, "Fig_DoE_balance.png"))
    plt.close()

    # --- Modal data ---
    f_tr = splits["train"][2]
    f_te = splits["test"][2]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2), facecolor="white")
    ax[0].violinplot([f_tr[:, j] for j in range(C.N_MODES)], showmedians=True)
    for pos in range(len(C.BENCHMARKS)):
        ax[0].plot(np.arange(1, C.N_MODES + 1), f_te[pos], "k.", ms=4)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("mode")
    ax[0].set_ylabel("frequency [Hz]")
    ax[0].set_title("Identified frequencies (train violins, benchmarks dots)", fontweight="bold")
    x_nodes = np.asarray(C.SENSOR_POSITIONS)
    for j in range(C.N_MODES):
        ax[1].plot(x_nodes, ref[0][j], "o-", label=f"mode {j + 1}")
        ax[2].plot(x_nodes, ref[1][j], "s-", label=f"mode {j + 1}")
    ax[1].set_title(f"Healthy vertical shapes at the {len(x_nodes)} sensors", fontweight="bold")
    ax[2].set_title(f"Healthy rotational shapes at the {len(x_nodes)} sensors", fontweight="bold")
    for a in ax[1:]:
        a.set_xlabel("x [m]")
        a.axhline(0, color="gray", lw=0.8)
        a.legend(fontsize=9)
    plt.tight_layout()
    save_figure(plt.gcf(), os.path.join(fig_dir, "Fig_modal_data.png"))
    plt.close()

    # --- Likelihood calibration ---
    fig, ax = plt.subplots(1, 3, figsize=(20, 4.8), facecolor="white")
    modes = np.arange(1, C.N_MODES + 1)
    for k, (name, lab) in enumerate((("test", "same model (10 el.)"), ("test_fine", "50-element model"))):
        cc = pcfg["calibration_check"][name]
        ax[0].bar(modes + (k - 0.5) * 0.4, cc["ratio_v"], 0.4, color=f"C{k}",
                  label=f"vertical shapes, {lab}; frequencies: {cc['ratio_f_total']:.2f}")
    ax[0].axhline(1.0, color="k", ls="--")
    ax[0].set_xlabel("mode")
    ax[0].set_ylabel("mean NLL term at z* / expected")
    ax[0].set_title("Calibration on the two test sets (1 = matches the data)", fontweight="bold")
    ax[0].legend(fontsize=9)
    lk = pcfg["likelihood"]
    ax[1].plot(modes, lk["sig_logf"], "o-", label=r"$\sigma_{\log f}$ (total)")
    ax[1].plot(modes, lk["parts"]["model_logf"], "o:", label=r"$\log f$: modelling error (mass + stiffness fields)")
    ax[1].plot(modes, lk["sig_v"], "s-", label=r"$\sigma_v$ (unit-norm shape)")
    ax[1].plot(modes, np.abs(lk["bias_logf"]), "x--", label=r"|bias| of $\log f$")
    ax[1].set_xlabel("mode")
    ax[1].set_title("Likelihood standard deviations", fontweight="bold")
    ax[1].legend(fontsize=9)
    im = ax[2].imshow(np.asarray(lk["corr_logf"]), cmap="RdBu_r", vmin=-1, vmax=1)
    for (i, j), v in np.ndenumerate(np.asarray(lk["corr_logf"])):
        ax[2].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    ax[2].set_xticks(range(C.N_MODES), modes)
    ax[2].set_yticks(range(C.N_MODES), modes)
    ax[2].set_title(r"Correlation of the $\log f$ errors (noise + modelling error)", fontweight="bold")
    fig.colorbar(im, ax=ax[2])
    plt.tight_layout()
    save_figure(plt.gcf(), os.path.join(fig_dir, "Fig_likelihood_calibration.png"))
    plt.close()

    plot_error_budget(plt, pcfg["model_error_budget"], fig_dir)
    print(f"  data figures saved to {fig_dir}")


def plot_error_budget(plt, budget, fig_dir):
    """Fig_model_error_budget.png: size of every error source per mode (frequencies and vertical shapes)."""
    modes = np.arange(1, C.N_MODES + 1)
    fig, ax = plt.subplots(1, 2, figsize=(17, 6), facecolor="white")
    w = 0.8 / len(budget)
    for k, (name, b) in enumerate(budget.items()):
        x = modes + (k - (len(budget) - 1) / 2) * w
        ax[0].bar(x, b["logf_std_pct"], w, color=f"C{k}", label=name)
        ax[0].plot(x, np.abs(b["logf_mean_pct"]), "kv", ms=5, label="|mean| (systematic part)" if k == 0 else None)
        ax[1].bar(x, b["one_minus_mac_v"], w, color=f"C{k}")
    ax[0].set_ylabel(r"std of $\log f_{data} - \log f_{decoder}$ [%]")
    ax[0].set_title("Frequencies: size of every error source", fontweight="bold")
    ax[1].set_yscale("log")
    ax[1].set_ylabel(r"mean $1 - \mathrm{MAC}$ (vertical shapes at the sensors)")
    ax[1].set_title("Mode shapes: size of every error source", fontweight="bold")
    for a in ax:
        a.set_xticks(modes, [f"mode {j}" for j in modes])
        a.grid(axis="y", ls=":", alpha=0.5)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=10, frameon=False)
    plt.tight_layout(rect=[0, 0.11, 1, 1])
    save_figure(plt.gcf(), os.path.join(fig_dir, "Fig_model_error_budget.png"))
    plt.close()


def plot_reference_examples(data_dir, pcfg, positions=None, workers=None):
    """Exact (SMC) posteriors of a few benchmarks of every test set, to see the target distributions of the data set."""
    from MODULES.review29oct_fixed.reference_posterior import compute_many, compute_reference_posterior
    from MODULES.review29oct_fixed.plotting import corner_single
    positions = positions or C.GT_EXAMPLES
    workers = workers or max(1, min(len(positions), (os.cpu_count() or 2) // 2))
    descs = {b["pos"]: b["desc"] for b in pcfg["benchmarks"]}
    fig_dir = os.path.join(data_dir, "figures")
    for split, el_per_zone in pcfg["test_sets"].items():
        print(f"--- Reference posteriors of benchmarks {positions} ({split}, {el_per_zone * C.N_ZONES}-element data) ---")
        compute_many(data_dir, positions, 1.0, None, 4000, workers, split=split)
        z = np.load(os.path.join(data_dir, f"alphas_{split}.npy"))
        for pos in positions:
            s, _ = compute_reference_posterior(data_dir, pos, 1.0, None, 4000, split=split)
            lo, hi = np.quantile(s, [0.025, 0.975], axis=0)
            print(f"  P{pos}: std {np.round(s.std(0), 3)}, widest 95 % interval {np.max(hi - lo):.2f}")
            corner_single(s, z[pos], C.LBOUND,
                          title=f"Sample {pos} ({descs[pos]}), {el_per_zone * C.N_ZONES}-element data: "
                                rf"reference posterior $p(\mathbf{{z}}|\mathbf{{m}})$",
                          cbar_label="relative posterior density (reference)",
                          save_path=os.path.join(fig_dir, f"Fig_reference_posterior_P{pos}_{split}.png"))


def main():
    parser = argparse.ArgumentParser(description="Balanced 7D bridge-on-bearings data set.")
    parser.add_argument("--data-dir", type=str, default=C.DATA_DIR)
    parser.add_argument("--n-train", type=int, default=C.N_TRAIN)
    parser.add_argument("--n-val", type=int, default=C.N_VAL)
    parser.add_argument("--n-test", type=int, default=C.N_TEST)
    parser.add_argument("--n-modes", type=int, default=C.N_MODES, help="identified modes (default: config.N_MODES)")
    parser.add_argument("--version", type=str, default=None, help="dataset_version (default: config.DATASET_VERSION); "
                        "its first '_'-separated token names the training run folders")
    args, _ = parser.parse_known_args()
    C.N_MODES = args.n_modes                      # every function of this module reads C.N_MODES at call time
    C.DATASET_VERSION = args.version or C.DATASET_VERSION
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(PROJECT_ROOT, args.data_dir)
    generate_full_benchmark_datasets(data_dir, args.n_train, args.n_val, args.n_test)


if __name__ == "__main__":
    main()
