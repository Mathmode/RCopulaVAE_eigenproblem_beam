#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed/localized_damage_study.py

Extra test scenarios with localized damage (reviewers R1.6, R2.3), to see case by case how the identification behaves:
  A. one damaged deck zone, every zone at severities 0.55, 0.70, 0.85 and 0.90 (20 cases);
  B. a short defect INSIDE a zone, which the zone-wise parametrization does not represent: one or two 0.2 m cells with
     a strong stiffness loss at different positions (6 cases). The reference value of the zone is its equivalent
     (harmonic-mean) stiffness factor;
  C. one degraded bearing, two adjacent damaged zones, two symmetric damaged zones (3 cases).
Undamaged parameters are 0.98. Every scenario is an as-built structure (mass and stiffness fields of the data set, a
fixed seed) computed with the 50-element model plus measurement noise, exactly like the test-fine set. For each one the
exact posterior (SMC, the likelihood of the data set) and, if a trained run is given, the VAE posterior are computed:
  Fig_localization_map.png     single-zone cases: 5 x 4 grid of violin plots (exact posterior vs VAE, truth)
  Fig_within_zone_defects.png  defects inside a zone: posteriors vs the equivalent zone stiffness
  Fig_other_scenarios.png      bearing and double-damage cases
  violins/S<nn>_<name>.png     one violin plot per scenario
  localized_scenarios.csv      mean, std, 95 % interval of every parameter; damage declared (upper 95 % bound < 0.95)

    python MODULES/review29oct_fixed/localized_damage_study.py --run-folder KSVAE_fixed_7D_3KS_obs-fv_Gamma1.0_Ep1500
    python MODULES/review29oct_fixed/localized_damage_study.py          (data set of config.py, exact posteriors only)
The data set (likelihood, noise, as-built scatter) is the one the run was trained on.
"""

import os
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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from multiprocessing import Pool

from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.fe_model import BeamMesh, BridgeModel, draw_asbuilt_fields, simulate_true_structure, \
    add_measurement_noise
from MODULES.review29oct_fixed.reference_posterior import smc_sample, load_problem_config

HEALTHY = 0.98
SEVERITIES = [0.55, 0.70, 0.85, 0.90]
THRESHOLD = C.DAMAGE_THRESHOLD
CELLS_PER_ZONE = C.EL_PER_ZONE_TEST          # 0.2 m cells (one element of the 50-element model each)


# =============================================================================
# 1. SCENARIOS
# =============================================================================
def build_scenarios():
    """List of dicts: name, kind, z (7,), defect = [(zone, [cells inside the zone], stiffness factor)] or []."""
    out = []
    for k in range(C.N_ZONES):
        for s in SEVERITIES:
            z = np.full(C.N_DIMS, HEALTHY)
            z[k] = s
            out.append({"name": f"zone{k + 1}_{s:.2f}", "kind": "single zone", "z": z, "defect": [],
                        "label": f"zone {k + 1} = {s:.2f}"})
    for zone, cells, d, where in ((2, [5], 0.30, "centre"), (2, [0], 0.30, "edge"), (2, [4, 5], 0.50, "centre, 0.4 m"),
                                  (0, [0], 0.30, "at the support"), (0, [5], 0.30, "centre"), (1, [5], 0.30, "centre")):
        out.append({"name": f"defect_zone{zone + 1}_{where.split(',')[0].replace(' ', '')}_{len(cells)}cell",
                    "kind": "defect inside a zone", "z": np.full(C.N_DIMS, HEALTHY), "defect": [(zone, cells, d)],
                    "label": f"zone {zone + 1}, {where}: {0.2 * len(cells):.1f} m with stiffness x{d:.2f}"})
    for z_mod, name, label in (({5: 0.55}, "bearingL_0.55", "left bearing = 0.55"),
                               ({1: 0.70, 2: 0.70}, "zones2+3_0.70", "zones 2 and 3 = 0.70"),
                               ({0: 0.70, 4: 0.70}, "zones1+5_0.70", "zones 1 and 5 = 0.70 (symmetric)")):
        z = np.full(C.N_DIMS, HEALTHY)
        for k, v in z_mod.items():
            z[k] = v
        out.append({"name": name, "kind": "other", "z": z, "defect": [], "label": label})
    return out


def equivalent_zone_factors(z, flex, zones=C.N_ZONES):
    """Equivalent (harmonic-mean) stiffness factor of every zone of the as-built structure (cells in series)."""
    zf = z.copy()
    zf[:zones] = z[:zones] / np.mean(1.0 + flex.reshape(zones, -1), axis=1)
    return zf


def simulate_scenario(sc, pcfg, seed):
    """Noisy measurement of the as-built scenario with the 50-element model; returns f, v, r and the reference z."""
    rng = np.random.default_rng(seed)
    mass, flex = draw_asbuilt_fields(1, rng, C.N_ZONES, int(pcfg["el_per_zone_decoder"]), CELLS_PER_ZONE,
                                     float(pcfg["mass_perturb"]), float(pcfg["within_zone_stiff_var"]))
    for zone, cells, d in sc["defect"]:
        idx = [zone * CELLS_PER_ZONE + c for c in cells]
        flex[0, idx] = (1.0 + flex[0, idx]) / d - 1.0        # stiffness of the defective cells x d
    mesh = BeamMesh(int(pcfg["n_zones"]), int(pcfg["el_per_zone_test"]), sensor_positions=pcfg["sensor_positions"],
                    **pcfg["structure"])
    ref = (np.asarray(pcfg["ref_vert"]), np.asarray(pcfg["ref_rot"]))
    f, v, r = simulate_true_structure(sc["z"][None], mesh, int(pcfg["n_modes"]), mass, flex, ref=ref)
    f, v, r = add_measurement_noise(f, v, r, rng, float(pcfg["freq_noise"]), float(pcfg["mode_noise"]), ref=ref)
    # the as-built scatter has zero zone mean; only a defect changes the equivalent factor of its zone
    z_ref = equivalent_zone_factors(sc["z"], flex[0]) if sc["defect"] else sc["z"].copy()
    return f[0], v[0], r[0], z_ref


def _exact_posterior(args):
    i, data_dir, f, v, r, n_particles = args
    pcfg = load_problem_config(data_dir)
    log_lik = BridgeModel(pcfg).log_likelihood_fn(f, v, r, gamma2=1.0, obs_types=pcfg["obs_types"])
    s, _ = smc_sample(log_lik, int(pcfg["n_dims"]), float(pcfg["lbound"]), n_particles=n_particles,
                      rng=np.random.default_rng(50_000 + i))
    return i, s


# =============================================================================
# 2. VAE POSTERIOR OF A TRAINED RUN
# =============================================================================
def load_trained_vae(run_folder, data_dir, info):
    """Rebuilds the VAE of a trained run and loads its weights (also files written by another Keras version)."""
    from MODULES.review29oct_fixed import tf_keras
    K = tf_keras()
    K.backend.set_floatx("float32")
    from MODULES.review29oct_fixed.preprocessing_and_eigen import load_baseline_matrices
    from MODULES.review29oct_fixed.GC_KSmix_models import My_CopulaKSVAE_withEigen, load_vae_weights
    pcfg = load_problem_config(data_dir)
    _, K_basis, L_inv = load_baseline_matrices(data_dir)
    n_modes, n_sens = int(pcfg["n_modes"]), len(pcfg["sensor_positions"])
    model = My_CopulaKSVAE_withEigen(
        input_dim=int(info["input_dim_enc"]), num_dofs=K_basis.shape[-1], n_modes=n_modes, Ke_matrices=K_basis,
        L_inv=L_inv, n_dims=int(info["n_dims"]), num_KS=int(info["num_KS"]), num_samples=int(info["n_samples"]),
        gamma=float(info["gamma"]), mean_f=info["mean_f"], std_f=info["std_f"], lbound=float(info["lbound"]),
        likelihood=info["likelihood"], obs_types=info["obs_types"], sensor_nodes=pcfg.get("sensor_nodes"))
    unit = np.full((2, n_modes, n_sens), 1.0 / np.sqrt(n_sens), np.float32)       # unit-norm placeholder shapes
    _ = model((np.zeros((2, n_modes), np.float32), unit, unit, np.ones((2, int(info["n_dims"])), np.float32)))
    how = load_vae_weights(model, os.path.join(run_folder, "final_model_weights.weights.h5"))
    print(f"  weights of {os.path.basename(run_folder)} loaded ({how})")
    return model


def vae_posteriors(run_folder, data_dir, obs, n_samples, info):
    """Samples of q(z|m) of the trained run for the observations obs = (f, v, r) [N, ...]."""
    from MODULES.review29oct_fixed.postprocessing import sample_vae_posterior
    model = load_trained_vae(run_folder, data_dir, info)
    f, v, r = (np.asarray(x, np.float32) for x in obs)
    f_s = ((np.log(f) - info["mean_f"]) / info["std_f"]).astype(np.float32)
    a = np.ones((len(f), int(info["n_dims"])), np.float32)
    ta, tb, tw, toff, tdiag = model.Encoder_model.predict((f_s, r, v, a), verbose=0)
    preds = {"test_a": ta, "test_b": tb, "test_weight_vals": tw, "test_offdiag_elems": toff, "test_diag_elems": tdiag}
    return [sample_vae_posterior(preds, i, n_samples, float(info["lbound"]))[0] for i in range(len(f))]


# =============================================================================
# 3. FIGURES
# =============================================================================
def violin_axes(ax, x_ref, x_vae, z_ref, lb, width=0.4):
    """Seven violins: VAE (filled) and exact posterior (dashed outline), true / equivalent value (star)."""
    from MODULES.review29oct_fixed.plotting import kde_1d, RED
    zg = np.linspace(lb, 1.0, 300)
    for d in range(len(z_ref)):
        g = kde_1d(x_ref[:, d], zg, lb)
        g = width * g / (g.max() + 1e-12)
        if x_vae is not None:
            q = kde_1d(x_vae[:, d], zg, lb)
            q = width * q / (q.max() + 1e-12)
            ax.fill_betweenx(zg, d + 1 - q, d + 1 + q, color="steelblue" if d < 5 else "#2CA02C", alpha=0.4, lw=0)
        ax.plot(d + 1 - g, zg, color=RED, ls="--", lw=1.0)
        ax.plot(d + 1 + g, zg, color=RED, ls="--", lw=1.0)
        ax.plot(d + 1, z_ref[d], "*", color="#FFD700", ms=10, mec="black", mew=0.8, zorder=10)
    ax.axhline(THRESHOLD, color="gray", ls=":", lw=0.8)
    ax.axvline(5.5, color="#555555", ls="-.", lw=0.8)
    ax.set_ylim(lb - 0.02, 1.02)
    ax.set_xlim(0.4, len(z_ref) + 0.6)


def plot_grid(scenarios, idx, refs, vaes, lb, nrows, ncols, title, save_path, panel_w=4.3, panel_h=3.3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from MODULES.review29oct_fixed.plotting import LABELS_7D, RED, PAPER_RC, save_figure
    with plt.style.context(["default", PAPER_RC]):
        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows + 1.0), squeeze=False,
                                 sharey=True, facecolor="white")
        for a, i in zip(axes.ravel(), idx):
            violin_axes(a, refs[i], vaes[i] if vaes else None, scenarios[i]["z_ref"], lb)
            a.set_title(scenarios[i]["label"], fontsize=10, fontweight="bold")
            a.set_xticks(np.arange(1, 8), LABELS_7D, fontsize=9)
            a.grid(axis="y", ls=":", alpha=0.4)
        for a in axes.ravel()[len(idx):]:
            a.axis("off")
        for a in axes[:, 0]:
            a.set_ylabel("stiffness factor z")
        handles = [mlines.Line2D([], [], color=RED, ls="--", label="exact posterior (SMC)"),
                   mlines.Line2D([], [], marker="*", color="#FFD700", mec="black", ls="", ms=12,
                                 label="true value (equivalent value of a zone with a defect)"),
                   mlines.Line2D([], [], color="gray", ls=":", label=f"damage threshold {THRESHOLD}")]
        if vaes:
            handles.insert(0, mpatches.Patch(color="steelblue", alpha=0.4, label="VAE posterior (deck)"))
            handles.insert(1, mpatches.Patch(color="#2CA02C", alpha=0.4, label="VAE posterior (bearings)"))
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=10, bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.035)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_figure(fig, save_path)
        plt.close(fig)


def summary_rows(scenarios, refs, vaes, lb):
    rows = []
    for i, sc in enumerate(scenarios):
        row = {"scenario": sc["name"], "kind": sc["kind"], "description": sc["label"]}
        for tag, xs in (("ref", refs[i]), ("vae", vaes[i] if vaes else None)):
            if xs is None:
                continue
            lo, hi = np.quantile(xs, [0.025, 0.975], axis=0)
            declared = hi < THRESHOLD
            truly = sc["z_ref"] < THRESHOLD
            row[f"{tag}_detected"] = int(np.sum(declared & truly))
            row[f"{tag}_missed"] = int(np.sum(~declared & truly))
            row[f"{tag}_false_alarms"] = int(np.sum(declared & ~truly))
            row[f"{tag}_zstar_in_95"] = int(np.sum((sc["z_ref"] >= lo) & (sc["z_ref"] <= hi)))
            for d, name in enumerate(C.PARAM_NAMES):
                row[f"{name}_{tag}_mean"] = float(xs[:, d].mean())
                row[f"{name}_{tag}_std"] = float(xs[:, d].std())
                row[f"{name}_{tag}_lo95"], row[f"{name}_{tag}_hi95"] = float(lo[d]), float(hi[d])
        for d, name in enumerate(C.PARAM_NAMES):
            row[f"{name}_true"] = float(sc["z_ref"][d])
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 4. DRIVER
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Localized-damage test scenarios: exact and VAE posteriors.")
    parser.add_argument("--run-folder", type=str, default=None,
                        help="trained run (name in Output/review29oct_fixed or full path); default: exact posteriors "
                             "only, data set of config.py")
    parser.add_argument("--n-particles", type=int, default=4000)
    parser.add_argument("--n-samples", type=int, default=4000, help="samples of q per scenario")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=777)
    args, _ = parser.parse_known_args()

    run_folder, info = None, None
    if args.run_folder:
        run_folder = args.run_folder if os.path.isabs(args.run_folder) else os.path.join(C.OUTPUT_DIR, args.run_folder)
        from MODULES.review29oct_fixed.postprocessing import load_run_info
        info = load_run_info(run_folder)
        data_dir = info["data_dir"]
        out_dir = os.path.join(run_folder, "Localized_scenarios")
    else:
        data_dir = C.DATA_DIR
        out_dir = os.path.join(C.OUTPUT_DIR, f"localized_scenarios_{os.path.basename(data_dir)}")
    os.makedirs(os.path.join(out_dir, "violins"), exist_ok=True)
    pcfg = load_problem_config(data_dir)
    lb = float(pcfg["lbound"])
    print(f"=== Localized-damage scenarios: data set {os.path.basename(data_dir)} "
          f"(mass scatter {pcfg['mass_perturb']}, noise {pcfg['freq_noise']}/{pcfg['mode_noise']}), "
          f"{'run ' + os.path.basename(run_folder) if run_folder else 'exact posteriors only'} ===")

    scenarios = build_scenarios()
    obs = []
    for i, sc in enumerate(scenarios):
        f, v, r, z_ref = simulate_scenario(sc, pcfg, args.seed + i)
        sc["z_ref"] = z_ref
        obs.append((f, v, r))

    workers = args.workers or max(1, (os.cpu_count() or 2) // 2)
    jobs = [(i, data_dir, o[0], o[1], o[2], args.n_particles) for i, o in enumerate(obs)]
    print(f"--- exact posteriors of {len(jobs)} scenarios (SMC, {args.n_particles} particles, {workers} workers) ---")
    with Pool(min(workers, len(jobs))) as pool:
        refs = dict(pool.map(_exact_posterior, jobs))
    refs = [refs[i] for i in range(len(scenarios))]

    vaes = None
    if run_folder:
        print("--- VAE posteriors ---")
        vaes = vae_posteriors(run_folder, data_dir, tuple(np.stack([o[k] for o in obs]) for k in range(3)),
                              args.n_samples, info)

    df = summary_rows(scenarios, refs, vaes, lb)
    df.to_csv(os.path.join(out_dir, "localized_scenarios.csv"), index=False)
    tag = "exact posterior (red) and VAE (filled)" if vaes else "exact posterior"
    single = [i for i, s in enumerate(scenarios) if s["kind"] == "single zone"]
    plot_grid(scenarios, single, refs, vaes, lb, C.N_ZONES, len(SEVERITIES),
              f"One damaged zone (others {HEALTHY}): {tag}, 50-element data",
              os.path.join(out_dir, "Fig_localization_map.png"))
    defects = [i for i, s in enumerate(scenarios) if s["kind"] == "defect inside a zone"]
    plot_grid(scenarios, defects, refs, vaes, lb, 2, 3,
              f"Short defects inside a zone (not representable by the zone parameters): {tag}",
              os.path.join(out_dir, "Fig_within_zone_defects.png"), panel_w=5.2)
    others = [i for i, s in enumerate(scenarios) if s["kind"] == "other"]
    plot_grid(scenarios, others, refs, vaes, lb, 1, 3, f"Bearing and double damage: {tag}",
              os.path.join(out_dir, "Fig_other_scenarios.png"), panel_w=5.2)

    from MODULES.review29oct_fixed.plotting import physical_profile
    for i, sc in enumerate(scenarios):
        physical_profile(refs[i], vaes[i] if vaes else refs[i], sc["z_ref"], lb,
                         os.path.join(out_dir, "violins", f"S{i:02d}_{sc['name']}.png"),
                         f"Scenario {i}: {sc['label']} (50-element data)")

    cols = ["scenario", "ref_detected", "ref_missed", "ref_false_alarms", "ref_zstar_in_95"]
    if vaes:
        cols += ["vae_detected", "vae_missed", "vae_false_alarms", "vae_zstar_in_95"]
    pd.set_option("display.width", 200)
    print(df[cols].to_string(index=False))
    print(f"\nFigures and CSV in {out_dir}")


if __name__ == "__main__":
    main()
