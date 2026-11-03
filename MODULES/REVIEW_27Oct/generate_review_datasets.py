# -*- coding: utf-8 -*-
"""
generate_review_datasets.py -- Synthetic datasets for the CPENG-8674 revision.

Run from the `code/` folder (same convention as main_GCopula_GMM.py):

    python -m MODULES.review_27oct.generate_review_datasets --list
    python -m MODULES.review_27oct.generate_review_datasets --group core
    python -m MODULES.review_27oct.generate_review_datasets --only TEST_E5_finemesh50 TEST_E5_healthy
    python -m MODULES.review_27oct.generate_review_datasets --group all --n-scale 0.1   # quick smoke test

Output: code/Data/review_27oct/<DATASET_NAME>/
    freqs_data_true.npy        (N, n_modes)          noisy natural frequencies [Hz]
    vertmodes_data_true.npy    (N, n_modes, n_s)     noisy vertical mode shapes at sensors
    rotmodes_data_true.npy     (N, n_modes, n_s)     noisy rotational mode shapes at sensors
    alpha_factors_true.npy     (N, n_latent)         labels [z_zone..., s_support...]
    *_clean.npy                                      noise-free counterparts
    z_element.npy / r_element.npy / g_factor.npy     element-level truth & nuisances (if used)
    Mass_matrix.npy, Ke_matrix.npy, L_inv.npy        legacy decoder inputs (nominal model!)
    Kzone_free.npy, Ksupport_free.npy, K0_free.npy   affine decoder inputs (nominal model!)
    split_indices.npz                                train/val/test indices (TRAIN sets only)
    metadata.json                                    full configuration + provenance

File names of the first four arrays are those expected by
MODULES.PREPROCESSING.preprocessing.load_data, so TRAIN sets can be used
directly with the existing pipeline.

IMPORTANT: decoder matrices are always those of the NOMINAL identification
model (the model the VAE "believes"), never of the perturbed data-generating
model.  This is what makes the TEST_* model-form sets a genuine test of
robustness to modelling error.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from dataclasses import replace

import numpy as np

from .fe_beam_model import BeamConfig, BeamFEModel, observe, save_json
from . import damage_scenarios as ds

# =============================================================================
# 0. STRUCTURAL CONFIGURATIONS
# =============================================================================
# (a) Paper model: 10 m steel beam, 5 elements = 5 zones, pinned, 6 sensors.
TOY_E5 = BeamConfig(name="toy_E5_pinned")

# (b) Same physical beam, finer FE meshes, SAME 5 zones and SAME 6 sensor locations (R1.1)
TOY_FINE = {n: replace(TOY_E5, name=f"toy_fine{n}_pinned", n_elements=n, n_zones=5, sensor_nodes=None)
            for n in (20, 50, 100)}

# (c) Toy beam on elastic bearings (R2.2/R2.3). kv_nom expressed through the
#     dimensionless ratio kappa = kv L^3 / EI. kappa = 200 places the bearings
#     in the range where their degradation is observable in the first modes
#     (see verify_fe_model.py, section 3) -- REVIEW THIS CHOICE.
_KAPPA_TOY = 200.0
TOY_ELASTIC = replace(TOY_E5, name="toy_fine20_elastic_latent", n_elements=20, n_zones=5,
                      support="elastic", kv_nom=_KAPPA_TOY * TOY_E5.EI / TOY_E5.L ** 3,
                      support_latent=True, sensor_nodes=None)

# (d) Representative short-span RC slab-girder bridge (R2.4, R1.7). PLACEHOLDER
#     values -- replace by the properties of a real bridge if available
#     (e.g. a FUTURAL case study).  L = 20 m, C40 concrete, 8 m deck,
#     superimposed dead load included in an effective density.
#     -> f1 ~ 4.3 Hz (pinned), typical of 20 m spans.
BRIDGE20 = BeamConfig(name="bridge20_RC_elastic", L=20.0, E=34e9, I=0.45, A=4.0, rho=3125.0,
                      n_elements=40, n_zones=10, support="elastic",
                      kv_nom=1.0e9, support_latent=True, sensor_nodes=None)
BRIDGE20_PINNED = replace(BRIDGE20, name="bridge20_RC_pinned", support="pinned", kv_nom=0.0,
                          support_latent=False)

# (e) Scalability family (R1.7): same 40-element bridge mesh, 11 fixed sensors,
#     increasing number of damage zones.
SCAL = {z: replace(BRIDGE20_PINNED, name=f"bridge20_Z{z}", n_zones=z, sensor_nodes=list(range(0, 41, 4)))
        for z in (5, 10, 20, 40)}

NOISE_PAPER = dict(sigma_f=0.025, sigma_phi=0.05)

# =============================================================================
# 1. DATASET REGISTRY
#    data_cfg  : model that GENERATES the data
#    ident_cfg : nominal model used by the VAE decoder (defaults to data_cfg)
# =============================================================================
REGISTRY = {}


def _reg(name, group, comment, data_cfg, sampler, n, ident_cfg=None, n_modes=5,
         noise=NOISE_PAPER, sampler_kw=None, nuisance_kw=None, train=False):
    REGISTRY[name] = dict(name=name, group=group, comment=comment, data_cfg=data_cfg,
                          ident_cfg=ident_cfg or data_cfg, sampler=sampler, n=n, n_modes=n_modes,
                          noise=dict(noise), sampler_kw=sampler_kw or {}, nuisance_kw=nuisance_kw or {},
                          train=train)


# ---------------------------- TRAINING SETS ---------------------------------
_reg("TRAIN_E5_uniform", "core", "Paper baseline re-generated with fixed seed (retraining after fixes; R3.2, R1.3, R1.4, R3.4, R3.7)",
     TOY_E5, "uniform", 10_000, train=True)
_reg("TRAIN_E5_realistic_doe", "core",
     "Realistic 4-regime Training DoE (15% healthy, 35% single-zone, 25% double-zone, 25% multi-zone; R1.6, R3.3)",
     TOY_E5, "realistic_bridge_doe", 10_000, train=True)
_reg("TRAIN_E5_finemesh20_realistic_doe", "core",
     "Observations from 20-FE mesh with realistic 4-regime DoE + 5-zone decoder matrices (R1.1, R1.6, R2.3)",
     TOY_FINE[20], "realistic_bridge_doe", 10_000, ident_cfg=TOY_E5, train=True)
_reg("06Oct_Data_Noisy_E5_Lvl25_MassPerturb5", "core",
     "Direct continuation of existing Oct work: 5% fixed element-mass modeling error (saves Mass_matrix_perturbed.npy + nominal Mass_matrix.npy; R3.1, R1.2)",
     TOY_E5, "uniform", 10_000, nuisance_kw=dict(fixed_mass_sigma=0.05), train=True)
_reg("TRAIN_E5_spikeslab", "core", "Healthy-inflated sampling density: p_healthy=0.3 per zone (R1.6, R3.3)",
     TOY_E5, "spike_slab", 10_000, sampler_kw=dict(p_healthy=0.3), train=True)
_reg("TRAIN_E5_massnuis", "uq", "Training data with random mass (5% element + 2.5% global) -> nuisance-latent / learned-noise models (R3.1, R1.2)",
     TOY_E5, "uniform", 10_000, nuisance_kw=dict(mass_sigma=0.05, mass_global_sigma=0.025), train=True)
_reg("TRAIN_toy_elastic", "supports", "Toy beam, 20 FE, 5 zones + 2 latent bearing factors s in [0.2,1] (R2.2, R2.3)",
     TOY_ELASTIC, "uniform", 10_000, nuisance_kw=dict(support_range=(0.2, 1.0)), train=True)
_reg("TRAIN_bridge20", "bridge", "Representative 20 m RC bridge, 40 FE, 10 zones + 2 bearings, 6 modes, 11 sensors (R2.4, R1.7)",
     BRIDGE20, "spike_slab", 20_000, n_modes=6, sampler_kw=dict(p_healthy=0.3),
     nuisance_kw=dict(support_range=(0.2, 1.0)), train=True)
for _z, _cfg in SCAL.items():
    _reg(f"TRAIN_scal_Z{_z}", "scaling", f"Scalability: {_z} zones, 40 FE, 11 sensors, 6 modes (R1.7)",
         _cfg, "uniform", 20_000, n_modes=6, train=True)

# ------------------------------ TEST SETS: paper model (TOY_E5) ---------------
_N = 1_000
_reg("TEST_E5_iid", "core", "Fresh in-distribution test set (unbiased final metrics; R3.2)", TOY_E5, "uniform", 2_000)
for _n, _cfg in TOY_FINE.items():
    _reg(f"TEST_E5_finemesh{_n}", "core", f"Data from a {_n}-element mesh, same 5 zones & sensors; decoder stays 5-element (R1.1 model-form error)",
         _cfg, "uniform", _N, ident_cfg=TOY_E5)
_reg("TEST_E5_healthy", "core", "Undamaged structure, noise only -> false-positive rate (R1.6)", TOY_E5, "healthy", 500)
_reg("TEST_E5_single", "core", "Mild damage (z in {0.95,0.9,0.8,0.7}) in ONE zone (R1.6)", TOY_E5, "single", _N)
_reg("TEST_E5_double", "core", "Mild damage in TWO zones, adjacent / non-adjacent (R1.6)", TOY_E5, "double", _N)
_reg("TEST_E5_oodsevere", "core", "1-2 zones with z in [0.30,0.45], below training support (R1.5)", TOY_E5, "ood_severe", _N)
_reg("TEST_E5_noise2x", "core", "Noise x2: sigma_f=5%, sigma_phi=10% (R1.5)", TOY_E5, "uniform", _N,
     noise=dict(sigma_f=0.05, sigma_phi=0.10))
_reg("TEST_E5_noise4x", "core", "Noise x4: sigma_f=10%, sigma_phi=20% (R1.5)", TOY_E5, "uniform", _N,
     noise=dict(sigma_f=0.10, sigma_phi=0.20))
for _s in (0.01, 0.025, 0.05, 0.10):
    _reg(f"TEST_E5_massR{int(_s * 1000):03d}", "core", f"i.i.d. element mass perturbation sigma={_s:.3f}; decoder uses nominal M (R3.1)",
         TOY_E5, "uniform", _N, nuisance_kw=dict(mass_sigma=_s))
_reg("TEST_E5_massG050", "core", "Global (unknown non-structural) mass offset sigma=5%: confounded with uniform damage (R3.1)",
     TOY_E5, "uniform", _N, nuisance_kw=dict(mass_global_sigma=0.05))
_reg("TEST_E5_eov", "uq", "Global stiffness factor U[0.97,1.03] (temperature/EOV surrogate) (R1.2)",
     TOY_E5, "uniform", _N, nuisance_kw=dict(g_range=(0.97, 1.03)))
_reg("TEST_E5_infprior", "uq", "z ~ zlb+(1-zlb)Beta(5,1): test set consistent with an informative prior (R3.3)",
     TOY_E5, "informative_prior", _N)
_reg("TEST_E5_localized_fine50", "core", "Crack-like damage on 1-2 of 50 FE inside one zone; label = harmonic zone equivalent (R2.3, R1.1)",
     TOY_FINE[50], "localized", _N, ident_cfg=TOY_E5)
_reg("TEST_E5_elasticdata", "supports", "Data from beam on elastic bearings (s in [0.2,1]) identified with pinned model (R2.2: cost of the pinned assumption)",
     TOY_ELASTIC, "uniform", _N, ident_cfg=TOY_E5, nuisance_kw=dict(support_range=(0.2, 1.0)))

# ------------------------------ TEST SETS: elastic toy beam ------------------
_reg("TEST_toy_elastic_iid", "supports", "In-distribution test, elastic toy beam", TOY_ELASTIC, "uniform", 2_000,
     nuisance_kw=dict(support_range=(0.2, 1.0)))
_reg("TEST_toy_elastic_supportonly", "supports", "Healthy beam, degraded bearings only -> can support damage be localised? (R2.3)",
     TOY_ELASTIC, "healthy", _N, nuisance_kw=dict(support_range=(0.2, 1.0)))

# ------------------------------ TEST SETS: 20 m bridge -----------------------
_reg("TEST_bridge20_iid", "bridge", "In-distribution test, 20 m bridge", BRIDGE20, "spike_slab", 2_000, n_modes=6,
     sampler_kw=dict(p_healthy=0.3), nuisance_kw=dict(support_range=(0.2, 1.0)))
_reg("TEST_bridge20_healthy", "bridge", "Healthy bridge, nominal bearings (false positives)", BRIDGE20, "healthy", 500, n_modes=6)
_reg("TEST_bridge20_single", "bridge", "Single-zone mild damage", BRIDGE20, "single", _N, n_modes=6)
_reg("TEST_bridge20_localized", "bridge", "Crack-like damage on 1-2 FE (0.5-1.0 m) inside a 2 m zone", BRIDGE20, "localized", _N,
     n_modes=6)
_reg("TEST_bridge20_mass05", "bridge", "Element mass perturbation 5% + global 2.5%", BRIDGE20, "uniform", _N, n_modes=6,
     nuisance_kw=dict(mass_sigma=0.05, mass_global_sigma=0.025))
for _z, _cfg in SCAL.items():
    _reg(f"TEST_scal_Z{_z}", "scaling", f"Scalability test, {_z} zones", _cfg, "uniform", 2_000, n_modes=6)

GROUPS = sorted({v["group"] for v in REGISTRY.values()})


# =============================================================================
# 2. GENERATION
# =============================================================================
def _seed(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def generate(spec: dict, out_root: str, n_scale: float = 1.0, verbose: bool = True) -> str:
    name, dcfg, icfg = spec["name"], spec["data_cfg"], spec["ident_cfg"]
    n = max(int(round(spec["n"] * n_scale)), 10)
    n_modes = spec["n_modes"]
    rng = np.random.default_rng(_seed(name))
    sampler = ds.SAMPLERS[spec["sampler"]]
    model = BeamFEModel(dcfg)
    ident = BeamFEModel(icfg)

    if icfg.n_zones != dcfg.n_zones or icfg.n_support_latents > dcfg.n_support_latents:
        raise ValueError(f"{name}: identification model latent space incompatible with data model")
    # ident sensors must be the same physical positions as data sensors
    x_d = np.asarray(dcfg.sensor_nodes) * dcfg.Le
    x_i = np.asarray(icfg.sensor_nodes) * icfg.Le
    if not np.allclose(x_d, x_i):
        raise ValueError(f"{name}: sensor positions differ between data and identification models")

    n_s = len(dcfg.sensor_nodes)
    F, V, R = (np.zeros((n, n_modes)), np.zeros((n, n_modes, n_s)), np.zeros((n, n_modes, n_s)))
    Fc, Vc, Rc = np.zeros_like(F), np.zeros_like(V), np.zeros_like(R)
    n_lat = icfg.n_zones + icfg.n_support_latents
    Z = np.zeros((n, n_lat))
    ZE = np.zeros((n, dcfg.n_elements))
    RE = np.ones((n, dcfg.n_elements))
    G = np.ones(n)
    metas = []

    t0 = time.time()
    nuis_kw = dict(spec["nuisance_kw"])
    fixed_mass_sigma = nuis_kw.pop("fixed_mass_sigma", 0.0)
    r_fixed = None
    if fixed_mass_sigma > 0:
        r_fixed = np.clip(rng.normal(1.0, fixed_mass_sigma, dcfg.n_elements), 0.7, 1.3)
        M_perturbed = model.mass(r_fixed)
    else:
        M_perturbed = None

    for i in range(n):
        d = sampler(dcfg, rng, **spec["sampler_kw"])
        d = ds.add_nuisances(d, dcfg, rng, **nuis_kw)
        if r_fixed is not None:
            d["r_element"] = r_fixed
        K = model.stiffness(d["z_zone"], d["s_support"], d["z_element"], d["g_factor"])
        M = M_perturbed if M_perturbed is not None else model.mass(d["r_element"])
        f, v, r = model.modal(K, M, n_modes)
        (fc, vc, rc), (fn, vn, rn) = observe(f, v, r, dcfg.sensor_nodes, rng, **spec["noise"])
        F[i], V[i], R[i] = fn, vn, rn
        Fc[i], Vc[i], Rc[i] = fc, vc, rc
        lab = list(d["z_zone"])
        if icfg.n_support_latents:
            s = d["s_support"] if d["s_support"] is not None else np.ones(icfg.n_support_latents)
            lab += list(s)
        Z[i] = lab
        ZE[i] = d["z_element"] if d["z_element"] is not None else np.repeat(d["z_zone"], dcfg.elements_per_zone)
        if d["r_element"] is not None:
            RE[i] = d["r_element"]
        G[i] = d["g_factor"]
        metas.append(d.get("meta", {}))

    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    sv = lambda fn, arr: np.save(os.path.join(out, fn), arr)
    sv("freqs_data_true.npy", F); sv("vertmodes_data_true.npy", V); sv("rotmodes_data_true.npy", R)
    sv("alpha_factors_true.npy", Z)
    sv("freqs_clean.npy", Fc); sv("vertmodes_clean.npy", Vc); sv("rotmodes_clean.npy", Rc)
    sv("z_element.npy", ZE); sv("r_element.npy", RE); sv("g_factor.npy", G)
    for k, arr in ident.decoder_matrices().items():   # NOMINAL identification model
        sv(f"{k}.npy", arr)
    if M_perturbed is not None:
        sv("Mass_matrix_perturbed.npy", M_perturbed)
    if spec["train"]:
        idx = rng.permutation(n)            # data are already i.i.d.; permutation kept for safety
        n_tr, n_va = int(0.7 * n), int(0.2 * n)
        np.savez(os.path.join(out, "split_indices.npz"), train=np.sort(idx[:n_tr]),
                 val=np.sort(idx[n_tr:n_tr + n_va]), test=np.sort(idx[n_tr + n_va:]))
    meta = dict(name=name, group=spec["group"], comment=spec["comment"], n_samples=n, n_modes=n_modes,
                sampler=spec["sampler"], sampler_kw=spec["sampler_kw"], nuisance_kw=spec["nuisance_kw"],
                noise=spec["noise"], seed=_seed(name), is_train=spec["train"],
                data_model=dcfg.to_dict(), identification_model=icfg.to_dict(),
                latent_names=icfg.latent_names(),
                sensor_positions_m=x_d.tolist(), scenario_meta=metas[:50],
                created=time.strftime("%Y-%m-%d %H:%M:%S"), generator="MODULES/review_27oct/generate_review_datasets.py")
    save_json(meta, os.path.join(out, "metadata.json"))
    if verbose:
        print(f"  {name:32s} N={n:6d} modes={n_modes} sensors={n_s:2d} latent={n_lat:2d} "
              f"f1=[{F[:, 0].min():6.2f},{F[:, 0].max():6.2f}] Hz  ({time.time() - t0:5.1f}s)")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", default=None, help=f"one of {GROUPS} or 'all'")
    p.add_argument("--only", nargs="*", default=None, help="explicit dataset names")
    p.add_argument("--out", default=os.path.join("Data", "review_27oct"))
    p.add_argument("--n-scale", type=float, default=1.0, help="scale all sample sizes (smoke tests)")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()

    if a.list or (a.group is None and not a.only):
        for g in GROUPS:
            print(f"\n[{g}]")
            for k, v in REGISTRY.items():
                if v["group"] == g:
                    print(f"  {k:32s} N={v['n']:6d}  {v['comment']}")
        return
    names = a.only if a.only else [k for k, v in REGISTRY.items() if a.group == "all" or v["group"] == a.group]
    os.makedirs(a.out, exist_ok=True)
    print(f"Generating {len(names)} datasets into {a.out}")
    for k in names:
        generate(REGISTRY[k], a.out, a.n_scale)

    # Write a clear human-readable index inside Data/review_27oct/MANIFEST.md
    manifest_path = os.path.join(a.out, "MANIFEST.md")
    lines = [
        "# Review 27Oct Datasets Manifest (`code/Data/review_27oct/`)\n",
        "All datasets generated for the MS CPENG-8674 Major Revision, mapped to Reviewer Comments.\n",
        "| Dataset Folder | Group | Type | Samples ($N$) | Modes | Addressed Comments & Description |",
        "|---|---|---|---:|---:|---|",
    ]
    for k, v in REGISTRY.items():
        exists_mark = "**[GENERATED]** " if os.path.exists(os.path.join(a.out, k, "metadata.json")) else ""
        dtype = "TRAIN (70/20/10 split)" if v["train"] else "TEST (evaluation)"
        lines.append(f"| `{k}` | `{v['group']}` | {dtype} | {v['n']} | {v['n_modes']} | {exists_mark}{v['comment']} |")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Updated dataset manifest at {manifest_path}")


if __name__ == "__main__":
    main()
