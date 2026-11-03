# -*- coding: utf-8 -*-
"""
ks28_dataset.py -- Design of experiments and dataset I/O for the extended case study.

Folder produced by `generate_all_datasets` (one self-contained folder per dataset):

    Data/<dataset_name>/
        config.json            BridgeConfig + NoiseConfig + PriorConfig + generation options
        decoder_basis.npz      affine matrices of the TRAINING model (used by the TF decoder)
        train.npz, val.npz     training / validation (decoder model, scenario prior design)
        test_id.npz            in-distribution test (decoder model, scenario prior)
        val_shift.npz          shifted VALIDATION set: used only to estimate the model-
                               discrepancy noise (never the test sets) -> Reviewer 3 c.2
        tests/<family>.npz     evaluation families (healthy, mild single-zone, ..., shifted)
        README_dataset.md      what is inside, with the numbers of this run
        figures/               overview figures

Every .npz holds:
    theta     (N, D)        latent truth (zone factors, bearing log2-factors)
    f_obs     (N, nm)       noisy natural frequencies [Hz]
    phi_obs   (N, nm, ns)   noisy vertical mode shapes at the sensors (unit peak)
    f_clean, phi_clean      noise-free counterparts (for diagnostics only)
    n_damaged (N,), damaged_mask (N, nz), support_anomaly (N, 2)  scenario metadata
    label_exact (N,) bool   False when theta is only an "equivalent" label
                            (crack-like damage on a finer mesh)
    family    str

Test families and the reviewer comments they answer
---------------------------------------------------
  id_prior          scenario prior, training model                               baseline
  healthy           all zones in the healthy band, nominal bearings              R1.6 (false positives)
  single_mild       one zone, z in [0.80, 0.95]                                  R1.6 (early damage)
  single_moderate   one zone, z in [0.60, 0.80]                                  R1.6
  double            two zones damaged                                            R1.6
  bearing_anomaly   healthy deck, one bearing softened or stiffened              R2.2, R2.3
  uniform_box       z ~ U[z_lb, 1]^nz (design of the original paper)             comparability, R3.3
  ood_severe        one zone below the training bound, z in [0.30, 0.48]         R1.5
  fine_mesh         truth model with 4x finer mesh                               R1.1, R1.7
  mass_perturbed    fine mesh + random element mass field (CoV 5 %)              R3.1
  temperature       fine mesh + global E factor U[0.96, 1.04] (EOV)              R1.2
  kr_mismatch       fine mesh + rotational bearing stiffness x U[0.5, 2]         R2.2 (support model-form)
  crack_localized   fine mesh + 1-2 fine elements of one zone lose 30-70 %       R2.3 (damage parametrisation)
  combined_shift    fine mesh + mass 3 % + temperature 2 % + kr x U[0.7, 1.4]   all model-form sources
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np

from .ks28_config import BridgeConfig, NoiseConfig, PriorConfig, save_configs, load_configs
from .ks28_fe_model import (FEBasis, build_basis, forward, forward_truth, add_measurement_noise,
                            split_theta)
from .ks28_priors import sample_prior, sample_eta, sample_uniform_box, prior_summary


# =============================================================================
# helpers
# =============================================================================
def _healthy_eta(N, cfg, prior, rng):
    """Nominal (non-anomalous) bearings."""
    eta, anom = sample_eta(N, cfg, prior, rng, force_anomaly=False)
    return eta, anom


def _assemble_theta(z, eta, cfg):
    return np.concatenate([z, eta], axis=1) if cfg.support_latent else z


def _empty_meta(N, cfg):
    return dict(n_damaged=np.zeros(N, int), damaged_mask=np.zeros((N, cfg.n_zones), bool),
                support_anomaly=np.zeros((N, 2), bool))


def _meta_from_z(z, cfg, prior):
    dm = z < (1.0 - prior.healthy_band)
    return dict(n_damaged=dm.sum(1), damaged_mask=dm, support_anomaly=np.zeros((len(z), 2), bool))


# =============================================================================
# scenario generators -> theta (+ optional truth-model options)
# =============================================================================
def scenarios_for_family(family: str, N: int, cfg: BridgeConfig, prior: PriorConfig,
                         rng: np.random.Generator, fine_epz: int):
    """Return theta, meta, truth_options (dict passed to forward_truth or None)."""
    nz = cfg.n_zones
    truth = None
    label_exact = np.ones(N, bool)

    if family in ("id_prior", "train", "val"):
        theta, meta = sample_prior(N, cfg, prior, rng)
        meta.setdefault("support_anomaly", np.zeros((N, 2), bool))

    elif family == "healthy":
        theta, meta = sample_prior(N, cfg, prior, rng, force_n_damaged=0)
        if cfg.support_latent:
            eta, anom = _healthy_eta(N, cfg, prior, rng)
            theta[:, nz:] = eta
            meta["support_anomaly"] = anom
        else:
            meta["support_anomaly"] = np.zeros((N, 2), bool)

    elif family in ("single_mild", "single_moderate", "double"):
        nd = 2 if family == "double" else 1
        sev = {"single_mild": (0.80, 0.95), "single_moderate": (0.60, 0.80), "double": None}[family]
        theta, meta = sample_prior(N, cfg, prior, rng, force_n_damaged=nd, severity_range=sev)
        if cfg.support_latent:
            eta, anom = _healthy_eta(N, cfg, prior, rng)
            theta[:, nz:] = eta
            meta["support_anomaly"] = anom
        else:
            meta["support_anomaly"] = np.zeros((N, 2), bool)

    elif family == "bearing_anomaly":
        if not cfg.support_latent:
            raise ValueError("bearing_anomaly needs support_latent=True")
        theta, meta = sample_prior(N, cfg, prior, rng, force_n_damaged=0)
        eta, _ = _healthy_eta(N, cfg, prior, rng)
        side = rng.integers(0, 2, size=N)
        soft = rng.random(N) < 0.6                       # softening more frequent than seizing
        val = np.where(soft, rng.uniform(cfg.eta_lb, -0.5, N), rng.uniform(0.5, cfg.eta_ub, N))
        eta[np.arange(N), side] = val
        theta[:, nz:] = eta
        anom = np.zeros((N, 2), bool)
        anom[np.arange(N), side] = True
        meta["support_anomaly"] = anom

    elif family == "uniform_box":
        theta = sample_uniform_box(N, cfg, rng)
        if cfg.support_latent:      # keep bearings realistic; the box design concerns the deck
            eta, anom = _healthy_eta(N, cfg, prior, rng)
            theta[:, nz:] = eta
        meta = _meta_from_z(theta[:, :nz], cfg, prior)

    elif family == "ood_severe":
        theta, meta = sample_prior(N, cfg, prior, rng, force_n_damaged=1, severity_range=(0.30, 0.48))
        if cfg.support_latent:
            eta, anom = _healthy_eta(N, cfg, prior, rng)
            theta[:, nz:] = eta
        meta["support_anomaly"] = np.zeros((N, 2), bool)

    elif family in ("fine_mesh", "mass_perturbed", "temperature", "kr_mismatch",
                    "combined_shift", "val_shift"):
        theta, meta = sample_prior(N, cfg, prior, rng)
        meta.setdefault("support_anomaly", np.zeros((N, 2), bool))
        truth = dict(elements_per_zone=fine_epz)
        if family == "mass_perturbed":
            truth.update(mass_cov=0.05)
        elif family == "temperature":
            truth.update(E_global=rng.uniform(0.96, 1.04, N))
        elif family == "kr_mismatch":
            truth.update(kr_factor_per_sample=np.exp(rng.uniform(np.log(0.5), np.log(2.0), N)))
        elif family in ("combined_shift", "val_shift"):
            truth.update(mass_cov=0.03, E_global=rng.uniform(0.98, 1.02, N),
                         kr_factor_per_sample=np.exp(rng.uniform(np.log(0.7), np.log(1.4), N)))

    elif family == "crack_localized":
        theta, meta = sample_prior(N, cfg, prior, rng, force_n_damaged=0)
        if cfg.support_latent:
            eta, anom = _healthy_eta(N, cfg, prior, rng)
            theta[:, nz:] = eta
        n_el = nz * fine_epz
        elem = np.ones((N, n_el))
        zone = rng.integers(0, nz, size=N)
        for i in range(N):
            w = rng.integers(1, 3)
            start = zone[i] * fine_epz + rng.integers(0, fine_epz - w + 1)
            elem[i, start:start + w] = 1.0 - rng.uniform(0.30, 0.70)
        # equivalent zone factor (flexibility / harmonic mean over the zone's elements)
        for i in range(N):
            k = zone[i]
            fe = theta[i, k] * elem[i, k * fine_epz:(k + 1) * fine_epz]
            theta[i, k] = 1.0 / np.mean(1.0 / fe)
            elem[i, k * fine_epz:(k + 1) * fine_epz] = fe / theta[i, k]   # keep product = truth
        truth = dict(elements_per_zone=fine_epz, element_damage=elem)
        dm = np.zeros((N, nz), bool)
        dm[np.arange(N), zone] = True
        meta = dict(n_damaged=np.ones(N, int), damaged_mask=dm, support_anomaly=np.zeros((N, 2), bool))
        label_exact[:] = False
    else:
        raise ValueError(f"unknown family {family}")

    meta["label_exact"] = label_exact
    return theta, meta, truth


def simulate(theta, cfg: BridgeConfig, noise: NoiseConfig, rng, basis_dec: FEBasis,
             truth: Optional[dict] = None, chunk: int = 1000):
    """theta -> noisy observations, with the training model or a truth model."""
    N = theta.shape[0]
    if truth is None:
        f, phi = forward(theta, basis_dec, cfg)
    else:
        epz = truth.get("elements_per_zone", cfg.elements_per_zone)
        basis_t = build_basis(cfg, epz)
        kr_ps = truth.get("kr_factor_per_sample", None)
        fs, ps = [], []
        for i in range(0, N, chunk):
            sl = slice(i, min(i + chunk, N))
            n = sl.stop - sl.start
            E_g = truth.get("E_global", None)
            ed = truth.get("element_damage", None)
            if kr_ps is None:
                f_, p_, _ = forward_truth(theta[sl], cfg, rng, basis=basis_t,
                                          element_damage=None if ed is None else ed[sl],
                                          mass_cov=truth.get("mass_cov", 0.0),
                                          E_global=None if E_g is None else E_g[sl])
            else:   # per-sample kr factor -> loop over samples of this chunk (cheap)
                f_ = np.zeros((n, cfg.n_modes)); p_ = np.zeros((n, cfg.n_modes, cfg.n_sensors))
                for jj, j in enumerate(range(sl.start, sl.stop)):
                    f_[jj], p_[jj], _ = forward_truth(theta[j:j + 1], cfg, rng, basis=basis_t,
                                                      element_damage=None if ed is None else ed[j:j + 1],
                                                      mass_cov=truth.get("mass_cov", 0.0),
                                                      E_global=None if E_g is None else E_g[j:j + 1],
                                                      kr_factor=float(kr_ps[j]))
            fs.append(f_); ps.append(p_)
        f, phi = np.concatenate(fs), np.concatenate(ps)
    f_obs, phi_obs = add_measurement_noise(f, phi, noise.sigma_f, noise.sigma_phi, rng)
    return f_obs, phi_obs, f, phi


def make_family(family, N, cfg, noise, prior, rng, basis_dec, fine_epz):
    theta, meta, truth = scenarios_for_family(family, N, cfg, prior, rng, fine_epz)
    f_obs, phi_obs, f, phi = simulate(theta, cfg, noise, rng, basis_dec, truth)
    out = dict(theta=theta, f_obs=f_obs, phi_obs=phi_obs, f_clean=f, phi_clean=phi,
               family=np.array(family))
    out.update({k: np.asarray(v) for k, v in meta.items()})
    if truth is not None:
        for k in ("E_global", "kr_factor_per_sample"):
            if k in truth:
                out[k] = np.asarray(truth[k])
        out["truth_elements_per_zone"] = np.array(truth.get("elements_per_zone", cfg.elements_per_zone))
    return out


def make_training_design(N, cfg, noise, prior, rng, basis_dec, uniform_fraction=0.1):
    """Training/validation design: scenario prior + a small uniform 'defensive' share.

    The uniform share only widens the coverage of the encoder (severe multi-zone damage);
    the prior used in the ELBO is still the scenario prior (decoupling, R3.3).
    """
    n_u = int(round(uniform_fraction * N))
    theta_p, meta_p = sample_prior(N - n_u, cfg, prior, rng)
    meta_p.setdefault("support_anomaly", np.zeros((N - n_u, 2), bool))
    if n_u > 0:
        theta_u = sample_uniform_box(n_u, cfg, rng)
        meta_u = _meta_from_z(theta_u[:, :cfg.n_zones], cfg, prior)
        meta_u["support_anomaly"] = (np.abs(theta_u[:, cfg.n_zones:]) > 0.3) if cfg.support_latent \
            else np.zeros((n_u, 2), bool)
        theta = np.concatenate([theta_p, theta_u])
        meta = {k: np.concatenate([meta_p[k], meta_u[k]]) for k in ("n_damaged", "damaged_mask",
                                                                       "support_anomaly")}
        perm = rng.permutation(N)
        theta = theta[perm]
        meta = {k: v[perm] for k, v in meta.items()}
    else:
        theta, meta = theta_p, {k: meta_p[k] for k in ("n_damaged", "damaged_mask", "support_anomaly")}
    f_obs, phi_obs, f, phi = simulate(theta, cfg, noise, rng, basis_dec, None)
    out = dict(theta=theta, f_obs=f_obs, phi_obs=phi_obs, f_clean=f, phi_clean=phi,
               label_exact=np.ones(N, bool), family=np.array("train_design"))
    out.update(meta)
    return out


TEST_FAMILIES = ["id_prior", "healthy", "single_mild", "single_moderate", "double",
                 "bearing_anomaly", "uniform_box", "ood_severe",
                 "fine_mesh", "mass_perturbed", "temperature", "kr_mismatch",
                 "crack_localized", "combined_shift"]


def generate_all_datasets(root: str, name: str, cfg: BridgeConfig, noise: NoiseConfig,
                          prior: PriorConfig, n_train=40000, n_val=5000, n_test_id=5000,
                          n_family=1000, n_val_shift=2000, fine_epz: int = 8,
                          uniform_fraction=0.1, seed=2810, families=None, make_figures=True):
    """Generate the full extended dataset in root/name. Returns the folder path."""
    folder = os.path.join(root, name)
    os.makedirs(os.path.join(folder, "tests"), exist_ok=True)
    rng = np.random.default_rng(seed)
    basis = build_basis(cfg)

    # --- decoder basis (affine matrices of the training model) ---------------------------
    np.savez(os.path.join(folder, "decoder_basis.npz"),
             K_zone=basis.K_zone, K_sup=basis.K_sup, M=basis.M,
             sensor_dofs=basis.sensor_dofs, sensor_nodes=basis.sensor_nodes,
             node_x=basis.node_x, zone_of_element=basis.zone_of_element,
             K_fixed=np.zeros_like(basis.M) if cfg.support_latent else basis.K_sup.sum(0))
    gen_opts = dict(n_train=n_train, n_val=n_val, n_test_id=n_test_id, n_family=n_family,
                    n_val_shift=n_val_shift, fine_epz=fine_epz, uniform_fraction=uniform_fraction,
                    seed=seed)
    save_configs(os.path.join(folder, "config.json"), cfg, noise, prior, extra=gen_opts)

    print(f"[KS28] generating training design ({n_train}) ...")
    np.savez_compressed(os.path.join(folder, "train.npz"),
                        **make_training_design(n_train, cfg, noise, prior, rng, basis, uniform_fraction))
    print(f"[KS28] generating validation design ({n_val}) ...")
    np.savez_compressed(os.path.join(folder, "val.npz"),
                        **make_training_design(n_val, cfg, noise, prior, rng, basis, uniform_fraction))
    print(f"[KS28] generating in-distribution test ({n_test_id}) ...")
    np.savez_compressed(os.path.join(folder, "test_id.npz"),
                        **make_family("id_prior", n_test_id, cfg, noise, prior, rng, basis, fine_epz))
    print(f"[KS28] generating shifted validation set ({n_val_shift}) ...")
    np.savez_compressed(os.path.join(folder, "val_shift.npz"),
                        **make_family("val_shift", n_val_shift, cfg, noise, prior, rng, basis, fine_epz))

    families = TEST_FAMILIES if families is None else families
    if not cfg.support_latent:
        families = [f for f in families if f != "bearing_anomaly"]
    for fam in families:
        print(f"[KS28]   test family '{fam}' ({n_family}) ...")
        np.savez_compressed(os.path.join(folder, "tests", f"{fam}.npz"),
                            **make_family(fam, n_family, cfg, noise, prior, rng, basis, fine_epz))

    write_dataset_readme(folder, cfg, noise, prior, gen_opts, families)
    if make_figures:
        try:
            from .ks28_dataset_plots import make_dataset_figures
            make_dataset_figures(folder)
        except Exception as exc:          # figures are optional
            print(f"[KS28] figures skipped: {exc}")
    print(f"[KS28] dataset written to {folder}")
    return folder


# =============================================================================
# README of a dataset
# =============================================================================
def write_dataset_readme(folder, cfg, noise, prior, gen_opts, families):
    basis = build_basis(cfg)
    f0, _ = forward(cfg.healthy_latent[None], basis, cfg)
    ps = prior_summary(cfg, prior)
    lines = [
        f"# Dataset `{os.path.basename(folder)}`", "",
        "Generated by `MODULES/KS_review28oct/ks28_dataset.py` (extended case study, review 28 Oct).", "",
        "## Structure",
        f"- Single-span bridge, L = {cfg.L} m, EI = {cfg.EI:.3e} N m^2, m = {cfg.mass_per_length:.0f} kg/m",
        f"- Elastic bearings at both ends: kv = {cfg.kv_nominal:.3e} N/m, kr = {cfg.kr_nominal:.3e} N m/rad",
        f"- {cfg.n_zones} damage zones, decoder mesh {cfg.n_zones * cfg.elements_per_zone} elements, "
        f"truth mesh for shifted tests {cfg.n_zones * gen_opts['fine_epz']} elements",
        f"- {cfg.n_sensors} vertical sensors at x/L = {np.round(cfg.sensor_x_over_L, 3).tolist()}",
        f"- {cfg.n_modes} modes; healthy frequencies [Hz] = {np.round(f0[0], 3).tolist()}",
        f"- Latent vector: {cfg.latent_names} (bounds {cfg.latent_lower.tolist()} - {cfg.latent_upper.tolist()})",
        "", "## Noise (= likelihood)",
        f"- sigma_f = {noise.sigma_f} (log-normal), sigma_phi = {noise.sigma_phi} (x peak amplitude)",
        "", "## Scenario prior", f"- {ps}",
        "", "## Files",
        f"- train.npz ({gen_opts['n_train']}), val.npz ({gen_opts['n_val']}): scenario prior + "
        f"{int(100 * gen_opts['uniform_fraction'])} % uniform-box share, training model",
        f"- test_id.npz ({gen_opts['n_test_id']}): scenario prior, training model",
        f"- val_shift.npz ({gen_opts['n_val_shift']}): shifted VALIDATION set (model discrepancy calibration only)",
        f"- tests/: {families} ({gen_opts['n_family']} each)",
        "", "See the docstring of ks28_dataset.py for the meaning of every family.",
    ]
    with open(os.path.join(folder, "README_dataset.md"), "w") as fh:
        fh.write("\n".join(lines))


# =============================================================================
# LOADING / PREPROCESSING FOR TRAINING
# =============================================================================
def load_npz(path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def load_dataset_folder(folder):
    cfg, noise, prior, extra = load_configs(os.path.join(folder, "config.json"))
    basis = load_npz(os.path.join(folder, "decoder_basis.npz"))
    return cfg, noise, prior, extra, basis


def list_test_families(folder):
    d = os.path.join(folder, "tests")
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npz"))


class InputScaler:
    """Encoder inputs: standardised log-frequencies + unit-peak mode shapes (flattened).

    Statistics are computed on the TRAINING split only and stored with the model.
    """

    def __init__(self, mean_logf=None, std_logf=None):
        self.mean_logf = mean_logf
        self.std_logf = std_logf

    def fit(self, f_obs):
        lf = np.log(f_obs)
        self.mean_logf = lf.mean(0)
        self.std_logf = lf.std(0) + 1e-8
        return self

    def transform(self, f_obs, phi_obs, n_modes=None, use_modes=True):
        nm = f_obs.shape[1] if n_modes is None else n_modes
        x_f = (np.log(f_obs[:, :nm]) - self.mean_logf[:nm]) / self.std_logf[:nm]
        if not use_modes:
            return x_f.astype(np.float32)
        x_p = phi_obs[:, :nm, :].reshape(len(f_obs), -1)
        return np.concatenate([x_f, x_p], axis=1).astype(np.float32)

    def to_dict(self):
        return dict(mean_logf=self.mean_logf.tolist(), std_logf=self.std_logf.tolist())

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["mean_logf"]), np.array(d["std_logf"]))
