# -*- coding: utf-8 -*-
"""
dataset_generation.py -- STEP (a): design of experiments and data generation.

Spyder: F5.  Console: python dataset_generation.py [--scale 0.1] [--no-plots]

Writes to ./Data:
  coarse_model.npz    affine matrices + noise levels of the inference model
  <set>.npz           train / val / test_* / ood_* datasets
  benchmarks.npz      the 10 benchmark scenarios, simulated with the coarse and fine models
  metadata.json       configuration used
  figures/            DoE, model-form error and identifiability figures

Every dataset contains
  theta (N, 7)                labels [Z1..Z5, BL, BR] (zone-equivalent stiffness for fine models)
  y_logf (N, 5)               noisy log-frequencies [log Hz]
  y_v, y_r (N, 5, 6)          noisy vertical / rotational unit-norm mode shapes at the 6 sensors
  logf_clean, v_clean, r_clean  noise-free values
  scenario (N,), damaged_mask (N, 7), log_prior (N,), log_doe (N,)
  logf_baseline_shift (5,)    healthy-state log-frequency offset fine - coarse (zero for coarse)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config as C  # noqa: E402  (sets KERAS_BACKEND before tensorflow is imported)
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt

from priors import build_prior, build_doe, sample_four_sites, SCENARIOS
from preprocessing_and_eigen import modal_arrays, ModalModelNP, align_to_reference


# =============================================================================
# Finite-element model
# =============================================================================
def beam_element_matrices(E, I, rho, Area, Le):
    """4x4 Euler-Bernoulli stiffness and consistent mass with rotary inertia (DOFs v1, r1, v2, r2)."""
    Ke = E * I / Le ** 3 * np.array([
        [12.0, 6.0 * Le, -12.0, 6.0 * Le],
        [6.0 * Le, 4.0 * Le ** 2, -6.0 * Le, 2.0 * Le ** 2],
        [-12.0, -6.0 * Le, 12.0, -6.0 * Le],
        [6.0 * Le, 2.0 * Le ** 2, -6.0 * Le, 4.0 * Le ** 2]])
    r2 = I / Area
    Me = rho * Area * Le * np.array([
        [13 / 35 + 6 * r2 / (5 * Le ** 2), 11 * Le / 210 + r2 / (10 * Le),
         9 / 70 - 6 * r2 / (5 * Le ** 2), -13 * Le / 420 + r2 / (10 * Le)],
        [11 * Le / 210 + r2 / (10 * Le), Le ** 2 / 105 + 2 * r2 / 15,
         13 * Le / 420 - r2 / (10 * Le), -Le ** 2 / 140 - r2 / 30],
        [9 / 70 - 6 * r2 / (5 * Le ** 2), 13 * Le / 420 - r2 / (10 * Le),
         13 / 35 + 6 * r2 / (5 * Le ** 2), -11 * Le / 210 - r2 / (10 * Le)],
        [-13 * Le / 420 + r2 / (10 * Le), -Le ** 2 / 140 - r2 / 30,
         -11 * Le / 210 - r2 / (10 * Le), Le ** 2 / 105 + 2 * r2 / 15]])
    return Ke, Me


class BeamFE:
    """Beam on elastic bearings with el_per_zone elements per damage zone."""

    def __init__(self, beam: C.BeamConfig, el_per_zone: int):
        self.beam, self.nsub = beam, el_per_zone
        self.n_el = beam.n_zones * el_per_zone
        self.n_dofs = 2 * (self.n_el + 1)
        self.Ke, self.Me = beam_element_matrices(beam.E, beam.I, beam.rho, beam.Area,
                                                 beam.L_total / self.n_el)
        self.zone_of_element = np.arange(self.n_el) // el_per_zone
        self.sensor_nodes = np.arange(0, self.n_el + 1, el_per_zone)      # zone boundaries

    def assemble(self, elem_k, elem_m, bearings):
        K = np.zeros((self.n_dofs, self.n_dofs))
        M = np.zeros((self.n_dofs, self.n_dofs))
        for e in range(self.n_el):
            s = slice(2 * e, 2 * e + 4)
            K[s, s] += elem_k[e] * self.Ke
            M[s, s] += elem_m[e] * self.Me
        b = self.beam
        K[0, 0] += bearings[0] * b.kv0
        K[1, 1] += bearings[0] * b.kr0
        K[-2, -2] += bearings[1] * b.kv0
        K[-1, -1] += bearings[1] * b.kr0
        return K, M

    def basis(self):
        """Affine basis K_p (zones, then bearings) and nominal mass."""
        K_basis = []
        for z in range(self.beam.n_zones):
            K_basis.append(self.assemble((self.zone_of_element == z).astype(float),
                                         np.ones(self.n_el), (0.0, 0.0))[0])
        for side in range(2):
            K_basis.append(self.assemble(np.zeros(self.n_el), np.ones(self.n_el),
                                         (1.0 - side, float(side)))[0])
        return np.stack(K_basis), self.assemble(np.ones(self.n_el), np.ones(self.n_el), (1.0, 1.0))[1]

    def modes(self, K, M, n_modes):
        lam, vec = sla.eigh(K, M, subset_by_index=[0, n_modes - 1])
        logf = 0.5 * np.log(lam) - np.log(2.0 * np.pi)
        v = vec[2 * self.sensor_nodes, :].T
        r = vec[2 * self.sensor_nodes + 1, :].T
        return logf, v / np.linalg.norm(v, axis=1, keepdims=True), r / np.linalg.norm(r, axis=1, keepdims=True)


def build_coarse_model(beam: C.BeamConfig, noise: C.NoiseConfig):
    """Inference model arrays, healthy reference shapes and per-mode noise levels."""
    fe = BeamFE(beam, C.COARSE_EL_PER_ZONE)
    K_basis, M = fe.basis()
    arrays = modal_arrays(K_basis, M, fe.sensor_nodes)
    logf_ref, v_ref, r_ref = ModalModelNP(arrays, beam.n_modes_max)(np.ones((1, C.N_DIMS)))
    arrays.update(
        logf_ref=logf_ref[0], v_ref=v_ref[0], r_ref=r_ref[0],
        sig_f=noise.freq_std,
        sig_v=noise.vert_rel_std * np.abs(v_ref[0]).max(-1),
        sig_r=noise.rot_rel_std * np.abs(r_ref[0]).max(-1),
        lower=C.param_bounds()[0], upper=C.param_bounds()[1],
        sensor_x=fe.sensor_nodes * beam.L_total / fe.n_el)
    return arrays


# =============================================================================
# Simulation
# =============================================================================
def element_factors(fe: BeamFE, theta_zones, jitter, rng):
    """
    Element stiffness factors of the zones. Within-zone variability is applied
    to the flexibility with zero mean, 1/f_e = (1 + j_e) / z, sum_e j_e = 0, so
    the zone keeps exactly the equivalent stiffness z (harmonic mean of f_e).
    """
    f = np.empty(fe.n_el)
    for z, th in enumerate(theta_zones):
        j = rng.normal(0.0, jitter, fe.nsub) if jitter > 0 else np.zeros(fe.nsub)
        j -= j.mean()
        f[fe.zone_of_element == z] = th / (1.0 + j)
    return f


def simulate(theta, variant: C.FEVariant, beam, coarse, rng):
    """Noise-free log-frequencies and reference-aligned unit mode shapes."""
    n_m = beam.n_modes_max
    pure = (variant.el_per_zone == C.COARSE_EL_PER_ZONE and variant.mass_std == 0
            and variant.stiffness_jitter == 0)
    if pure:
        logf, v, r = ModalModelNP(coarse, n_m)(theta)
    else:
        fe = BeamFE(beam, variant.el_per_zone)
        N = len(theta)
        logf, v = np.empty((N, n_m)), np.empty((N, n_m, len(fe.sensor_nodes)))
        r = np.empty_like(v)
        for i in range(N):
            ek = element_factors(fe, theta[i, :C.N_ZONES], variant.stiffness_jitter, rng)
            em = 1.0 + rng.normal(0.0, variant.mass_std, fe.n_el) if variant.mass_std > 0 else np.ones(fe.n_el)
            if em.min() <= 0.0:
                raise ValueError("mass_std too large: non-positive element mass")
            logf[i], v[i], r[i] = fe.modes(*fe.assemble(ek, em, theta[i, C.N_ZONES:]), n_m)
    v, r = align_to_reference(v, r, coarse["v_ref"], coarse["r_ref"], coarse["sig_v"], coarse["sig_r"])
    return logf, v, r


def add_noise(logf, v, r, coarse, factor, rng):
    """Gaussian measurement noise consistent with the likelihood (no re-normalization)."""
    y_logf = logf + factor * coarse["sig_f"] * rng.standard_normal(logf.shape)
    y_v = v + factor * coarse["sig_v"][:, None] * rng.standard_normal(v.shape)
    y_r = r + factor * coarse["sig_r"][:, None] * rng.standard_normal(r.shape)
    return y_logf, y_v, y_r


def baseline_shift(variant: C.FEVariant, beam, coarse):
    """Healthy-state log-frequency offset between the variant and the inference model."""
    if variant.el_per_zone == C.COARSE_EL_PER_ZONE:
        return np.zeros(beam.n_modes_max)
    fe = BeamFE(beam, variant.el_per_zone)
    logf, _, _ = fe.modes(*fe.assemble(np.ones(fe.n_el), np.ones(fe.n_el), (1.0, 1.0)), beam.n_modes_max)
    return logf - coarse["logf_ref"]


def generate_set(name, spec, beam, noise, coarse, prior, doe, scale=1.0):
    rng = np.random.default_rng(spec["seed"])
    n = max(int(round(spec["n"] * scale)), 20)
    variant = C.VARIANTS[spec["variant"]]
    if spec["sampler"] == "doe":
        theta, scen, dmg = doe.sample(n, rng, stratified=True)
    elif spec["sampler"] == "prior":
        theta, scen, dmg = prior.sample(n, rng)
    elif spec["sampler"] == "four_sites":
        theta, scen, dmg = sample_four_sites(n, rng)
    else:
        raise ValueError(spec["sampler"])
    t0 = time.time()
    logf, v, r = simulate(theta, variant, beam, coarse, rng)
    y_logf, y_v, y_r = add_noise(logf, v, r, coarse, variant.noise_factor, rng)
    out = dict(theta=theta, y_logf=y_logf, y_v=y_v, y_r=y_r, logf_clean=logf, v_clean=v, r_clean=r,
               scenario=scen.astype("U20"), damaged_mask=dmg,
               log_prior=prior.log_prob(theta), log_doe=doe.log_prob(theta),
               logf_baseline_shift=baseline_shift(variant, beam, coarse), variant=np.array(variant.name))
    np.savez_compressed(C.DATA_DIR / f"{name}.npz", **out)
    print(f"  {name:20s} n={n:6d}  variant={variant.name:12s} ({time.time() - t0:5.1f} s)")
    return out


def generate_benchmarks(beam, coarse):
    names = list(C.BENCHMARKS)
    theta = np.array([C.BENCHMARKS[k] for k in names], dtype=float)
    out = dict(names=np.array(names))
    for vname in ("coarse", "fine"):
        variant = C.VARIANTS[vname]
        logf, v, r = simulate(theta, variant, beam, coarse, np.random.default_rng(C.BENCHMARK_SEED))
        y_logf, y_v, y_r = add_noise(logf, v, r, coarse, 1.0, np.random.default_rng(C.BENCHMARK_SEED + 1))
        out.update({f"{vname}_theta": theta, f"{vname}_y_logf": y_logf, f"{vname}_y_v": y_v,
                    f"{vname}_y_r": y_r, f"{vname}_logf_clean": logf, f"{vname}_v_clean": v,
                    f"{vname}_r_clean": r,
                    f"{vname}_logf_baseline_shift": baseline_shift(variant, beam, coarse)})
    np.savez_compressed(C.DATA_DIR / "benchmarks.npz", **out)
    print(f"  benchmarks           n={len(names)} (coarse and fine model)")


# =============================================================================
# Figures
# =============================================================================
def _style():
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "mathtext.fontset": "cm",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def fig_doe(train, fig_dir):
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.2))
    for i, ax in enumerate(axs.ravel()[:C.N_DIMS]):
        ax.hist(train["theta"][:, i], bins=55, range=(C.LBOUND, C.UBOUND), color="#4c72b0")
        ax.set_xlabel(C.PARAM_LABELS[i])
        ax.set_yticks([])
    ax = axs.ravel()[-1]
    ax.barh(range(len(SCENARIOS)), [np.mean(train["scenario"] == s) for s in SCENARIOS], color="#4c72b0")
    ax.set_yticks(range(len(SCENARIOS)))
    ax.set_yticklabels([s.replace("_", " ") for s in SCENARIOS], fontsize=9)
    ax.set_xlabel("fraction of training samples")
    fig.suptitle("Training DoE: parameter histograms and scenario mix")
    fig.tight_layout()
    _save(fig, fig_dir / "Fig_DoE_marginals.png")


def location_severity_counts(d):
    bins = np.array(C.SEVERITY_BINS)
    counts = np.zeros((len(bins) - 1, C.N_DIMS), dtype=int)
    for k in range(C.N_DIMS):
        counts[:, k] = np.histogram(d["theta"][d["damaged_mask"][:, k], k], bins=bins)[0]
    return counts


def fig_doe_location_severity(train, fig_dir):
    counts = location_severity_counts(train)
    bins = C.SEVERITY_BINS
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(counts, cmap="Blues", origin="lower", aspect="auto")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j, i, counts[i, j], ha="center", va="center", fontsize=8,
                    color="w" if counts[i, j] > counts.max() / 2 else "k")
    ax.set_xticks(range(C.N_DIMS))
    ax.set_xticklabels(C.PARAM_NAMES)
    ax.set_yticks(range(len(bins) - 1))
    ax.set_yticklabels([f"[{bins[i]:.2f}, {bins[i + 1]:.2f})" for i in range(len(bins) - 1)])
    ax.set_xlabel("damaged parameter")
    ax.set_ylabel("value of the damaged parameter")
    fig.colorbar(im, ax=ax, label="training samples")
    ax.set_title("Training DoE: damaged samples per location and severity")
    fig.tight_layout()
    _save(fig, fig_dir / "Fig_DoE_location_severity.png")
    print("  damaged training samples per severity bin (rows) and location (columns):\n", counts)


def fig_model_form_error(beam, coarse, fig_dir):
    """Noise-free discrepancy fine - coarse on prior samples (raw and baseline-corrected)."""
    rng = np.random.default_rng(0)
    theta = build_prior().sample(300, rng)[0]
    fine = C.VARIANTS["fine"]
    logf_c, v_c, r_c = simulate(theta, C.VARIANTS["coarse"], beam, coarse, rng)
    logf_f, v_f, r_f = simulate(theta, fine, beam, coarse, rng)
    shift = baseline_shift(fine, beam, coarse)
    raw = 100 * (np.exp(logf_f - logf_c) - 1)
    cor = 100 * (np.exp(logf_f - shift - logf_c) - 1)
    mac_v = np.sum(v_f * v_c, -1) ** 2
    mac_r = np.sum(r_f * r_c, -1) ** 2
    M = beam.n_modes_max
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, data, title in ((axs[0], raw, "raw"), (axs[1], cor, "after healthy-baseline correction")):
        ax.boxplot([data[:, m] for m in range(M)], showfliers=False)
        ax.axhspan(-100 * coarse["sig_f"], 100 * coarse["sig_f"], color="0.88", label=r"$\pm\sigma_f$")
        ax.set_xlabel("mode")
        ax.set_ylabel(r"$f_{fine}/f_{coarse}-1$ [%]")
        ax.set_title(title)
        ax.legend(frameon=False)
    axs[2].boxplot([1 - mac_v[:, m] for m in range(M)], showfliers=False, positions=np.arange(M) + 0.85,
                   widths=0.25)
    axs[2].boxplot([1 - mac_r[:, m] for m in range(M)], showfliers=False, positions=np.arange(M) + 1.15,
                   widths=0.25)
    axs[2].set_yscale("log")
    axs[2].set_xticks(np.arange(1, M + 1))
    axs[2].set_xticklabels(range(1, M + 1))
    axs[2].set_xlabel("mode (left: vertical, right: rotational)")
    axs[2].set_ylabel("1 - MAC(fine, coarse)")
    fig.suptitle(f"Test model ({fine.el_per_zone} el./zone, mass std {fine.mass_std}, stiffness jitter "
                 f"{fine.stiffness_jitter}) vs inference model ({C.COARSE_EL_PER_ZONE} el./zone)")
    fig.tight_layout()
    _save(fig, fig_dir / "Fig_model_form_error.png")
    print("  median |f_fine/f_coarse - 1| per mode [%]: raw", np.round(np.median(np.abs(raw), 0), 3),
          " corrected", np.round(np.median(np.abs(cor), 0), 3))


def fig_identifiability(beam, coarse, fig_dir):
    """
    Cramer-Rao lower bound of each parameter at the healthy state as a function
    of the number of modes used (Fisher information of the Gaussian likelihood,
    finite-difference Jacobian). Shows how identifiability depends on the modes
    and the measured quantities (R3.4).
    """
    model = ModalModelNP(coarse, beam.n_modes_max)
    th0 = np.full(C.N_DIMS, 0.98)
    h = 1e-6

    def obs_vector(th, n_m, obs):
        logf, v, r = model(th[None])
        v, r = align_to_reference(v, r, coarse["v_ref"], coarse["r_ref"], coarse["sig_v"], coarse["sig_r"])
        parts = [logf[0, :n_m] / coarse["sig_f"]]
        if "v" in obs:
            parts.append((v[0, :n_m] / coarse["sig_v"][:n_m, None]).ravel())
        if "r" in obs:
            parts.append((r[0, :n_m] / coarse["sig_r"][:n_m, None]).ravel())
        return np.concatenate(parts)

    shown = [0, 1, 2, 5]          # Z5 = Z1, Z4 = Z2 and BR = BL at this symmetric state
    modes = np.arange(1, beam.n_modes_max + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for k, obs in enumerate(("fv", "fvr")):
        crb = np.full((beam.n_modes_max, C.N_DIMS), np.nan)
        for n_m in modes:
            J = np.stack([(obs_vector(th0 + h * np.eye(C.N_DIMS)[p], n_m, obs)
                           - obs_vector(th0 - h * np.eye(C.N_DIMS)[p], n_m, obs)) / (2 * h)
                          for p in range(C.N_DIMS)], axis=1)
            F = J.T @ J
            if np.linalg.matrix_rank(F) == C.N_DIMS:
                crb[n_m - 1] = np.sqrt(np.diag(np.linalg.inv(F)))
        for p in shown:
            ax[k].semilogy(modes, crb[:, p], "o-", ms=4, label=C.PARAM_NAMES[p])
        ax[k].set_title({"fv": "frequencies + vertical modes",
                         "fvr": "frequencies + vertical + rotational modes"}[obs])
        ax[k].set_xlabel("number of modes used")
        ax[k].set_xticks(modes)
    ax[0].set_ylabel(r"Cramer-Rao bound of std($\theta$)")
    ax[1].legend(title="(Z5 = Z1, Z4 = Z2, BR = BL)", frameon=False, fontsize=8, title_fontsize=8)
    fig.suptitle(r"Local identifiability at $\theta = 0.98$; frequencies alone cannot identify the 7 "
                 "parameters (singular Fisher matrix)", fontsize=10)
    fig.tight_layout()
    _save(fig, fig_dir / "Fig_identifiability_CRB.png")


def fig_beam(beam, coarse, fig_dir):
    L = beam.L_total
    fig, ax = plt.subplots(figsize=(9, 2.5))
    cols = plt.cm.Blues(np.linspace(0.35, 0.75, beam.n_zones))
    for z in range(beam.n_zones):
        x0 = z * L / beam.n_zones
        ax.add_patch(plt.Rectangle((x0, 0), L / beam.n_zones, 0.4, color=cols[z], ec="k", lw=0.8))
        ax.text(x0 + L / (2 * beam.n_zones), 0.2, f"Z{z + 1}", ha="center", va="center")
    for xb, lab in ((0.0, "BL"), (L, "BR")):
        yy = np.linspace(-0.05, -0.65, 9)
        ax.plot(xb + 0.15 * np.array([0, 1, -1, 1, -1, 1, -1, 1, 0]), yy, "k", lw=1.2)
        ax.plot([xb - 0.4, xb + 0.4], [-0.7, -0.7], "k", lw=2)
        ax.text(xb + (0.3 if xb == 0 else -0.3), -0.4, f"{lab}: $k_v$, $k_r$",
                ha="left" if xb == 0 else "right", va="center", fontsize=9)
    ax.plot(coarse["sensor_x"], np.full(len(coarse["sensor_x"]), 0.5), "v", color="#d62728", ms=7,
            label="sensors (vertical + rotation)")
    ax.set_xlim(-1, L + 1)
    ax.set_ylim(-0.9, 0.9)
    ax.set_yticks([])
    ax.set_xlabel("x [m]")
    ax.legend(loc="upper center", frameon=False)
    fig.tight_layout()
    _save(fig, fig_dir / "Fig_beam_model.png")


# =============================================================================
def main(scale=1.0, plots=True):
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    beam, noise = C.BeamConfig(), C.NoiseConfig()
    coarse = build_coarse_model(beam, noise)
    prior, doe = build_prior(), build_doe()
    np.savez(C.DATA_DIR / "coarse_model.npz", **coarse)
    C.save_json(dict(beam=beam, noise=noise, coarse_el_per_zone=C.COARSE_EL_PER_ZONE,
                     variants=C.VARIANTS, datasets=C.DATASETS, healthy=C.HEALTHY,
                     damaged_prior=C.DAMAGED_PRIOR, damaged_doe=C.DAMAGED_DOE,
                     prior_weights=C.PRIOR_SCENARIO_WEIGHTS, doe_weights=C.DOE_SCENARIO_WEIGHTS,
                     benchmarks=C.BENCHMARKS, scale=scale, param_names=C.PARAM_NAMES,
                     healthy_freqs_hz=np.exp(coarse["logf_ref"]),
                     sig_v=coarse["sig_v"], sig_r=coarse["sig_r"]),
                C.DATA_DIR / "metadata.json")
    print("Healthy inference-model frequencies [Hz]:", np.round(np.exp(coarse["logf_ref"]), 3))
    print("Mode-shape noise std per mode: vertical", np.round(coarse["sig_v"], 4),
          " rotational", np.round(coarse["sig_r"], 4))
    print(f"Generating datasets in {C.DATA_DIR} (scale={scale})")
    sets = {name: generate_set(name, spec, beam, noise, coarse, prior, doe, scale)
            for name, spec in C.DATASETS.items()}
    generate_benchmarks(beam, coarse)
    if plots:
        _style()
        fig_dir = C.DATA_DIR / "figures"
        fig_beam(beam, coarse, fig_dir)
        fig_doe(sets["train"], fig_dir)
        fig_doe_location_severity(sets["train"], fig_dir)
        fig_model_form_error(beam, coarse, fig_dir)
        fig_identifiability(beam, coarse, fig_dir)
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies all dataset sizes")
    ap.add_argument("--no-plots", action="store_true")
    a, _ = ap.parse_known_args()
    main(a.scale, not a.no_plots)
