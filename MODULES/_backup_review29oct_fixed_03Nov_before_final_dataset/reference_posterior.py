#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed/reference_posterior.py

Reference ("ground-truth") posterior of one observation m:

    p_gamma(z | m)  proportional to  exp(-NLL(z; m) / gamma^2) * U(z; [lbound, 1]^7)

sampled with adaptive-tempering Sequential Monte Carlo (SMC). It
  * starts from the prior and never uses z*,
  * uses the same nominal-mass 5-element model and the same likelihood (fe_model.neg_log_likelihood with the
    std's of problem_config.json) as the VAE loss, and the gamma of the VAE run (gamma = 1: exact posterior),
  * returns equally weighted samples (no filtering of light particles).

It replaces the "ground truth" of review29oct/postprocessing_7D_bridge.py, which (i) placed 4096 of its
points around z* (and its mirror), (ii) corrected that proposal with importance weights raised to the
power 0.55, which does not remove the bias towards z*, (iii) used a hand-tuned likelihood
(beta_eff = 0.55 hard-coded, MSE of standardised log f + 8 x MAC) that is neither the VAE loss nor the
noise of the data, and (iv) dropped the lightest 5 % of the weight.

Run from the project root (results are cached in <data_dir>/reference_posteriors/):
    python MODULES/review29oct_fixed/reference_posterior.py --gamma 1.0 --workers 6
"""

import os
import sys
import json
import time
import argparse

import numpy as np

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.fe_model import BridgeModel


# =============================================================================
# 1. ADAPTIVE-TEMPERING SMC
# =============================================================================
def systematic_resample(w, rng):
    n = len(w)
    cs = np.cumsum(w)
    cs[-1] = 1.0
    return np.searchsorted(cs, (rng.uniform() + np.arange(n)) / n)


def smc_sample(log_lik, n_dims, lbound, n_particles=4000, ess_frac=0.5, rng=None,
               min_moves=4, max_moves=40, final_moves=30, target_acc=0.25, verbose=False):
    """
    Targets pi_phi(z) ~ U(z) L(z)^phi for phi: 0 -> 1. Next phi keeps ESS = ess_frac * N; after resampling,
    particles move with random-walk Metropolis whose covariance is the particle covariance (scaled for an
    acceptance near target_acc), until each particle has moved with probability > 0.99.
    Returns samples (N, d) and a dict of diagnostics (log-evidence estimate, temperatures, acceptances).
    """
    rng = np.random.default_rng() if rng is None else rng
    N, d = n_particles, n_dims
    x = rng.uniform(lbound, 1.0, size=(N, d))
    ll = log_lik(x)
    phi, log_z, scale = 0.0, 0.0, 2.38 / np.sqrt(d)
    stages = []

    def move(x, ll, phi, scale, n_min, n_max):
        chol = np.linalg.cholesky(np.cov(x, rowvar=False) + 1e-12 * np.eye(d))
        accs = []
        while True:
            prop = x + scale * rng.normal(size=(N, d)) @ chol.T
            inside = np.all((prop >= lbound) & (prop <= 1.0), axis=1)
            ll_prop = np.full(N, -np.inf)
            if inside.any():
                ll_prop[inside] = log_lik(prop[inside])
            accept = np.log(rng.uniform(size=N)) < phi * (ll_prop - ll)
            x[accept], ll[accept] = prop[accept], ll_prop[accept]
            accs.append(accept.mean())
            scale *= np.exp(accs[-1] - target_acc)
            k = len(accs)
            if k >= n_max or (k >= n_min and (1.0 - np.mean(accs)) ** k < 0.01):
                return x, ll, scale, float(np.mean(accs)), k

    while phi < 1.0:
        llc = ll - ll.max()

        def ess(dp):
            w = np.exp(dp * llc)
            return w.sum() ** 2 / np.sum(w ** 2)

        if ess(1.0 - phi) >= ess_frac * N:
            dphi = 1.0 - phi
        else:
            lo, hi = 0.0, 1.0 - phi
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if ess(mid) >= ess_frac * N else (lo, mid)
            dphi = max(lo, 1e-15)
        logw = dphi * ll
        m = logw.max()
        log_z += m + np.log(np.mean(np.exp(logw - m)))
        w = np.exp(logw - m)
        idx = systematic_resample(w / w.sum(), rng)
        x, ll = x[idx], ll[idx]
        phi = min(phi + dphi, 1.0)
        x, ll, scale, acc, k = move(x, ll, phi, scale, min_moves, max_moves)
        stages.append((phi, acc, k))
        if verbose:
            print(f"    stage {len(stages):3d}: phi = {phi:.3e}  acc = {acc:.2f}  moves = {k}")

    x, ll, scale, acc, k = move(x, ll, 1.0, scale, final_moves, final_moves)
    info = {"log_evidence": float(log_z), "n_stages": len(stages), "final_acceptance": acc,
            "phis": [s[0] for s in stages], "acceptances": [s[1] for s in stages], "moves": [s[2] for s in stages]}
    return x, info


# =============================================================================
# 2. CACHED REFERENCE POSTERIOR OF A TEST CASE
# =============================================================================
def load_problem_config(data_dir):
    with open(os.path.join(data_dir, "problem_config.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def cache_path(data_dir, pos, gamma2, obs_types, n_particles, split="test"):
    prefix = "GT" if split == "test" else f"GT_{split}"
    return os.path.join(data_dir, "reference_posteriors",
                        f"{prefix}_P{pos:02d}_obs-{obs_types}_g2-{gamma2:.4g}_N{n_particles}.npz")


def compute_reference_posterior(data_dir, pos, gamma2=1.0, obs_types=None, n_particles=4000, seed=None,
                                recompute=False, verbose=False, split="test"):
    """Returns (samples (N, 7), info) for sample `pos` of the test set `split`; cached on disk."""
    pcfg = load_problem_config(data_dir)
    obs_types = obs_types or pcfg["obs_types"]
    path = cache_path(data_dir, pos, gamma2, obs_types, n_particles, split)
    f_obs = np.load(os.path.join(data_dir, f"freqs_{split}.npy"))[pos].astype(np.float64)
    v_obs = np.load(os.path.join(data_dir, f"vertmodes_{split}.npy"))[pos].astype(np.float64)
    r_obs = np.load(os.path.join(data_dir, f"rotmodes_{split}.npy"))[pos].astype(np.float64)
    if os.path.exists(path) and not recompute:
        d = np.load(path, allow_pickle=True)
        if "obs_f" in d and np.allclose(d["obs_f"], f_obs) and np.allclose(d["sig_logf"], pcfg["likelihood"]["sig_logf"]):
            return d["samples"], d["info"].item()

    model = BridgeModel(pcfg)
    log_lik = model.log_likelihood_fn(f_obs, v_obs, r_obs, gamma2=gamma2, obs_types=obs_types)

    t0 = time.time()
    rng = np.random.default_rng(10_000 + pos if seed is None else seed)
    samples, info = smc_sample(log_lik, int(pcfg["n_dims"]), float(pcfg["lbound"]), n_particles=n_particles,
                               rng=rng, verbose=verbose)
    info.update({"pos": pos, "split": split, "gamma2": gamma2, "obs_types": obs_types, "seconds": time.time() - t0,
                 "z_true": np.load(os.path.join(data_dir, f"alphas_{split}.npy"))[pos].tolist()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, samples=samples.astype(np.float32), info=np.array(info, dtype=object), obs_f=f_obs,
                        sig_logf=np.asarray(pcfg["likelihood"]["sig_logf"]))
    return samples, info


def _worker(args):
    data_dir, pos, gamma2, obs_types, n_particles, recompute, split = args
    s, info = compute_reference_posterior(data_dir, pos, gamma2, obs_types, n_particles, recompute=recompute, split=split)
    return pos, info


def compute_many(data_dir, positions, gamma2=1.0, obs_types=None, n_particles=4000, workers=1, recompute=False,
                 split="test"):
    jobs = [(data_dir, p, gamma2, obs_types, n_particles, recompute, split) for p in positions]
    if workers > 1 and len(jobs) > 1:
        from multiprocessing import Pool
        with Pool(min(workers, len(jobs))) as pool:
            results = pool.map(_worker, jobs)
    else:
        results = [_worker(j) for j in jobs]
    for pos, info in sorted(results):
        print(f"  P{pos}: {info['n_stages']} tempering stages, final acceptance {info['final_acceptance']:.2f}, "
              f"{info['seconds']:.0f} s")
    return results


def main():
    parser = argparse.ArgumentParser(description="Reference posteriors (SMC) of the benchmark test cases.")
    parser.add_argument("--data-dir", type=str, default=C.DATA_DIR)
    parser.add_argument("--gamma", type=float, default=1.0, help="gamma of the VAE run (likelihood ^ 1/gamma^2)")
    parser.add_argument("--obs", type=str, default=None, help="observed quantities (default: problem_config.json)")
    parser.add_argument("--positions", type=int, nargs="+", default=None)
    parser.add_argument("--n-particles", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--splits", type=str, nargs="+", default=None,
                        help="test sets (default: all, e.g. test = same model, test_fine = 50-element model)")
    args, _ = parser.parse_known_args()

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(PROJECT_ROOT, args.data_dir)
    pcfg = load_problem_config(data_dir)
    positions = args.positions if args.positions is not None else [b["pos"] for b in pcfg["benchmarks"]]
    splits = args.splits or list(pcfg.get("test_sets", {"test": None}))
    gamma2 = args.gamma ** 2
    descs = {b["pos"]: b["desc"] for b in pcfg["benchmarks"]}
    for split in splits:
        print(f"Reference posteriors of '{split}' (SMC, N={args.n_particles}, gamma^2={gamma2:.4g}), positions {positions}")
        compute_many(data_dir, positions, gamma2, args.obs, args.n_particles, args.workers, args.recompute, split)
        if args.no_plots:
            continue
        from MODULES.review29oct_fixed.plotting import corner_single
        fig_dir = os.path.join(data_dir, "reference_posteriors", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        z_test = np.load(os.path.join(data_dir, f"alphas_{split}.npy"))
        n_el = pcfg.get("test_sets", {}).get(split, 1) * int(pcfg["n_zones"])
        for pos in positions:
            s, _ = compute_reference_posterior(data_dir, pos, gamma2, args.obs, args.n_particles, split=split)
            corner_single(s, z_test[pos], float(pcfg["lbound"]),
                          title=f"Sample {pos} ({descs.get(pos, 'test case')}), {n_el}-element data: "
                                f"reference posterior $p(\\mathbf{{z}}|\\mathbf{{m}})$",
                          cbar_label="relative posterior density (reference)",
                          save_path=os.path.join(fig_dir, f"P{pos}_{split}_GT_obs-{args.obs or pcfg['obs_types']}_g2-{gamma2:.4g}.png"))
        print(f"Figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
