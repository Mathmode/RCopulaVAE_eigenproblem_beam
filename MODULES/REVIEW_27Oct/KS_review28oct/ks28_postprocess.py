# -*- coding: utf-8 -*-
"""
ks28_postprocess.py -- Metrics, ground-truth comparison and figures (NumPy only).

Works on a run folder produced by ks28_train (posteriors/*.npz hold the parameters of
q for every observation), so it runs on a laptop without TensorFlow.

Metrics per evaluation family (only cases with an exact label for accuracy/calibration)
  Accuracy      MAE / RMSE of the posterior mean (zones and bearings separately)
  Calibration   empirical coverage of the central p-credible intervals (quantiles of q,
                not mean +/- k sigma: q can be skewed/multimodal), MACE, 95 % coverage,
                mean 95 % width
  Sharpness/fit CRPS (sample estimator), ALL = mean log q(theta_true | m)
  Detection     P_k = q(z_k < 1 - healthy_band | m) for every zone
                FPR_struct : fraction of HEALTHY structures with any P_k > 0.5 (false alarm)
                FPR_zone   : fraction of healthy zones with P_k > 0.5
                TPR_zone   : fraction of damaged zones with P_k > 0.5
                Loc_acc    : argmax_k P_k is a damaged zone (single-zone families)
                AUC        : zone-level ROC AUC of P_k (damaged vs healthy zones)
  Bearings      P(|eta_j| > 0.3), TPR / FPR on the bearing family

Ground truth comparison (selected cases; SMC particles from ks28_ground_truth)
  W1 per coordinate, energy distance, |P_k(VAE) - P_k(GT)|, overlay corner plots
  (GT filled contours + VAE line contours + true value), split violins.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, wasserstein_distance

from .ks28_config import NoiseConfig, PriorConfig, BridgeConfig, load_configs
from .ks28_q_numpy import sample_q, log_q, subset_params

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
COL_VAE, COL_GT, COL_TRUE = "#2e86ab", "0.35", "#d1495b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


# =============================================================================
# loading
# =============================================================================
def load_run(run):
    with open(os.path.join(run, "run_config.json")) as fh:
        rc = json.load(fh)
    cfg, _, prior, extra = load_configs(os.path.join(rc["dataset_folder"], "config.json"))
    noise = NoiseConfig(**rc["noise_used"])
    return rc, cfg, noise, prior


def load_posterior(run, split):
    with np.load(os.path.join(run, "posteriors", f"{split}.npz")) as d:
        return {k: d[k] for k in d.files}


def list_splits(run):
    return sorted(f[:-4] for f in os.listdir(os.path.join(run, "posteriors")) if f.endswith(".npz"))


def q_params_of(post):
    return {k: post[k] for k in ("log_pi", "a", "b", "log_w", "L", "lo", "hi")}


def draw_samples(post, S=500, seed=0, chunk=200):
    rng = np.random.default_rng(seed)
    p = q_params_of(post)
    N = p["log_pi"].shape[0]
    out = []
    for i in range(0, N, chunk):
        s, _ = sample_q(subset_params(p, slice(i, min(i + chunk, N))), S, rng)
        out.append(s)
    return np.concatenate(out)


# =============================================================================
# metrics
# =============================================================================
def crps_samples(samples, y):
    """CRPS for samples (N, S, D) and truth (N, D) -> (N, D)."""
    xs = np.sort(samples, axis=1)
    S = xs.shape[1]
    t1 = np.mean(np.abs(xs - y[:, None, :]), axis=1)
    i = np.arange(1, S + 1)[None, :, None]
    t2 = np.sum((2 * i - S - 1) * xs, axis=1) / S ** 2          # = 0.5 E|X - X'|
    return t1 - t2


def _auc(scores_pos, scores_neg):
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return np.nan
    s = np.concatenate([scores_pos, scores_neg])
    r = np.argsort(np.argsort(s)) + 1.0
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv]
    n1, n0 = len(scores_pos), len(scores_neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def family_metrics(post, cfg: BridgeConfig, prior: PriorConfig, S=500, seed=0):
    """All metrics of one split. Returns (summary dict, per-case dict)."""
    th = post["theta"]
    N = th.shape[0]
    nz = cfg.n_zones
    smp = draw_samples(post, S, seed)                                   # (N, S, D)
    exact = post.get("label_exact", np.ones(N, bool)).astype(bool)
    lo, hi = cfg.latent_lower, cfg.latent_upper
    inside = np.all((th >= lo) & (th <= hi), axis=1)                     # OOD truths are outside
    ev = exact & inside

    mean = smp.mean(1)
    res = {}
    groups = {"zones": slice(0, nz)}
    if cfg.support_latent:
        groups["bearings"] = slice(nz, nz + 2)
    for g, sl in groups.items():
        err = mean[ev, sl] - th[ev, sl]
        res[f"MAE_{g}"] = float(np.mean(np.abs(err))) if ev.any() else np.nan
        res[f"RMSE_{g}"] = float(np.sqrt(np.mean(err ** 2))) if ev.any() else np.nan
        cov = []
        for p in LEVELS:
            ql = np.quantile(smp[ev][:, :, sl], (1 - p) / 2, axis=1)
            qh = np.quantile(smp[ev][:, :, sl], (1 + p) / 2, axis=1)
            cov.append(np.mean((th[ev, sl] >= ql) & (th[ev, sl] <= qh)) if ev.any() else np.nan)
        cov = np.array(cov)
        res[f"coverage_{g}"] = cov.tolist()
        res[f"COV95_{g}"] = float(cov[-1])
        res[f"MACE_{g}"] = float(np.mean(np.abs(cov - LEVELS)))
        w95 = np.quantile(smp[ev][:, :, sl], 0.975, 1) - np.quantile(smp[ev][:, :, sl], 0.025, 1)
        res[f"W95_{g}"] = float(np.mean(w95)) if ev.any() else np.nan
        res[f"CRPS_{g}"] = float(np.mean(crps_samples(smp[ev][:, :, sl], th[ev, sl]))) if ev.any() else np.nan
    if ev.any():
        res["ALL"] = float(np.mean(log_q(np.clip(th[ev], lo, hi)[:, None, :],
                                         subset_params(q_params_of(post), ev))))
    else:
        res["ALL"] = np.nan

    # ---- detection
    thr = 1.0 - prior.healthy_band
    Pk = np.mean(smp[:, :, :nz] < thr, axis=1)                             # (N, nz)
    dmask = post.get("damaged_mask", th[:, :nz] < thr).astype(bool)
    flag = Pk > 0.5
    healthy_struct = ~dmask.any(1)
    res["FPR_struct"] = float(np.mean(flag[healthy_struct].any(1))) if healthy_struct.any() else np.nan
    res["FPR_zone"] = float(np.mean(flag[~dmask])) if (~dmask).any() else np.nan
    res["TPR_zone"] = float(np.mean(flag[dmask])) if dmask.any() else np.nan
    single = dmask.sum(1) == 1
    res["Loc_acc"] = float(np.mean(dmask[single][np.arange(single.sum()), Pk[single].argmax(1)])) \
        if single.any() else np.nan
    res["AUC_zone"] = float(_auc(Pk[dmask], Pk[~dmask]))
    per_case = dict(mean=mean, Pk=Pk, samples=smp, damaged_mask=dmask)

    if cfg.support_latent:
        Pb = np.mean(np.abs(smp[:, :, nz:nz + 2]) > 0.3, axis=1)
        amask = post.get("support_anomaly", np.zeros((N, 2), bool)).astype(bool)
        res["bearing_TPR"] = float(np.mean(Pb[amask] > 0.5)) if amask.any() else np.nan
        res["bearing_FPR"] = float(np.mean(Pb[~amask] > 0.5)) if (~amask).any() else np.nan
        per_case["Pb"] = Pb
    res["n_cases"], res["n_exact"] = int(N), int(ev.sum())
    return res, per_case


def evaluate_run(run, splits=None, S=500, seed=0, out_name="metrics"):
    """Metrics for every split of a run -> CSV + JSON + calibration figure."""
    rc, cfg, noise, prior = load_run(run)
    splits = list_splits(run) if splits is None else splits
    allres = {}
    for s in splits:
        post = load_posterior(run, s)
        allres[s], _ = family_metrics(post, cfg, prior, S, seed)
        r = allres[s]
        print(f"{s:18s} MAEz {r['MAE_zones']:.4f}  COV95z {r['COV95_zones']:.3f}  MACEz {r['MACE_zones']:.3f}  "
              f"FPRs {r['FPR_struct']:.3f}  TPRz {r['TPR_zone']:.3f}  AUC {r['AUC_zone']:.3f}")
    od = os.path.join(run, "postprocessing")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, f"{out_name}.json"), "w") as fh:
        json.dump(allres, fh, indent=2)
    keys = [k for k in next(iter(allres.values())).keys() if not k.startswith("coverage_")]
    with open(os.path.join(od, f"{out_name}.csv"), "w") as fh:
        fh.write("split," + ",".join(keys) + "\n")
        for s, r in allres.items():
            fh.write(s + "," + ",".join(f"{r[k]:.5g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    plot_calibration(allres, os.path.join(od, f"{out_name}_calibration.png"))
    return allres


# =============================================================================
# ground truth comparison
# =============================================================================
def energy_distance(X, Y, n=1500, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    X = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    Y = Y[rng.choice(len(Y), min(n, len(Y)), replace=False)]
    d = lambda A, B: np.mean(np.linalg.norm(A[:, None] - B[None], axis=-1))
    return float(2 * d(X, Y) - d(X, X) - d(Y, Y))


def compare_with_ground_truth(run, split, case_ids, n_particles=4000, n_mcmc=10, S=4000,
                              seed=0, make_plots=True, recompute=False):
    """Run (or load) the SMC ground truth for some cases and compare with q."""
    from .ks28_fe_model import build_basis
    from .ks28_ground_truth import GroundTruthSMC
    rc, cfg, noise, prior = load_run(run)
    basis = build_basis(cfg)
    post = load_posterior(run, split)
    mcfg = rc["model"]
    noise_gt = NoiseConfig(**{**rc["noise_used"], "use_mode_shapes": mcfg["use_mode_shapes"]})
    gdir = os.path.join(run, "ground_truth")
    pdir = os.path.join(run, "postprocessing", "gt_comparison")
    os.makedirs(gdir, exist_ok=True)
    os.makedirs(pdir, exist_ok=True)
    rows = []
    rng = np.random.default_rng(seed)
    names = cfg.latent_names
    nz = cfg.n_zones
    thr = 1.0 - prior.healthy_band
    for c in case_ids:
        c = int(c)
        fpath = os.path.join(gdir, f"{split}_case{c}.npz")
        if os.path.exists(fpath) and not recompute:
            with np.load(fpath) as g:
                P = g["particles"]
                logZ, tsec = float(g["log_evidence"]), float(g["time_s"])
        else:
            smc = GroundTruthSMC(cfg, noise_gt, prior, basis, mcfg["prior"], mcfg["n_modes_used"],
                                 n_particles, n_mcmc, seed=seed + c)
            r = smc.run(post["f_obs"][c], post["phi_obs"][c])
            P, logZ, tsec = r["particles"], r["log_evidence"], r["time_s"]
            np.savez_compressed(fpath, particles=P, log_evidence=logZ, time_s=tsec,
                                theta_true=post["theta"][c])
        q_s, _ = sample_q(subset_params(q_params_of(post), [c]), S, rng)
        Q = q_s[0]
        row = dict(split=split, case=c, gt_time_s=tsec, log_evidence=logZ,
                   energy_distance=energy_distance(Q, P, rng=rng))
        for d, nmn in enumerate(names):
            row[f"W1_{nmn}"] = float(wasserstein_distance(Q[:, d], P[:, d]))
        Pk_q = np.mean(Q[:, :nz] < thr, 0)
        Pk_g = np.mean(P[:, :nz] < thr, 0)
        row["max_abs_dP_damage"] = float(np.max(np.abs(Pk_q - Pk_g)))
        rows.append(row)
        if make_plots:
            title = f"{split}, case {c}"
            plot_corner_overlay(P, Q, post["theta"][c], names, cfg,
                                os.path.join(pdir, f"{split}_case{c}_corner.png"), title)
            plot_violins(P, Q, post["theta"][c], names, cfg, prior,
                         os.path.join(pdir, f"{split}_case{c}_violins.png"), title, Pk_g, Pk_q)
        print(f"[GT] {split} case {c}: ED={row['energy_distance']:.4f}  max|dP|={row['max_abs_dP_damage']:.3f}")
    keys = list(rows[0].keys())
    path = os.path.join(pdir, f"{split}_gt_comparison.csv")
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    return rows


# =============================================================================
# figures
# =============================================================================
def _range(cfg, d):
    return (cfg.latent_lower[d], cfg.latent_upper[d])


def _zoom(P, Q, truth, d, cfg):
    lo, hi = _range(cfg, d)
    a = min(np.quantile(P[:, d], 0.002), np.quantile(Q[:, d], 0.002), truth[d])
    b = max(np.quantile(P[:, d], 0.998), np.quantile(Q[:, d], 0.998), truth[d])
    pad = 0.08 * (b - a + 1e-3)
    return max(lo, a - pad), min(hi, b + pad)


def plot_corner_overlay(P, Q, truth, names, cfg, path, title=""):
    """Lower triangle: GT (grey filled contours) + VAE (blue line contours) + truth (red star)."""
    D = len(names)
    fig, axs = plt.subplots(D, D, figsize=(1.9 * D, 1.9 * D))
    lims = [_zoom(P, Q, truth, d, cfg) for d in range(D)]
    for i in range(D):
        for j in range(D):
            ax = axs[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                bins = np.linspace(*lims[i], 41)
                ax.hist(P[:, i], bins=bins, density=True, color=COL_GT, alpha=0.45, label="ground truth")
                ax.hist(Q[:, i], bins=bins, density=True, histtype="step", color=COL_VAE, lw=1.6, label="VAE")
                ax.axvline(truth[i], color=COL_TRUE, lw=1.2)
                ax.set_yticks([])
            else:
                rx, ry = lims[j], lims[i]
                for data, kw in ((P, dict(cmap="Greys", filled=True)), (Q, dict(colors=COL_VAE, filled=False))):
                    H, xe, ye = np.histogram2d(data[:, j], data[:, i], bins=30, range=[rx, ry], density=True)
                    H = _smooth(H)
                    lv = _hpd_levels(H, [0.95, 0.68, 0.38])
                    X, Y = np.meshgrid(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), indexing="ij")
                    if kw.get("filled"):
                        ax.contourf(X, Y, H, levels=lv + [H.max() * 1.01], cmap="Greys", alpha=0.75)
                    else:
                        ax.contour(X, Y, H, levels=lv, colors=COL_VAE, linewidths=1.0)
                ax.plot(truth[j], truth[i], "*", color=COL_TRUE, ms=9, mec="white", mew=0.5)
                ax.set_xlim(rx); ax.set_ylim(ry)
            if i == D - 1:
                ax.set_xlabel(names[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(names[i])
            elif i != j:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
    h = [plt.Rectangle((0, 0), 1, 1, color=COL_GT, alpha=0.5), plt.Line2D([], [], color=COL_VAE),
         plt.Line2D([], [], marker="*", color=COL_TRUE, ls="")]
    fig.legend(h, ["ground truth (SMC)", "VAE posterior q", "true value"], loc="upper right",
               bbox_to_anchor=(0.97, 0.95), frameon=False, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=13)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _smooth(H):
    k = np.array([0.25, 0.5, 0.25])
    H = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 0, H)
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, H)


def _hpd_levels(H, masses):
    h = np.sort(H.ravel())[::-1]
    c = np.cumsum(h) / max(h.sum(), 1e-300)
    lv = sorted(set(float(h[min(np.searchsorted(c, m), len(h) - 1)]) for m in masses))
    lv = [v for v in lv if v > 0]
    return lv if lv else [H.max() * 0.5]


def plot_violins(P, Q, truth, names, cfg, prior, path, title, Pk_g, Pk_q):
    """Split violins per latent coordinate (left GT, right VAE) + damage probabilities."""
    D = len(names)
    nz = cfg.n_zones
    fig, axs = plt.subplots(1, 2 if cfg.support_latent else 1, figsize=(13, 4),
                            gridspec_kw=dict(width_ratios=[nz, 2]) if cfg.support_latent else None)
    axs = np.atleast_1d(axs)
    for ax, sl in zip(axs, [range(nz), range(nz, D)]):
        for k, d in enumerate(sl):
            for data, side, col in ((P[:, d], -1, COL_GT), (Q[:, d], 1, COL_VAE)):
                lo, hi = _range(cfg, d)
                g = np.linspace(lo, hi, 300)
                try:
                    dens = gaussian_kde(data, bw_method=0.15)(g)
                except Exception:
                    continue
                dens = 0.42 * dens / dens.max()
                ax.fill_betweenx(g, k, k + side * dens, color=col, alpha=0.6, lw=0)
            ax.plot([k - 0.45, k + 0.45], [truth[d]] * 2, color=COL_TRUE, lw=2)
            if d < nz:
                ax.text(k, cfg.z_lb - 0.03, f"GT {Pk_g[d]:.2f}\nVAE {Pk_q[d]:.2f}", ha="center",
                        va="top", fontsize=8)
        ax.set_xticks(range(len(sl)))
        ax.set_xticklabels([names[d] for d in sl])
    axs[0].set_ylabel("stiffness factor z")
    axs[0].set_ylim(cfg.z_lb - 0.11, 1.01)
    axs[0].set_title(r"zones (numbers: $P(z_k<%.2f\,|\,m)$)" % (1 - prior.healthy_band), fontsize=10)
    if cfg.support_latent:
        eb = np.concatenate([P[:, nz:], Q[:, nz:], truth[None, nz:]])
        lo_b, hi_b = np.quantile(eb, 0.001), np.quantile(eb, 0.999)
        pad = 0.1 * (hi_b - lo_b + 1e-3)
        axs[1].set_ylim(max(cfg.eta_lb, lo_b - pad), min(cfg.eta_ub, hi_b + pad))
        axs[1].set_ylabel(r"bearing factor $\eta$")
        axs[1].set_title("bearings", fontsize=10)
    h = [plt.Rectangle((0, 0), 1, 1, color=COL_GT, alpha=0.6), plt.Rectangle((0, 0), 1, 1, color=COL_VAE, alpha=0.6),
         plt.Line2D([], [], color=COL_TRUE, lw=2)]
    fig.legend(h, ["ground truth (left)", "VAE (right)", "true value"], loc="lower center", ncol=3,
               frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_calibration(allres, path):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    groups = ["zones", "bearings"]
    for ax, g in zip(axs, groups):
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        for i, (s, r) in enumerate(allres.items()):
            key = f"coverage_{g}"
            if key not in r or r.get("n_exact", 0) == 0:
                continue
            ax.plot(LEVELS, r[key], "-o", ms=3, lw=1.2, color=plt.cm.tab20(i % 20), label=s)
        ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage"); ax.set_title(g)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axs[1].legend(fontsize=7, ncol=2, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_history(run):
    import csv
    path = os.path.join(run, "history.csv")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    ep = np.array([float(r["epoch"]) for r in rows])
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for k, ax, lab in (("loss", axs[0], "loss (-objective / n_obs)"), ("nll", axs[1], "E_q[-log p(m|theta)]"),
                       ("entropy", axs[2], "entropy of q")):
        ax.plot(ep, [float(r[f"train_{k}"]) for r in rows], label="train")
        ax.plot(ep, [float(r[f"val_{k}"]) for r in rows], label="validation")
        ax.set_xlabel("epoch"); ax.set_title(lab)
    axs[0].legend(frameon=False)
    if np.all(np.array([float(r["train_loss"]) for r in rows]) > 0):
        axs[0].set_yscale("log")
    fig.tight_layout()
    os.makedirs(os.path.join(run, "postprocessing"), exist_ok=True)
    fig.savefig(os.path.join(run, "postprocessing", "training_history.png"), dpi=170)
    plt.close(fig)
