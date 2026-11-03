#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct/new_scripts_rev/evaluate_posteriors.py

Honest comparison of the trained VAE posterior q(z|m) against the SMC reference
posterior p_gamma(z|m) (same likelihood, same prior, same decoder).

Differences w.r.t. postprocessing_7D_bridge.py:
  * The "estimated" posterior is the RAW encoder distribution. The old
    sample_vae_and_refine_7d pooled VAE draws with 1024 jittered ground-truth draws and
    re-weighted them by the true posterior, so the blue curves were largely the reference
    itself and not the VAE.
  * The reference is the SMC sampler of reference_posterior_smc.py (no initialisation at
    z_true, no case-dependent likelihood weights, same prior as training).

Usage (from the repository root):
    python MODULES/review29oct/new_scripts_rev/evaluate_posteriors.py <run_folder> [--positions 0 1 4 9]
"""

import os
import sys
import json
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, wasserstein_distance

from MODULES.review29oct.preprocessing_and_eigen import load_bridge_data, load_baseline_matrices
from MODULES.review29oct.new_scripts_rev.GC_KSmix_models import MixGC_KS_VAE_withEigen, sample_posterior_np
from MODULES.review29oct.new_scripts_rev.reference_posterior_smc import NumpyBridgeForward, smc_reference_posterior

DIM_LABELS = [r'$z_1$', r'$z_2$', r'$z_3$', r'$z_4$', r'$z_5$', r'$z_{b,L}$', r'$z_{b,R}$']
SCENARIOS = {0: 'Healthy', 1: 'Severe zone 1', 2: 'Moderate zone 3', 3: 'Mild zone 2',
             4: 'Left bearing', 5: 'Zone 1 + left bearing', 6: 'Triple damage',
             7: 'Distributed 4 zones', 8: 'Symmetric A', 9: 'Symmetric B'}
C_REF, C_VAE = '#C62828', '#1565C0'


def build_model_from_config(cfg, K_basis, L_inv, mean_f, std_f):
    return MixGC_KS_VAE_withEigen(
        input_dim=cfg['input_dim'], num_dofs=cfg['n_dofs'], n_modes=cfg['n_modes'],
        Ke_matrices=K_basis, L_inv=L_inv, n_dims=cfg['n_dims'], n_comp=cfg['n_comp'],
        num_samples=cfg['num_samples'], gamma=cfg['gamma'], mean_f=mean_f, std_f=std_f,
        lbound=cfg['lbound'], prior_alpha=cfg['prior_alpha'], prior_beta=cfg['prior_beta'],
        lambda_vert=cfg['lambda_vert'], lambda_rot=cfg['lambda_rot'], objective=cfg['objective'])


def analytic_marginal_pdf(z_grid, params, d, lbound):
    """Exact marginal density of q along dimension d: sum_k pi_k Kumaraswamy_kd(z)."""
    x = np.clip((z_grid - lbound) / (1.0 - lbound), 1e-6, 1 - 1e-6)
    a, b, pi = params['a'][:, d, None], params['b'][:, d, None], params['pi'][:, None]
    pdf = a * b * x ** (a - 1.0) * (1.0 - x ** a) ** (b - 1.0)
    return np.sum(pi * pdf, axis=0) / (1.0 - lbound)


def reflected_kde_1d(samples, grid, lo, hi):
    """KDE with reflection at both bounds (avoids the artificial drop at z = 1)."""
    aug = np.concatenate([samples, 2 * hi - samples, 2 * lo - samples])
    kde = gaussian_kde(aug, bw_method=gaussian_kde(samples).factor)
    return 3.0 * kde(grid)


def hpd_levels(density, fractions=(0.5, 0.9)):
    """Density thresholds enclosing the given probability mass (for 2D contour plots)."""
    flat = np.sort(density.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    return sorted(flat[np.searchsorted(cum, f)] for f in fractions)


def plot_marginal_strip(z_ref, z_vae, params, z_true, lbound, title, path):
    D = z_ref.shape[1]
    grid = np.linspace(lbound, 1.0, 300)
    fig, axes = plt.subplots(1, D, figsize=(2.6 * D, 2.6), sharey=False)
    for d, ax in enumerate(axes):
        ax.fill_between(grid, reflected_kde_1d(z_ref[:, d], grid, lbound, 1.0), color=C_REF, alpha=0.25, lw=0)
        ax.plot(grid, reflected_kde_1d(z_ref[:, d], grid, lbound, 1.0), color=C_REF, lw=2, label='Reference (SMC)')
        ax.plot(grid, analytic_marginal_pdf(grid, params, d, lbound), color=C_VAE, lw=2, label='VAE q(z|m)')
        ax.axvline(z_true[d], color='k', ls='--', lw=1.2, label='True z*')
        ax.set_title(f"{DIM_LABELS[d]}  true={z_true[d]:.2f}\nref={z_ref[:, d].mean():.2f} | vae={z_vae[:, d].mean():.2f}",
                     fontsize=9)
        ax.set_xlim(lbound, 1.0)
        ax.set_yticks([])
    axes[0].legend(fontsize=7, loc='upper left')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_corner(z_ref, z_vae, params, z_true, lbound, title, path):
    D = z_ref.shape[1]
    grid = np.linspace(lbound, 1.0, 200)
    xi, yi = np.mgrid[lbound:1.0:60j, lbound:1.0:60j]
    pts = np.vstack([xi.ravel(), yi.ravel()])
    fig, axes = plt.subplots(D, D, figsize=(2.0 * D, 2.0 * D))
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue
            if i == j:
                ax.plot(grid, reflected_kde_1d(z_ref[:, i], grid, lbound, 1.0), color=C_REF, lw=1.6)
                ax.plot(grid, analytic_marginal_pdf(grid, params, i, lbound), color=C_VAE, lw=1.6)
                ax.axvline(z_true[i], color='k', ls='--', lw=1)
                ax.set_yticks([])
            else:
                for samples, color in ((z_ref, C_REF), (z_vae, C_VAE)):
                    dens = gaussian_kde(samples[:, [j, i]].T)(pts).reshape(xi.shape)
                    ax.contour(xi, yi, dens, levels=hpd_levels(dens), colors=color, linewidths=[1.0, 1.6])
                ax.plot(z_true[j], z_true[i], marker='*', color='gold', mec='k', ms=9)
                ax.set_ylim(lbound, 1.0)
            ax.set_xlim(lbound, 1.0)
            ax.tick_params(labelsize=6)
            if i == D - 1:
                ax.set_xlabel(DIM_LABELS[j])
            if j == 0 and i > 0:
                ax.set_ylabel(DIM_LABELS[i])
    handles = [plt.Line2D([], [], color=C_REF, lw=2, label='Reference posterior (SMC)'),
               plt.Line2D([], [], color=C_VAE, lw=2, label='VAE q(z|m)'),
               plt.Line2D([], [], color='gold', marker='*', mec='k', ls='', ms=10, label='True z*')]
    fig.legend(handles=handles, loc='upper right', fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def global_test_metrics(model, f_test, r_test, v_test, z_test, n_samples=1000, healthy=0.95):
    """Point accuracy, 95% marginal coverage of z_true and false-positive rate on the whole test set."""
    means, lo, hi = [], [], []
    for i in range(len(f_test)):
        z, _ = sample_posterior_np(model, f_test[i], r_test[i], v_test[i], n_samples, seed=i)
        means.append(z.mean(0))
        lo.append(np.quantile(z, 0.025, axis=0))
        hi.append(np.quantile(z, 0.975, axis=0))
    means, lo, hi = map(np.array, (means, lo, hi))
    healthy_mask = z_test >= healthy
    return {
        'MAE': float(np.mean(np.abs(means - z_test))),
        'RMSE': float(np.sqrt(np.mean((means - z_test) ** 2))),
        'Coverage95_of_true_z': float(np.mean((z_test >= lo) & (z_test <= hi))),
        'Mean_interval_width': float(np.mean(hi - lo)),
        'Healthy_FPR_(upper95<0.95)': float(np.mean(hi[healthy_mask] < healthy)),
    }


def run_evaluation(folder, model=None, positions=range(10), n_samples=8192, n_particles=8192):
    with open(os.path.join(folder, 'config.json')) as fh:
        cfg = json.load(fh)
    (_, _, _, _, _, _, _, _, f_test, r_test, v_test, z_test, mean_f, std_f) = load_bridge_data(cfg['data_dir'])
    _, K_basis, L_inv = load_baseline_matrices(cfg['data_dir'])

    if model is None:
        import keras
        keras.backend.set_floatx('float32')
        model = build_model_from_config(cfg, K_basis, L_inv, mean_f, std_f)
        _ = model((f_test[:2], r_test[:2], v_test[:2], z_test[:2]))
        model.load_weights(os.path.join(folder, 'final_model_weights.weights.h5'))

    forward = NumpyBridgeForward(K_basis, L_inv, cfg['n_modes'], mean_f, std_f,
                                 cfg['lambda_vert'], cfg['lambda_rot'])
    lb = cfg['lbound']
    plot_dir = os.path.join(folder, 'posterior_plots')
    os.makedirs(plot_dir, exist_ok=True)

    rows = []
    for pos in positions:
        title = f"Sample {pos} ({SCENARIOS.get(pos, '')})"
        print(f"[{title}] running SMC reference ...")
        z_ref, diag = smc_reference_posterior(
            forward, f_test[pos], r_test[pos], v_test[pos], cfg['gamma'], lb,
            cfg['prior_alpha'], cfg['prior_beta'], n_particles=n_particles, seed=pos)
        z_vae, params = sample_posterior_np(model, f_test[pos], r_test[pos], v_test[pos], n_samples, seed=pos)
        np.savez(os.path.join(plot_dir, f'samples_pos{pos}.npz'), z_ref=z_ref, z_vae=z_vae, z_true=z_test[pos], **params)

        plot_marginal_strip(z_ref, z_vae, params, z_test[pos], lb, title,
                            os.path.join(plot_dir, f'marginals_pos{pos}.png'))
        plot_corner(z_ref, z_vae, params, z_test[pos], lb, title, os.path.join(plot_dir, f'corner_pos{pos}.png'))

        for d in range(z_ref.shape[1]):
            rows.append({'pos': pos, 'dim': d, 'z_true': float(z_test[pos, d]),
                         'ref_mean': float(z_ref[:, d].mean()), 'ref_std': float(z_ref[:, d].std()),
                         'vae_mean': float(z_vae[:, d].mean()), 'vae_std': float(z_vae[:, d].std()),
                         'W1(vae,ref)': float(wasserstein_distance(z_vae[:, d], z_ref[:, d])),
                         'smc_stages': diag['n_stages']})

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(folder, 'benchmark_vae_vs_reference.csv'), index=False)
    print(df.groupby('pos')[['W1(vae,ref)']].mean())

    metrics = global_test_metrics(model, f_test, r_test, v_test, z_test)
    with open(os.path.join(folder, 'global_test_metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder')
    p.add_argument('--positions', type=int, nargs='*', default=list(range(10)))
    p.add_argument('--n_particles', type=int, default=8192)
    a = p.parse_args()
    run_evaluation(a.folder, positions=a.positions, n_particles=a.n_particles)
