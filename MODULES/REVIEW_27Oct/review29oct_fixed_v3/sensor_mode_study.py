#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed_v3/sensor_mode_study.py

Design study to choose a feasible measurement setup and realistic bearing stiffnesses BEFORE generating data:
  * sensor layouts (a few vertical accelerometers, never one per node),
  * number of identified modes,
  * bearing stiffness options.
For each combination it reports
  1. the linearized (Laplace / Fisher-information) posterior std of every parameter, i.e. how well the measurement
     setup identifies each zone and bearing (median over a set of damage states; a fast proxy of the exact posterior),
  2. spatial aliasing: the largest MAC, at the sensors, between two different modes (close to 1 = the sensors cannot
     tell those modes apart),
  3. the frequency of the last mode and its discretization error (decoder vs 50-element model).

Edit the LAYOUTS / MODES / BEARINGS dictionaries below and run from the project root:
    python MODULES/review29oct_fixed_v3/sensor_mode_study.py
Results: Output/review29oct_fixed/sensor_mode_study/ (CSV + figures).
"""

import os
import sys
import itertools

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MODULES.review29oct_fixed_v3 import config as C
from MODULES.review29oct_fixed_v3.fe_model import BeamMesh, modal_solve, align_signs, mac
from MODULES.review29oct_fixed_v3.plotting import fixed_style

# --- sensor layouts [m] (must be nodes of the decoder mesh: multiples of 1 m with 2 elements per zone) ---
LAYOUTS = {
    "6 sensors, zone boundaries (current)": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
    "4 sensors, interior zone boundaries": [2.0, 4.0, 6.0, 8.0],
    "5 sensors, zone centres": [1.0, 3.0, 5.0, 7.0, 9.0],
    "7 sensors, supports + zone centres": [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 10.0],
}
MODES = [3, 4, 5]

# --- bearing stiffness options (kv0 [N/m], kr0 [N m/rad]) ---
# Indicative laminated elastomeric pad for this light 10 m beam (reaction ~2 kN): 150 x 100 mm, 2 x 10 mm rubber
# layers, G = 0.9 MPa, shape factor S = 3, compression modulus Ec = 6 G S^2 = 48.6 MPa:
#   vertical    kv = Ec A / t_r            = 48.6e6 * 0.015 / 0.02          ~ 3.6e7 N/m
#   rotational  kr = Ec (b a^3 / 12) / t_r = 48.6e6 * 2.81e-5 / 0.02       ~ 6.8e4 N m/rad
#   horizontal  kh = G A / t_r             = 0.9e6 * 0.015 / 0.02           ~ 6.8e5 N/m  (shear: lateral loads)
BEARINGS = {
    "current (kv 1.5e8, kr 5e6)": (1.5e8, 5.0e6),
    "elastomeric pad (kv 3.6e7, kr 6.8e4)": (3.6e7, 6.8e4),
    "shear stiffness used vertically (kv 6.8e5, kr 6.8e4)": (6.8e5, 6.8e4),
}
STATES = [b[0] for b in C.BENCHMARKS]          # damage states at which the identifiability is evaluated


def decoder(mesh):
    K_basis, M = mesh.zone_basis()
    return K_basis, M


def modal(z, mesh, K_basis, M, n_modes, ref=None):
    nd = K_basis.shape[-1]
    K = (np.atleast_2d(z) @ K_basis.reshape(len(K_basis), -1)).reshape(-1, nd, nd)
    return modal_solve(K, M, mesh.sensor_nodes, n_modes, ref)


def linearized_std(z, mesh, K_basis, M, n_modes, ref, h=1e-4):
    """Laplace std of the 7 parameters: Fisher information of log f (sigma_f) and of the unit-norm vertical shapes
    (component std sigma_v = MODE_NOISE * peak), F = J_f^T J_f / s_f^2 + sum_j J_vj^T J_vj / s_vj^2."""
    f0, v0, _ = modal(z, mesh, K_basis, M, n_modes, ref)
    J_f = np.zeros((n_modes, 7))
    J_v = np.zeros((n_modes, v0.shape[-1], 7))
    for d in range(7):
        zp, zm = np.array(z, float), np.array(z, float)
        zp[d] += h
        zm[d] -= h
        fp, vp, _ = modal(zp, mesh, K_basis, M, n_modes, ref)
        fm, vm, _ = modal(zm, mesh, K_basis, M, n_modes, ref)
        J_f[:, d] = (np.log(fp[0]) - np.log(fm[0])) / (2 * h)
        J_v[:, :, d] = (vp[0] - vm[0]) / (2 * h)
    s_v = C.MODE_NOISE * np.abs(v0[0]).max(axis=1)
    F = J_f.T @ J_f / C.FREQ_NOISE ** 2 + sum(J_v[j].T @ J_v[j] / s_v[j] ** 2 for j in range(n_modes))
    return np.sqrt(np.diag(np.linalg.inv(F + 1e-9 * np.eye(7))))


@fixed_style
def main():
    out_dir = os.path.join(C.OUTPUT_DIR, "sensor_mode_study")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    mac_mats = {}
    for (lay_name, pos), (b_name, (kv, kr)) in itertools.product(LAYOUTS.items(), BEARINGS.items()):
        structure = dict(C.STRUCTURE, kv0=kv, kr0=kr)
        mesh = BeamMesh(C.N_ZONES, C.EL_PER_ZONE_DECODER, sensor_positions=pos, **structure)
        fine = BeamMesh(C.N_ZONES, C.EL_PER_ZONE_TEST, sensor_positions=pos, **structure)
        K_basis, M = decoder(mesh)
        Kf, Mf = decoder(fine)
        n_max = max(MODES)
        f_h, v_h, r_h = modal(np.ones(7), mesh, K_basis, M, n_max)
        ref = (v_h[0], r_h[0])
        f_fine, _, _ = modal(np.ones(7), fine, Kf, Mf, n_max)
        # aliasing over random states: largest MAC between two different modes at the sensors
        rng = np.random.default_rng(0)
        _, v_r, _ = modal(rng.uniform(C.LBOUND, 1, (500, 7)), mesh, K_basis, M, n_max, ref)
        cross = mac(v_r[:, :, None, :], v_r[:, None, :, :])
        for n_m in MODES:
            stds = np.array([linearized_std(z, mesh, K_basis, M, n_m, (ref[0][:n_m], ref[1][:n_m])) for z in STATES])
            off = cross[:, :n_m, :n_m] * (1 - np.eye(n_m))
            row = {"layout": lay_name, "n_sensors": len(pos), "n_modes": n_m, "bearings": b_name,
                   "f_last_Hz": f_h[0, n_m - 1], "mesh_error_last_mode_%": 100 * (f_h[0, n_m - 1] / f_fine[0, n_m - 1] - 1),
                   "max_MAC_between_modes": off.max()}
            row.update({f"std_{n}": v for n, v in zip(C.PARAM_NAMES, np.median(stds, axis=0))})
            rows.append(row)
            if b_name == next(iter(BEARINGS)):
                mac_mats[(lay_name, n_m)] = off.max(axis=0) + np.eye(n_m)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "sensor_mode_bearing_study.csv"), index=False)
    pd.set_option("display.width", 220)
    show = ["layout", "n_modes", "bearings", "f_last_Hz", "max_MAC_between_modes"] + [f"std_{n}" for n in C.PARAM_NAMES]
    print(df[show].round(3).to_string(index=False))

    # figure: median linearized std per parameter, current bearings, 5 modes vs fewer
    b0 = next(iter(BEARINGS))
    fig, axes = plt.subplots(1, len(MODES), figsize=(6 * len(MODES), 4.8), sharey=True, facecolor="white")
    for ax, n_m in zip(axes, MODES):
        sub = df[(df.bearings == b0) & (df.n_modes == n_m)]
        x = np.arange(7)
        for k, (_, r) in enumerate(sub.iterrows()):
            ax.bar(x + (k - 1.5) * 0.2, [r[f"std_{n}"] for n in C.PARAM_NAMES], 0.2, label=r["layout"])
        ax.set_xticks(x, C.PARAM_NAMES)
        ax.set_title(f"{n_m} modes, bearings: {b0}", fontweight="bold", fontsize=10)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, ls=":")
    axes[0].set_ylabel("median linearized posterior std")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "Fig_identifiability_layouts_modes.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor="white")
    sub = df[(df.layout == next(iter(LAYOUTS))) & (df.n_modes == 5)]
    for k, (_, r) in enumerate(sub.iterrows()):
        ax.bar(np.arange(7) + (k - 1) * 0.27, [r[f"std_{n}"] for n in C.PARAM_NAMES], 0.27, label=r["bearings"])
    ax.set_xticks(range(7), C.PARAM_NAMES)
    ax.set_yscale("log")
    ax.set_ylabel("median linearized posterior std")
    ax.set_title("Bearing stiffness options (6 sensors, 5 modes)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, ls=":")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "Fig_identifiability_bearings.png"), dpi=160)
    plt.close(fig)
    print(f"\nSaved to {out_dir}  (note: a linearized std above ~0.15 means the parameter is barely identifiable "
          f"within [{C.LBOUND}, 1])")


if __name__ == "__main__":
    main()
