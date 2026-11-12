#!/usr/bin/env python3
"""
Analysis of the whole test dataset (50-element data with localized damage, 1,200 structures) with the reference model
GCLogN_06Nov-5modes_7D_Gamma1.0_Conv6000, and selection of representative well-identified and failure cases.

Prediction and decision as in the response letter (R1.6): posterior mean z_hat (analytical, truncated log-normal
marginal), component declared damaged if z_hat < Z_C = 0.84; false positive: healthy component (z* >= 0.95) declared
damaged; false negative: component with relevant damage (z* < 0.80) declared healthy.

Outputs (in <run>/cases):
  location_summary.csv            per component: MAE, 95 % coverage, mean 95 % width, FP and FN rates, information
  Fig_location_summary.png/.pdf   (a) information (Cramer-Rao bound) (b) 95 % width (c) FP and FN rates, per component
  Fig_detection_vs_severity.png   detection rate vs true stiffness factor, end zones / interior zones / bearings
  candidates.csv                  ranking of the candidates of every case type
  selected_cases.json             the selected test positions, used by selected_cases_figures.py
Usage (project root): python MODULES/review29oct_fixed/test_case_analysis.py
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.fe_model import BeamMesh, modal_solve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = os.path.join(ROOT, "Output", "baseline_GaussCopula_GaussMarginals", "GCLogN_06Nov-5modes_7D_Gamma1.0_Conv6000")
DATA = os.path.join(ROOT, "Data", "06Nov_FINAL_SlabS10_7D_5sensors_5modes")
OUT = os.path.join(RUN, "cases")
os.makedirs(OUT, exist_ok=True)
LB, Z_C, Z_H, Z_R = 0.45, 0.84, 0.95, 0.80
NAMES = ["z1", "z2", "z3", "z4", "z5", "zBL", "zBR"]
LAB = [r"$z_1$", r"$z_2$", r"$z_3$", r"$z_4$", r"$z_5$", r"$z_{b,L}$", r"$z_{b,R}$"]
END, INTERIOR, BEAR = [0, 4], [1, 2, 3], [5, 6]


def posterior_summaries(path):
    p = np.load(path, allow_pickle=True).item()
    mu, s = p["test_logz_loc"].astype(float), p["test_logz_scale"].astype(float)
    a, b = (np.log(LB) - mu) / s, -mu / s
    Z = norm.cdf(b) - norm.cdf(a)
    mean = np.exp(mu + s ** 2 / 2) * (norm.cdf(b - s) - norm.cdf(a - s)) / Z
    q = lambda pr: np.exp(mu + s * norm.ppf(norm.cdf(a) + pr * Z))
    return mean, q(0.025), q(0.975)


z = np.load(os.path.join(DATA, "alphas_test_fine.npy")).astype(float)
cat = np.load(os.path.join(DATA, "labels_test_fine.npy"))
m, lo, hi = posterior_summaries(os.path.join(RUN, "Test_predicted_props_test_fine.npy"))
w, err = hi - lo, m - z
inside = (z >= lo) & (z <= hi)
dec = m < Z_C
H, R, Lt = z >= Z_H, z < Z_R, (z >= Z_R) & (z < Z_H)
FP, FN = H & dec, R & ~dec
n_dam = (z < Z_H).sum(1)
pcfg = json.load(open(os.path.join(DATA, "problem_config.json")))
lik = pcfg["likelihood"]

# ----------------------------------------------------------------------------- information per component (CRB)
mesh = BeamMesh(C.N_ZONES, C.EL_PER_ZONE_DECODER, sensor_positions=C.SENSOR_POSITIONS, **C.STRUCTURE)


def modal(zz, ref=None):
    K, M = mesh.assemble(zz[None, mesh.zone_of_el], zz[None, C.N_ZONES:])
    f, v, _ = modal_solve(K, M, mesh.sensor_nodes, C.N_MODES)
    v = v[0]
    if ref is not None:
        v = v * np.sign(np.sum(v * ref, axis=1, keepdims=True))
    return np.log(f[0]), v


z0, h = np.full(7, 0.90), 1e-4
lf0, v0 = modal(z0)
Jf, Jv = [], []
for d in range(7):
    zp, zm = z0.copy(), z0.copy(); zp[d] += h; zm[d] -= h
    (lp, vp), (lm, vm) = modal(zp, v0), modal(zm, v0)
    Jf.append((lp - lm) / (2 * h)); Jv.append(((vp - vm) / (2 * h)).ravel())
Jf, Jv = np.array(Jf).T, np.array(Jv).T
Wv = np.repeat(1.0 / np.asarray(lik["sig_v"]) ** 2, len(C.SENSOR_POSITIONS))
FIM = Jf.T @ np.asarray(lik["prec_logf"]) @ Jf + Jv.T @ (Wv[:, None] * Jv)
crb = np.sqrt(np.diag(np.linalg.inv(FIM)))

# ----------------------------------------------------------------------------- per-location summary
rows = []
for d in range(7):
    rows.append({"component": NAMES[d], "MAE": np.abs(err[:, d]).mean(), "coverage95": inside[:, d].mean(),
                 "width95_healthy": w[H[:, d], d].mean(), "width95_all": w[:, d].mean(),
                 "FP_rate": FP[:, d].sum() / H[:, d].sum(), "FN_rate": FN[:, d].sum() / R[:, d].sum(),
                 "n_healthy": int(H[:, d].sum()), "n_relevant": int(R[:, d].sum()), "CRB_std_z0.9": crb[d]})
loc = pd.DataFrame(rows)
loc.to_csv(os.path.join(OUT, "location_summary.csv"), index=False, float_format="%.4f")
print(loc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
cols = ["#c0392b" if d in END else ("#1f5aa6" if d in INTERIOR else "#2e7d4f") for d in range(7)]
xs = np.array([0, 1, 2, 3, 4, 5.6, 6.6])
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
axs[0].bar(xs, crb, color=cols, width=0.7)
axs[0].set_title("(a) Information in the data:\nCramér–Rao bound of the std", fontsize=8.5)
axs[1].bar(xs, loc["width95_healthy"], color=cols, width=0.7)
axs[1].set_title("(b) Mean width of the 95% credible\ninterval (healthy components)", fontsize=8.5)
axs[2].bar(xs - 0.18, 100 * loc["FP_rate"], width=0.36, color=cols, alpha=0.55)
axs[2].bar(xs + 0.18, 100 * loc["FN_rate"], width=0.36, color=cols, hatch="///", edgecolor="white")
axs[2].set_title("(c) Decision errors [%]", fontsize=8.5)
from matplotlib.patches import Patch
axs[2].legend(handles=[Patch(fc="0.75", label="False positives"), Patch(fc="0.35", hatch="///", ec="white", label="False negatives")],
              fontsize=7, frameon=False, handlelength=1.2, loc="upper center", bbox_to_anchor=(0.47, 0.98))
for ax in axs:
    ax.set_xticks(xs); ax.set_xticklabels(LAB, fontsize=8); ax.grid(axis="y", alpha=0.3, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Fig_location_summary.{e}"), dpi=300)
plt.close(fig)

# ----------------------------------------------------------------------------- detection rate vs severity
bins = np.array([0.45, 0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0001])
mid = 0.5 * (bins[1:] + bins[:-1])
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for grp, name, col, mk in ((END, "Deck end zones", "#c0392b", "s"), (INTERIOR, "Interior deck zones", "#1f5aa6", "o"),
                           (BEAR, "Bearings", "#2e7d4f", "^")):
    zz, dd = z[:, grp].ravel(), dec[:, grp].ravel()
    rate = [dd[(zz >= bins[k]) & (zz < bins[k + 1])].mean() for k in range(len(mid))]
    ax.plot(mid, rate, "-" + mk, color=col, ms=4, lw=1.3, label=name)
ax.axvspan(Z_H, 1.0, color="0.85", lw=0)
ax.axvspan(Z_R, Z_H, color="0.94", lw=0)
ax.axvline(Z_C, ls="--", color="k", lw=0.8)
ax.text(0.975, 0.55, "healthy", rotation=90, ha="center", fontsize=7.5, color="0.3")
ax.text(0.875, 0.55, "light damage", rotation=90, ha="center", fontsize=7.5, color="0.4")
ax.set_xlabel(r"True stiffness factor $z_d^*$"); ax.set_ylabel("Fraction declared damaged")
ax.set_xlim(0.45, 1.0); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=7.5, loc="center left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Fig_detection_vs_severity.{e}"), dpi=300)
plt.close(fig)

# ----------------------------------------------------------------------------- end zones: position of the damage
# localized damage of the test data: [sample, zone, first 0.2 m cell, number of cells, local stiffness factor]
Lrec = np.load(os.path.join(DATA, "asbuilt_fields", "localized_damage_test_fine.npy"))
dist, det_end = [], []
for r_ in Lrec:
    i, zone, c0, nc = int(r_[0]), int(r_[1]), int(r_[2]), int(r_[3])
    if zone in END and z[i, zone] < Z_R:
        x0, x1 = 2.0 * zone + 0.2 * c0, 2.0 * zone + 0.2 * (c0 + nc)
        dist.append(min(x0, C.STRUCTURE["L_total"] - x1)); det_end.append(dec[i, zone])
dist, det_end = np.array(dist), np.array(det_end)
edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.8])
mids = 0.5 * (edges[1:] + edges[:-1])
rate = np.array([det_end[(dist >= edges[k] - 1e-9) & (dist < edges[k + 1] - 1e-9)].mean() for k in range(len(mids))])
cnt = np.array([((dist >= edges[k] - 1e-9) & (dist < edges[k + 1] - 1e-9)).sum() for k in range(len(mids))])
interior_rate = dec[:, INTERIOR][R[:, INTERIOR]].mean()
fig, ax = plt.subplots(figsize=(4.4, 2.9))
xk = np.arange(len(mids))
ax.bar(xk, rate, width=0.7, color="#c0392b", alpha=0.8, label="Deck end zones")
for xm, rt, n in zip(xk, rate, cnt):
    ax.text(xm, rt / 2, f"n={n}", ha="center", va="center", fontsize=6.5, color="white", rotation=90)
ax.axhline(interior_rate, ls="--", color="#1f5aa6", lw=1.0, label="Interior deck zones (any position)")
ax.set_xticks(xk); ax.set_xticklabels([f"{edges[k]:.1f}–{edges[k + 1]:.1f}" for k in range(len(mids))], fontsize=7.5)
ax.set_xlabel("Distance from the damaged region to the support [m]")
ax.set_ylabel("Fraction of relevant damage\ndetected ($z_d^*<0.80$)")
ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.3, lw=0.5)
ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Fig_end_zone_damage_position.{e}"), dpi=300)
plt.close(fig)
print("\nend zones, relevant damage detected vs distance to the support:", dict(zip(np.round(mids, 2), np.round(rate, 2))),
      f"| interior zones: {interior_rate:.2f}")

# ----------------------------------------------------------------------------- candidates of every case type
mae_s, w_s = np.abs(err).mean(1), w.mean(1)
ok = inside.all(1) & ~(FP | FN).any(1)                      # every true value captured, no decision error
score = mae_s + 0.5 * w_s


def one_damaged(idx_set, zlo, zhi):
    d_idx = np.argmin(z, axis=1)
    return (n_dam == 1) & np.isin(d_idx, idx_set) & (z.min(1) >= zlo) & (z.min(1) < zhi), d_idx


cands = {}
cands["healthy_well"] = np.where((n_dam == 0) & ok)[0]
m1, d1 = one_damaged(INTERIOR, Z_R, 0.90)
cands["light_interior_well"] = np.where(m1 & ok & (np.abs(err[np.arange(len(z)), d1]) < 0.04) & (hi[np.arange(len(z)), d1] < Z_H))[0]
m2, d2 = one_damaged(BEAR, LB, Z_R)
cands["bearing_well"] = np.where(m2 & ok)[0]
two = (n_dam == 2) & ok & ((z < Z_R).sum(1) == 2)
cands["two_damaged_well"] = np.where(two)[0]
# failures
endFP = (n_dam == 0) & FP[:, END].any(1) & (FP.sum(1) == 1)
cands["healthy_FP_end_zone"] = np.where(endFP)[0]
m3, d3 = one_damaged(END, LB, 0.75)
cands["end_zone_missed"] = np.where(m3 & FN[np.arange(len(z)), d3] & (FP.sum(1) == 0))[0]
adj = {0: 5, 4: 6}
conf = [i for i in np.where(m3)[0] if FN[i, d3[i]] and dec[i, adj[d3[i]]]]
cands["end_zone_vs_bearing"] = np.array(conf, dtype=int)

rank_rows = []
for k, idx in cands.items():
    sc = score[idx] if "well" in k else -w_s[idx]           # failures: the clearest (widest) posteriors first
    if k == "end_zone_vs_bearing":
        sc = m[idx, [adj[d3[i]] for i in idx]]               # strongest compensation in the bearing first
    order = idx[np.argsort(sc)]
    for r_, i in enumerate(order[:8]):
        rank_rows.append({"type": k, "rank": r_ + 1, "pos": int(i), "category": int(cat[i]), "n_damaged": int(n_dam[i]),
                          **{f"{NAMES[d]}_true": z[i, d] for d in range(7)}, **{f"{NAMES[d]}_mean": m[i, d] for d in range(7)},
                          "MAE": mae_s[i], "mean_width95": w_s[i], "all_inside": bool(inside[i].all()),
                          "FP": int(FP[i].sum()), "FN": int(FN[i].sum())})
    print(f"\n{k}: {len(idx)} candidates; best: {list(order[:6])}")
cdf = pd.DataFrame(rank_rows)
cdf.to_csv(os.path.join(OUT, "candidates.csv"), index=False, float_format="%.3f")
for k in cands:
    sub = cdf[cdf.type == k].head(4)
    for _, r_ in sub.iterrows():
        print(f"  {k:22s} pos {r_.pos:4d}: z* = {[round(r_[f'{n}_true'], 2) for n in NAMES]}  mean = "
              f"{[round(r_[f'{n}_mean'], 2) for n in NAMES]}  MAE {r_.MAE:.3f} width {r_.mean_width95:.3f}")
