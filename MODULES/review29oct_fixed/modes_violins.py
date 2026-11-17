#!/usr/bin/env python3
"""
R3.4: violin plots of the same benchmark scenarios of the test dataset (50 elements, same design of experiments)
identified with 5 modes and with 3 modes. Uses only the predictions of the two runs (no reference posterior).
Usage (project root): python MODULES/review29oct_fixed/modes_violins.py
"""
import os, sys
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from MODULES.review29oct_fixed.posterior_families import family_from_preds
import MODULES.review29oct_fixed.selected_cases_figures as S

O = os.path.join(ROOT, "Output", "baseline_GaussCopula_GaussMarginals")
RUNS = {"5 modes": os.path.join(O, "GCLogN_13Nov-DoE30_7D_Gamma1.0_Conv6000"),
        "3 modes": os.path.join(O, "GCLogN_13Nov-DoE30-3modes_7D_Gamma1.0_Conv6000")}
CASES = {3: r"Mild damage in zone 2, $z_2^{\mathrm{true}}=0.78$", 1: r"Severe damage in end zone 1, $z_1^{\mathrm{true}}=0.56$"}
OUT = os.path.join(ROOT, "MODULES", "review29oct_fixed", "Latex_docs", "figures_final", "Fig_modes_violins")
LB, TAU = 0.45, 0.80

z = np.load(os.path.join(ROOT, "Data", "13Nov_FINAL_SlabS10_7D_DoE30", "alphas_test_fine_uniform.npy")).astype(float)
pred = {}
for k, run in RUNS.items():
    p = np.load(os.path.join(run, "Test_predicted_props_test_fine_uniform.npy"), allow_pickle=True).item()
    mu, s = p["test_logz_loc"].astype(float), p["test_logz_scale"].astype(float)
    a, b = (np.log(LB) - mu) / s, -mu / s
    Zn = norm.cdf(b) - norm.cdf(a)
    pred[k] = dict(fam=family_from_preds(p, LB),
                   mean=np.exp(mu + s ** 2 / 2) * (norm.cdf(b - s) - norm.cdf(a - s)) / Zn,
                   lo=np.exp(mu + s * norm.ppf(norm.cdf(a) + 0.025 * Zn)),
                   hi=np.exp(mu + s * norm.ppf(norm.cdf(a) + 0.975 * Zn)),
                   pd=(norm.cdf((np.log(TAU) - mu) / s) - norm.cdf(a)) / Zn)

plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm", "pdf.fonttype": 42})  # TrueType, no Type 3
fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4), sharey=True)
k = 0
for r, (pos, desc) in enumerate(CASES.items()):
    for c, run in enumerate(RUNS):
        P = pred[run]
        smp = P["fam"].sample(4000, seed=pos, idx=[pos])[0]
        S.decision_violins(axs[r, c], smp, z[pos], P["mean"][pos], P["lo"][pos], P["hi"][pos],
                           f"({chr(97 + k)}) Test case {pos}, {run}" + "\n" + desc, P["pd"][pos])
        k += 1
for ax in axs[:, 0]:
    ax.set_ylabel("Stiffness factor $z$")
fig.legend(handles=S.legend_handles(), loc="lower center", ncol=4, fontsize=7.2, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=3.0)
for e in ("pdf", "png"):
    fig.savefig(f"{OUT}.{e}", dpi=300)
print("saved", OUT)
for pos in CASES:
    for run in RUNS:
        print(pos, run, "P_d", np.round(pred[run]["pd"][pos], 2), "mean", np.round(pred[run]["mean"][pos], 2))
