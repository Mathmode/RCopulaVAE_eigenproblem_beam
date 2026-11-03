# -*- coding: utf-8 -*-
"""
verify_fe_model.py -- Sanity checks of the review FE engine + numbers for the
                      response letter.

    python -m MODULES.review_27oct.verify_fe_model

1. Bit-level reproduction of the ORIGINAL generator (Noisy_random_dataset_generation.py
   algorithm, copied verbatim below) for the 5-element pinned beam.
2. Discretisation (model-form) error of the 5-element model w.r.t. a converged
   mesh and the analytical Euler-Bernoulli solution, mode by mode  -> R1.1.
3. Sensitivity of the frequencies to bearing degradation on the elastic toy
   beam and on the 20 m bridge  -> choice of kv_nom (R2.2/R2.3).
4. Mass/stiffness confounding: uniform mass increase vs uniform stiffness loss -> R3.1/R1.2.
"""

import numpy as np

from .fe_beam_model import BeamFEModel, observe
from .generate_review_datasets import TOY_E5, TOY_FINE, TOY_ELASTIC, BRIDGE20, BRIDGE20_PINNED
from dataclasses import replace


def legacy_modal(alphas, n_elements=5, n_modes=5):
    """Verbatim algorithm of PREPROCESSING/Noisy_random_dataset_generation.py (noise-free part)."""
    E, I, rho, Area, L_total = 210e9, 8.33e-6, 7850, 0.005, 10.0
    L_e = L_total / n_elements
    n_nodes = n_elements + 1
    n_dofs = 2 * n_nodes
    k_const = E * I / L_e ** 3
    Ke_base = k_const * np.array([[12, 6 * L_e, -12, 6 * L_e], [6 * L_e, 4 * L_e ** 2, -6 * L_e, 2 * L_e ** 2],
                                  [-12, -6 * L_e, 12, -6 * L_e], [6 * L_e, 2 * L_e ** 2, -6 * L_e, 4 * L_e ** 2]])
    m_const = rho * Area * L_e
    Me = m_const * np.array([
        [13 / 35 + 6. * I / (5. * Area * L_e ** 2), 11. * L_e / 210. + I / (10. * Area * L_e), 9 / 70 - 6 * I / (5 * Area * L_e ** 2), -13 * L_e / 420 + I / (10 * Area * L_e)],
        [11. * L_e / 210. + I / (10 * Area * L_e), L_e ** 2 / 105 + 2 * I / (15 * Area), 13 * L_e / 420 - I / (10 * Area * L_e), -1. * L_e ** 2 / 140 - I / (30 * Area)],
        [9 / 70 - 6 * I / (5 * Area * L_e ** 2), 13 * L_e / 420 - I / (10 * Area * L_e), 13 / 35 + 6 * I / (5 * Area * L_e ** 2), -11 * L_e / 210 - I / (10 * Area * L_e)],
        [-13 * L_e / 420 + I / (10 * Area * L_e), -L_e ** 2 / 140 - I / (30 * Area), -11 * L_e / 210 - I / (10 * Area * L_e), L_e ** 2 / 105 + 2 * I / (15 * Area)]])
    M = np.zeros((n_dofs, n_dofs)); K = np.zeros((n_dofs, n_dofs))
    for i in range(n_elements):
        s = 2 * i
        M[s:s + 4, s:s + 4] += Me
        K[s:s + 4, s:s + 4] += alphas[i] * Ke_base
    free = np.delete(np.arange(n_dofs), [0, n_dofs - 2])
    Mf, Kf = M[np.ix_(free, free)], K[np.ix_(free, free)]
    Linv = np.linalg.inv(np.linalg.cholesky(Mf))
    w, y = np.linalg.eigh(Linv @ Kf @ Linv.T)
    phi = Linv.T @ y
    f = np.sqrt(np.maximum(w[:n_modes], 0)) / (2 * np.pi)
    full = np.zeros((n_dofs, n_modes)); full[free] = phi[:, :n_modes]
    v, r = full[0::2], full[1::2]
    sg = np.sign(v[np.argmax(np.abs(v), axis=0), range(n_modes)])
    return f, (v * sg).T, (r * sg).T


def mac(a, b):
    return (a @ b) ** 2 / ((a @ a) * (b @ b))


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ 1
    section("1. Reproduction of the original generator (5 elements, pinned)")
    m = BeamFEModel(TOY_E5)
    err_f, err_v = 0.0, 0.0
    for _ in range(200):
        z = rng.uniform(0.5, 1.0, 5)
        f0, v0, r0 = legacy_modal(z)
        f1, v1, r1 = m.modal(m.stiffness(z), m.mass(), 5)
        (f1, v1, r1), _ = observe(f1, v1, r1, TOY_E5.sensor_nodes, rng, 0.0, 0.0)
        err_f = max(err_f, np.max(np.abs(f1 - f0) / f0))
        err_v = max(err_v, np.max(np.abs(v1 - v0)) / np.max(np.abs(v0)))
    print(f"max rel. freq. diff = {err_f:.2e}   max rel. mode diff = {err_v:.2e}   -> {'OK' if max(err_f, err_v) < 1e-8 else 'MISMATCH'}")

    # ------------------------------------------------------------------ 2
    section("2. Discretisation error of the 5-element model (healthy beam)  [R1.1]")
    c = TOY_E5
    fa = [(n * np.pi / c.L) ** 2 * np.sqrt(c.EI / (c.rho * c.A)) / (2 * np.pi) for n in range(1, 6)]
    rows = {}
    for n_el in (5, 10, 20, 50, 100):
        cfg = replace(TOY_E5, n_elements=n_el, n_zones=5, sensor_nodes=None)
        mm = BeamFEModel(cfg)
        f, v, _ = mm.modal(mm.stiffness(np.ones(5)), mm.mass(), 5)
        rows[n_el] = (f, v[:, cfg.sensor_nodes])
    print("mode | analytic EB [Hz] |" + "".join(f"  FE{n:<4d}" for n in rows) + "| err(5 vs 100) | MAC(5 vs 100 @sensors)")
    for k in range(5):
        e = 100 * (rows[5][0][k] - rows[100][0][k]) / rows[100][0][k]
        mc = mac(rows[5][1][k], rows[100][1][k])
        print(f"  {k + 1}  | {fa[k]:16.3f} |" + "".join(f" {rows[n][0][k]:7.3f}" for n in rows) + f"| {e:+12.3f}% | {mc:.5f}")
    print("NOTE: the analytic EB values neglect rotary inertia (included in the FE mass) -> small offset at high modes.")
    print("Compare the 5 vs 100 error with the 2.5 % frequency noise: if larger, model-form error dominates noise.")

    # ------------------------------------------------------------------ 3
    section("3. Bearing degradation sensitivity  [R2.2/R2.3]")
    for cfg, nz, nm in ((TOY_ELASTIC, 5, 5), (BRIDGE20, 10, 6)):
        mm = BeamFEModel(cfg)
        ref, _, _ = mm.modal(mm.stiffness(np.ones(nz), np.ones(2)), mm.mass(), nm)
        pin = BeamFEModel(replace(cfg, support="pinned", kv_nom=0.0, support_latent=False))
        fp, _, _ = pin.modal(pin.stiffness(np.ones(nz)), pin.mass(), nm)
        print(f"\n{cfg.name}:  kv_nom = {cfg.kv_nom:.3e} N/m  (kappa = kv L^3/EI = {cfg.kv_nom * cfg.L ** 3 / cfg.EI:.1f})")
        print("   nominal f [Hz]      : " + " ".join(f"{x:7.3f}" for x in ref))
        print("   pinned  f [Hz]      : " + " ".join(f"{x:7.3f}" for x in fp))
        for s in (0.8, 0.5, 0.2):
            f, _, _ = mm.modal(mm.stiffness(np.ones(nz), np.array([s, 1.0])), mm.mass(), nm)
            print(f"   left bearing s={s:.1f} : " + " ".join(f"{100 * (a - b) / b:+6.2f}%" for a, b in zip(f, ref)))
        for zk in (0.8, 0.5):
            zz = np.ones(nz); zz[0] = zk
            f, _, _ = mm.modal(mm.stiffness(zz, np.ones(2)), mm.mass(), nm)
            print(f"   zone1 z={zk:.1f} (ref)  : " + " ".join(f"{100 * (a - b) / b:+6.2f}%" for a, b in zip(f, ref)))

    # ------------------------------------------------------------------ 4
    section("4. Mass-stiffness confounding (5-element beam)  [R3.1/R1.2]")
    f0, v0, _ = m.modal(m.stiffness(np.ones(5)), m.mass(), 5)
    f_m, v_m, _ = m.modal(m.stiffness(np.ones(5)), m.mass(1.05 * np.ones(5)), 5)
    f_k, v_k, _ = m.modal(m.stiffness(np.ones(5) / 1.05), m.mass(), 5)
    print("+5% uniform mass      : " + " ".join(f"{100 * (a - b) / b:+6.3f}%" for a, b in zip(f_m, f0)))
    print("-4.76% uniform stiff. : " + " ".join(f"{100 * (a - b) / b:+6.3f}%" for a, b in zip(f_k, f0)))
    print("MAC between the two mode sets: " + " ".join(f"{mac(a, b):.6f}" for a, b in zip(v_m[:, TOY_E5.sensor_nodes], v_k[:, TOY_E5.sensor_nodes])))
    print("-> a global mass change is indistinguishable from uniform damage with MAC + frequencies only.")


if __name__ == "__main__":
    main()
