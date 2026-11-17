#!/usr/bin/env python3
"""
R1.6: decisions healthy / damaged for every component of the test structures, from the posterior of a GC-LogN run.

FP: healthy component (z* >= ETA = 0.95) declared damaged; FN: component with relevant damage (z* < 0.80) declared
healthy; light damage (0.80 <= z* < 0.95) is not counted. All posterior quantities are closed-form for the truncated
log-normal marginals, except the prior update (importance weights on joint samples of the Gaussian copula posterior).

  1. Posterior-mean rule: damaged if z_hat_d < tau, tau selected on the VALIDATION set with --criterion: "pod" (default,
     letter since 13 Nov 2026): the smallest tau on a 0.01 grid that detects at least --pod (0.95) of the components
     with relevant damage (Neyman-Pearson: fixed detection rate, fewest false alarms); "eer": equal FP and FN rates
     (rounded to 0.01). FP/FN rates, ROC curve and AUC on the TEST set; 95 % bootstrap intervals (resampling the test
     structures).
  2. Probability rule (equal costs of both errors): damaged if P(z_d < 0.80 | m) > P(z_d >= ETA | m), i.e. if relevant
     damage is more probable than the healthy state, (a) with the uniform prior of the training, and (b) with a prior that
     gives the healthy state the probability pi_H (default: share of healthy components in the TRAINING set), obtained
     without retraining: p_new(z | m) ∝ q(z | m) prod_d r(z_d), with r = pi_H / P_U(healthy) on [ETA, 1] and
     (1 - pi_H) / P_U(damaged) on [z_lb, ETA).
  3. Probability that each decision of rule 2b is wrong (posterior mass of the other state) vs observed error rate.

Outputs (<run>/decision/, or <run>/decision_testA/ with --split test): decision_table.csv, error_probability.csv,
summary.json, Fig_ROC_components.png/.pdf
Usage (project root):
    python MODULES/review29oct_fixed/decision_analysis.py --run Output/baseline_GaussCopula_GaussMarginals/<run>
"""
import os, re, sys, json, argparse
import numpy as np
from scipy.stats import norm

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from MODULES.review29oct_fixed.posterior_families import family_from_preds

LB, ETA, ZR = 0.45, 0.95, 0.80
END = [0, 4]
PRED_FILE = {"test_fine": "Test_predicted_props_test_fine.npy", "test": "Test_predicted_props.npy",
             "test_fine_uniform": "Test_predicted_props_test_fine_uniform.npy",
             "val": "Val_predicted_props.npy"}


def data_dir_of(run):
    """Data set of a run (Problem_info.npy); the folder of the same name in <project>/Data if trained elsewhere."""
    d = str(np.load(os.path.join(run, "Problem_info.npy"), allow_pickle=True).item()["data_dir"])
    if os.path.isdir(d):
        return d
    local = os.path.join(ROOT, "Data", re.split(r"[\\/]", d.rstrip("\\/"))[-1])
    if not os.path.isdir(local):
        raise FileNotFoundError(f"data set of the run not found: {d} (nor {local})")
    return local


def posterior(run, split):
    preds = np.load(os.path.join(run, PRED_FILE[split]), allow_pickle=True).item()
    mu, s = preds["test_logz_loc"].astype(float), preds["test_logz_scale"].astype(float)
    a, b = (np.log(LB) - mu) / s, -mu / s
    Z = norm.cdf(b) - norm.cdf(a)
    cdf = lambda c: (norm.cdf((np.log(c) - mu) / s) - norm.cdf(a)) / Z
    mean = np.exp(mu + s ** 2 / 2) * (norm.cdf(b - s) - norm.cdf(a - s)) / Z
    return preds, mean, 1.0 - cdf(ETA), cdf(ZR)          # posterior mean, P(healthy), P(relevant damage)


def rates(dec, z):
    h, r = z >= ETA, z < ZR
    return int((dec & h).sum()), int(h.sum()), int((~dec & r).sum()), int(r.sum())


def auc(score, z):
    h, r = z >= ETA, z < ZR
    rk = np.concatenate([score[r], score[h]]).argsort().argsort() + 1.0
    n = r.sum()
    return float((rk[:n].sum() - n * (n + 1) / 2) / (n * h.sum()))


def eer_tau(mean, z):
    """Threshold of the posterior-mean rule with equal FP and FN rates."""
    h, r = z >= ETA, z < ZR
    cand = np.quantile(mean[h | r], np.linspace(0.01, 0.99, 981))
    gaps = [abs((mean[h] < c).mean() - (mean[r] >= c).mean()) for c in cand]
    return float(cand[int(np.argmin(gaps))])


def pod_tau(mean, z, pod):
    """Smallest threshold on a 0.01 grid that detects at least a fraction pod of the components with relevant damage."""
    r = z < ZR
    for t in np.round(np.arange(0.70, 1.0001, 0.01), 2):
        if (mean[r] < t).mean() >= pod:
            return float(t)
    raise ValueError(f"no threshold reaches a detection rate of {pod}")


def prior_update(preds, pi_h, n_samples, chunk=100):
    """P(healthy) and P(relevant damage) of every component under a prior with P(healthy) = pi_h (exact, by sampling)."""
    p_u = (1 - ETA) / (1 - LB)
    lw_h, lw_d = np.log(pi_h / p_u), np.log((1 - pi_h) / (1 - p_u))
    fam = family_from_preds(preds, LB)
    n = fam.n
    ph, pd = np.zeros((n, fam.d)), np.zeros((n, fam.d))
    for i0 in range(0, n, chunk):
        idx = np.arange(i0, min(i0 + chunk, n))
        x = fam.sample(n_samples, seed=i0, idx=idx)                       # (len(idx), S, D)
        lw = np.where(x >= ETA, lw_h, lw_d).sum(-1)
        w = np.exp(lw - lw.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        ph[idx] = (w[..., None] * (x >= ETA)).sum(1)
        pd[idx] = (w[..., None] * (x < ZR)).sum(1)
    return ph, pd


def roc_figure(mean, z, tau, out_dir, extra_dirs=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, r = z >= ETA, z < ZR
    score = -mean
    grid = np.concatenate([[np.inf], np.sort(np.unique(score))[::-1], [-np.inf]])
    fpr = np.array([(score[h] > c).mean() for c in grid])
    tpr = np.array([(score[r] > c).mean() for c in grid])
    fp, fn = (mean[h] < tau).mean(), (mean[r] >= tau).mean()
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(figsize=(3.9, 3.6))
    ax.plot(fpr, tpr, "-", color="#1f5aa6", lw=1.6, label=f"Proposed method (AUC = {auc(score, z):.2f})")
    ax.plot(fp, 1 - fn, "o", color="#1f5aa6", ms=5.5, mec="k", mew=0.6, zorder=5)
    ax.annotate(f"$\\hat z_d<{tau:.2f}$\nFP rate = {fp:.1%}\nFN rate = {fn:.1%}", (fp, 1 - fn), xytext=(14, -46),
                textcoords="offset points", fontsize=8, color="#1f5aa6")
    ax.plot([0, 1], [0, 1], ":", color="0.5", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.grid(alpha=0.3, lw=0.5); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    for d in (out_dir, *extra_dirs):
        for e in ("pdf", "png"):
            fig.savefig(os.path.join(d, f"Fig_ROC_components.{e}"), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder (absolute, or relative to the project root)")
    ap.add_argument("--split", default="test_fine", choices=["test_fine", "test", "test_fine_uniform"],
                    help="test_fine_uniform: test set A of the letter (50 elements, same design of experiments); test_fine: test set B (localized damage); test: 10-element test data")
    ap.add_argument("--tau", type=float, default=None, help="threshold of the mean rule (default: --criterion on validation)")
    ap.add_argument("--criterion", default="pod", choices=["pod", "eer"], help="selection of tau on the validation set")
    ap.add_argument("--pod", type=float, default=0.95, help="detection rate of relevant damage for --criterion pod")
    ap.add_argument("--pi-h", type=float, default=None, help="prior P(healthy) (default: share in the training set)")
    ap.add_argument("--samples", type=int, default=2000, help="joint posterior samples per structure (prior update)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples of the test structures")
    ap.add_argument("--fig-dir", default=None, help="also save the ROC figure here (e.g. the letter figures folder)")
    a = ap.parse_args()
    run = a.run if os.path.isabs(a.run) else os.path.join(ROOT, a.run)
    data = data_dir_of(run)
    out = os.path.join(run, {"test_fine": "decision", "test": "decision_testA10", "test_fine_uniform": "decision_testA"}[a.split])
    os.makedirs(out, exist_ok=True)

    z = np.load(os.path.join(data, f"alphas_{a.split}.npy")).astype(float)
    preds, mean, ph, pd = posterior(run, a.split)
    zv = np.load(os.path.join(data, "alphas_val.npy")).astype(float)
    _, mean_v, _, _ = posterior(run, "val")
    tau_eer = eer_tau(mean_v, zv)
    tau_pod = pod_tau(mean_v, zv, a.pod)
    tau = a.tau if a.tau is not None else (tau_pod if a.criterion == "pod" else round(tau_eer, 2))
    ztr = np.load(os.path.join(data, "alphas_train.npy"))
    pi_h = a.pi_h if a.pi_h is not None else float((ztr >= ETA).mean())
    h, r = z >= ETA, z < ZR
    healthy_structures = (z < ETA).sum(1) == 0
    end = np.zeros_like(h); end[:, END] = True
    print(f"run {os.path.basename(run)} | {a.split}: {len(z)} structures, {h.sum()} healthy and {r.sum()} damaged "
          f"(z*<{ZR}) components | tau on validation: POD {a.pod:.2f} -> {tau_pod:.2f}, EER {tau_eer:.3f} | used tau = {tau:.2f} | pi_H = {pi_h:.3f}")

    # 1. posterior-mean rule, ROC and bootstrap
    dec_mean = mean < tau
    rng = np.random.default_rng(0)
    boot = {"AUC": [], "FP": [], "FN": []}
    for _ in range(a.boot):
        i = rng.integers(0, len(z), len(z))
        fp, nh, fn, nr = rates(mean[i] < tau, z[i])
        boot["AUC"].append(auc(-mean[i], z[i])); boot["FP"].append(fp / nh); boot["FN"].append(fn / nr)
    ci = {k: [float(q) for q in np.quantile(v, [0.025, 0.975])] for k, v in boot.items()}
    roc_figure(mean, z, tau, out, [a.fig_dir] if a.fig_dir else [])

    # 2. probability rules
    ph_n, pd_n = prior_update(preds, pi_h, a.samples)
    rules = [("Posterior mean below tau", dec_mean, ph, pd),
             ("P(damage) > P(healthy), uniform prior", pd > ph, ph, pd),
             (f"P(damage) > P(healthy), prior P(healthy) = {pi_h:.2f}", pd_n > ph_n, ph_n, pd_n)]
    rows = []
    for name, dec, p_h, p_d in rules:
        fp, nh, fn, nr = rates(dec, z)
        rows.append({"rule": name, "FP": fp, "healthy_components": nh, "FP_rate": fp / nh, "FN": fn,
                     "damaged_components": nr, "FN_rate": fn / nr, "FP_predicted": float(p_h[dec].sum()),
                     "FN_predicted": float(p_d[~dec].sum()),
                     "FP_rate_healthy_structures": float((dec & h)[healthy_structures].sum() / h[healthy_structures].sum()),
                     "FP_rate_end_zones": float((dec & h & end).sum() / (h & end).sum()),
                     "FP_rate_other_components": float((dec & h & ~end).sum() / (h & ~end).sum()),
                     "FN_rate_severe_z<0.70": float((~dec & (z < 0.70)).sum() / (z < 0.70).sum())})
    import pandas as pd_
    pd_.DataFrame(rows).to_csv(os.path.join(out, "decision_table.csv"), index=False, float_format="%.4f")

    # 3. probability that each decision of the prior-updated rule is wrong vs observed
    dec = pd_n > ph_n
    p_wrong = np.where(dec, ph_n, pd_n)
    wrong = (dec & h) | (~dec & r)
    bins = []
    for lo, hi in ((0.0, 0.05), (0.05, 0.20), (0.20, 1.0001)):
        sel = (p_wrong >= lo) & (p_wrong < hi)
        bins.append({"P_wrong_from": lo, "P_wrong_to": min(hi, 1.0), "decisions": int(sel.sum()),
                     "share_of_decisions": float(sel.mean()), "mean_predicted": float(p_wrong[sel].mean()),
                     "observed_error_rate": float(wrong[sel].mean()), "share_of_errors": float(wrong[sel].sum() / wrong.sum())})
    pd_.DataFrame(bins).to_csv(os.path.join(out, "error_probability.csv"), index=False, float_format="%.4f")

    # 4. breakdown of the errors of the posterior-mean rule (numbers of the R1.6 text)
    def fp_fn(mn, zz, t=tau):
        fp, nh, fn, nr = rates(mn < t, zz)
        return [fp / nh, fn / nr]
    zA = np.load(os.path.join(data, "alphas_test.npy")).astype(float)
    _, mean_A, _, _ = posterior(run, "test")
    FPm, FNm = dec_mean & h, ~dec_mean & r
    sev, mild = z < 0.70, (z >= 0.70) & r
    inter, bear = np.zeros_like(h), np.zeros_like(h)
    inter[:, [1, 2, 3]] = True
    bear[:, [5, 6]] = True
    mu, s = preds["test_logz_loc"].astype(float), preds["test_logz_scale"].astype(float)
    aa, bb = (np.log(LB) - mu) / s, -mu / s
    Zt = norm.cdf(bb) - norm.cdf(aa)
    q = lambda pr: np.exp(mu + s * norm.ppf(norm.cdf(aa) + pr * Zt))
    width = q(0.975) - q(0.025)
    breakdown = {
        "FP_FN_validation": fp_fn(mean_v, zv), "FP_FN_test_A": fp_fn(mean_A, zA), "FP_FN_this_split": fp_fn(mean, z),
        "healthy_structures": int(healthy_structures.sum()),
        "healthy_structures_without_FP": int((~FPm[healthy_structures].any(1)).sum()),
        "FP_rate_end_zones": float(FPm[end].sum() / h[end].sum()), "FP_rate_interior": float(FPm[inter].sum() / h[inter].sum()),
        "FP_rate_bearings": float(FPm[bear].sum() / h[bear].sum()), "FP_share_end_zones": float(FPm[end].sum() / FPm.sum()),
        "FN_rate_severe_z<0.70": float(FNm[sev].sum() / sev.sum()), "FN_rate_mild_0.70-0.80": float(FNm[mild].sum() / mild.sum()),
        "FN_share_mild": float(FNm[mild].sum() / FNm.sum()),
        "FN_rate_end_zones": float(FNm[end].sum() / r[end].sum()), "FN_rate_interior": float(FNm[inter].sum() / r[inter].sum()),
        "FN_rate_bearings": float(FNm[bear].sum() / r[bear].sum()), "FN_share_end_zones": float(FNm[end].sum() / FNm.sum()),
        "FN_share_mean_below_0.90": float((mean[FNm] < 0.90).mean()),
        "scenarios_with_severe_damage_missed": int((FNm & sev).any(1).sum()),
        "mean_width95_FP": float(width[FPm].mean()), "mean_width95_healthy_declared_healthy": float(width[h & ~dec_mean].mean()),
        "tau_tradeoff_FP_FN": {f"{t:.2f}": fp_fn(mean, z, t) for t in (0.82, 0.84, 0.85, 0.86, 0.88)},
    }
    rec_path = os.path.join(data, "asbuilt_fields", "localized_damage_test_fine.npy")
    if a.split == "test_fine" and os.path.exists(rec_path):
        gap = {}
        for i, k, st, n, _ in np.load(rec_path):
            i, k, st, n = int(i), int(k), int(st), int(n)
            if k in END:
                gap[(i, k)] = 0.2 * (st if k == END[0] else 10 - st - n)   # distance damaged region - support [m]
        miss = [gap[(i, k)] for i in range(len(z)) for k in END if (FNm & sev)[i, k] and (i, k) in gap]
        breakdown["severe_end_zone_misses"] = len(miss)
        breakdown["severe_end_zone_misses_within_0.4m_of_support"] = int(np.sum(np.array(miss) < 0.4 + 1e-9))
    summary = {"run": os.path.basename(run), "split": a.split, "criterion": a.criterion if a.tau is None else "fixed",
               "pod_target": a.pod, "tau_pod_validation": tau_pod, "tau_eer_validation": tau_eer, "tau": tau, "pi_H": pi_h,
               "AUC": auc(-mean, z), "bootstrap_95": ci, "rules": rows, "error_probability_bins": bins,
               "breakdown": breakdown, "samples_per_structure": a.samples, "bootstrap_resamples": a.boot}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(pd_.DataFrame(rows)[["rule", "FP", "FP_rate", "FN", "FN_rate", "FP_predicted", "FN_predicted"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"AUC {summary['AUC']:.4f}; bootstrap 95 %: " + ", ".join(f"{k} [{v[0]:.4f}, {v[1]:.4f}]" for k, v in ci.items()))
    print(pd_.DataFrame(bins).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(json.dumps(breakdown, indent=1, default=float))
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
