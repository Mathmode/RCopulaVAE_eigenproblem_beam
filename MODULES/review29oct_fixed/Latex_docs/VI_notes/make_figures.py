#!/usr/bin/env python3
"""
Figures of the background notes VI_notes.tex (no data, no model: only the formulas of the notes).
Usage: python make_figures.py   (writes figures/*.pdf next to this script)
"""
import os
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm", "pdf.fonttype": 42})
LB, UB = 0.45, 1.0
rng = np.random.default_rng(1)


# ----------------------------------------------------------------------------- marginal families on [LB, UB]
class TruncGauss:
    """Gaussian N(mu, s^2) truncated to [LB, UB]."""
    def __init__(self, mu, s):
        self.mu, self.s = mu, s
        self.a, self.b = (LB - mu) / s, (UB - mu) / s
        self.Z = norm.cdf(self.b) - norm.cdf(self.a)
    def pdf(self, z):
        return np.where((z >= LB) & (z <= UB), norm.pdf((z - self.mu) / self.s) / (self.s * self.Z), 0.0)
    def cdf(self, z):
        return np.clip((norm.cdf((np.clip(z, LB, UB) - self.mu) / self.s) - norm.cdf(self.a)) / self.Z, 0, 1)
    def ppf(self, u):
        return self.mu + self.s * norm.ppf(norm.cdf(self.a) + u * self.Z)


class TruncLogNormal:
    """ln z ~ N(m, s^2) truncated to [ln LB, ln UB]."""
    def __init__(self, m, s):
        self.w = TruncGaussLog(m, s)
    def pdf(self, z):
        zc = np.clip(z, 1e-9, None)
        return self.w.pdf(np.log(zc)) / zc
    def cdf(self, z):
        return self.w.cdf(np.log(np.clip(z, 1e-9, None)))
    def ppf(self, u):
        return np.exp(self.w.ppf(u))


class TruncGaussLog(TruncGauss):
    """helper: Gaussian truncated to [ln LB, ln UB]"""
    def __init__(self, m, s):
        self.mu, self.s = m, s
        self.a, self.b = (np.log(LB) - m) / s, (np.log(UB) - m) / s
        self.Z = norm.cdf(self.b) - norm.cdf(self.a)
    def pdf(self, w):
        return np.where((w >= np.log(LB)) & (w <= np.log(UB)), norm.pdf((w - self.mu) / self.s) / (self.s * self.Z), 0.0)
    def cdf(self, w):
        return np.clip((norm.cdf((np.clip(w, np.log(LB), np.log(UB)) - self.mu) / self.s) - norm.cdf(self.a)) / self.Z, 0, 1)


class Mixture:
    """Truncated mixture sum_k w_k TN_k: pdf and cdf in closed form, quantile by bisection."""
    def __init__(self, comps, weights):
        self.c, self.w = comps, np.asarray(weights) / np.sum(weights)
    def pdf(self, z):
        return sum(w * c.pdf(z) for w, c in zip(self.w, self.c))
    def cdf(self, z):
        return sum(w * c.cdf(z) for w, c in zip(self.w, self.c))
    def ppf(self, u, iters=60):
        lo, hi = np.full_like(u, LB), np.full_like(u, UB)
        for _ in range(iters):                      # F is increasing: bisection always converges
            mid = 0.5 * (lo + hi)
            left = self.cdf(mid) < u
            lo, hi = np.where(left, mid, lo), np.where(left, hi, mid)
        return 0.5 * (lo + hi)


class Kumaraswamy:
    """Kumaraswamy(a, b) on x = (z - LB) / (UB - LB): closed-form pdf, cdf and quantile."""
    def __init__(self, a, b):
        self.a, self.b = a, b
    def pdf(self, z):
        x = np.clip((z - LB) / (UB - LB), 1e-12, 1 - 1e-12)
        return self.a * self.b * x ** (self.a - 1) * (1 - x ** self.a) ** (self.b - 1) / (UB - LB)
    def cdf(self, z):
        x = np.clip((z - LB) / (UB - LB), 0, 1)
        return 1 - (1 - x ** self.a) ** self.b
    def ppf(self, u):
        return LB + (UB - LB) * (1 - (1 - u) ** (1 / self.b)) ** (1 / self.a)


# ----------------------------------------------------------------------------- Gaussian copula
def copula_logdensity(u1, u2, rho):
    y1, y2 = norm.ppf(u1), norm.ppf(u2)
    det = 1 - rho ** 2
    return -0.5 * np.log(det) - (rho ** 2 * (y1 ** 2 + y2 ** 2) - 2 * rho * y1 * y2) / (2 * det)


def copula_samples(n, rho):
    L = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho ** 2)]])
    eps = rng.standard_normal((n, 2))
    y = eps @ L.T
    return norm.cdf(y)


# ============================================================================= Fig. 1: inverse-CDF sampling
tn = TruncGauss(0.93, 0.08)
zz = np.linspace(LB, UB, 600)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
ax[0].plot(zz, norm.pdf((zz - tn.mu) / tn.s) / tn.s, "--", color="0.5", lw=1, label=r"$\mathcal{N}(\mu,\sigma^2)$ (not truncated)")
ax[0].plot(zz, tn.pdf(zz), color="#1d4e89", lw=1.5, label=r"truncated, $q_d(z)$")
ax[0].axvline(UB, color="k", lw=0.6); ax[0].axvline(LB, color="k", lw=0.6)
ax[0].set_xlabel("$z$"); ax[0].set_ylabel("density"); ax[0].set_title(r"(a) density: $\varphi(\xi)/(\sigma Z)$")
ax[0].legend(fontsize=6.5, frameon=False, loc="upper left")
ax[1].plot(zz, tn.cdf(zz), color="#1d4e89", lw=1.5)
for u in (0.15, 0.5, 0.85):
    z = tn.ppf(np.array([u]))[0]
    ax[1].annotate("", xy=(z, u), xytext=(LB, u), arrowprops=dict(arrowstyle="-", color="#b03a2e", lw=0.8))
    ax[1].annotate("", xy=(z, 0), xytext=(z, u), arrowprops=dict(arrowstyle="->", color="#b03a2e", lw=0.8))
ax[1].set_xlabel("$z$"); ax[1].set_ylabel("$F_d(z)$"); ax[1].set_title(r"(b) $z=F_d^{-1}(u)$")
ax[1].set_xlim(LB, UB); ax[1].set_ylim(0, 1.02)
u = rng.uniform(size=20000)
ax[2].hist(tn.ppf(u), bins=60, density=True, color="#a9cbe8", edgecolor="none", label="samples $F_d^{-1}(u)$")
ax[2].plot(zz, tn.pdf(zz), color="#1d4e89", lw=1.2, label="$q_d(z)$")
ax[2].set_xlabel("$z$"); ax[2].set_title("(c) 20,000 samples")
ax[2].legend(fontsize=6.5, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "inverse_cdf.pdf"))
plt.close(fig)

# ============================================================================= Fig. 2: copula with four marginal families
rho = -0.75
fams = [
    ("(a) trunc. Gaussian", TruncGauss(0.80, 0.10), TruncGauss(0.76, 0.09)),
    ("(b) trunc. log-normal", TruncLogNormal(np.log(0.80), 0.12), TruncLogNormal(np.log(0.76), 0.12)),
    ("(c) trunc. Gaussian mixture", Mixture([TruncGauss(0.62, 0.05), TruncGauss(0.85, 0.05)], [0.4, 0.6]),
     TruncGauss(0.76, 0.09)),
    ("(d) Kumaraswamy mixture", Mixture([Kumaraswamy(2.0, 8.0), Kumaraswamy(9.0, 1.5)], [0.45, 0.55]),
     Kumaraswamy(3.0, 2.5)),
]
g = np.linspace(LB + 1e-4, UB - 1e-4, 220)
G1, G2 = np.meshgrid(g, g)
fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.9), gridspec_kw=dict(height_ratios=[1.6, 1]))
for k, (title, q1, q2) in enumerate(fams):
    u1, u2 = np.clip(q1.cdf(G1), 1e-9, 1 - 1e-9), np.clip(q2.cdf(G2), 1e-9, 1 - 1e-9)
    dens = np.exp(copula_logdensity(u1, u2, rho)) * q1.pdf(G1) * q2.pdf(G2)
    a = ax[0, k]
    a.contourf(G1, G2, dens, levels=14, cmap="Blues")
    U = copula_samples(300, rho)
    a.plot(q1.ppf(U[:, 0]), q2.ppf(U[:, 1]), ".", ms=1.6, color="#b03a2e", alpha=0.7)
    a.set_xlim(LB, UB); a.set_ylim(LB, UB); a.set_title(title, fontsize=8)
    a.set_xlabel("$z_1$"); a.set_aspect("equal")
    if k == 0:
        a.set_ylabel("$z_2$")
    U = copula_samples(20000, rho)
    b = ax[1, k]
    b.hist(q1.ppf(U[:, 0]), bins=50, range=(LB, UB), density=True, color="#a9cbe8", edgecolor="none")
    b.plot(g, q1.pdf(g), color="#1d4e89", lw=1.1)
    b.set_xlim(LB, UB); b.set_xlabel("$z_1$")
    if k == 0:
        b.set_ylabel(r"$q_1(z_1)$")
fig.tight_layout(w_pad=0.6, h_pad=0.4)
fig.savefig(os.path.join(OUT, "copula_marginals.pdf"))
plt.close(fig)

# ============================================================================= Fig. 3: tempering and reverse KL
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
z = np.linspace(LB, UB, 800)
loglik = -0.5 * ((z - 0.72) / 0.06) ** 2               # Gaussian likelihood in z (illustration)
for gam, ls in ((0.5, ":"), (1.0, "-"), (2.0, "--")):
    p = np.exp(loglik / gam ** 2)
    p /= np.trapezoid(p, z)
    ax[0].plot(z, p, ls, color="#1d4e89", lw=1.3, label=rf"$\gamma={gam:g}$")
ax[0].set_xlabel("$z$"); ax[0].set_ylabel("density")
ax[0].set_title(r"(a) $p_\gamma(z|\mathbf{m})\propto p(\mathbf{m}|z)^{1/\gamma^2}p(z)$")
ax[0].legend(frameon=False, fontsize=7)
# bimodal target, best Gaussian under reverse KL (mode seeking) and forward KL (mass covering), on a grid
x = np.linspace(-3, 7.5, 4001)
target = 0.6 * norm.pdf(x, 0, 0.5) + 0.4 * norm.pdf(x, 4.5, 0.5)
logt = np.log(target + 1e-300)
best, bestv = None, np.inf
for m in np.linspace(-1, 5.5, 261):
    for s in np.linspace(0.3, 2.5, 111):
        q = norm.pdf(x, m, s)
        v = np.trapezoid(q * (np.log(q + 1e-300) - logt), x)
        if v < bestv:
            best, bestv = (m, s), v
mf = np.trapezoid(x * target, x); sf = np.sqrt(np.trapezoid((x - mf) ** 2 * target, x))
ax[1].fill_between(x, target, color="0.85", label=r"target $p$")
ax[1].plot(x, norm.pdf(x, *best), color="#1d4e89", lw=1.3, label=r"$\arg\min_q \mathrm{KL}(q\,\|\,p)$ (VI)")
ax[1].plot(x, norm.pdf(x, mf, sf), "--", color="#b03a2e", lw=1.2, label=r"$\arg\min_q \mathrm{KL}(p\,\|\,q)$")
ax[1].set_xlabel("$z$"); ax[1].set_title("(b) reverse vs. forward KL, Gaussian $q$")
ax[1].legend(frameon=False, fontsize=6.5, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tempering_kl.pdf"))
plt.close(fig)
print("reverse-KL optimum (m, s) =", np.round(best, 3), "| moment match", round(mf, 3), round(sf, 3))

# ============================================================================= Fig. 4: worked examples of Section 3
# Example 3.6: y = a z + eps, eps ~ N(0, sig^2), z ~ U[LB, UB]  ->  posterior = N(y/a, (sig/a)^2) truncated to [LB, UB]
a_s, sig = 0.5, 0.025
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35))
z = np.linspace(0.35, 1.1, 1500)
for k, (zhat, title) in enumerate(((0.75, "(a) $y/a=0.75$ (damaged)"), (0.98, "(b) $y/a=0.98$ (near the bound)"))):
    lik = np.exp(-0.5 * ((z - zhat) / (sig / a_s)) ** 2)                     # likelihood as a function of z
    prior = np.where((z >= LB) & (z <= UB), 1.0 / (UB - LB), 0.0)
    post = lik * prior
    post /= np.trapezoid(post, z)
    lik_n = lik / np.trapezoid(lik, z)                                          # normalised only for display
    ax[k].plot(z, prior, color="0.55", lw=1.0, label="prior $p(z)$")
    ax[k].plot(z, lik_n, "--", color="#b03a2e", lw=1.1, label=r"likelihood $p(y|z)$ (scaled)")
    ax[k].plot(z, post, color="#1d4e89", lw=1.5, label="posterior $p(z|y)$")
    ax[k].set_xlabel("$z$"); ax[k].set_title(title, fontsize=8.5); ax[k].set_xlim(0.35, 1.1)
ax[0].set_ylabel("density"); ax[0].legend(frameon=False, fontsize=6.3, loc="upper left")
# Example 3.8: y = a1 z1 + a2 z2 + eps, uniform prior on the box -> ridge
a1, a2, y = 0.5, 0.3, 0.5 * 0.75 + 0.3 * 0.85
g = np.linspace(LB, UB, 400)
Z1, Z2 = np.meshgrid(g, g)
post2 = np.exp(-0.5 * ((y - a1 * Z1 - a2 * Z2) / sig) ** 2)
ax[2].contourf(Z1, Z2, post2, levels=12, cmap="Blues")
ax[2].plot(g, (y - a1 * g) / a2, "--", color="#b03a2e", lw=0.9, label=r"$a_1z_1+a_2z_2=y$")
ax[2].set_xlim(LB, UB); ax[2].set_ylim(LB, UB); ax[2].set_aspect("equal")
ax[2].set_xlabel("$z_1$"); ax[2].set_ylabel("$z_2$"); ax[2].set_title("(c) two unknowns, one measurement", fontsize=8.5)
ax[2].legend(frameon=False, fontsize=6.3, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "examples.pdf"))
plt.close(fig)
print("figures written to", OUT)
