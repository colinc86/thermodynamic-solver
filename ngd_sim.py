#!/usr/bin/env python3
"""Does the measured error model still allow natural gradient descent to beat SGD?

THE BRIDGE THIS BUILDS
    The breadboard measured how a thermodynamic linear solver's SYSTEMATIC error behaves as
    the matrix gets worse conditioned. The ladder's answer was p = 0: the floor does not grow
    with kappa, up to kappa = 21. The pre-registered alternative, p = 1, is what perturbation
    theory predicts and what the ladder excluded at 16.6 sigma.

    That result only matters if it changes something downstream. The downstream application
    thermodynamic linear algebra is funded for is natural gradient descent, which solves

        (F + lam I) d = g                      F = Fisher information matrix

    once per step. Fisher spectra are badly conditioned, so the damping lam is what brings the
    solved system inside a device's reach - and damping also degrades the step back toward
    plain gradient descent. Hence the question the paper poses:

        How much damping brings kappa_eff inside a given device's reach,
        and does NGD still beat SGD at that damping?

    This script answers it by running the optimisation, with the linear solve corrupted by the
    measured error model, on a model small enough that the Fisher is formed and inverted EXACTLY
    - no K-FAC, no block-diagonal approximation, no autodiff. 730 parameters, explicit backprop.

WHAT IS AND IS NOT CLAIMED
    Claimed: given an error model of the form eps_sys(kappa) = delta * kappa^p acting on the
    eigenvalues of the solved system, this is what NGD does. The p = 0 / p = 1 contrast is the
    contrast the ladder measured, and delta = 0.7 % is the coefficient it was measured at.

    NOT claimed: that a fabricated device has this board's delta (it will not - see the paper's
    Limitations), nor that a 2x2 result extrapolates to N = 730. What transfers is the EXPONENT,
    which is the thing the ladder was built to measure and the thing the runtime theory omits.

MODELLING CHOICES, STATED
    - The device returns the inverse with a relative error on each EIGENVALUE of the solved
      system: Sigma_hat = U diag(s_i (1 + e_i)) U^T, e_i ~ N(0, eps^2). That is the quantity the
      ladder measured (kappa_meas is a ratio of two eigenvalues of the measured covariance), so
      it is the quantity the measurement licenses us to perturb.
    - The e_i are drawn ONCE PER RUN, not per step. A systematic floor is a bias: a real device
      makes a repeatable error, and a bias that is resampled every step would average out over
      the optimisation and flatter the result. Fixed per run is the conservative choice.
    - The headline sweep is in the infinite-integration-time limit, so that ONLY the systematic
      floor acts. The statistical term sqrt(2 kappa tau / T) is reported separately, as the
      integration time a device would need for sampling error to fall below its own floor.

Run:  python3 ngd_sim.py            (writes ../images/fig-ngd.png and ../data/ngd-damping.csv)
      python3 ngd_sim.py --quick    (fewer steps / fewer lambdas, for a smoke test)
"""
import argparse
import gzip
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MNIST = os.environ.get("MNIST", os.path.join(HERE, "..", "data", "mnist"))
IMG = os.path.join(HERE, "images")
DAT = os.path.join(HERE, "data")

# ---------------------------------------------------------------- the measured error model
DELTA0 = 0.007       # systematic floor coefficient, measured at kappa 2.21 (error_model.py)
KAPPA0 = 2.21        # the kappa it was measured at
TAU_FAST = 11e-6     # s, the ladder network's fast relaxation time (kappa_ladder.py)


def eps_sys(kappa, p):
    """Pre-registered systematic floor: relative error on the solved system's eigenvalues."""
    return DELTA0 * (kappa / KAPPA0) ** p


def integration_time_for(kappa, p, margin=1.0):
    """Seconds of integration for the statistical error to fall to `margin` x the floor.

    eps_stat = sqrt(2 kappa tau / T)  (Aifer et al.), set equal to margin * eps_sys(kappa, p).
    """
    return 2 * kappa * TAU_FAST / (margin * eps_sys(kappa, p)) ** 2


# ---------------------------------------------------------------- data
def _idx(path):
    with gzip.open(path, "rb") as f:
        buf = f.read()
    n_dim = buf[3]
    dims = [int.from_bytes(buf[4 + 4 * i:8 + 4 * i], "big") for i in range(n_dim)]
    return np.frombuffer(buf, dtype=np.uint8, offset=4 + 4 * n_dim).reshape(dims)


def load_mnist(n_train=8192, n_test=4096, pool=4, seed=0):
    """MNIST, average-pooled 4x4 so that 28x28 -> 7x7 = 49 inputs, standardised.

    Pooling is what keeps the parameter count low enough to form the EXACT Fisher. It costs
    accuracy (the model tops out well below a full MLP) and that is fine: the question here is
    about the linear solve, not about the state of the art on MNIST.
    """
    xs = _idx(os.path.join(MNIST, "train-images-idx3-ubyte.gz")).astype(np.float64) / 255.0
    ys = _idx(os.path.join(MNIST, "train-labels-idx1-ubyte.gz")).astype(np.int64)
    xt = _idx(os.path.join(MNIST, "t10k-images-idx3-ubyte.gz")).astype(np.float64) / 255.0
    yt = _idx(os.path.join(MNIST, "t10k-labels-idx1-ubyte.gz")).astype(np.int64)
    rng = np.random.default_rng(seed)
    i, j = rng.permutation(len(xs))[:n_train], rng.permutation(len(xt))[:n_test]
    xs, ys, xt, yt = xs[i], ys[i], xt[j], yt[j]

    def prep(x):
        n, h, w = x.shape
        return x.reshape(n, h // pool, pool, w // pool, pool).mean((2, 4)).reshape(n, -1)

    a, b = prep(xs), prep(xt)
    mu, sd = a.mean(0), a.std(0) + 1e-6
    return (a - mu) / sd, ys, (b - mu) / sd, yt


# ---------------------------------------------------------------- model: D -> H -> C, tanh
class MLP:
    """Explicit forward / backward / logit-Jacobian. No autodiff, so every term is auditable."""

    def __init__(self, d, h, c, seed=0):
        rng = np.random.default_rng(seed)
        self.d, self.h, self.c = d, h, c
        self.shapes = [(d, h), (h,), (h, c), (c,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.n = sum(self.sizes)
        w1 = rng.normal(0, np.sqrt(1.0 / d), (d, h))
        w2 = rng.normal(0, np.sqrt(1.0 / h), (h, c))
        self.theta = np.concatenate([w1.ravel(), np.zeros(h), w2.ravel(), np.zeros(c)])

    def unpack(self, th):
        out, k = [], 0
        for s, sz in zip(self.shapes, self.sizes):
            out.append(th[k:k + sz].reshape(s))
            k += sz
        return out

    def forward(self, th, x):
        w1, b1, w2, b2 = self.unpack(th)
        a = x @ w1 + b1
        z = np.tanh(a)
        return z @ w2 + b2, z

    @staticmethod
    def softmax(logits):
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def loss_grad(self, th, x, y):
        """Mean cross-entropy and its exact gradient."""
        w1, b1, w2, b2 = self.unpack(th)
        logits, z = self.forward(th, x)
        p = self.softmax(logits)
        n = len(x)
        loss = -np.log(np.clip(p[np.arange(n), y], 1e-300, None)).mean()
        dl = p.copy()
        dl[np.arange(n), y] -= 1.0
        dl /= n
        gw2, gb2 = z.T @ dl, dl.sum(0)
        dz = (dl @ w2.T) * (1 - z ** 2)
        gw1, gb1 = x.T @ dz, dz.sum(0)
        return loss, np.concatenate([gw1.ravel(), gb1, gw2.ravel(), gb2])

    def logit_jacobian(self, th, x):
        """dlogits/dtheta, shape (N, C, n_params). Exact, vectorised."""
        w1, b1, w2, b2 = self.unpack(th)
        n, d, h, c = len(x), self.d, self.h, self.c
        logits, z = self.forward(th, x)
        dz = 1 - z ** 2                                     # (n, h)
        # d logit_k / d w2[:, k] = z ;  d logit_k / d b2[k] = 1 ;  both zero for the other logits
        jw2 = np.zeros((n, c, h, c))
        jb2 = np.zeros((n, c, c))
        for k in range(c):
            jw2[:, k, :, k] = z
            jb2[:, k, k] = 1.0
        # d logit_c / d a_h = w2[h, c] * (1 - z_h^2)
        da = w2.T[None, :, :] * dz[:, None, :]              # (n, c, h)
        jw1 = da[:, :, None, :] * x[:, None, :, None]       # (n, c, d, h)
        jb1 = da                                            # (n, c, h)
        return np.concatenate([jw1.reshape(n, c, d * h), jb1.reshape(n, c, h),
                               jw2.reshape(n, c, h * c), jb2.reshape(n, c, c)], axis=2)

    def fisher(self, th, x):
        """Exact Fisher = Gauss-Newton for softmax cross-entropy: (1/N) sum J^T (diag(p)-pp^T) J."""
        logits, _ = self.forward(th, x)
        p = self.softmax(logits)
        J = self.logit_jacobian(th, x)                      # (n, c, P)
        n = len(x)
        # A = (diag(p) - p p^T)^(1/2) applied to J, so F = (1/n) sum (A J)^T (A J)
        # use the exact factorisation via eigh of the small (c x c) blocks
        F = np.zeros((self.n, self.n))
        for i in range(n):
            Hi = np.diag(p[i]) - np.outer(p[i], p[i])
            w, V = np.linalg.eigh(Hi)
            w = np.clip(w, 0, None)
            B = (V * np.sqrt(w)).T @ J[i]                   # (c, P)
            F += B.T @ B
        return F / n


# ---------------------------------------------------------------- the device
def device_solve(A, g, p_exp, rng, e_fixed=None, exact=False):
    """Solve A d = g the way a thermodynamic computer would, with the measured error model.

    The device returns the INVERSE (a covariance), not a factorisation, so the error lands on
    the inverse's eigenvalues. kappa_eff is the condition number of the matrix it is asked to
    invert - the x-axis of the ladder.
    """
    w, U = np.linalg.eigh(A)
    w = np.clip(w, 1e-300, None)
    kappa = w.max() / w.min()
    if exact:
        return U @ ((U.T @ g) / w), kappa
    eps = eps_sys(kappa, p_exp)
    e = e_fixed if e_fixed is not None else rng.normal(0, 1.0, len(w))
    inv = (1.0 / w) * (1.0 + eps * e)
    inv = np.clip(inv, 0.0, None)          # a covariance the device returns is PSD by construction
    return U @ (inv * (U.T @ g)), kappa


# ---------------------------------------------------------------- training
def run(net, data, method, lam=0.0, p_exp=0.0, lr=0.1, steps=300, batch=256,
        fisher_every=10, seed=1):
    """One optimisation run. method in {'sgd', 'ngd'}; 'ngd' with p_exp=None means exact solve."""
    xtr, ytr, xte, yte = data
    rng = np.random.default_rng(seed)
    th = net.theta.copy()
    e_fixed = rng.normal(0, 1.0, net.n) if (method == "ngd" and p_exp is not None) else None
    F = None
    hist, kap = [], np.nan
    for t in range(steps):
        idx = rng.integers(0, len(xtr), batch)
        xb, yb = xtr[idx], ytr[idx]
        loss, g = net.loss_grad(th, xb, yb)
        hist.append(loss)
        if method == "sgd":
            d = g
        else:
            if F is None or t % fisher_every == 0:
                F = net.fisher(th, xb)
            A = F + lam * np.eye(net.n)
            d, k = device_solve(A, g, p_exp, rng, e_fixed, exact=(p_exp is None))
            if t == 0:
                kap = k        # at the INITIAL point: a diverged run collapses F and would
                               # otherwise report a meaningless kappa_eff near 1
        th = th - lr * d
        if not np.all(np.isfinite(th)) or loss > 50:
            return dict(diverged=True, final=np.inf, acc=0.0, kappa=kap, hist=hist)
    ltr, _ = net.loss_grad(th, xtr, ytr)
    logits, _ = net.forward(th, xte)
    acc = float((logits.argmax(1) == yte).mean())
    return dict(diverged=False, final=float(ltr), acc=acc, kappa=float(kap), hist=hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--from-csv", action="store_true",
                    help="redraw the figure from the committed CSV; do not re-run the sweep")
    a = ap.parse_args()
    if a.from_csv:
        return refigure()
    steps = 120 if a.quick else a.steps
    lams = [1e-4, 1e-3, 1e-2] if a.quick else [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    seeds = 1 if a.quick else a.seeds

    data = load_mnist()
    net = MLP(data[0].shape[1], a.hidden, 10, seed=0)
    print(f"model {data[0].shape[1]} -> {a.hidden} -> 10, {net.n} parameters, exact Fisher")
    print(f"{len(data[0])} train / {len(data[2])} test, {steps} steps, {seeds} seed(s)\n")

    # SGD baseline: tune the learning rate, same step budget, best of a grid. The grid is wide
    # enough that the optimum is interior - a baseline tuned to its boundary is not a baseline.
    best = None
    for lr in (0.1, 0.3, 1.0, 3.0, 10.0):
        r = [run(net, data, "sgd", lr=lr, steps=steps, seed=s + 1) for s in range(seeds)]
        f = float(np.mean([x["final"] for x in r]))
        sd = float(np.std([x["final"] for x in r], ddof=1)) if seeds > 1 else 0.0
        print(f"  sgd lr {lr:<5} loss {f:.4f} +/- {sd:.4f}  acc {np.mean([x['acc'] for x in r]):.4f}")
        if best is None or f < best[1]:
            best = (lr, f, float(np.mean([x["acc"] for x in r])), sd)
    print(f"  -> SGD baseline lr {best[0]}, loss {best[1]:.4f} +/- {best[3]:.4f}, "
          f"acc {best[2]:.4f}\n")

    # For each lambda the learning rate is tuned on the EXACT solver and then reused for the two
    # device conditions, so that the only difference between the three is the solve error.
    ngd_lrs = (0.01, 0.03, 0.1, 0.3, 1.0)
    rows = []
    for lam in lams:
        tuned = None
        for lr in ngd_lrs:
            r = [run(net, data, "ngd", lam=lam, p_exp=None, lr=lr, steps=steps, seed=s + 1)
                 for s in range(seeds)]
            f = float(np.mean([x["final"] for x in r]))
            if tuned is None or f < tuned[1]:
                tuned = (lr, f, float(np.mean([x["acc"] for x in r])),
                         float(np.nanmean([x["kappa"] for x in r])),
                         float(np.std([x["final"] for x in r], ddof=1)) if seeds > 1 else 0.0)
        lr, out = tuned[0], {"exact": (tuned[1], tuned[2], tuned[3], tuned[4])}
        for tag, pe in (("p0", 0.0), ("p1", 1.0)):
            r = [run(net, data, "ngd", lam=lam, p_exp=pe, lr=lr, steps=steps, seed=s + 1)
                 for s in range(seeds)]
            out[tag] = (float(np.mean([x["final"] for x in r])),
                        float(np.mean([x["acc"] for x in r])), np.nan,
                        float(np.std([x["final"] for x in r], ddof=1)) if seeds > 1 else 0.0)
        k = out["exact"][2]
        rows.append([lam, k, eps_sys(k, 0.0), eps_sys(k, 1.0),
                     out["exact"][0], out["p0"][0], out["p1"][0],
                     out["exact"][3], out["p0"][3], out["p1"][3],
                     out["exact"][1], out["p0"][1], out["p1"][1],
                     integration_time_for(k, 0.0), lr])
        print(f"  lam {lam:<8g} kappa_eff {k:10.3g}  lr {lr:<5} loss exact {out['exact'][0]:8.4f} "
              f"p0 {out['p0'][0]:8.4f} p1 {out['p1'][0]:10.4f}   "
              f"acc {out['exact'][1]:.4f}/{out['p0'][1]:.4f}/{out['p1'][1]:.4f}")

    os.makedirs(DAT, exist_ok=True)
    hdr = ("lambda,kappa_eff,eps_sys_p0,eps_sys_p1,loss_exact,loss_p0,loss_p1,"
           "sd_exact,sd_p0,sd_p1,acc_exact,acc_p0,acc_p1,integration_time_s_p0,lr_tuned")
    with open(os.path.join(DAT, "ngd-damping.csv"), "w") as f:
        f.write(f"# Damped NGD on a {net.n}-parameter MLP (MNIST, 4x4-pooled to 49 inputs), EXACT "
                f"Fisher, {steps} steps, mean over {seeds} seeds (sd_* are the seed spread).\n")
        f.write(f"# SGD baseline at the same step budget: loss {best[1]:.4f} +/- {best[3]:.4f}, "
                f"acc {best[2]:.4f} (lr {best[0]}, best of a 5-point grid).\n")
        f.write(f"# eps_sys = {DELTA0} (kappa_eff/{KAPPA0})^p, the ladder's pre-registered form, "
                f"applied to the eigenvalues of the inverse the device returns. p=0 is a flat floor, "
                f"which the ladder allows; p=1 is the kappa^1 floor it ruled out.\n")
        f.write(f"# lr_tuned is tuned on the EXACT solver per lambda and reused for p0/p1, so the "
                f"only difference between the three columns is the solve error.\n")
        f.write(hdr + "\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" for v in r) + "\n")
    print(f"\nwrote {os.path.join(DAT, 'ngd-damping.csv')}")
    figure(rows, best, net, steps, seeds)
    return rows, best, net


def refigure():
    """Redraw fig-ngd.png from the committed CSV, so a plotting change needs no 45-minute sweep."""
    path = os.path.join(DAT, "ngd-damping.csv")
    hdr, rows = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                hdr.append(line)
                continue
            if line.startswith("lambda"):
                continue
            rows.append([float(v) for v in line.strip().split(",")])
    base = [h for h in hdr if "SGD baseline" in h][0]
    loss = float(base.split("loss")[1].split("+/-")[0])
    sd = float(base.split("+/-")[1].split(",")[0])
    steps = int([h for h in hdr if "steps" in h][0].split("Fisher,")[1].split("steps")[0])
    n = int(hdr[0].split("-parameter")[0].split()[-1])
    figure(rows, (None, loss, None, sd), type("N", (), {"n": n}), steps, 3)


def figure(rows, best, net, steps, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = np.array(rows, dtype=float)
    lam, kap = a[:, 0], a[:, 1]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6))
    finite = a[:, 4:7][np.isfinite(a[:, 4:7])]
    lo, hi = finite.min(), finite.max()
    y_div = hi + 0.12 * (hi - lo)          # a clear lane above every real point
    for col, sd, c, ls, lab in ((4, 7, "#1a1a1a", "-", "exact solve"),
                                (5, 8, "#4a6fa5", "-", r"device, $p=0$  (flat floor)"),
                                (6, 9, "#c1440e", "--", r"device, $p=1$  (ruled out by the ladder)")):
        y, e = a[:, col].copy(), a[:, sd].copy()
        ok = np.isfinite(y)
        ax.errorbar(kap[ok], y[ok], yerr=e[ok], fmt="o" + ls, color=c, ms=5, lw=1.6,
                    capsize=3, label=lab)
        if (~ok).any():           # diverged runs get their own lane, not a fake y value
            ax.plot(kap[~ok], np.full((~ok).sum(), y_div), "x", color=c, ms=9, mew=2)
    if not np.isfinite(a[:, 6]).all():
        ax.text(kap[~np.isfinite(a[:, 6])].min() * 0.55, y_div, "diverged",
                fontsize=8.5, color="#c1440e", va="center", ha="right")
    ax.set_ylim(lo - 0.06 * (hi - lo), y_div + 0.06 * (hi - lo))
    ax.axhline(best[1], color="#888", ls=":", lw=1.8, label=f"tuned SGD, same budget")
    ax.axhspan(best[1] - best[3], best[1] + best[3], color="#888", alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.set_xlabel(r"$\kappa_\mathrm{eff}$ of the solved system $(F+\lambda I)$")
    ax.set_ylabel(f"training loss after {steps} steps")
    ax.set_title(r"Training loss against $\kappa_\mathrm{eff}$")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25, which="both")

    ax2.loglog(kap, 100 * a[:, 2], "-", color="#4a6fa5", lw=1.8, label=r"$p=0$ (flat): 0.7 %")
    ax2.loglog(kap, 100 * a[:, 3], "--", color="#c1440e", lw=1.8, label=r"$p=1$: $0.7\,\%\times\kappa/2.21$")
    ax2.set_xlabel(r"$\kappa_\mathrm{eff}$"); ax2.set_ylabel("systematic error on the solve [%]")
    ax2.set_title(r"Systematic error on the solve against $\kappa_\mathrm{eff}$")
    ax2.axhline(100, color="#888", ls=":", lw=1.2)
    ax2.text(kap.min() * 1.2, 115, "100 % — the step is noise", fontsize=8, color="#666")
    ax2.legend(frameon=False, fontsize=9); ax2.grid(alpha=0.25, which="both")
    fig.tight_layout()
    os.makedirs(IMG, exist_ok=True)
    fig.savefig(os.path.join(IMG, "fig-ngd.png"), dpi=130)
    print(f"wrote {os.path.join(IMG, 'fig-ngd.png')}")


if __name__ == "__main__":
    main()
