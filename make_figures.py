#!/usr/bin/env python3
"""Regenerate every figure and CSV in this writeup from the raw bench captures.

The raw captures are ~4 GB of .npz and are NOT shipped with this repo. This script is
here so the path from raw samples to every published number is auditable: point RAW at a
directory of captures and everything in images/ and data/ is rebuilt.

Each capture is a numpy archive with keys:
    dt        sample interval [s]
    ch1, ch2  node 1 and node 2 voltages [V]

Capture sets used:
    gnd2_1..30  FINAL configuration - matched capacitors, probe grounds tied to the
                shield, thermally settled. Every record passed the scope cross-check
                on both channels.
    plat_01..30 the SAME board immediately before the probe grounds were tied to the
                shield. Used only for the ground-loop comparison in the floor figure.
    v10k_1..4   R3 OUT, 10 kohm source   } the 4kT pair, taken 2026-09-11 on the
    r1k_1..4    R3 OUT, 1 kohm source    } original board (C1 = 26.8 nF, unshielded)

Run:  python3 make_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("RAW", os.path.join(HERE, "..", "data"))
IMG = os.path.join(HERE, "images")
CSV = os.path.join(HERE, "data")
DPI = 130

# ---------------------------------------------------------------- the board, as measured
# Every value below came off the DMM with the part out of circuit. Nothing is nominal.
K_B = 1.380649e-23
T_K = 297.0                     # bench temperature, assumed (+/- 3 K = 1 % on 4kT)
RN1, RN2 = 4690.0, 4680.0       # R1, R2  - source resistors into each node
RC = 6830.0                     # R3      - the coupling resistor (nominal 6.8k)
RG = 9900.0                     # R4      - node 2 to ground
C1, C2 = 23.50e-9, 23.51e-9     # MATCHED pair, 0.04 % apart, chosen from a batch of nine
C1_ORIG = 26.80e-9              # the capacitor in the C1 slot when the 4kT pair was taken
R11, R12 = 9883.0, 9911.0       # the Johnson sources themselves
EN, IN_ = 5.2e-9, 165e-15       # OPA2189 voltage / current noise (datasheet)
EFB = 3.86e-9                   # 4kT(R7||R9), the stage-1 feedback network
# Gain-setting resistors, measured out of circuit. The front-end gain is therefore
# COMPUTED, not fitted - there is no free parameter anywhere in this model.
R7, R9, R13, R15 = 9944.0, 991.0, 99650.0, 987.2      # channel A
R8, R10, R14, R16 = 10081.0, 1003.5, 99680.0, 983.8   # channel B
GAIN_A = (1 + R7 / R9) * (1 + R13 / R15)              # 1124.9
GAIN_B = (1 + R8 / R10) * (1 + R14 / R16)             # 1130.2
BAND = (200.0, 50000.0)         # analysis band - see README Limitations, "Band-limited"
NSEG = 1 << 20

G = np.array([[1 / RN1 + 1 / RC, -1 / RC],
              [-1 / RC, 1 / RN2 + 1 / RC + 1 / RG]])
CM = np.diag([C1, C2])


def drive_input_referred(r_source):
    """Total input-referred noise density [V/rtHz] for a given source resistance."""
    return np.sqrt(4 * K_B * T_K * r_source + EN ** 2 + EFB ** 2 + (IN_ * r_source) ** 2)


SI1 = (drive_input_referred(R11) * GAIN_A / RN1) ** 2   # injected current PSD, node 1
SI2 = (drive_input_referred(R12) * GAIN_B / RN2) ** 2   # injected current PSD, node 2


def node_psd(freqs, si1=SI1, si2=SI2, coupled=True):
    """Two-sided-equivalent node PSDs [V^2/Hz] from injected current PSDs si1, si2."""
    g = G if coupled else np.array([[1 / RN1, 0.0], [0.0, 1 / RN2 + 1 / RG]])
    out = np.empty((len(freqs), 2))
    d = np.diag([si1, si2])
    for i, f in enumerate(freqs):
        mi = np.linalg.inv(g + 2j * np.pi * f * CM)
        s = mi @ d @ mi.conj().T
        out[i] = (s[0, 0].real, s[1, 1].real)
    return out


def sigma_band(lo, hi, si1=SI1, si2=SI2, n=40000):
    """Predicted covariance matrix integrated over [lo, hi]."""
    ff = np.linspace(lo, hi, n)
    s = np.zeros((2, 2))
    d = np.diag([si1, si2])
    for f in ff:
        mi = np.linalg.inv(G + 2j * np.pi * f * CM)
        s += np.real(mi @ d @ mi.conj().T)
    return s * (ff[1] - ff[0])


def tau_slow():
    """Slowest relaxation time of the coupled network [s]."""
    return 1.0 / np.linalg.eigvals(np.linalg.solve(CM, G)).real.min()


def kappa():
    lam = np.linalg.eigvals(np.linalg.solve(CM, G)).real
    return lam.max() / lam.min()


def identity_offdiag(s):
    """Off-diagonal of Sigma.G as a percentage of its diagonal. Zero if Sigma = c G^-1."""
    m = s @ G
    return m[0, 1] / np.sqrt(abs(m[0, 0] * m[1, 1])) * 100


def lyap_offdiag(s):
    """Normalised off-diagonal of G S C + C S G. Zero for ANY diagonal drive."""
    r = G @ s @ CM + CM @ s @ G
    return r[0, 1] / np.sqrt(abs(r[0, 0] * r[1, 1])) * 100


# ---------------------------------------------------------------- io
def load(name):
    d = np.load(os.path.join(RAW, name))
    return float(d["dt"]), d["ch1"].astype(float), d["ch2"].astype(float)


def welch(x, dt, nseg=NSEG):
    w = np.hanning(nseg)
    acc, n = None, 0
    for i in range(0, len(x) - nseg + 1, nseg // 2):
        s = np.abs(np.fft.rfft((x[i:i + nseg] - x[i:i + nseg].mean()) * w)) ** 2
        acc = s if acc is None else acc + s
        n += 1
    psd = acc / n * 2 * dt / (w ** 2).sum()
    return np.fft.rfftfreq(nseg, dt), psd, n


def pooled_psd(names, ch):
    acc, f, dt = None, None, None
    for nm in names:
        dt, x, y = load(nm)
        f, p, _ = welch(x if ch == 1 else y, dt)
        acc = p if acc is None else acc + p
    return f, acc / len(names)


def bandpass(v, dt, lo, hi):
    f = np.fft.rfftfreq(len(v), dt)
    V = np.fft.rfft(v - v.mean())
    V[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(V, len(v))


def band_limited(names):
    """Every record of a set, band-limited, as two lists."""
    X, Y = [], []
    for nm in names:
        dt, x, y = load(nm)
        X.append(bandpass(x, dt, *BAND))
        Y.append(bandpass(y, dt, *BAND))
    return X, Y, dt


def write_csv(name, header, rows, note):
    path = os.path.join(CSV, name)
    with open(path, "w") as fh:
        fh.write(f"# {note}\n{header}\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    print(f"  wrote {name} ({len(rows)} rows)")


FINAL = [f"gnd2_{i}.npz" for i in range(1, 31)]
BEFORE_REGROUND = [f"plat_{i:02d}.npz" for i in range(1, 31)]
STYLE = dict(color="#1a1a1a", lw=1.2)
ACC = "#c1440e"
BLUE = "#4a6fa5"


# ---------------------------------------------------------------- 1. the excess
def excess_over_budget():
    """How much louder the board is than the datasheet noise budget allows.

    With the gain measured there is nothing left to fit, so this is a residual, not a
    knob: the ratio of the measured node spectrum to the model, and the extra
    input-referred noise density needed to explain it.
    """
    f, pm = pooled_psd(FINAL, 1)
    b = (f >= 300) & (f < 40000)
    ratio = float(np.sqrt(np.median(pm[b] / node_psd(f[b])[:, 0])))
    modelled = drive_input_referred(R11)
    implied = modelled * ratio
    extra = np.sqrt(max(implied ** 2 - modelled ** 2, 0))
    return ratio, modelled, implied, extra


# ---------------------------------------------------------------- 2. results table
def results():
    X, Y, _ = band_limited(FINAL)
    sp = sigma_band(*BAND)
    rows = []
    for x, y in zip(X, Y):
        s = np.cov(np.vstack([x, y]))
        rows.append([np.sqrt(s[0, 0]) * 1e3, np.sqrt(s[1, 1]) * 1e3,
                     np.sqrt(s[0, 0] / s[1, 1]), s[0, 1] / np.sqrt(s[0, 0] * s[1, 1]),
                     identity_offdiag(s), lyap_offdiag(s)])
    a = np.array(rows)
    # pooled estimate over all records - the best single number for each quantity
    A, B = np.concatenate(X), np.concatenate(Y)
    sP = np.cov(np.vstack([A, B]))
    pooled = [np.sqrt(sP[0, 0]) * 1e3, np.sqrt(sP[1, 1]) * 1e3, np.sqrt(sP[0, 0] / sP[1, 1]),
              sP[0, 1] / np.sqrt(sP[0, 0] * sP[1, 1]), identity_offdiag(sP), lyap_offdiag(sP)]
    preds = [np.sqrt(sp[0, 0]) * 1e3, np.sqrt(sp[1, 1]) * 1e3, np.sqrt(sp[0, 0] / sp[1, 1]),
             sp[0, 1] / np.sqrt(sp[0, 0] * sp[1, 1]), identity_offdiag(sp), lyap_offdiag(sp)]
    labels = ["v1_mV", "v2_mV", "v1_over_v2", "rho12", "identity_offdiag_pct", "lyapunov_offdiag_pct"]
    n = len(rows)
    out = []
    for j, lab in enumerate(labels):
        m, se, pr = pooled[j], a[:, j].std(ddof=1) / np.sqrt(n), preds[j]
        if lab.endswith("_pct"):
            err = f"{m - pr:+.2f} pp"       # a percentage OF a near-zero quantity is meaningless
        else:
            err = f"{100*(m-pr)/abs(pr):+.1f} %"
        out.append([lab, f"{m:.4f}", f"{se:.4f}", f"{pr:.4f}", err])
    write_csv("results-summary.csv", "quantity,measured,std_err,predicted,deviation", out,
              f"Final configuration, band {BAND[0]:.0f}-{BAND[1]:.0f} Hz, {n} records of 0.6 s "
              f"pooled (18 s). std_err is the standard error of the mean across records. "
              f"Gain is COMPUTED from measured resistors (A {GAIN_A:.0f}, B {GAIN_B:.0f}), "
              f"capacitors are a matched pair - nothing in this model is free. The node voltages "
              f"v1 and v2 carry a real residual (the board is louder than its noise budget); the "
              f"scale-free quantities are immune to that.")
    return a, pooled, preds


# ---------------------------------------------------------------- 3. spectra vs model
def fig_psd():
    f, p1 = pooled_psd(FINAL, 1)
    _, p2 = pooled_psd(FINAL, 2)
    b = (f >= 150) & (f <= 60000)
    pred = node_psd(f[b])
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2.6, 1.0], hspace=0.08))
    ax.loglog(f[b], np.sqrt(p1[b]) * 1e6, label="node 1, measured", **STYLE)
    ax.loglog(f[b], np.sqrt(p2[b]) * 1e6, label="node 2, measured", color=BLUE, lw=1.2)
    ax.loglog(f[b], np.sqrt(pred[:, 0]) * 1e6, "--", color=ACC, lw=1.6, label="model (no shape fitted)")
    ax.loglog(f[b], np.sqrt(pred[:, 1]) * 1e6, "--", color=ACC, lw=1.6)
    ax.axvspan(*BAND, color="#000", alpha=0.05)
    ax.set_ylabel(r"node noise density [$\mu$V/$\sqrt{\mathrm{Hz}}$]")
    ax.set_title("Node spectra against the model")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25, which="both")

    # the gap, on its own axis: a flat offset would be a horizontal line, and it is not
    for pp, nm, st in ((p1, "node 1", STYLE), (p2, "node 2", dict(color=BLUE, lw=1.2))):
        r = np.sqrt(pp[b]) / np.sqrt(pred[:, 0 if nm == "node 1" else 1])
        axr.semilogx(f[b], r, label=nm, **st)
    axr.axhline(1.0, color=ACC, ls="--", lw=1.6)
    axr.axvspan(*BAND, color="#000", alpha=0.05)
    axr.set_ylim(0.9, 1.75)
    axr.set_xlabel("frequency [Hz]"); axr.set_ylabel("data / model")
    axr.grid(alpha=0.25, which="both")
    fig.savefig(os.path.join(IMG, "fig-psd-vs-model.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig-psd-vs-model.png")
    for idx, (p, nm) in enumerate(((p1, "psd-node1.csv"), (p2, "psd-node2.csv"))):
        sel = np.unique(np.geomspace(np.where(b)[0][0], np.where(b)[0][-1], 400).astype(int))
        write_csv(nm, "freq_Hz,measured_V_per_rtHz,model_V_per_rtHz",
                  [[f"{f[i]:.2f}", f"{np.sqrt(p[i]):.4e}",
                    f"{np.sqrt(node_psd([f[i]])[0, idx]):.4e}"] for i in sel],
                  f"Node {idx+1} spectrum, {len(FINAL)} records pooled. Model uses measured "
                  f"components and the COMPUTED gain (A {GAIN_A:.0f}, B {GAIN_B:.0f}) - nothing fitted.")


# ---------------------------------------------------------------- 4. rho by band
def fig_rho():
    bands = [(200, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 12000), (12000, 30000)]
    meas, pred, ctr = [], [], []
    acc = None
    for nm in FINAL:
        dt, x, y = load(nm)
        w = np.hanning(NSEG)
        for i in range(0, len(x) - NSEG + 1, NSEG // 2):
            X = np.fft.rfft((x[i:i+NSEG] - x[i:i+NSEG].mean()) * w)
            Y = np.fft.rfft((y[i:i+NSEG] - y[i:i+NSEG].mean()) * w)
            t = np.array([np.abs(X) ** 2, np.abs(Y) ** 2, np.real(X * np.conj(Y))])
            acc = t if acc is None else acc + t
    f = np.fft.rfftfreq(NSEG, dt)
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        meas.append(acc[2][m].sum() / np.sqrt(acc[0][m].sum() * acc[1][m].sum()))
        sp = sigma_band(lo, hi, n=8000)
        pred.append(sp[0, 1] / np.sqrt(sp[0, 0] * sp[1, 1]))
        ctr.append(np.sqrt(lo * hi))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogx(ctr, pred, "--o", color=ACC, lw=1.6, ms=6, label="model")
    ax.semilogx(ctr, meas, "o", color="#1a1a1a", ms=8, label="measured")
    ax.set_xlabel("band centre [Hz]"); ax.set_ylabel(r"correlation $\rho_{12}$")
    ax.set_title(r"Correlation vs frequency, measured against the model")
    ax.legend(frameon=False); ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig-rho-by-band.png"), dpi=DPI)
    plt.close(fig); print("  wrote fig-rho-by-band.png")
    write_csv("rho-by-band.csv", "band_lo_Hz,band_hi_Hz,rho_measured,rho_model",
              [[lo, hi, f"{m:.4f}", f"{p:.4f}"] for (lo, hi), m, p in zip(bands, meas, pred)],
              f"Band-limited correlation between the two nodes, {len(FINAL)} records pooled. rho "
              "varies several-fold across the band because the nodes decouple above the corners.")
    return meas


# ---------------------------------------------------------------- 5. precision law
def fig_precision():
    tau = tau_slow()
    X, Y, dt = band_limited(FINAL)
    pts, rows = [], []
    for t_ms in (10, 20, 40, 80, 150, 300):
        Tw, vals = t_ms * 1e-3, []
        k = int(Tw / dt)
        for xb in X:
            for j in range(len(xb) // k):
                vals.append(np.mean(xb[j * k:(j + 1) * k] ** 2))
        v = np.array(vals)
        eps = v.std(ddof=1) / v.mean()
        pts.append([t_ms, len(v), eps])
        rows.append([t_ms, len(v), f"{eps:.5f}", f"{np.sqrt(2*tau/Tw):.5f}"])
    a = np.array(pts, float)
    slope = np.polyfit(np.log(a[:, 0]), np.log(a[:, 2]), 1)[0]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    tt = np.geomspace(7, 450, 50)
    ax.loglog(tt, np.sqrt(2 * tau / (tt * 1e-3)) * 100, "--", color=ACC, lw=1.8, zorder=1,
              label=r"$\sqrt{2\tau/T}$  (no fitted parameter)")
    ax.errorbar(a[:, 0], a[:, 2] * 100, yerr=a[:, 2] * 100 / np.sqrt(2 * a[:, 1]),
                fmt="o", color="#1a1a1a", ms=7, capsize=3, zorder=2,
                label=f"measured   slope {slope:+.3f}")
    ax.set_xlim(7, 450); ax.set_ylim(1.6, 22)
    ax.set_xlabel("integration time T [ms]")
    ax.set_ylabel(r"relative error on $\Sigma_{11}$ [%]")
    ax.set_title(r"Precision improves as $\sqrt{T}$")
    ax.legend(frameon=False, fontsize=9, loc="lower left"); ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig-precision-vs-time.png"), dpi=DPI)
    plt.close(fig); print("  wrote fig-precision-vs-time.png")
    write_csv("precision-vs-time.csv", "T_ms,n_windows,eps_measured,eps_theory", rows,
              f"Relative error on the variance vs integration time, {len(FINAL)} records. "
              f"Theory is sqrt(2*tau/T) with tau = {tau*1e6:.1f} us from the measured R and C - "
              f"nothing fitted. log-log slope {slope:+.3f} (required -0.500).")
    return slope, tau


# ---------------------------------------------------------------- 6. the systematic floor
def floor_curve(X, Y, rho_model, tau, dt):
    """Scatter and accuracy of rho vs integration time.

    scatter  = std of the windowed estimate. Falls as 1/sqrt(T) forever - a bias is
               common to every window and cancels here.
    accuracy = RMS deviation of the windowed estimate FROM THE MODEL. Falls as
               sqrt(2 tau/T + eps_sys^2): it flattens at the systematic floor.
    """
    out = []
    for t_ms in (2, 5, 10, 20, 40, 80, 150, 300, 600):
        k, est = int(t_ms * 1e-3 / dt), []
        for x, y in zip(X, Y):
            for j in range(len(x) // k):
                est.append(np.corrcoef(x[j*k:(j+1)*k], y[j*k:(j+1)*k])[0, 1])
        e = np.array(est)
        out.append([t_ms * 1e-3, len(e), e.std(ddof=1), abs(e.mean() - rho_model),
                    np.sqrt(np.mean((e - rho_model) ** 2))])
    for g in (2, 3, 5):                                  # pool records for T > one record
        est = []
        for s in range(0, len(X) - g + 1, g):
            est.append(np.corrcoef(np.concatenate(X[s:s+g]), np.concatenate(Y[s:s+g]))[0, 1])
        e = np.array(est)
        out.append([g * 0.6, len(e), e.std(ddof=1), abs(e.mean() - rho_model),
                    np.sqrt(np.mean((e - rho_model) ** 2))])
    return np.array(out)


def fig_floor():
    tau = tau_slow()
    sp = sigma_band(*BAND)
    rho_m = sp[0, 1] / np.sqrt(sp[0, 0] * sp[1, 1])
    X, Y, dt = band_limited(FINAL)
    A = floor_curve(X, Y, rho_m, tau, dt)
    Xb, Yb, _ = band_limited(BEFORE_REGROUND)
    Bf = floor_curve(Xb, Yb, rho_m, tau, dt)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    tt = np.geomspace(1.5e-3, 4, 60)
    ax.loglog(tt * 1e3, np.sqrt(2 * tau / tt) * rho_m, "--", color="#999", lw=1.4,
              label=r"$\sqrt{2\tau/T}\cdot\rho$  (statistical only)")
    ax.loglog(Bf[:, 0] * 1e3, Bf[:, 4], "s-", color="#bbb", ms=5, lw=1.2,
              label=f"accuracy, before regrounding  (floor {Bf[-1, 3]:.3f})")
    ax.loglog(A[:, 0] * 1e3, A[:, 2], "o-", color=BLUE, ms=6, lw=1.4,
              label="scatter across windows  (keeps falling)")
    ax.loglog(A[:, 0] * 1e3, A[:, 4], "o-", color="#1a1a1a", ms=6, lw=1.6,
              label="accuracy vs model  (final configuration)")
    ax.set_xlabel("integration time T [ms]")
    ax.set_ylabel(r"error on $\rho_{12}$  (absolute)")
    ax.set_title(r"Error on $\rho_{12}$ against integration time")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left"); ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig-systematic-floor.png"), dpi=DPI)
    plt.close(fig); print("  wrote fig-systematic-floor.png")
    rows = [["final", f"{r[0]*1e3:.0f}", int(r[1]), f"{r[2]:.5f}", f"{r[3]:.5f}", f"{r[4]:.5f}"] for r in A]
    rows += [["before_regrounding", f"{r[0]*1e3:.0f}", int(r[1]), f"{r[2]:.5f}", f"{r[3]:.5f}", f"{r[4]:.5f}"] for r in Bf]
    write_csv("systematic-floor.csv", "configuration,T_ms,n_windows,scatter,abs_bias,accuracy_rms", rows,
              f"rho12 estimated over windows of length T, {len(FINAL)} records. scatter = std across "
              f"windows (falls as 1/sqrt(T) regardless of any bias). accuracy_rms = RMS deviation "
              f"from the model rho = {rho_m:.4f}; it flattens at the systematic floor. "
              f"'before_regrounding' is the same board minutes earlier with the probe grounds not "
              f"yet tied to the shield.")
    return A, Bf, rho_m


# ---------------------------------------------------------------- 7. the 4kT difference
def fig_4kt(gain=GAIN_A):
    """S(10k) - S(1k) cancels every term that does not scale with R.

    NOTE: these captures are from the 2026-09-11 board (C1 = 26.8 nF, R3 OUT, unshielded).
    They test the FRONT END - source resistor and amplifier - which has not changed since,
    so they remain valid; but they are not from the final network configuration.
    """
    f, p10 = pooled_psd([f"v10k_{i}.npz" for i in (1, 2, 3, 4)], 1)
    _, p1 = pooled_psd([f"r1k_{i}.npz" for i in (1, 2, 3, 4)], 1)
    diff = p10 - p1
    fc = 1 / (2 * np.pi * RN1 * C1_ORIG)          # R3 out: single pole, the C1 of THAT day
    h2 = 1 / (1 + (f / fc) ** 2)
    m = (f >= 300) & (f < 20000) & (diff > 0)
    flat = np.sqrt(diff[m] / h2[m]) / gain * 1e9  # input-referred, nV/rtHz
    true = np.sqrt(4 * K_B * T_K * (R11 - 1000.0)) * 1e9
    edges = np.geomspace(300, 20000, 26)
    idx = np.digitize(f[m], edges)
    bx = [f[m][idx == i].mean() for i in range(1, len(edges)) if (idx == i).sum() > 3]
    by = [np.median(flat[idx == i]) for i in range(1, len(edges)) if (idx == i).sum() > 3]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.semilogx(f[m], flat, color="#bbb", lw=0.7, label="difference spectrum (raw)")
    ax.semilogx(bx, by, "o-", color="#1a1a1a", ms=5, lw=1.4, label="binned median")
    # Current noise also scales with R, so it SURVIVES the subtraction: the expected height is
    # sqrt(4kT.dR + i_n^2 (R10k^2 - R1k^2)), not the Johnson term alone.
    i_meas = 0.83e-12
    with_in = np.sqrt(4 * K_B * T_K * (R11 - 1000.0) + i_meas ** 2 * (R11 ** 2 - 1000.0 ** 2)) * 1e9
    ax.axhline(true, ls=":", color="#888", lw=1.6,
               label=rf"Johnson alone, $\sqrt{{4k_BT\,\Delta R}}$ = {true:.1f}")
    ax.axhline(with_in, ls="--", color=ACC, lw=1.8,
               label=rf"+ measured $i_n$ = 0.83 pA/$\sqrt{{\mathrm{{Hz}}}}$  $\rightarrow$ {with_in:.1f} nV/$\sqrt{{\mathrm{{Hz}}}}$")
    ax.set_ylim(0, 30)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"input-referred [nV/$\sqrt{\mathrm{Hz}}$]")
    ax.set_title(r"S(10 k$\Omega$) $-$ S(1 k$\Omega$), network response divided out")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig-4kt-difference.png"), dpi=DPI)
    plt.close(fig); print("  wrote fig-4kt-difference.png")
    write_csv("fourkt-difference.csv", "freq_Hz,input_referred_nV_per_rtHz",
              [[f"{x:.1f}", f"{y:.3f}"] for x, y in zip(bx, by)],
              f"Difference spectrum with the network response removed, referred to the input at "
              f"gain {gain:.0f}. Flat would mean the R-dependent noise is white; it is close and not exact, rising ~20 % across 300 Hz - 20 kHz (log-log slope +0.044). "
              f"sqrt(4kT*dR) = {true:.2f} nV/rtHz. Captures are from the 2026-09-11 board "
              f"(C1 = 26.8 nF); they test the front end, which is unchanged.")
    lo = np.array(bx) < 2000
    hi = np.array(bx) > 6000
    return np.median(np.array(by)[lo]) / np.median(np.array(by)[hi]), true


# ---------------------------------------------------------------- 8. the condition-number ladder
# (prefix, R1, R2, R3) - all measured out of circuit 2026-09-17 (SDM3045X). The same R3 stayed in
# for every rung. These are NOT 0.1 % parts: the 10 k pair differs by 1.36 %, the 1 k pair by 1.67 %.
LADDER = [("r1", 1006.5, 990.0, 1000.6), ("r2", 2172.0, 2174.3, 1000.6),
          ("r3", 4685.6, 4692.3, 1000.6), ("r4b", 10049.0, 9914.0, 1000.6)]
LADDER_BAND = (200.0, 30000.0)
DELTA0, KAPPA0 = 0.007, 2.21          # pre-registered coefficient and the kappa it was set at


def fig_ladder():
    """Does the error floor grow with condition number?

    Symmetric network (R4 removed, R1 = R2 = RL, R3 = 1 k), four rungs, 30 records each. The
    observable is the measured condition number kappa_meas = var(v1+v2)/var(v1-v2), compared to the
    same quantity from the band-limited, pole-weighted model (kappa_ladder.py). Predictions were
    registered before any rung was built: |error| = DELTA0 * (kappa/KAPPA0)^p, p in {1, 0.5, 0}.
    Every rung resistor in LADDER is an out-of-circuit DMM reading; the pairs are not matched, so
    the model is evaluated with each pair's two values rather than an assumed symmetry.
    """
    import kappa_ladder as K
    rows, pts = [], []
    for pre, rl1, rl2, rc in LADDER:
        G = K.conductance(rl1, rl2, rc)
        ls, lf = K.eig(G)
        kap = lf / ls
        per = []
        for i in range(1, 31):
            dt, x, y = load(f"{pre}_{i:02d}.npz")
            x, y = bandpass(x, dt, *LADDER_BAND), bandpass(y, dt, *LADDER_BAND)
            per.append(K.observables(np.cov(np.vstack([x, y])), G)["kappa"])
        per = np.array(per)
        km = K.observables(K.sigma_net(G, K.FE_PAPER, rl1, rl2, LADDER_BAND), G)["kappa"]
        # pooled estimate = mean over records (Vc/Vd of the pooled covariance agrees to <0.1 %)
        kmeas, se = per.mean(), per.std(ddof=1) / np.sqrt(len(per))
        dev, sdev = kmeas / km - 1, se / km
        idx = np.arange(len(per)); sl = np.polyfit(idx, per, 1)[0] * (len(per) - 1) / per.mean()
        pts.append((kap, dev, sdev))
        rows.append([f"{kap:.2f}", f"{rl1:.1f}", f"{rl2:.1f}", f"{rc:.1f}", len(per), f"{kmeas:.3f}", f"{se:.3f}", f"{km:.3f}",
                     f"{100*dev:+.2f}", f"{100*sdev:.2f}",
                     f"{100*DELTA0*(kap/KAPPA0)**1:.2f}", f"{100*DELTA0*(kap/KAPPA0)**0.5:.2f}", f"{100*DELTA0:.2f}",
                     f"{100*sl:+.1f}"])
        print(f"  rung {pre}: kappa {kap:5.1f}  kappa_meas {kmeas:7.3f} +/- {se:.3f}  model {km:7.3f}  "
              f"dev {100*dev:+.2f} +/- {100*sdev:.2f} %   drift over run {100*sl:+.1f} %")
    a = np.array(pts)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    kk = np.geomspace(2, 30, 100)
    for p, ls_, lab in ((1.0, "--", "p = 1  (perturbation-theory null)"), (0.5, "-.", "p = 0.5"),
                        (0.0, ":", "p = 0  (no dependence on conditioning)")):
        ax.plot(kk, 100 * DELTA0 * (kk / KAPPA0) ** p, ls_, color=ACC, lw=1.5, label=lab)
        ax.plot(kk, -100 * DELTA0 * (kk / KAPPA0) ** p, ls_, color=ACC, lw=1.5)
    ax.errorbar(a[:, 0], 100 * a[:, 1], yerr=100 * a[:, 2], fmt="o", color="#1a1a1a", ms=7, capsize=4,
                zorder=3, label="measured, 30 records per rung")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xscale("log"); ax.set_xlim(2, 30); ax.set_ylim(-3, 6)
    # label the ticks at the rungs actually built, not at decades - there is only one decade here
    ax.set_xticks(list(a[:, 0]), [f"{k:.1f}" for k in a[:, 0]], minor=False)
    ax.set_xticks([], minor=True)
    ax.set_xlabel(r"condition number $\kappa$ of the network (model)")
    ax.set_ylabel(r"$\kappa_\mathrm{meas}/\kappa_\mathrm{model} - 1$   [%]")
    ax.set_title("Deviation from the model against condition number")
    ax.legend(frameon=False, fontsize=9, loc="lower left"); ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig-ladder.png"), dpi=DPI)
    plt.close(fig); print("  wrote fig-ladder.png")
    write_csv("ladder-rungs.csv",
              "kappa_model,R1_ohm,R2_ohm,R3_ohm,n_records,kappa_meas,std_err,kappa_model_bandlimited,"
              "deviation_pct,deviation_std_err_pct,pred_p1_pct,pred_p05_pct,pred_p0_pct,drift_over_run_pct",
              rows,
              f"Condition-number ladder: R4 removed, R1 ~ R2 = RL, R3 = 1 k, band "
              f"{LADDER_BAND[0]:.0f}-{LADDER_BAND[1]:.0f} Hz. R1, R2 are the out-of-circuit multimeter readings "
              f"(they are NOT matched to 0.1 %: the pairs differ by up to 1.67 %); R3 measured 1000.6 ohm. The model uses "
              f"the measured R1 and R2 separately. kappa_meas = var(v1+v2)/var(v1-v2), per record, mean and "
              f"standard error over 30 records; kappa_model_bandlimited is the same quantity from the pole-weighted "
              f"model with the measured asymmetry. pred_* are the PRE-REGISTERED floors |err| = {DELTA0} "
              f"(kappa/{KAPPA0})^p. The kappa 5.3 rung started before the board had thermally settled; its drift "
              f"column shows it.")
    return a


# ---------------------------------------------------------------- 9. a look at the noise
def fig_trace():
    dt, x, y = load(FINAL[0])
    n = int(8e-3 / dt)
    t = np.arange(n) * dt * 1e3
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(t, bandpass(x, dt, *BAND)[:n] * 1e3, label="node 1", **STYLE)
    ax.plot(t, bandpass(y, dt, *BAND)[:n] * 1e3, label="node 2", color=BLUE, lw=1.0)
    ax.set_xlabel("time [ms]"); ax.set_ylabel("node voltage [mV]")
    ax.set_title("The signal: 8 ms of thermal noise at both nodes")
    ax.legend(frameon=False, ncol=2); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig-noise-trace.png"), dpi=DPI)
    plt.close(fig); print("  wrote fig-noise-trace.png")


# ---------------------------------------------------------------- components
def components_csv():
    rows = [["R1", "R_n1  source resistor into node 1", 4700, RN1],
            ["R2", "R_n2  source resistor into node 2", 4700, RN2],
            ["R3", "R_c   coupling resistor", 6800, RC],
            ["R4", "R_g   node 2 to ground", 10000, RG],
            ["C1", "node 1 to ground [F]  (matched pair)", 22e-9, C1],
            ["C2", "node 2 to ground [F]  (matched pair)", 22e-9, C2],
            ["R11", "Johnson source (channel A)", 10000, R11],
            ["R12", "Johnson source (channel B)", 10000, R12]]
    write_csv("components-measured.csv", "part,role,nominal,measured,deviation_pct",
              [[a, b, c, d, f"{100*(d-c)/c:+.1f}"] for a, b, c, d in rows],
              "Measured out of circuit with an SDM3045X. C1 and C2 are a matched pair 0.04 % "
              "apart, selected from nine candidates; the original C1 was 26.8 nF, 14 % from C2.")


# ---------------------------------------------------------------- main
# ---------------------------------------------------------------- 9. detailed balance / reversibility
REV_NSEG = 1 << 18
REV_NOTCH = (800.0, 1600.0)      # set by the pgnd POWER spectrum, not by any node-data phase
REV_BAND = (200.0, 5000.0)
REV_EDGES = [200, 400, 800, 1600, 3200, 6400, 12800, 25600, 50000]


def _rev_mask(f, lo, hi, notch=True):
    m = (f >= lo) & (f < hi)
    if notch:
        m &= ~((f >= REV_NOTCH[0]) & (f < REV_NOTCH[1]))
    return m


def _rev_pooled(files, nseg=REV_NSEG):
    import equilibrium as E
    acc, n, dt = None, 0, None
    for nm in files:
        dt, x, y = load(nm)
        a, k = E.record_spectrum(x, y, dt, nseg)
        acc = a if acc is None else acc + a
        n += k
    return E.normalise(acc, n, dt, nseg) + (dt,)


def fig_reversibility():
    """Is the board time-reversible - i.e. actually at equilibrium?

    Detailed balance requires S_i/C_i equal at both nodes, which is the SAME condition as
    Sigma = c G^-1. It is tested here on the DYNAMICS: a reversible process has a purely real
    cross-spectrum, so arg S12(f) is the observable. See equilibrium.py for the pre-registration,
    the confound budget, the notch policy, the synthetic-data validation of the estimator and
    the surrogate null.
    """
    import equilibrium as E
    f, P, _ = _rev_pooled([f"pgnd_{i}.npz" for i in range(1, 7)])     # pickup reference
    rows, pts = [], []
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))

    # --- panel 1: where the phase lives, board vs pure pickup
    fg, S, _ = _rev_pooled(E.DATASETS[0].files())
    ctr, pb, pp, frac = [], [], [], []
    ptot = P[(f >= 200) & (f < 50000), 0, 0].real.sum()
    for lo, hi in zip(REV_EDGES[:-1], REV_EDGES[1:]):
        m = (fg >= lo) & (fg < hi)
        ctr.append(np.sqrt(lo * hi))
        pb.append(np.angle(S[m, 0, 1].sum()))
        pp.append(np.angle(P[m, 0, 1].sum()))
        frac.append(P[m, 0, 0].real.sum() / ptot)
    ax.axvspan(*REV_NOTCH, color=ACC, alpha=0.12, lw=0)
    ax.semilogx(ctr, pp, "s--", color=ACC, ms=6, lw=1.5, label="probe pickup alone (tips on cage)")
    ax.semilogx(ctr, pb, "o-", color="#1a1a1a", ms=6, lw=1.6, label="the board, at its nodes")
    ax.axhline(0, color="#999", lw=0.8)
    ax.text(1130, 0.62, f"{100*max(frac):.0f} % of the\npickup's power", fontsize=8,
            color=ACC, ha="center")
    ax.set_xlabel("band centre [Hz]"); ax.set_ylabel(r"$\arg S_{12}$  [rad]")
    ax.set_title("Phase by band: the board against pure pickup")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25, which="both")

    # --- panel 2: five datasets, measured vs model, pickup band excluded
    for ds in E.DATASETS:
        fd, Sd, dt = _rev_pooled(ds.files())
        per = []
        for nm in ds.files():
            dtr, x, y = load(nm)
            a, k = E.record_spectrum(x, y, dtr, REV_NSEG)
            fr, Sr = E.normalise(a, k, dtr, REV_NSEG)
            per.append(np.angle(Sr[_rev_mask(fr, *REV_BAND), 0, 1].sum()))
        per = np.array(per)
        se = per.std(ddof=1) / np.sqrt(len(per))
        val = float(np.angle(Sd[_rev_mask(fd, *REV_BAND), 0, 1].sum()))
        full = float(np.angle(Sd[_rev_mask(fd, *REV_BAND, notch=False), 0, 1].sum()))
        mod = float(np.angle(E.model_cross_spectrum(fd[_rev_mask(fd, *REV_BAND)], ds)[:, 0, 1].sum()))
        lam = np.linalg.eigvals(np.linalg.solve(ds.C, ds.G)).real
        kap = lam.max() / lam.min()
        pts.append((kap, val, se, mod))
        rows.append([ds.tag.split()[0], f"{kap:.2f}", len(per), f"{full:+.5f}", f"{val:+.5f}",
                     f"{se:.5f}", f"{mod:+.5f}", f"{abs(val - mod) / se:.1f}"])
        print(f"  {ds.tag:18s} notched {val:+.3e} +/- {se:.1e}  model {mod:+.3e}  "
              f"({abs(val-mod)/se:.1f} sigma)", flush=True)
    a = np.array(pts)
    o = np.argsort(a[:, 0])
    ax2.errorbar(a[o, 0], 1e3 * a[o, 1], yerr=1e3 * a[o, 2], fmt="o", color="#1a1a1a", ms=6,
                 capsize=4, lw=1.5, label="measured (pickup band excluded)")
    ax2.plot(a[o, 0], 1e3 * a[o, 3], "^--", color=BLUE, ms=7, lw=1.5,
             label="detailed-balance model")
    ax2.axhline(0, color="#999", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xticks(list(a[o, 0]), [f"{k:.0f}" if k > 3 else f"{k:.1f}" for k in a[o, 0]])
    ax2.set_xticks([], minor=True)
    ax2.set_xlabel(r"condition number $\kappa$"); ax2.set_ylabel(r"$\arg S_{12}$  [mrad]")
    ax2.set_title(r"$\arg S_{12}$ against condition number")
    ax2.legend(frameon=False, fontsize=9); ax2.grid(alpha=0.25, which="both")

    fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig-reversibility.png"), dpi=DPI)
    plt.close(fig); print("  wrote fig-reversibility.png")
    write_csv("reversibility.csv",
              "dataset,kappa,n_records,arg_S12_full_band,arg_S12_notched,std_err,"
              "arg_S12_model,sigma_from_model", rows,
              f"Detailed-balance test: a reversible process has a PURELY REAL cross-spectrum, so "
              f"arg S12 is the observable. Band {REV_BAND[0]:.0f}-{REV_BAND[1]:.0f} Hz; the "
              f"'notched' column additionally excludes {REV_NOTCH[0]:.0f}-{REV_NOTCH[1]:.0f} Hz, "
              f"where an independent capture (pgnd, both probe tips on one point of the shield) "
              f"shows the probe-lead pickup carries 88.5 % of its power with its own phase of "
              f"+0.79 rad. The exclusion is defined by that POWER spectrum, never by node-data "
              f"phase. std_err is from the record-to-record spread. Pre-registration, confound "
              f"budget, notch policy, synthetic-data validation of the estimator and the "
              f"surrogate null: equilibrium.py.")
    return a


# ---------------------------------------------------------------- 10. the trace with interference removed
CLEAN_NOTCH = (800.0, 1600.0)     # the probe-pickup band, set by pgnd's power spectrum (see eq. py)
CLEAN_THRESH = 3.0                # a bin is a "line" if it exceeds this x its local median


def line_intervals(nseg=1 << 20, w=101, cache=[]):
    """Frequency intervals carrying discrete interference, found on the POOLED spectrum.

    This cannot be done on one record's raw periodogram: at that resolution each bin is
    exponentially distributed, so ~12 % of bins exceed 3x their local median by chance and the
    test flags a quarter of the band. Averaging the thirty records first reduces the per-bin
    scatter enough for the threshold to pick out real lines - the same 27-bin set quoted in
    Limitations. Flagged on the union of the two channels so both get an identical filter.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    if cache:
        return cache[0]
    acc, n, dt = None, 0, None
    for nm in FINAL:
        dt, x, y = load(nm)
        win = np.hanning(nseg)
        for i in range(0, len(x) - nseg + 1, nseg // 2):
            A = np.fft.rfft((x[i:i + nseg] - x[i:i + nseg].mean()) * win)
            B = np.fft.rfft((y[i:i + nseg] - y[i:i + nseg].mean()) * win)
            s = np.stack([np.abs(A) ** 2, np.abs(B) ** 2])
            acc = s if acc is None else acc + s
            n += 1
    f = np.fft.rfftfreq(nseg, dt)
    S = acc / n

    def med(v):
        return np.median(sliding_window_view(np.pad(v, w // 2, mode="edge"), w), axis=-1)

    band = (f >= BAND[0]) & (f <= BAND[1])
    flag = ((S[0] > CLEAN_THRESH * med(S[0])) | (S[1] > CLEAN_THRESH * med(S[1]))) & band
    df = f[1] - f[0]
    iv = [(f[i] - df / 2, f[i] + df / 2) for i in np.flatnonzero(flag)]
    cache.append(iv)
    print(f"    {len(iv)} interference lines identified on the pooled spectrum "
          f"({100*len(iv)/band.sum():.2f} % of the band)")
    return iv


def clean(x, y, dt, drop_notch=True):
    """Band-limit, then remove the interference this write-up has identified.

    Two removals with very different costs:
      - the discrete lines: ~0.5 % of the band, so almost no genuine signal goes with them;
      - the 800-1600 Hz probe-pickup band: broadband, so real network signal is removed too.
    Returns the cleaned pair and the fraction of each node's power taken out by each step.
    """
    n = len(x)
    f = np.fft.rfftfreq(n, dt)
    X, Y = np.fft.rfft(x - x.mean()), np.fft.rfft(y - y.mean())
    band = (f >= BAND[0]) & (f <= BAND[1])
    X[~band] = 0
    Y[~band] = 0
    p0 = (np.abs(X) ** 2).sum(), (np.abs(Y) ** 2).sum()
    lines = np.zeros_like(band)
    for lo, hi in line_intervals():
        lines |= (f >= lo) & (f < hi)
    X[lines] = 0
    Y[lines] = 0
    p1 = (np.abs(X) ** 2).sum(), (np.abs(Y) ** 2).sum()
    if drop_notch:
        nb = (f >= CLEAN_NOTCH[0]) & (f < CLEAN_NOTCH[1])
        X[nb] = 0
        Y[nb] = 0
    p2 = (np.abs(X) ** 2).sum(), (np.abs(Y) ** 2).sum()
    frac = dict(lines=[1 - p1[i] / p0[i] for i in (0, 1)],
                notch=[(p1[i] - p2[i]) / p0[i] for i in (0, 1)],
                n_lines=len(line_intervals()))
    return np.fft.irfft(X, n), np.fft.irfft(Y, n), frac


def fig_trace_clean():
    """The board's own noise, with the discrete interference removed.

    The write-up refers to the pickup repeatedly without ever showing it. Top left: one 8 ms segment
    as measured, the interference alone, and what is left. Top right: that signal on both nodes.
    Bottom left: the same record in the network's own coordinates, common against differential.
    Bottom right: the spectrum, so what came out is visible rather than asserted.

    The 800-1600 Hz probe-pickup band is MARKED but NOT removed. It is broadband, and most of the
    power in it is genuine network signal - deleting it would take 17.6 % of node 1's power in this
    record, far more than the pickup it contains - so a trace with that band cut out would not be "the board's
    own noise" either.
    """
    dt, x, y = load(FINAL[0])
    xc, yc, frac = clean(x, y, dt, drop_notch=False)
    _, _, frac_n = clean(x, y, dt, drop_notch=True)
    xr, yr = bandpass(x, dt, *BAND), bandpass(y, dt, *BAND)
    rem = xr - xc
    n = int(8e-3 / dt)
    t = np.arange(n) * dt * 1e3
    fig, axx = plt.subplots(2, 2, figsize=(11.6, 6.8))
    ax, axs, axm, ax2 = axx[0, 0], axx[0, 1], axx[1, 0], axx[1, 1]

    # panel 1: measured = signal + interference, stacked so all three are visible at once
    off = 2.6
    ax.plot(t, xr[:n] * 1e3 + off, color="#666", lw=0.9)
    ax.plot(t, rem[:n] * 1e3, color=ACC, lw=0.9)
    ax.plot(t, xc[:n] * 1e3 - off, lw=0.9, color="#1a1a1a")
    for yv, lab in ((off, f"as measured   {1e3*xr.std():.3f} mV rms"),
                    (0.0, f"interference   {1e3*rem.std():.3f} mV rms"),
                    (-off, f"signal   {1e3*xc.std():.3f} mV rms")):
        ax.text(0.15, yv + 1.02, lab, fontsize=7.5, va="bottom", zorder=5,
                color={off: "#666", 0.0: ACC, -off: "#1a1a1a"}[yv],
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    ax.set_ylabel("node 1 [mV, offset]"); ax.set_yticks([])
    ax.set_title("measured  =  signal  +  interference")
    ax.grid(alpha=0.25, axis="x")

    # panel 2: the signal itself, both nodes
    axs.plot(t, xc[:n] * 1e3, label="node 1", color="#1a1a1a", lw=1.2)
    axs.plot(t, yc[:n] * 1e3, color=BLUE, lw=1.0, label="node 2")
    axs.set_ylabel("node voltage [mV]")
    axs.set_title("The signal, both nodes")
    axs.legend(frameon=False, fontsize=8, ncol=2); axs.grid(alpha=0.25)

    # panel 3: the same signal in the network's own coordinates. The two nodes look correlated
    # because they ARE - that shared part is the off-diagonal of Sigma, i.e. the answer. Splitting
    # it into common and differential modes is the useful move, not removing it: the RATIO of their
    # variances is the measured condition number, which is the observable the ladder uses.
    com, dif = (xc + yc), (xc - yc)
    ratio = com.var() / dif.var()
    lam = np.linalg.eigvals(np.linalg.solve(CM, G)).real
    axm.plot(t, com[:n] * 1e3 + 3.0, color="#1a1a1a", lw=0.9)
    axm.plot(t, dif[:n] * 1e3 - 3.0, color=ACC, lw=0.9)
    for yv, lab, c in ((3.0, f"common  $v_1+v_2$   {1e3*com.std():.3f} mV rms", "#1a1a1a"),
                       (-3.0, f"differential  $v_1-v_2$   {1e3*dif.std():.3f} mV rms", ACC)):
        axm.text(0.15, yv + 1.25, lab, fontsize=7.5, va="bottom", zorder=5, color=c,
                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    axm.set_xlabel("time [ms]"); axm.set_ylabel("mode [mV, offset]"); axm.set_yticks([])
    axm.set_title(f"var ratio {ratio:.2f}  vs  model κ {lam.max()/lam.min():.2f}")
    axm.grid(alpha=0.25, axis="x")

    fw, pr, _ = welch(xr, dt)
    _, pc, _ = welch(xc, dt)
    b = (fw >= BAND[0]) & (fw <= BAND[1])
    ax2.axvspan(*CLEAN_NOTCH, color=ACC, alpha=0.12, lw=0)
    ax2.loglog(fw[b], np.sqrt(pr[b]) * 1e6, color="#c9c9c9", lw=1.4, label="as measured")
    ax2.loglog(fw[b], np.sqrt(pc[b]) * 1e6, color="#1a1a1a", lw=0.8, label="signal")
    ax2.set_ylim(0.3, 40)
    ax2.text(1130, 0.45, "probe pickup band\n(marked, not removed)", fontsize=7.5,
             color=ACC, ha="center")
    ax2.set_xlabel("frequency [Hz]")
    ax2.set_ylabel(r"node 1 [$\mu$V/$\sqrt{\mathrm{Hz}}$]")
    ax2.set_title("Where the interference came from")
    ax2.legend(frameon=False, fontsize=8, loc="upper right"); ax2.grid(alpha=0.25, which="both")

    fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig-noise-clean.png"), dpi=DPI)
    plt.close(fig); print("  wrote fig-noise-clean.png")
    print(f"    {frac['n_lines']} lines remove {100*frac['lines'][0]:.2f} % / "
          f"{100*frac['lines'][1]:.2f} % of node power")
    print(f"    measured {1e3*xr.std():.4f} / {1e3*yr.std():.4f} mV   "
          f"interference {1e3*rem.std():.4f} mV   signal {1e3*xc.std():.4f} / {1e3*yc.std():.4f} mV")
    print(f"    the 800-1600 Hz band would remove a further {100*frac_n['notch'][0]:.1f} % / "
          f"{100*frac_n['notch'][1]:.1f} % (mostly genuine signal) - not removed")
    com, dif = (xc + yc), (xc - yc)
    print(f"    modes: common {1e3*com.std():.3f} mV, differential {1e3*dif.std():.3f} mV, "
          f"var ratio {com.var()/dif.var():.3f}")
    return frac, frac_n


if __name__ == "__main__":
    print(f"front-end gain, COMPUTED from measured resistors (nothing fitted):")
    print(f"  channel A {GAIN_A:.1f}   channel B {GAIN_B:.1f}   (nominal 1111)")
    print(f"network: tau_slow = {tau_slow()*1e6:.1f} us   kappa = {kappa():.2f}   "
          f"c1/c2 = {SI1/C1/(SI2/C2):.4f}")
    ratio, modelled, implied, extra = excess_over_budget()
    print(f"\nresidual against the datasheet noise budget:")
    # NOT flat: the published node spectra rise from x1.04 at 300 Hz to x1.17 at 40 kHz in
    # amplitude. The median below is a single summary of a frequency-dependent ratio, which is
    # why the C_eff fit that assumed a white excess was retracted. See README "Limitations".
    print(f"  measured / model        {ratio:.3f}  (median over 300 Hz - 40 kHz; rises with f)")
    print(f"  input-referred modelled {modelled*1e9:.2f} nV/rtHz")
    print(f"  input-referred implied  {implied*1e9:.2f} nV/rtHz")
    print(f"  UNACCOUNTED             {extra*1e9:.2f} nV/rtHz\n")
    print("writing data/ ...")
    components_csv()
    a, pooled, preds = results()
    print("\nbuilding figures ...")
    fig_psd()
    fig_rho()
    slope, tau = fig_precision()
    A, Bf, rho_m = fig_floor()
    flatness, true4kt = fig_4kt()
    ladder = fig_ladder()
    fig_trace_clean()   # supersedes fig_trace(), which is kept but no longer published
    fig_reversibility()
    n = len(a)
    se = a.std(0, ddof=1) / np.sqrt(n)
    print(f"\nheadline ({n} records pooled):")
    for j, lab in enumerate(["v1 [mV]", "v2 [mV]", "v1/v2", "rho12", "Sigma.G offdiag [%]", "Lyapunov offdiag [%]"]):
        dev = f"{pooled[j]-preds[j]:+.2f} pp" if "[%]" in lab else f"{100*(pooled[j]-preds[j])/abs(preds[j]):+.1f} %"
        print(f"  {lab:22s} {pooled[j]:+.4f} +/- {se[j]:.4f}   model {preds[j]:+.4f}   {dev}")
    print(f"  precision slope {slope:+.3f} (theory -0.500), tau = {tau*1e6:.1f} us")
    print(f"  floor on rho: final {A[-1,3]:.4f} ({100*A[-1,3]/rho_m:.2f} %)   "
          f"before regrounding {Bf[-1,3]:.4f} ({100*Bf[-1,3]/rho_m:.2f} %)")
    print(f"  4kT difference flatness (LF/HF) {flatness:.3f} (white = 1.000)")
    print(f"  ladder: kappa_meas/model - 1 = " + "  ".join(f"{100*d:+.2f}%@{k:.0f}" for k, d, _ in ladder))
