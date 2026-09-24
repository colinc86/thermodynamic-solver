#!/usr/bin/env python3
"""Is the board an equilibrium sampler? - TWO-TIME STRUCTURE, PRE-REGISTERED.

WRITTEN BEFORE THE ANALYSIS IS RUN. Everything in the PREDICTIONS block below is computed from
measured components only. Do not edit it once the data has been touched; append results beside it.

WHAT THE PAPER ALREADY SHOWS, AND WHAT IT DOES NOT
    Every number in the write-up is a SECOND MOMENT AT LAG ZERO: rho12, v1/v2, the Sigma.G
    off-diagonal, the Lyapunov off-diagonal, the whole kappa ladder. Nothing touches the DYNAMICS,
    and nothing tests whether the device is a SAMPLER rather than a filter that happens to land on
    the right covariance. The field is branded "thermodynamic" - an equilibrium system relaxing into
    a Gibbs distribution - but this board is driven by AMPLIFIED Johnson noise through an active
    front end, at a Johnson-equivalent temperature of ~1e9 K. That is not obviously equilibrium.
    (The ~2.5e11 K this line carried until 2026-09-20 came from the abandoned AWG drive in
    DESIGN.md. For the Johnson front end: SI*R/(4k) = 1.010e9 K at node 1, 1.024e9 at node 2, and
    equivalently 3.400e6 x the power a passive 4.69 k delivers at 297 K. The argument is unchanged.)

THE TEST
    For  C x' = -G x + xi  with G symmetric and xi diagonal white, write A = C^-1 G. The stationary
    two-time correlation is C(t) = <x(t+s) x(s)^T> = exp(-A t) Sigma. Time reversibility (detailed
    balance) requires C(t) = C(t)^T for all t, i.e.  A Sigma = Sigma A^T.

    Working that through with C diagonal and the drive diagonal:
        (A D)_ij = G_ij d_j / C_i ,   (D A^T)_ij = d_i G_ij / C_j ,   d_i ~ S_i / C_i^2
        equal for i != j  <=>  d_i C_i = d_j C_j  <=>  S_i / C_i equal at both nodes  <=>  c1 = c2

    which is EXACTLY the condition for Sigma = c G^-1. So this is not a restatement of the paper's
    identity test - it is an INDEPENDENT DYNAMICAL OBSERVABLE of the same physical condition,
    sensitive to different failure modes.

    Note what the condition does NOT contain: G. Symmetrising the network does not make it more
    reversible. G enters only through HOW LARGE the observable is for a given violation. An earlier
    draft of the plan got this backwards; the derivation above is why it was wrong.

    In the frequency domain the statement is sharp:

        REVERSIBLE  <=>  the cross-spectrum S12(f) is PURELY REAL.
        Im[S12(f)] != 0 is a probability current - the standard non-equilibrium signature.

    Verified analytically below (`_selftest`): with C1 = C2 and S1 = S2, G + i.omega.C is normal,
    its eigenvectors are G's real orthogonal ones, and S12 comes out real to machine precision.

THE COMMON PARAMETER (what makes five datasets over-determined)
    c1/c2 is NOT common across the datasets: c_i = S_i/(4 C_i) and S_i ~ (V_i/R_i)^2, and the series
    resistors R_i differ per rung (and are measured). What IS common is the FRONT-END OUTPUT VOLTAGE
    RATIO, since gnd2 and all four rungs use the identical front end (FE_PAPER, same R7/R8):

        rho_V(f) = [drive_A . |H_A(f)|] / [drive_B . |H_B(f)|]          <- ONE function, all five
        S1(f)/S2(f) = rho_V(f)^2 . (R_2/R_1)^2                          <- R's measured per dataset

    So: one common front-end asymmetry, five independent network geometries, five predicted phase
    profiles. A single dataset could be fitted by anything; five cannot.

PREDICTIONS (frozen)
    P1. NOT zero. A first draft of this block guessed "|arg S12| < 1e-3 rad"; running the model -
        components only, no records read - falsified that guess immediately. The board is KNOWN to
        have c1 != c2 (drive asymmetry 0.984, from R11 != R12 and the two gains), so the honest
        prediction is a small but NON-ZERO probability current, and the numbers are these:

            dataset            arg S12, 200 Hz - 5 kHz      5 - 50 kHz
            gnd2                    -2.42e-3 rad            -2.05e-2
            r1  (kappa 3.0)         -2.26e-3                -1.51e-2
            r2  (kappa 5.3)         -5.77e-4                -4.15e-3
            r3  (kappa 10.4)        -4.82e-4                -4.46e-3
            r4b (kappa 21.0)        -1.62e-3                -2.04e-2

        Against a band-averaged resolution of ~2e-3 rad this is MARGINAL by design: gnd2 and r1 sit
        right at the edge of detectability, r2 and r3 below it. That is the honest state of the
        experiment and it is recorded here rather than discovered afterwards. The sign is a
        prediction too - all five are NEGATIVE - and a measured current of the opposite sign would
        be as informative as one of the wrong size.
    P1b. The measurement is therefore a two-sided test: significant agreement with the table above
        confirms the equilibrium picture QUANTITATIVELY (a predicted current, found); a measured
        magnitude far above it means an unmodelled non-equilibrium term.
    P2. The five measured phase profiles are jointly consistent with the one rho_V(f) above and with
        the measured R's - i.e. a single-parameter fit describes all five.
    P3. tau_slow from the even part of C(t) matches R_L.C: 92.0 us on gnd2, and tracking a factor of
        ten across the rungs.

DECISION RULE (frozen)
    - DETECTION requires BOTH: >= 5 sigma against the record-to-record spread, AND a magnitude
      exceeding the modelled confound sum for that band. 2-5 sigma is reported as "not resolved",
      never as a hint.
    - The estimator must first pass the SURROGATE NULL: x1 from record i paired with x2 from record
      j != i is genuinely independent, so its odd part must come out zero. If it does not, the
      estimator is biased and NO result is reported.
      CORRECTED AFTER FIRST USE: the null statistic must be Im(COHERENCE), not arg S12. For
      independent signals |S12| collapses (measured: 0.377 -> 0.013) and the ARGUMENT of a near-zero
      complex number is uniformly random, so the first version of this test "failed" at -2.47 rad for
      that reason alone. The estimator was fine; the statistic was wrong.
      The estimator was then validated end-to-end on synthetic OU data generated by filtering white
      noise with the analytic transfer function on numpy's own FFT grid (so generator and reference
      share a convention by construction, and no sign argument is needed):
          numpy-FFT of h(t) = exp(-At)C^-1  ==  (G + i.2.pi.f.C)^-1  to 6.7e-4  [no conjugate flip]
          equal drives      -> measured +2.6e-3 +/- 4.0e-3 vs model +1.1e-4     [unbiased, 0.6 sigma]
          s1/s2 = 0.96      -> measured -1.66e-2 +/- 4.6e-3 vs model -1.05e-2   [right sign, 1.3 s]

CONFOUNDS, sized before measuring (this is what sets the band)
    scope inter-channel skew (dt ~ 1 ns, grows as f) : ~3e-5 rad at 1-5 kHz, ~3e-4 at 50 kHz
    probe capacitance mismatch (~10 pF on 23.5 nF)   : ~4e-4 rad at 1-5 kHz, growing
    front-end pole mismatch (gains differ 0.5 %)     : negligible at 1-5 kHz, ~5e-4 at 50 kHz
    Expected resolution, 30 records, nseg 2^18, coherence^2 ~ 0.4:
        per bin  sqrt((1-g2)/(2 n g2)) ~ 0.023 rad ;  averaged over 200 Hz-5 kHz (~126 bins) ~ 0.002

    *** THE CONFOUND THIS TABLE MISSED, recorded because it turned out to dominate everything ***
    The three rows above are the ones the first draft thought of. It did NOT list the AMBIENT
    PICKUP'S OWN PHASE - even though the write-up already establishes that the pickup is coherent
    (0.90), sits at 1-5 kHz, and is DIFFERENT between the two probe loops. A coherent interferer
    carrying a phase of its own, landing inside the primary band, is precisely what this measurement
    is most vulnerable to, and it is ~25x larger than everything tabulated above.
    Measured from pgnd_1..6 (both tips on one point of the cage): the pickup carries 88.5 % of its
    power in 800-1600 Hz, with |coherence| 0.989 and arg S12 = +0.79 rad.

    => PRIMARY BAND 200 Hz - 5 kHz, EXCLUDING 800-1600 Hz. The exclusion is defined by the pgnd
       POWER spectrum - an independent dataset - and not by any node-data phase. Within the notched
       band every known confound is 5-50x below the resolution. The un-notched value and the full
       band-by-band breakdown are always reported alongside, never replaced.

Run:  python3 equilibrium.py          prints the frozen predictions and the self-test
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_figures as M          # noqa: E402  - measured components, load(), bandpass()
import kappa_ladder as K          # noqa: E402  - FrontEnd / FE_PAPER pole model

PRIMARY_BAND = (200.0, 5000.0)     # where the confounds are provably below resolution
DIAGNOSTIC_BAND = (5000.0, 50000.0)
NSEG = 1 << 18                     # 38 Hz resolution at 10 MSa/s
SIGMA_DETECT = 5.0


# ---------------------------------------------------------------- datasets
class Dataset:
    """A board state: its network, its two series resistors, and its records."""

    def __init__(self, tag, prefix, n, r1, r2, rc, rg=np.inf, fmt="{p}_{i}.npz"):
        self.tag, self.prefix, self.n = tag, prefix, n
        self.r1, self.r2, self.rc, self.rg = r1, r2, rc, rg
        self.fmt = fmt
        self.G = np.array([[1 / r1 + 1 / rc, -1 / rc],
                           [-1 / rc, 1 / r2 + 1 / rc + (1 / rg if np.isfinite(rg) else 0.0)]])
        self.C = np.diag([M.C1, M.C2])

    def files(self):
        return [self.fmt.format(p=self.prefix, i=i) for i in range(1, self.n + 1)]

    def tau(self):
        """Relaxation times [s] from the measured R and C: eigenvalues of C^-1 G."""
        lam = np.linalg.eigvals(np.linalg.solve(self.C, self.G)).real
        return 1.0 / lam.max(), 1.0 / lam.min()      # (fast, slow)


DATASETS = [
    Dataset("gnd2", "gnd2", 30, M.RN1, M.RN2, M.RC, M.RG, "{p}_{i}.npz"),
    Dataset("r1  (kappa 3.0)", "r1", 30, 1006.5, 990.0, 1000.6, np.inf, "{p}_{i:02d}.npz"),
    Dataset("r2  (kappa 5.3)", "r2", 30, 2172.0, 2174.3, 1000.6, np.inf, "{p}_{i:02d}.npz"),
    Dataset("r3  (kappa 10.4)", "r3", 30, 4685.6, 4692.3, 1000.6, np.inf, "{p}_{i:02d}.npz"),
    Dataset("r4b (kappa 21.0)", "r4b", 30, 10049.0, 9914.0, 1000.6, np.inf, "{p}_{i:02d}.npz"),
]


# ---------------------------------------------------------------- the model
def drive_psd(f, ds, fe=None, rho_v=1.0):
    """Injected-current PSD at each node, including the front-end poles.

    rho_v multiplies channel A's drive VOLTAGE - the single common free parameter of P2.
    Only |H|^2 enters: the two channels are independent, so the poles shape each drive's
    magnitude and contribute no phase of their own.
    """
    fe = fe or K.FE_PAPER
    va = fe.drive(0) * np.abs(fe.H(f, 0)) * rho_v
    vb = fe.drive(1) * np.abs(fe.H(f, 1))
    return (va / ds.r1) ** 2, (vb / ds.r2) ** 2


def model_cross_spectrum(f, ds, fe=None, rho_v=1.0, cp1=0.0, cp2=0.0):
    """S(f) = H diag(s1, s2) H^dagger with H = (G + i.2.pi.f.C)^-1.  cp* are probe capacitances."""
    f = np.atleast_1d(np.asarray(f, float))
    s1, s2 = drive_psd(f, ds, fe, rho_v)
    s1, s2 = np.broadcast_to(s1, f.shape), np.broadcast_to(s2, f.shape)
    C = ds.C + np.diag([cp1, cp2])
    out = np.empty((len(f), 2, 2), complex)
    for i, fi in enumerate(f):
        H = np.linalg.inv(ds.G + 2j * np.pi * fi * C)
        out[i] = H @ np.diag([s1[i], s2[i]]) @ H.conj().T
    return out


def model_phase(f, ds, **kw):
    """Predicted arg S12(f) [rad]. Zero for a reversible (detailed-balance) system."""
    S = model_cross_spectrum(f, ds, **kw)
    return np.angle(S[:, 0, 1])


def band_phase(f, S, lo, hi, weight=True):
    """Coherence-weighted mean of arg S12 over a band.

    Weighting by |S12| is the right estimator: bins where the two nodes barely share power carry
    almost no phase information, and an unweighted mean would let them dominate the variance.
    """
    m = (f >= lo) & (f < hi)
    z = S[m, 0, 1]
    if not weight:
        return float(np.mean(np.angle(z)))
    return float(np.angle(z.sum()))      # |S12|-weighted circular mean


# ---------------------------------------------------------------- the estimator
def spectral_matrix(X, Y, dt, nseg=NSEG, step=None):
    """Full 2x2 COMPLEX cross-spectral matrix, Welch-averaged.

    Lifted unchanged from find_floor.py. The Hanning window is real and symmetric, so it scales
    the spectrum without rotating it - it cannot manufacture an imaginary part.
    """
    acc, n = np.zeros((nseg // 2 + 1, 2, 2), complex), 0
    for x, y in zip(X, Y):
        a, k = record_spectrum(x, y, dt, nseg, step)
        acc += a
        n += k
    return normalise(acc, n, dt, nseg)


def record_spectrum(x, y, dt, nseg=NSEG, step=None):
    """UNNORMALISED accumulator for ONE record pair, plus its segment count.

    Streaming form of the above: the 150 records are ~100 MB each and will not all fit in memory.
    """
    w = np.hanning(nseg)
    step = step or nseg // 2
    acc, n = np.zeros((nseg // 2 + 1, 2, 2), complex), 0
    for i in range(0, len(x) - nseg + 1, step):
        A = np.fft.rfft((x[i:i + nseg] - x[i:i + nseg].mean()) * w)
        B = np.fft.rfft((y[i:i + nseg] - y[i:i + nseg].mean()) * w)
        acc[:, 0, 0] += np.abs(A) ** 2
        acc[:, 1, 1] += np.abs(B) ** 2
        acc[:, 0, 1] += A * np.conj(B)
        n += 1
    acc[:, 1, 0] = np.conj(acc[:, 0, 1])
    return acc, n


def normalise(acc, n, dt, nseg=NSEG):
    w = np.hanning(nseg)
    return np.fft.rfftfreq(nseg, dt), acc / n * (2 * dt / (w ** 2).sum())


def areal_velocity(x, y, dt):
    """<x1 xdot2 - x2 xdot1> / 2 - the phase-plane circulation rate. Zero iff reversible.

    Independent time-domain route to the same quantity as the odd part of the cross-spectrum;
    the two must agree (verification step 3). Centred differences, edges dropped.
    """
    xd = (x[2:] - x[:-2]) / (2 * dt)
    yd = (y[2:] - y[:-2]) / (2 * dt)
    return 0.5 * float(np.mean(x[1:-1] * yd - y[1:-1] * xd))


def measure(ds, band=PRIMARY_BAND, nseg=NSEG, verbose=True):
    """Per-record and pooled band phase, the surrogate null, and the areal velocity.

    Streams: each record is loaded once, contributes to the real pooled accumulator, to its own
    per-record phase, to the time-domain areal velocity, and (paired with the PREVIOUS record's
    node-2 trace) to the surrogate accumulator. The surrogate pairing is independent by
    construction, so its odd part must come out zero - that is the estimator's own null.
    """
    files = ds.files()
    tot, ntot, per, areal = None, 0, [], []
    sur_tot, sur_n, prev_y, dt = None, 0, None, None
    for j, nm in enumerate(files):
        dt, x, y = M.load(nm)
        acc, n = record_spectrum(x, y, dt, nseg)
        f, S = normalise(acc, n, dt, nseg)
        per.append(band_phase(f, S, *band))
        xb, yb = M.bandpass(x, dt, *band), M.bandpass(y, dt, *band)
        areal.append(areal_velocity(xb, yb, dt))
        tot = acc if tot is None else tot + acc
        ntot += n
        if prev_y is not None:
            sacc, sn = record_spectrum(x, prev_y, dt, nseg)
            sur_tot = sacc if sur_tot is None else sur_tot + sacc
            sur_n += sn
        prev_y = y
        if verbose and (j + 1) % 10 == 0:
            print(f"      {ds.tag}: {j + 1}/{len(files)}", flush=True)
    f, S = normalise(tot, ntot, dt, nseg)
    _, Ssur = normalise(sur_tot, sur_n, dt, nseg)
    per, areal = np.array(per), np.array(areal)
    return dict(f=f, S=S, S_sur=Ssur, per=per, dt=dt, areal_per=areal,
                phase=band_phase(f, S, *band),
                se=float(per.std(ddof=1) / np.sqrt(len(per))),
                sur_phase=band_phase(f, Ssur, *band),
                hi=band_phase(f, S, *DIAGNOSTIC_BAND),
                areal=float(areal.mean()),
                areal_se=float(areal.std(ddof=1) / np.sqrt(len(areal))))


# ---------------------------------------------------------------- self-test of the model
def _selftest():
    """With C1 = C2 and equal drives the cross-spectrum must be exactly real."""
    ds = Dataset("sym", "-", 0, 5000.0, 5000.0, 6800.0)
    ds.C = np.diag([23.5e-9, 23.5e-9])
    f = np.geomspace(10, 1e5, 200)
    s = 1e-20
    H = np.array([np.linalg.inv(ds.G + 2j * np.pi * fi * ds.C) for fi in f])
    S12 = np.einsum("fij,jk,flk->fil", H, np.diag([s, s]), H.conj())[:, 0, 1]
    worst = np.max(np.abs(S12.imag) / np.abs(S12))
    ok = worst < 1e-12
    print(f"  self-test  equal C, equal drives -> |Im S12|/|S12| max = {worst:.2e}   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print(__doc__.split("Run:")[0])
    print("=" * 78)
    print("MODEL SELF-TEST\n")
    _selftest()

    print("\n" + "=" * 78)
    print("P1 / P3 - FROZEN PREDICTIONS, from measured components only, before any record is read\n")
    f = np.geomspace(PRIMARY_BAND[0], DIAGNOSTIC_BAND[1], 400)
    print(f"  {'dataset':18s} {'tau_slow':>10s} {'tau_fast':>10s} "
          f"{'arg S12, 200Hz-5kHz':>21s} {'5-50kHz':>12s}")
    for ds in DATASETS:
        tf, ts = ds.tau()
        S = model_cross_spectrum(f, ds)
        p_lo = band_phase(f, S, *PRIMARY_BAND)
        p_hi = band_phase(f, S, *DIAGNOSTIC_BAND)
        print(f"  {ds.tag:18s} {ts*1e6:9.1f}us {tf*1e6:9.2f}us "
              f"{p_lo:+20.2e} {p_hi:+11.2e}")

    print("\n  Common front-end voltage ratio rho_V (identical for all five datasets):")
    fe = K.FE_PAPER
    for fq in (200.0, 1000.0, 5000.0, 50000.0):
        r = (fe.drive(0) * abs(fe.H(fq, 0))) / (fe.drive(1) * abs(fe.H(fq, 1)))
        print(f"    {fq:8.0f} Hz   rho_V = {r:.6f}")
    print("\n  DECISION RULE: detection needs >= 5 sigma AND magnitude above the confound sum.")
    print("  The surrogate null must pass first, or nothing is reported.")
