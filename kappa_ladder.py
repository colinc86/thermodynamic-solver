#!/usr/bin/env python3
"""The condition-number ladder, modelled — v2.1 (2026-09-15).

What v1 established: the naive ladder is vacuous because shrinking R3 raises kappa without
touching tau_slow; the meaningful ladder pins tau_fast (the hardware's speed) and lets kappa push
tau_slow out. Symmetric network, R4 out, R1 = R2 = RL swept, Rc solved per rung so tau_fast is
constant.

What v2 added, from measurements of 2026-09-13/14:
  FRONT-END POLES.  Each OPA2189 stage is a pole at GBW/gain. The ladder raises stage 1 to x101
      (R7, R8 -> ~100 k) so both stages sit at 137 kHz. Modelled.
  MEASURED AMPLIFIER NOISE.  e_n ~ 9 nV/rtHz (inputs shorted), i_n ~ 0.83 pA/rtHz (R-dependence).
      CAVEAT: the e_n figure was derived from a shorted-input capture that still contained the
      probe pickup; it is ~10 % high, and the model's absolute amplitude is now ~+14 % rather than
      the paper's -17 %. The truth is between. Absolute amplitude is NOT what the ladder measures.
  MEASURED PROBE PICKUP.  Tips on the shield, board OFF: 140 / 99 uV in-band, coherence 0.93.
      Additive, post-network, entered as a fixed covariance per band (chk_1..6) x GEOM (1.0 +/- 0.4).
  WHY THE GAIN GOES UP.  Pickup enters after the network; gain acts before it; scale-free results
      are gain-invariant; the amplifier's own excess scales with gain. Only the pickup shrinks.

What v2.1 CORRECTS, from running v2:
  rho IS THE WRONG OBSERVABLE.  For the symmetric network rho = (kappa-1)/(kappa+1): it saturates
      at 1 and its sensitivity to any error falls as 1/kappa. The right observable is the measured
      condition number itself,
          kappa_meas = var(v1 + v2) / var(v1 - v2) = V_common / V_diff,
      whose relative error IS the kappa-amplified quantity (the soft eigenvalue's error ~ delta x
      kappa). The floor predictions are restated on kappa_meas.
  THE PICKUP'S DAMAGE IS TO THE DIFFERENTIAL MODE.  The network's differential mode carries 1/kappa
      of the power, so the pickup's small differential component (it is 87 % coherent, so 13 % of it
      is differential) can be a large fraction of V_diff even when it is 1 % of the total. That
      fraction is computed and printed; it is the number that decides which rungs are usable.
  THE DRIVE ASYMMETRY IS ALSO AMPLIFIED.  c1/c2 = 0.985 makes the model's own Sigma.G off-diagonal
      non-zero, and at high kappa the model predicts tens of percent. That is the same mechanism
      the ladder is built to measure, visible in the model. The test at high kappa is against the
      model's prediction, not against zero.
  PER-RECORD SCATTER DEPENDS ON THE RUNG.  eps ~ sqrt(2 (tau_slow + tau_fast) / T) per record,
      calibrated x1.4 to the measured 3.0 % on rho at kappa 2.2; at the top rung tau_slow = 2.35 ms
      and a 0.6 s record scatters ~12 %. N is sized to split adjacent hypotheses, not to resolve p=0.

VALIDATION.  The model is run on the CURRENT board (asymmetric, R4 in, paper gain) with the chk
pickup added and compared to ver_1..6.

Run:  python3 kappa_ladder.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_figures as M                                   # load(), bandpass(), constants

K_B, T_K = 1.380649e-23, 297.0
T_REC = 0.6                     # one record [s]

# ---------------------------------------------------------------- front end
GBW = 14e6                      # OPA2189 gain-bandwidth, datasheet typical
R9, R10 = 991.0, 1003.5         # stage-1 gain-set resistors, measured
R13, R15 = 99650.0, 987.2       # stage-2, channel A, measured
R14, R16 = 99680.0, 983.8       # stage-2, channel B, measured
R11, R12 = 9883.0, 9911.0       # the Johnson sources, measured
EN_MEAS = 9.05e-9               # effective e_n, MEASURED (inputs shorted; ~10 % high, see docstring)
IN_MEAS = 0.83e-12              # i_n, MEASURED (R-dependence of the excess)   datasheet 165e-15
# Neither figure affects the ladder: kappa_meas is a ratio and the front end scales both modes
# alike, so swapping these for the OPA2189 datasheet values leaves every rung unchanged to four
# decimals. They matter only to the model's ABSOLUTE amplitude, which the ladder does not use.


class FrontEnd:
    """Two non-inverting stages; each contributes a pole at GBW / (its gain)."""
    def __init__(self, r7, r8, label):
        self.label = label
        self.g1 = (1 + r7 / R9, 1 + r8 / R10)
        self.g2 = (1 + R13 / R15, 1 + R14 / R16)
        self.gain = (self.g1[0] * self.g2[0], self.g1[1] * self.g2[1])
        self.efb = (np.sqrt(4 * K_B * T_K * (r7 * R9 / (r7 + R9))),
                    np.sqrt(4 * K_B * T_K * (r8 * R10 / (r8 + R10))))

    def H(self, f, ch):
        f1, f2 = GBW / self.g1[ch], GBW / self.g2[ch]
        return self.gain[ch] / ((1 + 1j * f / f1) * (1 + 1j * f / f2))

    def drive(self, ch):
        r = (R11, R12)[ch]
        return np.sqrt(4 * K_B * T_K * r + EN_MEAS ** 2 + self.efb[ch] ** 2 + (IN_MEAS * r) ** 2)

    def poles_khz(self):
        return (GBW / self.g1[0] / 1e3, GBW / self.g2[0] / 1e3)


FE_PAPER = FrontEnd(9944.0, 10081.0, "paper  x11 x x101")        # measured R7, R8
FE_LADDER = FrontEnd(100e3, 100e3, "ladder x101 x x101")         # R7, R8 NOMINAL - TO MEASURE


# ---------------------------------------------------------------- network
C = 23.505e-9                   # the matched pair, measured


def conductance(rl1, rl2, rc, rg=np.inf):
    return np.array([[1 / rl1 + 1 / rc, -1 / rc],
                     [-1 / rc, 1 / rl2 + 1 / rc + (0.0 if np.isinf(rg) else 1 / rg)]])


def coupling_for(rl, tau_fast):
    return 2.0 / (C / tau_fast - 1.0 / rl)


def eig(G):
    lam = np.sort(np.linalg.eigvals(G / C).real)
    return lam[0], lam[1]                                     # slow, fast [1/s]


# ---------------------------------------------------------------- pickup (measured, additive)
_PK = {}


def pickup_cov(band):
    """Band-limited covariance of the tips-on-cage capture chk_1..6 [V^2]. Cached per band."""
    if band not in _PK:
        X, Y = [], []
        for nm in [f"chk_{i}.npz" for i in range(1, 7)]:
            dt, x, y = M.load(nm)
            X.append(M.bandpass(x, dt, *band)); Y.append(M.bandpass(y, dt, *band))
        _PK[band] = np.cov(np.vstack([np.concatenate(X), np.concatenate(Y)]))
    return _PK[band]


GEOM = 1.0                      # tips-on-cage -> tips-on-node geometry factor, +/- 0.4
# 2026-09-20: 1.0 is the deliberate UPPER BOUND (Limitations: "subtracting the pickup is an
# upper bound"), not a best estimate. The write-up's empirical figure is ~0.2 - adding the full
# pickup back predicts +9 % on rho_12 where +1.8 % is seen - which sits outside the +/- 0.4 band
# quoted on this line. The ladder result's published model uses GEOM = 0 (network alone) and
# pickup_sensitivity.py sweeps 0 / 0.2 / 0.5 / 1.0 rather than relying on either default.


# ---------------------------------------------------------------- prediction
def sigma_net(G, fe, rl1, rl2, band, n=6000):
    ff = np.linspace(band[0], band[1], n)
    S = np.zeros((2, 2))
    for f in ff:
        si = np.diag([(fe.drive(0) * abs(fe.H(f, 0)) / rl1) ** 2,
                      (fe.drive(1) * abs(fe.H(f, 1)) / rl2) ** 2])
        mi = np.linalg.inv(G + 2j * np.pi * f * np.diag([C, C]))
        S += np.real(mi @ si @ mi.conj().T)
    return S * (ff[1] - ff[0])


def observables(S, G):
    m = S @ G
    vc = S[0, 0] + S[1, 1] + 2 * S[0, 1]                   # common-mode variance
    vd = S[0, 0] + S[1, 1] - 2 * S[0, 1]                   # differential-mode variance
    return dict(v1=np.sqrt(S[0, 0]) * 1e3, v2=np.sqrt(S[1, 1]) * 1e3,
                ratio=np.sqrt(S[0, 0] / S[1, 1]), rho=S[0, 1] / np.sqrt(S[0, 0] * S[1, 1]),
                offdiag=m[0, 1] / np.sqrt(abs(m[0, 0] * m[1, 1])) * 100,
                vc=vc, vd=vd, kappa=vc / vd)                 # kappa exact for the symmetric network


def predict(G, fe, rl1, rl2, band, geom=GEOM):
    Sn = sigma_net(G, fe, rl1, rl2, band)
    Pk = geom * pickup_cov(band)
    St = Sn + Pk
    on, ot, ok = observables(Sn, G), observables(St, G), observables(Pk, G)
    share_total = (np.diag(St) - np.diag(Sn)) / np.diag(St) * 100
    share_diff = ok["vd"] / ot["vd"] * 100                  # pickup's share of the DIFFERENTIAL mode
    return on, ot, share_total, share_diff


# ---------------------------------------------------------------- pre-registered floor
KAPPA_NOW = 2.21
DELTA_NOW = 0.007               # UPPER BOUND on the intrinsic floor at kappa 2.21 (notebook 09-14)


def floor(kappa, p):
    """Predicted relative systematic error on kappa_meas."""
    return DELTA_NOW * (kappa / KAPPA_NOW) ** p


def scatter_per_record(tau_s, tau_f):
    """Statistical scatter of kappa_meas from one 0.6 s record, relative.
    sqrt(2 tau/T) per mode, in quadrature, x1.4 calibrated to the measured 3.0 % on rho."""
    return 1.4 * np.sqrt(2 * (tau_s + tau_f) / T_REC)


def records_to_split(f_a, f_b, eps_rec, sigma=3.0):
    """Records needed to separate two floor hypotheses at `sigma`."""
    gap = abs(f_a - f_b)
    return max(1, int(np.ceil((sigma * eps_rec / gap) ** 2)))


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    print("front ends:")
    for fe in (FE_PAPER, FE_LADDER):
        print(f"  {fe.label:20s} gain A {fe.gain[0]:7.0f}  B {fe.gain[1]:7.0f}   poles "
              f"{fe.poles_khz()[0]:6.0f} / {fe.poles_khz()[1]:6.0f} kHz   drive A {fe.drive(0)*1e9:5.2f} nV/rtHz")

    # ------------------------------------------------ validation on the current board
    print("\n=== VALIDATION: current board (asymmetric, R4 in, paper gain) + measured pickup vs ver_1..6 ===")
    G0 = conductance(4690.0, 4680.0, 6830.0, 9900.0)
    band0 = (200.0, 50000.0)
    on, ot, sh, shd = predict(G0, FE_PAPER, 4690.0, 4680.0, band0)
    X, Y = [], []
    for nm in [f"ver_{i}.npz" for i in range(1, 7)]:
        dt, x, y = M.load(nm); X.append(M.bandpass(x, dt, *band0)); Y.append(M.bandpass(y, dt, *band0))
    ov = observables(np.cov(np.vstack([np.concatenate(X), np.concatenate(Y)])), G0)
    print(f"  {'':22s} {'v1 [mV]':>8s} {'v1/v2':>7s} {'rho':>8s} {'Sig.G offd':>10s} {'Vc/Vd':>7s}")
    for lab, o in (("network alone (model)", on), (f"+ pickup x{GEOM:.1f}", ot), ("ver MEASURED", ov)):
        print(f"  {lab:22s} {o['v1']:8.4f} {o['ratio']:7.4f} {o['rho']:+8.4f} {o['offdiag']:+9.3f} % {o['kappa']:7.3f}")
    print(f"  pickup share: total {sh[0]:.1f} / {sh[1]:.1f} %   of the differential mode {shd:.1f} %")
    print(f"  model+pickup vs measured:  rho {100*(ot['rho']/ov['rho']-1):+.1f} %   Vc/Vd {100*(ot['kappa']/ov['kappa']-1):+.1f} %   "
          f"(network alone: rho {100*(on['rho']/ov['rho']-1):+.1f} %, Vc/Vd {100*(on['kappa']/ov['kappa']-1):+.1f} %)")
    for g in (0.6, 1.4):
        _, o, _, _ = predict(G0, FE_PAPER, 4690.0, 4680.0, band0, geom=g)
        print(f"    geometry x{g:.1f}:  rho {o['rho']:+.4f}  offdiag {o['offdiag']:+.3f} %  Vc/Vd {o['kappa']:.3f}")
    print("  (Vc/Vd is only approximately kappa on this asymmetric board; exact on the symmetric ladder)")

    # ------------------------------------------------ the ladder
    TAU_F = 11e-6
    RLS = [1000., 2200., 4700., 10000., 22000., 47000., 100000.]
    for band in ((200.0, 30000.0), (20.0, 30000.0)):
        print(f"\n=== LADDER  band {band[0]:.0f} Hz - {band[1]/1e3:.0f} kHz   {FE_LADDER.label}   pickup x{GEOM:.1f} ===")
        print(f"  {'RL':>7s} {'Rc':>5s} {'kappa':>6s} {'f_slow':>6s} | {'v1 mV':>6s} {'rho':>6s} | "
              f"{'kappa_meas':>10s} {'+pickup':>8s} {'pk err':>7s} {'pk%Vd':>6s} | "
              f"{'floor p=1':>9s} {'p=.5':>6s} {'p=0':>6s} | {'eps/rec':>7s} {'N(1|.5)':>7s} {'N(.5|0)':>7s}")
        for rl in RLS:
            rc = coupling_for(rl, TAU_F)
            G = conductance(rl, rl, rc)
            ls, lf = eig(G)
            k = lf / ls
            on, ot, sh, shd = predict(G, FE_LADDER, rl, rl, band)
            fl = [floor(k, p) for p in (1.0, 0.5, 0.0)]
            eps = scatter_per_record(1 / ls, 1 / lf)
            n1, n2 = records_to_split(fl[0], fl[1], eps), records_to_split(fl[1], fl[2], eps)
            pk_err = 100 * (ot["kappa"] / on["kappa"] - 1)
            flag = "" if abs(pk_err) < 0.3 * 100 * fl[1] else "   <- pickup err > 30 % of the p=.5 floor"
            print(f"  {rl/1e3:6.1f}k {rc:5.0f} {k:6.1f} {ls/2/np.pi:6.0f} | {ot['v1']:6.3f} {ot['rho']:+6.3f} | "
                  f"{on['kappa']:10.2f} {ot['kappa']:8.2f} {pk_err:+6.1f}% {shd:5.1f}% | "
                  f"{100*fl[0]:8.1f}% {100*fl[1]:5.1f}% {100*fl[2]:5.2f}% | {100*eps:6.1f}% {n1:7d} {n2:7d}{flag}")
        print("  kappa_meas = Vc/Vd from the model, with and without pickup; pk err = the pickup's bias on it.")
        print("  pk%Vd = pickup's share of the differential-mode variance (the one that matters).")
        print("  N(a|b) = records to separate hypotheses a and b at 3 sigma, given eps/rec.")

    # ------------------------------------------------ without the gain change, for the record
    print("\n=== same ladder, band 20 Hz - 30 kHz, WITHOUT the gain change (paper front end) ===")
    for rl in (4700., 22000., 100000.):
        rc = coupling_for(rl, TAU_F); G = conductance(rl, rl, rc); ls, lf = eig(G)
        on, ot, sh, shd = predict(G, FE_PAPER, rl, rl, (20.0, 30000.0))
        print(f"  RL {rl/1e3:6.1f}k  kappa {lf/ls:6.1f}   pickup share: total {sh[0]:5.1f} %, of Vd {shd:5.1f} %   "
              f"kappa_meas {on['kappa']:7.2f} -> {ot['kappa']:7.2f}  ({100*(ot['kappa']/on['kappa']-1):+.0f} %)")
