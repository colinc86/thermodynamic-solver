#!/usr/bin/env python3
"""Extended error model for a thermodynamic linear solver - PRE-REGISTERED PREDICTIONS.

WRITTEN BEFORE THE LADDER DATA EXISTS. The point of this file is to commit to falsifiable
predictions in advance so that whatever the ladder shows cannot be fitted after the fact.
Do not edit the PREDICTIONS block once captures begin; add results alongside it.

THE CLAIM UNDER TEST
    Aifer et al. (2308.05660) and the thermodynamic-AI line that follows it bound runtime
    with eps = sqrt(2*tau/T): error falls as 1/sqrt(T), so accuracy is bought with time.
    That is SAMPLING error only. Physical hardware also carries a systematic floor -
    coupling asymmetry, unequal drive, stray reactance, amplifier excess noise - which
    integration cannot remove. Their complexity analysis has no term for it.

    Standard perturbation theory for a matrix inverse says that floor is amplified by the
    condition number: d(G^-1) = -G^-1 dG G^-1, so ||d(G^-1)||/||G^-1|| <= kappa ||dG||/||G||.

        eps_total^2(T, kappa) = 2*kappa*tau_fast/T  +  eps_sys^2(kappa)
        eps_sys(kappa)        = A * delta * kappa^p

    p = 1 is the null hypothesis from numerical analysis. p = 0 would mean the floor is
    conditioning-independent and the technology is far healthier than this model fears.

WHY IT MATTERS
    Time to a target error T = 2*kappa*tau_fast / (eps^2 - eps_sys^2) DIVERGES at finite
    kappa. There is a hardest problem the device can solve at ANY runtime:

        kappa_max = (eps_target / (A*delta))^(1/p)

    The claimed advantage of thermodynamic computing is on ill-conditioned problems. If
    p = 1, ill-conditioning is also what destroys its accuracy, and the two effects fight.
    Which one wins is an experimental question nobody has asked.

THE ERROR METRIC  (revised 2026-09-15 - see kappa_ladder.py v2.1)
    Sigma is measured only up to the unknown scale c, so the observable must be scale-free.
    rho12 is NOT it: for the symmetric ladder rho = (kappa-1)/(kappa+1) saturates at 1 and its
    sensitivity to any error falls as 1/kappa. The observable is the measured condition number,

        kappa_meas = var(v1 + v2) / var(v1 - v2)

    exact for the symmetric network (common and differential modes ARE its eigenvectors), and
    its relative error is the kappa-amplified soft-eigenvalue error that perturbation theory
    predicts. eps_sys below is the relative systematic error on kappa_meas, tested against the
    band-limited, pole-weighted model's own kappa_meas (which runs ~1.2x the eigenvalue ratio).

Run:  python3 error_model.py
"""
import numpy as np

# ---------------------------------------------------------------- PREDICTIONS (frozen)
P_HYPOTHESES = {
    "p = 1   (conditioning-amplified, numerical-analysis null)": 1.0,
    "p = 0.5 (partial cancellation)": 0.5,
    "p = 0   (floor independent of conditioning)": 0.0,
}

# History of this coefficient, all before any ladder data:
#   0.034  09-12  four untrusted records, single board state
#   0.018  09-13  30-record plateau, matched caps, probe grounds on the shield (5.1 % before that)
#   <=0.007 09-14  the 1.8 % was itself probe-lead pickup: subtracting the measured tips-on-cage
#                  covariance brings rho to 0.1-0.2 % of the model. The remaining bound comes from
#                  the +/-40 % geometry factor on that subtraction. This is an UPPER BOUND on the
#                  intrinsic (component/coupling) floor at kappa 2.21, not a measurement of it.
#                  With the ladder's gain increase the pickup share drops ~84x and rung 1 measures
#                  the intrinsic floor directly to ~0.05 %.
KAPPA_NOW = 2.21
E_SYS_NOW = 0.007          # upper bound on the intrinsic floor on rho12 at kappa 2.21

# ---------------------------------------------------------------- RESULTS (added as they land)
# 2026-09-15  rung 4, kappa_model 21.0 (nominal 10k/10k/1k, R4 out), 30 records, all trusted:
#     kappa_meas = 23.970 +/- 0.095   vs network-alone model 24.034   ->  -0.27 % +/- 0.40 %
#     pre-registered floor at kappa 21:  p=1  6.7 %   p=0.5  2.2 %   p=0  0.70 %
#     p = 1 and p = 0.5 both excluded; consistent with p = 0.  (Superseded by the 09-17
#     re-evaluation with measured resistors; see the final exclusion table below.)
#     The floor does not grow with kappa to at least 21. Caveats: nominal component values
#     (DMM readings pending), single rung, pickup term crude. See notebook 2026-09-15.
# 2026-09-16  rung 3, kappa_model 10.4 (nominal 4.7k/4.7k/1k, R4 out), 30 records, all trusted:
#     kappa_meas = 13.190 +/- 0.060  vs model 13.284  ->  -0.70 % +/- 0.45 %   (superseded, see 09-17)
# 2026-09-16  rung 2, kappa_model 5.4 (nominal 2.2k/2.2k/1k, R4 out), 30 records, all trusted:
#     kappa_meas = 7.361 +/- 0.020  vs model 7.255  ->  +1.46 % +/- 0.28 % over all 30,
#     (superseded, see 09-17: against the measured-resistor model 7.182 this is +2.50 %,
#      and the settled last ten are ~+1.5 % rather than the ~+0.5 % below)
#     BUT the run started unsoaked and drifted -2.3 %; the settled last ten give ~+0.5 %.
#     Floors at 5.4 are 1.71 / 1.09 / 0.70 % - within 1 % of each other; this rung constrains
#     the slope, not the exponent on its own. Recorded as the settled value with a wide error.
# 2026-09-16  rung 1, kappa_model 3.0 (nominal 1k/1k/1k, R4 out), 30 records, all trusted, no drift:
#     kappa_meas = 4.290 +/- 0.008  vs model 4.122  ->  +4.09 % +/- 0.20 %  (20 sigma)
#     (superseded, see 09-17: against the measured-resistor model 4.115 this is +4.27 %)
#     EXCEEDS every pre-registered floor (<= 0.95 % at kappa 3) by ~16 sigma. A +2.05 % error on rho
#     amplified by Vc/Vd's sensitivity at low kappa. Mechanism open; see notebook.
#
# LADDER VERDICT (final, all resistors measured): deviation +4.27, +2.50, -0.44, -0.05 % at
# kappa 3.0, 5.3, 10.4, 21.0. The floor does NOT grow with kappa - it shrinks. p = 1 excluded
# by the two high rungs. The pre-registered form delta*kappa^p describes none of it; what
# exists is a LOW-kappa excess of unknown origin.
# 2026-09-17  R1/R2 MEASURED (SDM3045X): 1006.5/990.0, 2172.0/2174.3, 4685.6/4692.3, 10049/9914 ohm.
#     R3 = 1000.6 ohm (same part, every rung). Every network resistor is now measured.
#     Not 0.1 % parts - pairs differ by up to 1.7 %. Model re-evaluated with each pair's two values.
#     The kappa-3 excess is NOT the resistors: the 1.7 % mismatch moves the model 0.12 %.
#     High rungs got cleaner. FINAL exclusion, z = (predicted floor - |measured deviation|) / sigma:
#         kappa 21.0:  p=1 16.6 sigma   p=0.5  5.3 sigma   p=0  1.6 sigma (consistent)
#         kappa 10.4:  p=1  6.2 sigma   p=0.5  2.4 sigma   p=0  0.6 sigma (consistent)
#     Earlier comments in this block quoting 17 and 8 sigma predate the measured resistors and
#     used prediction/sigma rather than (prediction - |deviation|)/sigma. The two low rungs do not
#     exclude anything: they EXCEED every prediction (kappa 3 by 16 sigma, kappa 5.3 by 2.9 sigma
#     even against p = 1), which is the open low-kappa anomaly, not support for amplification.
# 2026-09-20  Two notes added alongside, nothing above altered.
#     (a) The exclusion z above mixes rounding conventions. 16.6 sigma at kappa 21 uses the
#     unrounded standard error (0.3961 %); 6.2 sigma at kappa 10.4 uses the rounded 0.46 rather
#     than 0.4529, which gives 6.3. The write-up quotes 16.6 and 6.3, both unrounded. No
#     conclusion moves; recorded so the two documents do not appear to disagree.
#     (b) The component-accuracy block printed at the end of __main__ hardcodes A = 1
#     (delta <= eps_target/kappa) instead of calling a_delta(), so it is a generic p = 1 scaling
#     and NOT a result from this board. The write-up is right not to quote it. Left as written,
#     since it is pre-registration text.
RESULTS = {21.0: (-0.0005, 0.0040),   # kappa_model: (relative error on kappa_meas, std err)
           10.4: (-0.0044, 0.0046),   # all four with every network resistor measured (R3 = 1000.6)
            5.3: (+0.0250, 0.0028),   # full 30, drifting -2.3 % (unsoaked); settled ~ +0.015
            3.0: (+0.0427, 0.0020)}
TAU_FAST_LADDER = 11e-6    # pinned by design, see kappa_ladder.py
LADDER_KAPPA = np.array([2.1, 4.7, 10.0, 21.4, 47.0, 100.4, 213.7])   # kappa_ladder.py v2.1, C = 23.505 nF


def a_delta(p):
    """Back the hardware error coefficient out of the one board we have measured."""
    return E_SYS_NOW / KAPPA_NOW ** p


def eps_sys(kappa, p):
    return a_delta(p) * kappa ** p


def eps_total(T, kappa, p, tau_fast=TAU_FAST_LADDER):
    return np.sqrt(2 * kappa * tau_fast / T + eps_sys(kappa, p) ** 2)


def time_to(eps_target, kappa, p, tau_fast=TAU_FAST_LADDER):
    """Runtime for a target error, or inf if the systematic floor already exceeds it."""
    head = eps_target ** 2 - eps_sys(kappa, p) ** 2
    return np.inf if head <= 0 else 2 * kappa * tau_fast / head


def kappa_max(eps_target, p):
    return np.inf if p == 0 else (eps_target / a_delta(p)) ** (1.0 / p)


if __name__ == "__main__":
    print(__doc__.split("Run:")[0].split("THE ERROR METRIC")[0])
    print("=" * 78)
    print("PRE-REGISTERED: predicted systematic floor at each ladder rung\n")
    print("  kappa " + "".join(f"{n.split('(')[0].strip():>14}" for n in P_HYPOTHESES))
    for k in LADDER_KAPPA:
        row = "".join(f"{100*eps_sys(k, p):>13.1f}%" for p in P_HYPOTHESES.values())
        print(f"  {k:6.1f}{row}")
    print("\n  The columns separate ~20x at rung 5 (kappa 47) and ~100x at rung 7. Rung 5 is the")
    print("  deciding rung: rungs 6-7 carry 5-18 % of probe pickup on the differential mode")
    print("  (kappa_ladder.py v2.1) and cannot separate p = 0.5 from p = 0.")

    print("\n" + "=" * 78)
    print("CONSEQUENCE: hardest problem solvable at ANY runtime\n")
    print(f"  {'target':>8}" + "".join(f"{n.split('(')[0].strip():>14}" for n in P_HYPOTHESES))
    for eps in (0.10, 0.03, 0.01, 0.003):
        row = ""
        for p in P_HYPOTHESES.values():
            km = kappa_max(eps, p)
            row += f"{'unbounded':>14}" if np.isinf(km) else f"{km:>14.1f}"
        print(f"  {100*eps:7.1f}%{row}")

    print("\n" + "=" * 78)
    print("CONSEQUENCE: runtime to 1 % vs kappa (p = 1), against their prediction\n")
    print(f"  {'kappa':>6}{'theirs (stat only)':>22}{'with floor':>16}")
    for k in LADDER_KAPPA:
        t_them = 2 * k * TAU_FAST_LADDER / 0.01 ** 2
        t_us = time_to(0.01, k, 1.0)
        us = "UNREACHABLE" if np.isinf(t_us) else f"{t_us:.2f} s"
        print(f"  {k:6.1f}{t_them:>20.2f} s{us:>16}")

    print("\n" + "=" * 78)
    print("WHAT THIS WOULD MEAN FOR THERMODYNAMIC NGD\n")
    print("  Natural gradient descent solves F x = g with F the Fisher matrix. Fisher and")
    print("  Hessian spectra in deep networks are notoriously ill-conditioned, kappa ~ 1e4")
    print("  to 1e8 undamped. Under p = 1 with this board's error coefficient:\n")
    for k in (1e2, 1e4, 1e6):
        print(f"    kappa {k:.0e}   systematic floor {100*eps_sys(k, 1.0):,.0f} %")
    print("\n  i.e. undamped thermodynamic NGD would be pure noise. But NGD is never run")
    print("  undamped - Tikhonov damping (F + lam*I) caps the effective conditioning at")
    print("  kappa_eff ~ lam_max/lam. So the real question is constructive, not fatal:\n")
    print("    How much damping brings kappa_eff inside the device's reach, and does NGD")
    print("    still beat SGD at that damping?\n")
    print("  That is answerable in simulation on the 3090 using the error model MEASURED")
    print("  here - and it is the honest bridge from this breadboard to the AI claim.")
    print("\n  Required component/coupling accuracy for a target kappa_eff at 1 % error:")
    for k in (10, 100, 1000):
        print(f"    kappa_eff {k:5d}  ->  delta <= {100*0.01/k:.4f} %  "
              f"({'reachable' if 0.01/k > 1e-4 else 'beyond trimmed analog hardware'})")
