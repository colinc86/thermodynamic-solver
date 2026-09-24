#!/usr/bin/env python3
"""How much of the ladder result depends on leaving probe pickup out of the model?

The ladder compares a measured kappa_meas = var(v1+v2)/var(v1-v2) against the same
quantity computed from the model. The measurement is a raw covariance of the captured
records, so whatever the probe leads pick up is in it. The model (rung_result.py, and
the model column of the ladder table) is the network alone. That asymmetry is
harmless at the bottom rung, where pickup is 0.3 % of the differential-mode power, and
is not at the top rung, where it is 14 %.

Pickup has its own Vc/Vd of about 10.6. Adding it to a network whose ratio is higher
drags the total down, so including it *raises* the measured-minus-model deviation at
the top rung and barely moves the rest.

How much of it reaches the nodes is not known to better than a factor of a few. The
pickup covariance is measured with both probe tips on one point of the shield, which
over-reads: Limitations puts the fraction actually coupling into the nodes at about a
fifth, from the +9 %-predicted / +1.8 %-seen discrepancy on rho_12. This script sweeps
that fraction and reports what each power law does under it.

Result, and why the ladder result is worded the way it is: p = 1 is excluded under every
coupling, and the kappa = 10.4 rung excludes it on its own. p = 0.5 is excluded only at
coupling near zero. Its exclusion is therefore not a robust claim, and the write-up
does not make it.

    python3 pickup_sensitivity.py            # the sweep behind the ladder result
    python3 pickup_sensitivity.py --selftest
"""
import argparse
import csv
import sys

import numpy as np

import kappa_ladder as K

BAND = (200.0, 30000.0)
ROWS = "data/ladder-rungs.csv"
COUPLINGS = (0.0, 0.2, 0.5, 1.0)
DELTA0, KAPPA0 = 0.7, 2.21          # pre-registered floor, per error_model.py


def load(path=ROWS):
    """The four rungs, from the same CSV the ladder table is built from."""
    with open(path) as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    return [
        dict(kappa=float(r["kappa_model"]), r1=float(r["R1_ohm"]),
             r2=float(r["R2_ohm"]), rc=float(r["R3_ohm"]),
             meas=float(r["kappa_meas"]),
             # std_err is on kappa_meas; the table quotes it relative to the model
             se_pct=100 * float(r["std_err"]) / float(r["kappa_model_bandlimited"]))
        for r in rows
    ]


def deviation(rung, geom, pickup):
    """Measured kappa against the model, in percent, with `geom` x pickup added."""
    G = K.conductance(rung["r1"], rung["r2"], rung["rc"])
    S = K.sigma_net(G, K.FE_PAPER, rung["r1"], rung["r2"], BAND) + geom * pickup
    return 100 * (rung["meas"] / K.observables(S, G)["kappa"] - 1)


def excluded_by(rung, dev, p):
    """Sigmas between |deviation| and the pre-registered floor at this rung.

    Positive: the law over-predicts the deviation. Negative: it under-predicts.
    Either sign is an exclusion; only |value| small means the law survives.
    """
    return (DELTA0 * (rung["kappa"] / KAPPA0) ** p - abs(dev)) / rung["se_pct"]


def table(rungs, pickup):
    for geom in COUPLINGS:
        devs = [deviation(r, geom, pickup) for r in rungs]
        print(f"\n--- pickup x {geom}")
        print(f"{'kappa':>6} {'deviation':>16} | {'p=1':>9} {'p=0.5':>9} {'p=0':>9}")
        for r, d in zip(rungs, devs):
            sig = "".join(f"{excluded_by(r, d, p):>+8.1f}s" for p in (1, 0.5, 0))
            print(f"{r['kappa']:>6.1f} {d:>+9.2f} +- {r['se_pct']:.2f} |{sig}")


def _selftest():
    rungs = load()
    pickup = K.pickup_cov(BAND)

    vc = pickup[0, 0] + pickup[1, 1] + 2 * pickup[0, 1]
    vd = pickup[0, 0] + pickup[1, 1] - 2 * pickup[0, 1]
    assert 9 < vc / vd < 13, f"pickup Vc/Vd {vc / vd:.1f} outside the expected range"

    # geom = 0 must reproduce the deviations already published in the CSV.
    with open(ROWS) as fh:
        pub = [float(r["deviation_pct"])
               for r in csv.DictReader(line for line in fh if not line.startswith("#"))]
    for r, want in zip(rungs, pub):
        got = deviation(r, 0.0, pickup)
        assert abs(got - want) < 0.015, f"kappa {r['kappa']}: {got:+.3f} vs published {want:+.3f}"

    # Pickup must bite at the top rung and not at the bottom one.
    bottom = abs(deviation(rungs[0], 1.0, pickup) - deviation(rungs[0], 0.0, pickup))
    top = abs(deviation(rungs[-1], 1.0, pickup) - deviation(rungs[-1], 0.0, pickup))
    assert bottom < 0.6 and top > 6, f"pickup moved bottom {bottom:.2f}, top {top:.2f}"

    # The two claims the ladder result rests on.
    for geom in COUPLINGS:
        assert max(excluded_by(r, deviation(r, geom, pickup), 1.0) for r in rungs) > 5, \
            f"p=1 not excluded at pickup x{geom}"
    at02 = excluded_by(rungs[-1], deviation(rungs[-1], 0.2, pickup), 0.5)
    assert abs(at02) < 2, f"p=0.5 top rung is {at02:+.1f}s at x0.2, expected consistent"

    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        sys.exit(0)
    table(load(), K.pickup_cov(BAND))
    print("\nPositive sigma: the law over-predicts the deviation. Negative: under-predicts.")
    print("Either way the law is excluded; only a small |sigma| means it survives.")
