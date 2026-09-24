# Data

Distilled datasets behind every figure and number in the paper. Each file carries a `#` comment on
its first line describing how it was produced; the columns are defined below.

The raw captures these come from are ~4 GB of oscilloscope records and are not distributed. The
script that reduces them — [`../make_figures.py`](../make_figures.py) — is included so the pipeline
is auditable: `RAW=/path/to/captures python3 make_figures.py` rebuilds everything here.

Unless stated otherwise, numbers come from the **final board configuration** — matched capacitors,
probe grounds tied to the shield, thermally settled — **thirty independent 0.6 s records** (18 s in
total), every one of which passed the oscilloscope cross-check on both channels, analyzed over
**200 Hz – 50 kHz**.

---

### `results-summary.csv`

The headline table.

| column | meaning |
|---|---|
| `quantity` | `v1_mV`, `v2_mV` — node RMS; `v1_over_v2` — amplitude ratio; `rho12` — correlation between the nodes; `identity_offdiag_pct` — off-diagonal of `Σ·G` as a percentage of its diagonal, which vanishes when `Σ = c·G⁻¹`; `lyapunov_offdiag_pct` — normalized off-diagonal of `GΣC + CΣG`, which vanishes for any diagonal drive |
| `measured` | the estimate from all thirty records pooled |
| `std_err` | standard error of the mean, from the spread across the thirty records |
| `predicted` | from the measured components and the computed gain — nothing fitted |
| `deviation` | percent, except the two `_pct` rows, which are quoted in **percentage points** — a percentage of a near-zero quantity would be meaningless |

`v1` and `v2` carry a real residual (the board is louder than its noise budget allows); the four
scale-free rows are immune to it.

### `components-measured.csv`

| column | meaning |
|---|---|
| `part` | reference designator |
| `role` | what it does in the network |
| `nominal` | marked value (ohms, or farads for C1/C2) |
| `measured` | out-of-circuit multimeter reading, same units |
| `deviation_pct` | percent from nominal |

C1 and C2 are a matched pair, 0.04 % apart, chosen from nine candidates. The original C1 was
26.8 nF — 14 % from C2 — which is what made the naive identity test unusable in earlier drafts.

### `precision-vs-time.csv`

The √T scaling.

| column | meaning |
|---|---|
| `T_ms` | integration window length |
| `n_windows` | number of independent windows averaged; the uncertainty on `eps_measured` is roughly `1/√(2n)` |
| `eps_measured` | relative standard deviation of the variance estimate across windows |
| `eps_theory` | `√(2τ/T)` with τ from the measured R and C — **nothing fitted** |

### `systematic-floor.csv`

Where more integration time stops helping.

| column | meaning |
|---|---|
| `configuration` | `final`, or `before_regrounding` — the same board minutes earlier, before the probe grounds were tied to the shield |
| `T_ms` | window length; rows above 600 ms pool consecutive records |
| `n_windows` | number of windows at that length |
| `scatter` | standard deviation of the windowed ρ₁₂ estimate. Falls as 1/√T regardless of any bias — a bias is common to every window and cancels here |
| `abs_bias` | \|mean of the windowed estimate − model\| |
| `accuracy_rms` | RMS deviation of the windowed estimate **from the model** = √(scatter² + bias²). This is the one that flattens |

The gap that opens between `scatter` and `accuracy_rms` at long T is the systematic floor: the
error that integration cannot remove.

### `ladder-rungs.csv`

The condition-number ladder: does the floor grow with κ?

| column | meaning |
|---|---|
| `kappa_model` | condition number of the network from the measured R and measured C (eigenvalue ratio of `C⁻¹G`) |
| `R1_ohm`, `R2_ohm`, `R3_ohm` | all measured out of circuit. R1, R2 are not a matched pair (up to 1.7 % apart) and the model uses both; R3 is the same 1000.6 Ω part for every rung. R4 removed |
| `n_records` | 0.6 s records at this rung |
| `kappa_meas` | the measured condition number `var(v₁+v₂)/var(v₁−v₂)`, mean over records, band 200 Hz – 30 kHz |
| `std_err` | standard error of `kappa_meas` from the record-to-record spread |
| `kappa_model_bandlimited` | the same quantity from the model — band-limited and amplifier-pole-weighted, so it runs 1.14–1.37× `kappa_model` |
| `deviation_pct`, `deviation_std_err_pct` | `kappa_meas / kappa_model_bandlimited − 1`, in percent, with its error |
| `pred_p1_pct`, `pred_p05_pct`, `pred_p0_pct` | the **pre-registered** floors `0.7 % × (κ/2.21)^p` for p = 1, 0.5, 0 — written before any rung was built |
| `drift_over_run_pct` | linear drift of `kappa_meas` across the 30 records; the κ = 5.3 rung started before the board had settled |

### `psd-node1.csv`, `psd-node2.csv`

| column | meaning |
|---|---|
| `freq_Hz` | frequency, logarithmically sampled to ~350 rows |
| `measured_V_per_rtHz` | amplitude spectral density at the node, thirty records pooled |
| `model_V_per_rtHz` | model from measured components and the computed gain; **no shape parameter is fitted** |

These rows are log-sampled from a linearly-spaced spectrum, so the file is for plotting, not for
statistics. A median taken over these rows lands near 3.5 kHz and gives ×1.16; the ×1.26 in the
write-up is the median over all 4163 linear bins, which lands near 20 kHz where the ratio is
genuinely higher. Band integrals are unaffected — integrate the square, since these are amplitude
densities, and node 1 gives ×1.173 against results-summary's ×1.172.

### `rho-by-band.csv`

| column | meaning |
|---|---|
| `band_lo_Hz`, `band_hi_Hz` | band edges |
| `rho_measured` | band-limited correlation between the two nodes |
| `rho_model` | same band, from the model |

ρ falls several-fold across these bands as the capacitors decouple the nodes. Reproducing that
variation is a stronger test than matching one band-averaged number.

### `fourkt-difference.csv`

| column | meaning |
|---|---|
| `freq_Hz` | bin center (logarithmically binned; the difference of two noisy spectra is itself noisy) |
| `input_referred_nV_per_rtHz` | `S(10 kΩ) − S(1 kΩ)` with the network response divided out and referred to the amplifier input |

Flat ⇒ the R-dependent part of the noise is white. It is close to flat and not exactly: over
300 Hz – 20 kHz the rms scatter about the median is 7.4 %, and the log-log slope of +0.044 amounts
to a ~20 % rise across the band, with the binned median below 2 kHz sitting 10 % under the one
above 6 kHz.

The expected **height** is not `√(4k_BT·ΔR)` = 12.07 nV/√Hz alone: amplifier current noise also
scales with R and survives the subtraction as `i_n·√(R₁₀ₖ² − R₁ₖ²)`. With the measured
0.83 pA/√Hz the expectation is 14.57 nV/√Hz against a measured median of 14.53; inverting the
measured height for `i_n` gives 0.822 pA/√Hz. With the datasheet 165 fA it would be 12.18.

**These captures come from the original board** (2026-09-11: C1 = 26.8 nF, unshielded). They test
the front end — source resistor and amplifier — which has not changed since, so they remain valid;
but they are not from the final network configuration.

### `reversibility.csv`

The detailed-balance test in *It is at equilibrium*: a time-reversible process has a **purely real** cross-spectrum, so
`arg S₁₂` is the observable and zero is the equilibrium prediction.

| column | meaning |
|---|---|
| `dataset` | board state — `gnd2` is the paper's board; `r1`…`r4b` are the four ladder rungs |
| `kappa` | condition number of that network |
| `n_records` | 0.6 s records behind the estimate |
| `arg_S12_full_band` | 200 Hz – 5 kHz, everything included. **Contaminated** — see below |
| `arg_S12_notched` | the same band excluding 800–1600 Hz, where an independent capture (`pgnd`, both probe tips on one point of the shield) shows the probe-lead pickup carries 88.5 % of its power with a phase of +0.79 rad. The exclusion is set by that **power** spectrum, never by node-data phase |
| `std_err` | from the record-to-record spread, not an analytic formula |
| `arg_S12_model` | the detailed-balance model with the measured drive asymmetry `c₁/c₂ = 0.984`. Note this is **not zero**: a known asymmetry implies a small, predicted probability current |
| `sigma_from_model` | \|measured − model\| / std_err |

The gap between the `full_band` and `notched` columns is the measurement's whole story: un-notched,
`gnd2` reads +53.8 mrad and looks like a 10σ detection of non-equilibrium physics; notched, it reads
−5.85 ± 5.90 mrad against a model of −4.78, and agrees at 0.2σ. Pre-registration, confound budget,
notch policy, the synthetic-data validation of the estimator and the surrogate null are in
[`../equilibrium.py`](../equilibrium.py).

### `moments.csv`

The third and fourth moments behind *It is at equilibrium* — the only numbers here that are not second moments. Thirty
records, mean ± SEM over records. Produced by [`../moments.py`](../moments.py).

| column | meaning |
|---|---|
| `band` | `200-50k` is the analysis band; `0-200` and `50k-up` are the slices either side of it |
| `quantity` | `skewness`, `excess_kurtosis` — band-limited, the same band-limiter every figure uses. `isserlis_*` — relative residuals of the two fourth-moment identities a jointly Gaussian pair satisfies exactly, `⟨x₁²x₂²⟩ = ⟨x₁²⟩⟨x₂²⟩ + 2⟨x₁x₂⟩²` and `⟨x₁³x₂⟩ = 3⟨x₁²⟩⟨x₁x₂⟩`. The `*_unfiltered` rows slice the **raw** record instead, to show what the band limit removes |
| `node` | 1 or 2; `0` means the quantity is joint |
| `value`, `std_err` | mean and standard error over the thirty records |

A single channel can be made Gaussian by any linear filter, so the `isserlis_*` rows carry the
weight: they constrain the joint distribution, which is what the covariance claim is about. Three
quarters of the raw record's power sits in `50k-up`, where the excess kurtosis is +1.5 and +3.4 —
that is the oscilloscope, not the board, and it is what band-limiting removes.

### `ngd-damping.csv`

The bridge from the ladder to the application: damped natural gradient descent, with the linear solve
corrupted by the measured error model. Produced by [`../ngd_sim.py`](../ngd_sim.py), which needs no
raw captures — only MNIST.

| column | meaning |
|---|---|
| `lambda` | Tikhonov damping in `(F + λI) d = g` |
| `kappa_eff` | condition number of the matrix actually solved, at the initial point. This is the ladder's x-axis, extended to the range a real Fisher lives in |
| `eps_sys_p0`, `eps_sys_p1` | the pre-registered floor `0.7 % × (κ_eff/2.21)^p` at that conditioning, for a flat floor, p = 0, and the ruled-out p = 1 |
| `loss_exact`, `loss_p0`, `loss_p1` | training loss after the step budget, with an exact solve and with each device error model. `inf` means the run diverged |
| `sd_exact`, `sd_p0`, `sd_p1` | spread across seeds (standard deviation, not standard error) |
| `acc_exact`, `acc_p0`, `acc_p1` | test accuracy for the same three |
| `integration_time_s_p0` | seconds of integration for the device's *sampling* error to fall to its own systematic floor, evaluated at the **ladder network's** τ_fast of 11 µs. Note that is not the headline board, whose τ_fast is 41.6 µs and which would give ~3.8× longer. Divide by τ for the transferable, dimensionless version: `T/τ = 2κ_eff/ε_sys²` |
| `lr_tuned` | learning rate, tuned per λ on the **exact** solver and reused for both device conditions, so the only difference across the three loss columns is the solve error |

The header rows of the file carry the SGD baseline at the same step budget, and the model and step
counts. The comparison is **per optimization step**, not per FLOP: a digital NGD step costs far more
than an SGD step, and making that step cheap is the entire premise of building the device.
