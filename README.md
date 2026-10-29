# airfoil-bo

Bayesian optimisation of a 2D airfoil with XFoil in the loop. A Gaussian
process learns L/D from the shape, a second GP learns where XFoil fails to
converge, and the next shape to simulate is the one with the best
feasibility-weighted expected improvement.

## The problem

- **Design space:** 7 numbers in [0, 1]. Each one sets a CST coefficient:
  3 control the thickness distribution and 4 control the camber line
  (`airfoil_bo/geometry.py`). Thickness is always rescaled to exactly **12%
  of the chord**, so every candidate meets the thickness requirement by
  construction.
- **Objective:** mean L/D at **CL = 0.4, 0.6 and 0.8**, at **Re = 10⁶**,
  with free transition (Ncrit 9). Averaging over three lift coefficients
  stops the optimiser from tuning a shape that only works at one point.
- **Unknown constraint:** a design fails if XFoil can't produce a clean
  polar through all three target CLs. That happens on about 20% of random
  designs, mostly ones that stall before CL 0.8.

## Methods compared

| Name | What it does |
|---|---|
| `gp_ei` | 10 Sobol points, then a GP on L/D plus a GP classifier on success. The next design maximises log EI + log P(success). |
| `gp_ei_impute` | Same, but with no classifier. Each failed design is scored as the worst L/D seen so far, a common shortcut. |
| `random` | Uniform random designs. |
| `scripts/reference.py` | Differential evolution with thousands of simulations. This is the brute-force "best known" reference. |

## Setup

```bash
tools/setup_xfoil.sh                 # XFoil from the Ubuntu package, no sudo needed
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python scripts/benchmark.py --seeds 10 --budget 100   # ~35 min on 12 cores
.venv/bin/python scripts/reference.py                            # ~15 min
.venv/bin/python scripts/analyze.py                              # table + figures in results/
```

## XFoil gotchas (already handled in `airfoil_bo/xfoil.py`)

1. **The Ubuntu binary crashes with SIGFPE** in the viscous solver. It is
   compiled with floating-point traps enabled. The fix is to preload
   `tools/libnofpe.so`, which turns the trap setup off.
2. **Jumping straight to a target CL gives spurious solutions.** The
   `CL 0.4` command started from α = 0 converged to drag below a
   *fully laminar flat plate*. The objective now sweeps α outward from 0°
   in 0.5° steps and interpolates at the target CLs. Points where CD < CDf
   are rejected.
3. **One failed point poisons the rest of a scripted sweep.** Every later
   point warm-starts from the broken boundary layer. XFoil is now driven
   interactively: each point that fails is retried once from a fresh
   boundary layer (`INIT`). That cut the failure rate on random designs
   from 42% to 20%.

## Results

Each method ran with 10 seeds and 100 XFoil simulations
(`results/summary.md`). The GP methods start from 10 Sobol points.

| Method | Median best L/D after 20 / 50 / 100 sims | Median gap to best known at 100 sims | Sims to match random search's 100-sim median (107.2) | Failed sims | Time per proposal |
|---|---|---|---|---|---|
| GP + EI, failure classifier | 106.2 / 111.7 / **113.8** | 4.1 (3.5%) | **30** | 7% | 9.3 s |
| GP + EI, failures imputed | 103.3 / 109.8 / 112.3 | 5.6 (4.7%) | 38 | 4% | 2.7 s |
| Random search | 105.8 / 107.0 / 107.2 | 10.7 (9.1%) | - | 18% | 0.0 s |

References:
- Differential evolution: **117.9** after 2,870 simulations (the best known value).
- Standard airfoils: NACA 4412 93.5, NACA 2412 88.9, NACA 0012 65.1.

![Convergence](results/convergence.png)

What the numbers say:

- **Bayesian optimisation clearly beats random search.** Both GP variants
  match random search's 100-simulation result in 30–38 simulations
  (one-sided Mann–Whitney on the final best: p ≈ 0.0001). With 100
  simulations they get within 3.5–4.7% of an optimum that took
  differential evolution 29× as many.
- **Classifier vs imputation is suggestive, not proven.** The classifier
  run ends higher in median (113.8 vs 112.3), but at 10 seeds the
  difference is not significant (p = 0.15).
- **The classifier fails more often, and that's expected.** Imputing a
  failure as "worst L/D" teaches the GP that failure regions are bad, so
  it avoids them (1% failures in sims 51–100). The classifier only
  down-weights them, so it keeps probing the stall boundary, where the
  best designs sit (4% failures in sims 51–100). About 76% of failures
  in the GP runs are designs that stall before CL 0.8. Most of the rest
  fail only at CL 0.4, which is probably a solver artifact (see next
  steps).
- **Random search beats NACA 4412 within about 10 simulations.** The
  design space contains many shapes better than NACA 4412 for this
  objective. NACA 4412 is a sanity check, not a strong baseline.

![Best airfoil](results/best_airfoil.png)

**The optimum is a laminar-flow airfoil.** The best shape found here
(L/D 116.2) and the differential-evolution optimum (117.9) are nearly
the same design:

| | Max thickness at | Max camber |
|---|---|---|
| Best found here | 40% chord | 3.2% at 59% chord |
| Differential evolution | 41% chord | 3.4% at 57% chord |
| NACA 4412 (for comparison) | 30% chord | 4% at 40% chord |

Moving the thickness and camber aft keeps the pressure gradient
favourable over more of the chord. That delays transition, so drag stays
flat at about 50 counts across CL 0.4–0.8, with the edge of the
low-drag bucket placed just above CL 0.8.

## Caveats

- **This is the XFoil optimum, not the real-world optimum.** At Ncrit 9
  (a clean wind tunnel), XFoil rewards long laminar runs. Surface
  roughness or turbulence would trip transition and erase most of the
  gain. Averaging the objective over Ncrit (e.g. 5 and 9) would make the
  design more robust.
- **XFoil is a viscous–inviscid panel method.** Drag near stall and in
  the post-stall region is unreliable.
- **The GPs are refitted from scratch every step.** That's fine at
  ≤100 points, but proposal time grows roughly as n³.

## Next steps

0. **Fix the low-CL sweep.** The downward α sweep restarts from a
   re-initialised boundary layer at −0.5°. Some good designs (including
   the best one, whose polar stops at CL 0.36) lose their low-CL points
   this way. Running the downward sweep first, warm-started from the
   α = 0 solution, should remove most of the CL 0.4-only failures. Re-run
   the benchmark after this change so the results match the code.
1. **Calibration (PAI):** fit on runs from some seeds and test on held-out
   designs. Check NLL and the empirical coverage of the 50/90% predictive
   intervals. Compare the GP against a deep ensemble.
2. **Other acquisition functions:** UCB, Thompson sampling, and batch
   qLogEI, which could propose 12 designs at once to use all cores.
3. **NeuralFoil baseline:** it is an off-the-shelf XFoil surrogate.
   Compare how good its optimum is under XFoil.
4. **Multi-fidelity:** use XFoil as the cheap source and AirfRANS RANS
   results on NACA shapes as the expensive one.
