# airfoil-bo

Bayesian optimisation of a 2D airfoil with XFoil in the loop. A surrogate
model learns L/D from the shape, a GP classifier learns where XFoil fails
to converge, and the next shape to simulate is the one with the best
feasibility-weighted expected improvement. On top of that, a calibration
study asks whether the surrogate's uncertainty is honest, and whether
honest uncertainty actually finds better airfoils.

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
  polar through all three target CLs. That happens on about 18% of random
  designs, mostly ones that stall before CL 0.8.

## Methods compared

| Name | Surrogate | Failed designs |
|---|---|---|
| `gp_ei` | GP | GP classifier. The next design maximises log EI + log P(success). |
| `gp_ei_impute` | GP | Scored as the worst L/D seen so far, a common shortcut. |
| `ens_ei` | Deep ensemble of 5 MLPs, each trained on a bootstrap resample | GP classifier |
| `ens_ei_plain` | Deep ensemble with the plain recipe (no bootstrap) | GP classifier |
| `random` | — | — |
| `scripts/reference.py` | Differential evolution with 2,870 simulations, the brute-force "best known" reference | |

Every BO method starts from 10 Sobol points and gets 100 XFoil simulations
in total.

## Setup

```bash
tools/setup_xfoil.sh                 # XFoil from the Ubuntu package, no sudo needed
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python scripts/benchmark.py --seeds 30                          # ~2.5 h on 12 cores
.venv/bin/python scripts/benchmark.py --methods ens_ei_plain --seeds 10   # ~25 min
.venv/bin/python scripts/reference.py                                     # ~10 min
.venv/bin/python scripts/dataset.py --seed 1000 --out results/data/train_pool.json
.venv/bin/python scripts/dataset.py --seed 2000 --out results/data/test.json
.venv/bin/python scripts/calibration.py                                   # ~10 min
.venv/bin/python scripts/analyze.py                                       # tables + figures in results/
```

## Results

### Optimisation

The table below covers 30 seeds per method (10 for `ens_ei_plain`); it is
also in `results/summary.md` along with the full significance table.

| Method | Median best L/D after 20 / 50 / 100 sims | Median gap to best known at 100 sims | Sims to match random search's 100-sim median (105.2) | Failed sims | Time per proposal |
|---|---|---|---|---|---|
| GP + EI, failure classifier | 104.0 / 110.9 / **113.2** | 4.8 (4.1%) | **22** | 6% | 5.4 s |
| GP + EI, failures imputed | 101.7 / 109.1 / 112.7 | 5.4 (4.6%) | 25 | 5% | 2.3 s |
| Random search | 99.8 / 104.0 / 105.2 | 12.9 (10.9%) | - | 18% | - |
| Bootstrapped ensemble + EI | 97.2 / 100.1 / 103.1 | 15.0 (12.7%) | not reached | 34% | 9.3 s |
| Plain ensemble + EI (10 seeds) | 97.4 / 98.6 / 102.0 | 16.1 (13.6%) | not reached | 51% | 8.1 s |

References:
- Differential evolution: **118.1** after 2,870 simulations (the best known value).
- Standard airfoils: NACA 4412 93.5, NACA 2412 88.9, NACA 0012 65.1.

![Convergence](results/convergence.png)

- **GP-based BO beats random search by a wide margin** (one-sided
  Mann–Whitney on the final best: p < 10⁻¹⁰). Both GP variants match
  random search's 100-simulation result in 22–25 simulations. With 100
  simulations they get within 4–5% of an optimum that took differential
  evolution 29× as many.
- **The failure classifier is not clearly better than imputation.** In
  the first 10-seed run it looked slightly better. With 30 seeds the gap shrinks
  to about 0.5 L/D (p = 0.09), and the classifier costs more than twice
  as much per proposal. When simulations are this cheap, imputation is
  the pragmatic default. Where the classifier does earn its place is
  calibration (below).
- **Deep-ensemble BO is worse than random search** (p = 0.02 bootstrapped,
  p = 0.04 plain). It fails a third to half of its simulations. The next
  section explains why.

### Is the uncertainty honest?

![Calibration](results/calibration.png)

Full tables are in `results/calibration.md`.

**On held-out random designs** (trained on n random designs, tested on
1,000 others):

| Training designs | GP: 90% interval holds | Bootstrapped ensemble: 90% holds | Plain ensemble: 90% holds |
|---|---|---|---|
| 20 | 67% | 79% | 52% |
| 50 | 79% | 83% | 65% |
| 100 | 82% | 88% | 67% |
| 200 | 88% | 85% | 73% |

- **The GP is overconfident with little data and becomes honest as data
  grows.** That's the usual symptom of fitting kernel hyperparameters by
  point estimate on a handful of points. It has the lowest prediction
  error from 50 points up.
- **The bootstrapped ensemble is better calibrated at small n.** Its NLL
  is essentially tied with the GP's. Bootstrapping is the fix that
  matters: the plain recipe is overconfident at every size. (The ensemble
  settings were chosen on data from the first benchmark, not on this test
  set.)
- **The failure classifier clearly beats guessing the base rate**
  (Brier score 0.064 vs 0.150 at 100 points), but it helps little at 20
  points.

**Inside the BO loop.** Each proposal logged its prediction before XFoil
ran:

| Method | 90% interval holds | Median (actual − predicted) L/D | Predicted P(success) | Actual success |
|---|---|---|---|---|
| GP + EI, failure classifier | 76% | −1.7 | 95.4% | 95.4% |
| GP + EI, failures imputed | 78% | −2.0 | - | - |
| Bootstrapped ensemble + EI | 91% | −11.5 | 72.9% | 64.8% |
| Plain ensemble + EI | 73% | −7.5 | 51.9% | 45.8% |

- **Every method is optimistic at its own proposals: the winner's curse.**
  EI picks the designs where the model's errors happen to be optimistic,
  so outcomes land below the predictions even for a model that is
  calibrated on random points.
- **The GP classifier is almost perfectly calibrated in the loop.** It
  predicted 95.4% success on average, and 95.4% of designs succeeded.

**So does honest uncertainty find better airfoils? Not here.** The
bootstrapped ensemble had the most nominal coverage in *both* tests and
still optimised worst. What decides it is behaviour away from the data:

- **A GP falls back to its prior mean there**, so EI weighs mean against
  variance sensibly.
- **Neural networks extrapolate trends linearly**, so EI is largest on
  the edges of the box. **86%** of the bootstrapped ensemble's proposals
  had at least 5 of the 7 parameters at a bound, against **3%** for the
  GP.
- **The ensemble kept proposing corners it had already tried.** A lone
  point at a corner is left out of about a third of the bootstrap
  resamples, so those members keep extrapolating there.
- **The classifier can't stop this.** Far from the data it predicts about
  90% success.

Calibration is necessary but not sufficient. How the model extrapolates
matters as much as how its errors are distributed.

### The best airfoil

![Best airfoil](results/best_airfoil.png)

The best shape found by any BO run (L/D 117.2) and the
differential-evolution optimum (118.1) are nearly the same laminar-flow
design:

| | Max thickness at | Max camber |
|---|---|---|
| Best BO run | 41% chord | 3.3% at 57% chord |
| Differential evolution | 41% chord | 3.5% at 56% chord |
| NACA 4412 (for comparison) | 30% chord | 4% at 40% chord |

Moving the thickness and camber aft keeps the pressure gradient
favourable over more of the chord. That delays transition, so drag stays
flat at about 50 counts across CL 0.3–0.8, with the edge of the low-drag
bucket placed just above CL 0.8.

## XFoil gotchas (handled in `airfoil_bo/xfoil.py`)

1. **The Ubuntu binary crashes with SIGFPE** in the viscous solver. It is
   compiled with floating-point traps enabled. The fix is to preload
   `tools/libnofpe.so`, which turns the trap setup off.
2. **Jumping straight to a target CL gives spurious solutions.** The
   `CL 0.4` command started from α = 0 converged to drag below a
   *fully laminar flat plate*. The objective now sweeps α outward from 0°
   in 0.5° steps and interpolates at the target CLs. Points where CD < CDf
   are rejected.
3. **One failed point poisons the rest of a scripted sweep.** XFoil is now
   driven interactively, and each point that fails is retried once from a
   fresh boundary layer. That cut failures on random designs from 42% to
   20%. Both sweep directions now start from the α = 0 solution.
4. **Most low-CL failures are real, not artifacts (tested and rejected).**
   In the first GP runs, about a quarter of the failures were at the
   low-CL end (CL 0.4, sometimes also 0.6) while CL 0.8 was fine. I
   suspected the cold restart of the downward sweep, but fixing it
   rescued only 3 of the 52 designs that failed at CL 0.4. They are
   heavily cambered shapes whose lower surface separates at the negative
   α needed for CL 0.4. Finer steps and 200 iterations rescue about half
   of them, but their drag at CL 0.4 is 55–190 counts (against ~50 for
   good designs), so they are genuinely poor there. The solver settings
   stay as they are and the classifier absorbs these failures.

## Caveats

- **This is the XFoil optimum, not the real-world optimum.** At Ncrit 9
  (a clean wind tunnel), XFoil rewards long laminar runs. Surface
  roughness or turbulence would trip transition and erase most of the
  gain.
- **XFoil is a viscous–inviscid panel method.** Drag near stall is
  unreliable.
- **The ensemble result covers this setup only:** 7 dimensions, ≤100
  points, EI over the whole box, no trust region. Ensembles are known to
  do better with more data or with local search; see next steps.

## Next steps

1. **Trust regions (TuRBO):** optimise the acquisition only inside a box
   around the incumbent. This removes the corner-seeking directly and is
   the fair way to give ensembles a second chance. Run the GPs with the
   same change.
2. **Fully Bayesian GP hyperparameters** (MCMC / SAAS) to fix the GP's
   overconfidence at small n, then check whether that helps optimisation.
3. **Thompson sampling and batch qLogEI:** propose 12 designs per step
   and compare wall-clock time rather than simulation count.
4. **A harder, robust objective:** average over Ncrit 5 and 9, and use
   12–14 shape parameters, where random search should fall much further
   behind.
5. **Multi-fidelity:** NeuralFoil (milliseconds) as the cheap source and
   XFoil as the accurate one.

## Changes since the first version

- **Solver:** both sweep directions now warm-start from α = 0.
- **Benchmark:** 30 seeds instead of 10, plus the two ensemble methods.
- **Classifier fit:** L-BFGS instead of Adam. Proposals are about 1.7×
  faster (9.3 s → 5.4 s), and the old Adam fit hadn't converged.
- **Calibration:** every proposal logs its prediction, and there is a new
  calibration study (`scripts/dataset.py`, `scripts/calibration.py`).
- **Revised conclusion:** the first version's hint that the classifier
  beats imputation (p = 0.15 at 10 seeds) didn't hold up. With 30 seeds
  the difference is ~0.5 L/D and still not significant (p = 0.09).
