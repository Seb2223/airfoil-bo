| Method | Median best L/D after 10 / 20 / 50 / 100 sims | Median gap to best known at 100 sims | Sims to match random search's 100-sim median (105.2) | Failed sims | Proposals at a corner | Time per proposal |
|---|---|---|---|---|---|---|
| GP + EI, failure classifier | 96.1 / 104.0 / 110.9 / 113.2 | 4.8 (4.1%) | 22 | 6% | 3% | 5.4 s |
| GP + EI, failures imputed | 96.1 / 101.7 / 109.1 / 112.7 | 5.4 (4.6%) | 25 | 5% | 4% | 2.3 s |
| Random search | 94.7 / 99.8 / 104.0 / 105.2 | 12.9 (10.9%) | - | 18% | - | 0.0 s |
| Bootstrapped ensemble + EI, failure classifier | 96.1 / 97.2 / 100.1 / 103.1 | 15.0 (12.7%) | not reached | 34% | 86% | 9.3 s |
| Plain ensemble + EI, failure classifier | 97.4 / 97.4 / 98.6 / 102.0 | 16.1 (13.6%) | not reached | 51% | 80% | 8.1 s |
| GP + EI, failure classifier, trust region | 96.1 / 104.0 / 111.3 / 114.2 | 3.9 (3.3%) | 24 | 4% | 1% | 3.7 s |
| Bootstrapped ensemble + EI, failure classifier, trust region | 96.1 / 97.9 / 105.4 / 110.1 | 8.0 (6.7%) | 50 | 10% | 6% | 9.0 s |
| Claude agent, via Claude Code | 107.9 / 111.2 / 114.5 / 116.6 | 1.5 (1.2%) | 2 | 1% | 0% | 0.9 s |

Final best L/D, one-sided Mann-Whitney U (row better than column):

| | p |
|---|---|
| GP + EI, failure classifier > GP + EI, failures imputed | 0.093 |
| GP + EI, failure classifier > Random search | 1.7e-11 |
| GP + EI, failure classifier > Bootstrapped ensemble + EI, failure classifier | 1.8e-11 |
| GP + EI, failure classifier > Plain ensemble + EI, failure classifier | 1.5e-06 |
| GP + EI, failure classifier, trust region > GP + EI, failure classifier | 0.13 |
| GP + EI, failure classifier > Bootstrapped ensemble + EI, failure classifier, trust region | 1.1e-05 |
| Claude agent, via Claude Code > GP + EI, failure classifier | 5.7e-05 |
| GP + EI, failures imputed > Random search | 8.9e-11 |
| GP + EI, failures imputed > Bootstrapped ensemble + EI, failure classifier | 5e-11 |
| GP + EI, failures imputed > Plain ensemble + EI, failure classifier | 2e-06 |
| GP + EI, failure classifier, trust region > GP + EI, failures imputed | 0.031 |
| GP + EI, failures imputed > Bootstrapped ensemble + EI, failure classifier, trust region | 0.00035 |
| Claude agent, via Claude Code > GP + EI, failures imputed | 1.2e-05 |
| Random search > Bootstrapped ensemble + EI, failure classifier | 0.018 |
| Random search > Plain ensemble + EI, failure classifier | 0.036 |
| GP + EI, failure classifier, trust region > Random search | 1.1e-10 |
| Bootstrapped ensemble + EI, failure classifier, trust region > Random search | 8.4e-05 |
| Claude agent, via Claude Code > Random search | 1.5e-06 |
| Bootstrapped ensemble + EI, failure classifier > Plain ensemble + EI, failure classifier | 0.46 |
| GP + EI, failure classifier, trust region > Bootstrapped ensemble + EI, failure classifier | 5e-11 |
| Bootstrapped ensemble + EI, failure classifier, trust region > Bootstrapped ensemble + EI, failure classifier | 1.1e-06 |
| Claude agent, via Claude Code > Bootstrapped ensemble + EI, failure classifier | 1.5e-06 |
| GP + EI, failure classifier, trust region > Plain ensemble + EI, failure classifier | 2.4e-06 |
| Bootstrapped ensemble + EI, failure classifier, trust region > Plain ensemble + EI, failure classifier | 0.00044 |
| Claude agent, via Claude Code > Plain ensemble + EI, failure classifier | 9.1e-05 |
| GP + EI, failure classifier, trust region > Bootstrapped ensemble + EI, failure classifier, trust region | 2.9e-06 |
| Claude agent, via Claude Code > GP + EI, failure classifier, trust region | 0.00061 |
| Claude agent, via Claude Code > Bootstrapped ensemble + EI, failure classifier, trust region | 1.5e-06 |

Plateau and waste:

| Method | Median gain over sims 51-100 | Median sim where the final best was found | Repeated designs | Mean unused sims |
|---|---|---|---|---|
| GP + EI, failure classifier | +1.9 | 82 | 0.1% | 0.0 |
| Random search | +0.3 | 56 | 0.0% | 0.0 |
| Claude agent, via Claude Code | +2.2 | 97 | 0.0% | 0.0 |

Claude agent, via Claude Code: 10 runs, model claude-opus-5-5, effort high, stop reasons: end_turn x10. Median per run: 274,454 input tokens (241,708 of them cache reads), 7,615 output tokens, $0.46 API-equivalent (billed to the plan), 2 min, 13 simulate calls.
