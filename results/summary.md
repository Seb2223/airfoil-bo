| Method | Median best L/D after 20 / 50 / 100 sims | Median gap to best known at 100 sims | Sims to match random search's 100-sim median (105.2) | Failed sims | Time per proposal |
|---|---|---|---|---|---|
| GP + EI, failure classifier | 104.0 / 110.9 / 113.2 | 4.8 (4.1%) | 22 | 6% | 5.4 s |
| GP + EI, failures imputed | 101.7 / 109.1 / 112.7 | 5.4 (4.6%) | 25 | 5% | 2.3 s |
| Random search | 99.8 / 104.0 / 105.2 | 12.9 (10.9%) | - | 18% | 0.0 s |
| Bootstrapped ensemble + EI, failure classifier | 97.2 / 100.1 / 103.1 | 15.0 (12.7%) | not reached | 34% | 9.3 s |
| Plain ensemble + EI, failure classifier | 97.4 / 98.6 / 102.0 | 16.1 (13.6%) | not reached | 51% | 8.1 s |

Final best L/D, one-sided Mann-Whitney U (row better than column):

| | p |
|---|---|
| GP + EI, failure classifier > GP + EI, failures imputed | 0.093 |
| GP + EI, failure classifier > Random search | 1.7e-11 |
| GP + EI, failure classifier > Bootstrapped ensemble + EI, failure classifier | 1.8e-11 |
| GP + EI, failure classifier > Plain ensemble + EI, failure classifier | 1.5e-06 |
| GP + EI, failures imputed > Random search | 8.9e-11 |
| GP + EI, failures imputed > Bootstrapped ensemble + EI, failure classifier | 5e-11 |
| GP + EI, failures imputed > Plain ensemble + EI, failure classifier | 2e-06 |
| Random search > Bootstrapped ensemble + EI, failure classifier | 0.018 |
| Random search > Plain ensemble + EI, failure classifier | 0.036 |
| Bootstrapped ensemble + EI, failure classifier > Plain ensemble + EI, failure classifier | 0.46 |
