Static: trained on n random designs, scored on 1,000 held-out random designs.

| n | Model | RMSE | NLL | 50% interval holds | 90% interval holds |
|---|---|---|---|---|---|
| 20 | GP | 14.4 | 4.99 | 35% | 67% |
| 20 | Deep ensemble (bootstrapped) | 13.4 | 4.45 | 44% | 79% |
| 20 | Deep ensemble (plain) | 13.8 | 19.83 | 28% | 52% |
| 50 | GP | 10.1 | 3.86 | 42% | 79% |
| 50 | Deep ensemble (bootstrapped) | 10.9 | 3.94 | 47% | 83% |
| 50 | Deep ensemble (plain) | 10.8 | 6.68 | 36% | 65% |
| 100 | GP | 8.8 | 3.69 | 45% | 82% |
| 100 | Deep ensemble (bootstrapped) | 9.4 | 3.59 | 51% | 88% |
| 100 | Deep ensemble (plain) | 9.3 | 5.46 | 34% | 67% |
| 200 | GP | 7.0 | 3.33 | 53% | 88% |
| 200 | Deep ensemble (bootstrapped) | 7.7 | 3.38 | 46% | 85% |
| 200 | Deep ensemble (plain) | 7.2 | 4.01 | 36% | 73% |

| n | Failure classifier Brier | Base-rate Brier |
|---|---|---|
| 20 | 0.138 | 0.155 |
| 50 | 0.091 | 0.151 |
| 100 | 0.064 | 0.150 |
| 200 | 0.057 | 0.149 |

In the loop: model predictions at each proposed design vs the XFoil result.

| Method | 50% interval holds | 90% interval holds | Median (actual - predicted) L/D | Mean predicted P(success) | Actual success |
|---|---|---|---|---|---|
| GP + EI, failure classifier | 46% | 76% | -1.7 | 95.4% | 95.4% |
| GP + EI, failures imputed | 47% | 78% | -2.0 | - | - |
| Bootstrapped ensemble + EI, failure classifier | 54% | 91% | -11.5 | 72.9% | 64.8% |
| Plain ensemble + EI, failure classifier | 41% | 73% | -7.5 | 51.9% | 45.8% |
