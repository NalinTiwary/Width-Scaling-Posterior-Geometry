# Paper figures after cleanup (figure width 6.75 in)

## Figure 1: spectral statistic $T=\|W_2\|_{op}/\sqrt m$ and coverage

The 99th-percentile curve decreases from 2.112 at width 32 to 2.029 at width 256, illustrating a narrowing upper tail above the theorem's admissibility threshold; at $a=2.1$ the median coverage rises from 98.7% to 100.0%. Between 13.0% and 14.9% of the 16,384 inspected states per target lie above 2 at every width, which the theorem does not exclude.

| width | above 2 (r0/r1/r2 of 16384) | % above 2 median [range] | max T | posterior q99 | prior q99 | coverage at a=2.1 median [range] |
|---|---|---|---|---|---|---|
| 32 | 2217/2436/2131 | 13.53 [13.01, 14.87] | 2.260 | 2.112 | 2.111 | 98.7% [98.4, 98.7] |
| 64 | 2238/2398/2206 | 13.66 [13.46, 14.64] | 2.182 | 2.073 | 2.075 | 99.7% [99.7, 99.8] |
| 128 | 2302/2327/2330 | 14.20 [14.05, 14.22] | 2.119 | 2.046 | 2.050 | 100.0% [100.0, 100.0] |
| 256 | 2391/2318/2329 | 14.22 [14.15, 14.59] | 2.082 | 2.029 | 2.028 | 100.0% [100.0, 100.0] |

States within the numerical guard of 2: 0.

## Figure 2: loss autocorrelation

Unchanged accepted curves (lags 0-400 exported, 0-200 displayed); see `paper_figures/figure_results.md` for the per-lag numbers.

## Supplement S1: prediction autocorrelation

Not yet computed (requires the cluster export).

## Supplement S2: step-size sensitivity (deep endpoints)

Not yet computed (requires the cluster export).
