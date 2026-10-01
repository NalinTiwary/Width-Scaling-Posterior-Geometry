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

At lag 100, the narrowest-minus-widest difference of the mean curve (positive: lower autocorrelation at the widest width), as median [range] over replicates, is deep +0.065 [+0.063, +0.072] (24/24 point-level differences positive); shallow +0.017 [+0.015, +0.022] (23/24 point-level differences positive).

Lag horizon K = 400 (history in `tables/prediction_acf_settings.json`). Diagnostics: 216/216 series pass R-hat<1.01, bulk ESS>=1000, tail ESS>=400; drift flags: 0.

| arch | lag | D (mean of 8 points) median [range] | replicates positive | points positive |
|---|---|---|---|---|
| shallow | 25 | +0.003 [-0.003, +0.011] | 2/3 | 15/24 |
| shallow | 50 | +0.012 [+0.008, +0.018] | 3/3 | 20/24 |
| shallow | 100 | +0.017 [+0.015, +0.022] | 3/3 | 23/24 |
| shallow | 200 | +0.009 [+0.007, +0.015] | 3/3 | 20/24 |
| deep | 25 | +0.047 [+0.044, +0.060] | 3/3 | 24/24 |
| deep | 50 | +0.070 [+0.069, +0.087] | 3/3 | 24/24 |
| deep | 100 | +0.065 [+0.063, +0.072] | 3/3 | 24/24 |
| deep | 200 | +0.030 [+0.020, +0.032] | 3/3 | 23/24 |

## Supplement S2: step-size sensitivity (deep endpoints)

At $kh=0.5$ the width-32-minus-width-256 autocorrelation difference, as median [range] over replicates, is +0.162 [+0.156, +0.185] at $h=0.01$ (3/3 replicates positive) and +0.189 [+0.165, +0.204] at $h=0.005$ (3/3 replicates positive).

Matched duration 512, display range kh in [0, 2]. Diagnostics: 0 of the series fail a protocol gate; drift flags: 0.

| h | kh | D median [range] | replicates positive |
|---|---|---|---|
| 0.01 | 0.25 | +0.145 [+0.135, +0.164] | 3/3 |
| 0.01 | 0.5 | +0.162 [+0.156, +0.185] | 3/3 |
| 0.01 | 1 | +0.121 [+0.089, +0.141] | 3/3 |
| 0.01 | 2 | +0.053 [+0.013, +0.070] | 3/3 |
| 0.005 | 0.25 | +0.155 [+0.144, +0.186] | 3/3 |
| 0.005 | 0.5 | +0.189 [+0.165, +0.204] | 3/3 |
| 0.005 | 1 | +0.155 [+0.123, +0.155] | 3/3 |
| 0.005 | 2 | +0.071 [+0.062, +0.082] | 3/3 |
