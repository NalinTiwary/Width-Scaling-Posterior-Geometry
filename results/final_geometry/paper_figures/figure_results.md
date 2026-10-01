# Figure results (two-figure package)

Generated from the computed tables in `tables/`. Scope: empirical spectral-domain coverage and fixed-step loss autocorrelation. No PI/LSI constant, integrated autocorrelation time, continuous-time rate or fitted width exponent is estimated.

## Figure 1: posterior coverage of the deep spectral domain

Across widths m = 32-256 (n = 128, two hidden layers), all 196,608 inspected unrestricted posterior states (16,384 per target, 12 targets) lie inside the spectral restriction G_2.5 (0 exits). The replicate-median 99th percentile of S = ||W2||op/(2.5 sqrt(m)) ranges from 0.812 to 0.845 across widths (largest single-replicate value 0.848), below the boundary S = 1; medians lie between 0.763 and 0.791. The posterior median, 95th and 99th percentiles differ from the iid Gaussian-prior quantiles by at most 0.0027 in S, so these norm statistics are close to their prior values.

| m | q50 | q95 | q99 | replicate range q99 | prior q99 |
|---|---|---|---|---|---|
| 32 | 0.7632 | 0.8187 | 0.8447 | 0.8436-0.8478 | 0.8444 |
| 64 | 0.7772 | 0.8124 | 0.8291 | 0.8288-0.8310 | 0.8298 |
| 128 | 0.7859 | 0.8081 | 0.8186 | 0.8186-0.8188 | 0.8199 |
| 256 | 0.7913 | 0.8051 | 0.8116 | 0.8114-0.8118 | 0.8112 |

Per-target counts and quantiles: `tables/spectral_replicate_summary.csv`; statewise values: `tables/spectral_states.csv.gz`; cross-check against the campaign summary: `tables/spectral_crosscheck.csv`.

## Figure 2: width and loss autocorrelation

Status `fixed_step_loss_acf_status = pass`, `analysis_scope = fixed_step_discrete_sampler`; displayed lag range K = 400 (extension history: [400]); window 102,400 transitions per chain at h = 0.01.

In the deep model, the widest network decorrelates faster at early/moderate lags (25, 50, 100; all 3 replicates), while at lags 200, 400 the replicate differences are mixed (3/3, 2/3 negative) as the curves approach zero. At lag 100 the pointwise-median ACF is 0.19 (m=32), 0.13 (m=64), 0.08 (m=128), 0.07 (m=256), decreasing monotonically with width. In the shallow model, the widest network decorrelates faster at early/moderate lags (25, 50, 100; all 3 replicates), while at lags 200, 400 the replicate differences are mixed (3/3, 2/3 negative) as the curves approach zero. At lag 100 the pointwise-median ACF is 0.12 (m=64), 0.09 (m=256), 0.09 (m=1024), 0.09 (m=4096), not monotone in width.

_The two sentences above are generated mechanically from `tables/acf_width_contrasts.csv`; check them against the plotted curves before submission._

### deep: pointwise-median ACF at fixed lags

| lag | m=32 | m=64 | m=128 | m=256 | widest - narrowest | replicates negative |
|---|---|---|---|---|---|---|
| 25 | 0.560 | 0.475 | 0.424 | 0.400 | -0.160 | 3/3 |
| 50 | 0.375 | 0.274 | 0.214 | 0.195 | -0.180 | 3/3 |
| 100 | 0.195 | 0.129 | 0.078 | 0.069 | -0.126 | 3/3 |
| 200 | 0.065 | 0.032 | 0.023 | 0.012 | -0.053 | 3/3 |
| 400 | 0.016 | 0.005 | 0.002 | 0.006 | -0.010 | 2/3 |

### shallow: pointwise-median ACF at fixed lags

| lag | m=64 | m=256 | m=1024 | m=4096 | widest - narrowest | replicates negative |
|---|---|---|---|---|---|---|
| 25 | 0.503 | 0.467 | 0.466 | 0.461 | -0.043 | 3/3 |
| 50 | 0.293 | 0.245 | 0.251 | 0.236 | -0.057 | 3/3 |
| 100 | 0.124 | 0.087 | 0.089 | 0.089 | -0.035 | 3/3 |
| 200 | 0.028 | 0.027 | 0.012 | 0.010 | -0.018 | 3/3 |
| 400 | 0.003 | -0.000 | -0.001 | -0.004 | -0.007 | 2/3 |

### Selected-window sampling checks (V)

R-hat max 1.0028 (threshold < 1.01); bulk ESS min 3212 (threshold >= 1000); window acceptance 0.988-0.994; drift flags 0; failed targets: none.

## Regeneration

```bash
python -m bnn_geometry.paper_figures.render --tables results/final_geometry/paper_figures/tables --figures results/final_geometry/paper_figures/figures
```
