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

Pending: the h=0.01 loss-trace NPZs have not been exported yet (`export-loss-traces`).

## Regeneration

```bash
python -m bnn_geometry.paper_figures.render --tables results/final_geometry/paper_figures/tables --figures results/final_geometry/paper_figures/figures
```
