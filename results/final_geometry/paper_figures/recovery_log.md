# Recovery log

Inputs were recovered from existing campaign outputs before computing anything new (guide §10.1). No posterior sampling was rerun for this package.

- Spectral states: existing statewise table `tables/spectral_states.csv` (re-exported as `paper_figures/tables/spectral_states.csv.gz`) of the unrestricted reference archive (every archived state, stride 16, 4 chains x 4,096 per deep target), cross-checked against the campaign's pooled per-target summary (quantiles identical, counts identical) and its float64 SVD backend check.
- Prior reference: existing 4,096 iid Gaussian W2 matrices per width (`controls/prior_spectral_values.npz`, seeds in `tables/spectral_prior_summary.csv`).
- Loss traces: pending cluster export.
