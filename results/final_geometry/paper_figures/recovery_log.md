# Recovery log

Inputs were recovered from existing campaign outputs before computing anything new (guide §10.1). No posterior sampling was rerun for this package.

- Spectral states: existing statewise table `tables/spectral_states.csv` (re-exported as `paper_figures/tables/spectral_states.csv.gz`) of the unrestricted reference archive (every archived state, stride 16, 4 chains x 4,096 per deep target), cross-checked against the campaign's pooled per-target summary (quantiles identical, counts identical) and its float64 SVD backend check.
- Prior reference: existing 4,096 iid Gaussian W2 matrices per width (`controls/prior_spectral_values.npz`, seeds in `tables/spectral_prior_summary.csv`).
- Loss traces: 24 targets exported from the existing h=0.01 dynamics HDF5 traces (full per-transition V incl. rejected repeats; priority 1 of §10.3), 0 failures. Each export verified target/execution hashes, kernel and step, contiguous unit-stride transitions, unchanged V at rejected transitions, agreement of the mean V with the campaign analysis and the final-checkpoint state's recomputed V.
  - shallow_m0064_r0: targets/shallow_m0064_r0/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - shallow_m0064_r1: targets/shallow_m0064_r1/dynamics/h_0p01 segments discard,stage_1,stage_2,stage_3; retained 204,800/chain; window transitions 104,401-206,800; status recovered_existing_trace
  - shallow_m0064_r2: targets/shallow_m0064_r2/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - shallow_m0256_r0: targets/shallow_m0256_r0/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - shallow_m0256_r1: targets/shallow_m0256_r1/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - shallow_m0256_r2: targets/shallow_m0256_r2/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - shallow_m1024_r0: targets/shallow_m1024_r0/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - shallow_m1024_r1: targets/shallow_m1024_r1/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - shallow_m1024_r2: targets/shallow_m1024_r2/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - shallow_m4096_r0: targets/shallow_m4096_r0/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - shallow_m4096_r1: targets/shallow_m4096_r1/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - shallow_m4096_r2: targets/shallow_m4096_r2/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - deep_m0032_r0: targets/deep_m0032_r0/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - deep_m0032_r1: targets/deep_m0032_r1/dynamics/h_0p01 segments discard,stage_1,stage_2,stage_3; retained 204,800/chain; window transitions 104,401-206,800; status recovered_existing_trace
  - deep_m0032_r2: targets/deep_m0032_r2/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - deep_m0064_r0: targets/deep_m0064_r0/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - deep_m0064_r1: targets/deep_m0064_r1/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - deep_m0064_r2: targets/deep_m0064_r2/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - deep_m0128_r0: targets/deep_m0128_r0/dynamics/h_0p01 segments discard,stage_1,stage_2,stage_3; retained 204,800/chain; window transitions 104,401-206,800; status recovered_existing_trace
  - deep_m0128_r1: targets/deep_m0128_r1/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - deep_m0128_r2: targets/deep_m0128_r2/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - deep_m0256_r0: targets/deep_m0256_r0/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - deep_m0256_r1: targets/deep_m0256_r1/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace
  - deep_m0256_r2: targets/deep_m0256_r2/dynamics/h_0p01 segments discard,stage_1,stage_2; retained 102,400/chain; window transitions 2,001-104,400; status recovered_existing_trace

No targeted new run (§10.4) was needed or made.
