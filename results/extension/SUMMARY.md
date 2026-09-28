# Larger-sample extension results

Generated 2026-09-28T22:07:08+00:00 on `cc-login1.campuscluster.illinois.edu`, commit `ad953959f5`, config `config.ext.yaml`.
Targets done: 27/27.

## Validation

| check | result |
|---|---|
| arrowhead_vs_dense | PASS |
| brackets_vs_dense_hessian | PASS |
| dense_bracket_matches_pilot | PASS |
| hessian_decomposition | PASS |
| matched_full_vs_reduced_sampling | PASS |
| matched_sampling_negative_control | PASS |
| reduction_exact | PASS |

## Settings (medians over replicates)

| data | n | m | n/√m | B | D_th | coverage (min) | exits | q99 H/B | median d₊ | q95 [d₋, d₊] | q95 d₊/D_th | between-rep SD | diag pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fmnist | 256 | 4096 | 4 | 21.08 | 15.33 | 1 (1) | 0 | 0.156 | 0.0859 | [0.109, 0.11] | 0.00718 | 0.0044 | False |
| fmnist | 256 | 16384 | 2 | 21.11 | 7.677 | 1 (1) | 0 | 0.163 | 0.049 | [0.0614, 0.0621] | 0.00808 | 0.00098 | False |
| orth | 64 | 1024 | 2 | 11.53 | 0.8731 | 1 (1) | 0 | 0.271 | 0.0329 | [0.0343, 0.035] | 0.04 | 0.00045 | True |
| orth | 64 | 4096 | 1 | 11.6 | 0.4389 | 1 (1) | 0 | 0.283 | 0.0167 | [0.0176, 0.0178] | 0.0406 | 0.0001 | True |
| orth | 128 | 1024 | 4 | 15.44 | 1.611 | 1 (1) | 0 | 0.202 | 0.0443 | [0.0449, 0.0462] | 0.0287 | 0.00031 | False |
| orth | 128 | 4096 | 2 | 15.49 | 0.8079 | 1 (1) | 0 | 0.212 | 0.0224 | [0.023, 0.0236] | 0.0292 | 0.00016 | False |
| orth | 128 | 16384 | 1 | 15.54 | 0.4051 | 1 (1) | 0 | 0.221 | 0.0114 | [0.0117, 0.0119] | 0.0293 | 1.5e-05 | False |
| orth | 256 | 4096 | 4 | 21.08 | 1.522 | 1 (1) | 0 | 0.156 | 0.0305 | [0.0308, 0.0316] | 0.0208 | 6.2e-05 | False |
| orth | 256 | 16384 | 2 | 21.11 | 0.7625 | 1 (1) | 0 | 0.162 | 0.0154 | [0.0156, 0.0159] | 0.0208 | 4.8e-05 | False |

## Targets

| target | status | T | diag pass | R̂ max | bulk ESS min | coverage | budget hit |
|---|---|---|---|---|---|---|---|
| orth_n128_m4096_r0 | done | 16000 | True | 1 | 1164 | 1 | False |
| orth_n128_m4096_r1 | done | 16000 | False | 1.01 | 937 | 1 | False |
| orth_n128_m4096_r2 | done | 16000 | True | 1 | 1028 | 1 | False |
| orth_n128_m1024_r0 | done | 16000 | False | 1 | 905 | 1 | False |
| orth_n128_m1024_r1 | done | 16000 | True | 1 | 1081 | 1 | False |
| orth_n128_m1024_r2 | done | 16000 | True | 1 | 1058 | 1 | False |
| orth_n128_m16384_r0 | done | 16000 | False | 1 | 930 | 1 | False |
| orth_n128_m16384_r1 | done | 16000 | True | 1 | 1049 | 1 | False |
| orth_n128_m16384_r2 | done | 16000 | True | 1 | 1040 | 1 | False |
| orth_n64_m1024_r0 | done | 16000 | True | 1 | 1903 | 1 | False |
| orth_n64_m1024_r1 | done | 16000 | True | 1 | 2002 | 1 | False |
| orth_n64_m1024_r2 | done | 16000 | True | 1 | 1854 | 1 | False |
| orth_n64_m4096_r0 | done | 16000 | True | 1 | 1791 | 1 | False |
| orth_n64_m4096_r1 | done | 16000 | True | 1 | 2030 | 1 | False |
| orth_n64_m4096_r2 | done | 16000 | True | 1 | 1868 | 1 | False |
| orth_n256_m4096_r0 | done | 16000 | False | 1.01 | 475 | 1 | False |
| orth_n256_m4096_r1 | done | 16000 | False | 1.01 | 509 | 1 | False |
| orth_n256_m4096_r2 | done | 16000 | False | 1.02 | 439 | 1 | False |
| orth_n256_m16384_r0 | done | 16000 | False | 1.01 | 504 | 1 | False |
| orth_n256_m16384_r1 | done | 16000 | False | 1.01 | 535 | 1 | False |
| orth_n256_m16384_r2 | done | 16000 | False | 1.01 | 451 | 1 | False |
| fmnist_n256_m4096_r0 | done | 16000 | False | 1.01 | 648 | 1 | False |
| fmnist_n256_m4096_r1 | done | 16000 | False | 1.01 | 476 | 1 | False |
| fmnist_n256_m4096_r2 | done | 16000 | False | 1.01 | 496 | 1 | False |
| fmnist_n256_m16384_r0 | done | 4000 | False | 1.04 | 122 | 1 | False |
| fmnist_n256_m16384_r1 | done | 4000 | False | 1.04 | 93 | 1 | False |
| fmnist_n256_m16384_r2 | done | 4000 | False | 1.05 | 132 | 1 | False |

## Figures

![ext_figure1_coverage](figures/ext_figure1_coverage.png)
![ext_figure2_curvature](figures/ext_figure2_curvature.png)
![ext_figureA_realdata](figures/ext_figureA_realdata.png)
![ext_figureB_cutoff](figures/ext_figureB_cutoff.png)
