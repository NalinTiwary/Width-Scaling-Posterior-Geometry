# Cylinder experiment results

Generated 2026-09-27T00:25:16+00:00 on `Apples-MacBook-Pro-2.local`, commit `d4eb1c012d` (dirty tree), config `config.yaml`.

## Target status

| target | status | T | diag pass | coverage | exits | R̂ max | bulk ESS min | hit budget |
|---|---|---|---|---|---|---|---|---|
| m256_seed0_ess | done | 4000 | False | 1 | 0 | 1 | 720 | False |
| m256_seed1_ess | done | 4000 | False | 1 | 0 | 1 | 880 | False |
| m256_seed2_ess | done | 4000 | False | 1 | 0 | 1 | 850 | False |
| m1024_seed0_ess | done | 4000 | True | 1 | 0 | 1 | 1100 | False |
| m1024_seed1_ess | done | 4000 | False | 1 | 0 | 1 | 950 | False |
| m1024_seed2_ess | done | 4000 | False | 1 | 0 | 1 | 890 | False |
| m4096_seed0_ess | missing | — | — | None | — | None | None | — |
| m4096_seed1_ess | missing | — | — | None | — | None | None | — |
| m4096_seed2_ess | missing | — | — | None | — | None | None | — |
| m1024_seed0_pcn | missing | — | — | None | — | None | None | — |

## Theorem checks (ESS targets)

| m | seed | coverage | H/B_m q95 | H/B_m q99 | q95 d₋ | q95 d₊ | D_th | d₊/D_th | E[V] post | E[V] prior |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 0 | 1 | 0.3199 | 0.3428 | 0.05234 | 0.05357 | 0.966 | 0.0555 | 20.57 | 23.75 |
| 256 | 1 | 1 | 0.3165 | 0.3394 | 0.05104 | 0.05257 | 0.966 | 0.0544 | 20.7 | 23.75 |
| 256 | 2 | 1 | 0.3162 | 0.3403 | 0.05235 | 0.05371 | 0.966 | 0.0556 | 20.7 | 23.75 |
| 1024 | 0 | 1 | 0.3336 | 0.3549 | 0.02713 | 0.0275 | 0.4876 | 0.0564 | 20.82 | 23.73 |
| 1024 | 1 | 1 | 0.3334 | 0.3559 | 0.02629 | 0.0266 | 0.4876 | 0.0546 | 20.72 | 23.77 |
| 1024 | 2 | 1 | 0.3325 | 0.3545 | 0.027 | 0.02736 | 0.4876 | 0.0561 | 20.68 | 23.78 |

## pCN cross-check (m=1024, seed 0)

Status: missing

## Figures

![figure1_coverage](figures/figure1_coverage.png)
![figure2_curvature](figures/figure2_curvature.png)
![figureS1_cutoff](figures/figureS1_cutoff.png)
![figureS2_efficiency](figures/figureS2_efficiency.png)
