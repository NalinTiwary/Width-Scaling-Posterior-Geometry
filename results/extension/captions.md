# Extension figure captions (draft)

**Figure 1 (main).** Posterior coverage of the theorem cylinder G_{B_{m,n}} versus width for
n = 64, 128, 256 on orthonormal inputs in d = 256. Small markers are the three joint
data/prior-center replicates (open = no observed exits, which does not imply zero Monte Carlo
uncertainty); diamonds are replicate medians. The dashed curve is the analytic floor 1 − 1/m,
common to all n because s = 1. B_{m,n} is fixed from the addendum's formula before sampling and
grows roughly like sqrt(n); see Appendix Figure B and the H/B table for how much of the cutoff
the posterior uses. [Describe observed results here.]

**Figure 2 (main).** Negative likelihood curvature d_H = σ²[−λ_min(∇²V)]₊ over 256 prespecified
posterior states per target (64 evenly spaced retained states per chain), restricted to the
cylinder. For each replicate, the thick bar with a triangle brackets the empirical 95th
percentile and the thin bar with a circle brackets the median; endpoints are numerical bounds
(block residual spectrum below; Rayleigh–Ritz on the 32 worst neuron blocks above), not
confidence intervals. Solid lines join replicate medians of the upper 95th-percentile bound;
dashed lines show the exact envelope D_th(m, n) for each n; the dotted line marks d = 1.
[Describe observed results here, including the matched n/sqrt(m) = 2 diagonal from
matched_diagonal.csv.] No PI/LSI constants, width thresholds or exponents are estimated.

**Appendix Figure A.** Fashion-MNIST T-shirt/top vs shirt, n = 256 (128 per class), inputs
centered and projected to 32 principal components fit on the official two-class training pool,
then unit-normalized; widths 4,096 and 16,384; three paired subset/prior-center replicates.
Left: coverage. Right: median and 95th-percentile d_H brackets (full-coordinate dense block
code). The dashed "safe envelope" uses the dataset's M₂ and the bound M₄² ≤ max_i‖x_i‖ M₂; it may
be far above one and is not informative at these settings. [Describe observed results here.]

**Appendix Figure B.** CDF of H/B_{m,n} at m = 4,096 (available for every n), pooled over all
retained draws of each replicate (solid, one line per replicate) with the exact Gaussian-prior
maximum distribution (dashed). The vertical line is the cylinder boundary. The horizontal
separation between colors shows how much of the higher-n coverage relies on the larger cutoff.
