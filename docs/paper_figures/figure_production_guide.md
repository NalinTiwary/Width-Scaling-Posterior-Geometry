---
title: "Two-figure production guide"
subtitle: "Posterior-domain coverage and loss autocorrelation from the existing BNN campaign"
author: "Implementation and presentation specification"
date: "1 October 2026"
---

# 1. The deliverable and the scientific scope

Produce exactly two main figures from the existing campaign:

1. **Posterior coverage of the deep spectral domain:** show where the posterior spectral-norm distribution lies relative to the domain boundary, with a prior reference.
2. **Width and loss autocorrelation:** show whether the implemented sampler forgets the likelihood-loss value more quickly as width increases, at one fixed step size.

Neither figure estimates a PI constant, an LSI constant, an integrated autocorrelation time, or a continuous-time convergence rate. No entropy tilts, variational maxima, fitted exponential decay, or fitted width exponent are needed. This guide replaces the previous figure-selection and analysis instructions for the paper. Preserve the original campaign, its diagnostics, and its audit as historical records.

Use existing outputs wherever they contain the required information. If a necessary array is absent, recover it from saved states or replay checkpoints. If recovery is impossible, make the targeted run specified in Section 10. Do not reconstruct an autocorrelation curve from an integrated-time estimate or from a plotted PDF.

The scientific setup remains the completed campaign: n=128, d=32, binary classification, tanh, sigma=1, shallow L=2 at widths 64, 256, 1024, 4096; deep L=3 at widths 32, 64, 128, 256; three matched data/center replicates; four chains per target. All chains are unrestricted. The final deep event is

$$
G_{2.5}=\left\{\frac{\|W_2\|_{\mathrm{op}}}{\sqrt m}\le2.5\right\}.
$$

The head and first layer are unrestricted. W2 is the later hidden matrix in the two-hidden-layer model. There is no head cutoff in this figure.

## 1.1 What has and has not been available for this guide

The supplied execution summary and five result PDFs have been inspected. They report complete reference runs, final dynamics at h=0.01, and zero exits among 196,608 inspected deep states. The actual scalar chains, per-state spectral tables, and project source code were not supplied with those attachments, and no matching raw exports were found in the available file search. Consequently, the exact project paths below are **logical locations to resolve**, not verified filesystem paths.

The included `figure_metrics.py` is implemented and tested calculation code accepting NumPy arrays. It contains no assumed HDF5 key names, proprietary loaders, or sampler implementation. `plot_style.json` fixes the presentation choices. The plotting adapter must map the real project outputs to the schemas in this guide.

The existing figures are not a source of numerical coordinates. Their role is to identify the relevant outputs and verify that the new plots address the same campaign.

## 1.2 Expected final package

Export:

- `figure_1_spectral_domain.pdf` and `.png`;
- `figure_2_loss_acf.pdf` and `.png`;
- the exact source tables for both figures;
- a short `figure_results.md` giving the numerical observations;
- `captions.tex` with the final captions;
- a compact provenance/diagnostics table and `recovery_log.md`;
- a plotting command that regenerates both figures from the exported tables without posterior sampling.

One small appendix table can carry predictive improvement and sampling diagnostics. Do not add PI/LSI ratio figures back into the paper to fill space.

# 2. Find the correct existing inputs

Work from the completed campaign export and its manifest. The earlier project called its result directory `results/final_geometry/`; use that location if it exists, otherwise identify its equivalent by the campaign metadata. Search by **content and identifiers**, not by the newest modification time.

| Logical input | What it must contain | Likely place to look |
|---|---|---|
| Target specification | architecture, m, n, sigma, center/data hashes, sampler identity | target manifest or spec JSON |
| Spectral state table | one S value or singular norm for each inspected deep reference state | analysis tables; spectral-state export |
| Spectral summary | per-target quantiles, inspected count, exit count | spectral summary CSV |
| Prior spectral reference | raw prior norms or q50/q95/q99 for each deep width | prior-control table |
| Dynamics loss trace | ordered retained post-transition likelihood potential V, including repeated rejected states | dynamics HDF5/NPZ scalar traces |
| Dynamics execution metadata | exact step, draw phase, spacing, acceptance, run lineage | checkpoint metadata and diagnostics |
| Predictive check | prior/posterior predictive negative log scores | predictive-score table |

Do not search the old `results/extension/` campaign for these figures. Those targets have different prior centers and a different scientific setup.

For each input record the actual path, internal array/dataset name, shape, dtype, source hash, and mapping to target/chain. Write this to `input_map.csv`. If only a family summary such as `relaxation_families.csv` is found, continue searching: that table alone does not contain the loss time series.

## 2.1 Select the dynamics by metadata, not by directory name

Use only the final fixed-step dynamics at **h=0.01**, for every width and both architectures. The supplied summary reports that these runs exist for all 24 targets. Some endpoint traces may be stored under `half_step`, `validation`, or `refinement` because that trajectory later became the production trajectory. Such a trace is eligible if its actual kernel and target match.

Exclude the h=0.02 trajectories and the h=0.005 validation trajectories from this figure. Exclude step calibration, burn-in, the reference elliptical-slice sampler, and any conditional/restricted trajectories. Do not rescale another step's horizontal axis and pool it with h=0.01.

An eligible trajectory has:

- the correct target/data/prior hashes;
- the identical adjusted sampler implementation and settings apart from width;
- h=0.01 in the production sigma=1 convention;
- four identifiable independent chains;
- chronological retained observations after the documented discarded segment;
- loss values at every transition, including rejections;
- a known save stride and no missing transitions.

Do not count a copied endpoint run twice under its old and new logical roles. Use a unique execution/run identifier plus state/draw indices to identify duplicates.

## 2.2 Identify V correctly

Use the **likelihood potential**, which in this campaign is the summed training cross-entropy,

$$
V(\theta)=\sum_i\left[\operatorname{softplus}(\sqrt2 f_\theta(x_i))
-y_i\sqrt2 f_\theta(x_i)\right].
$$

Do not use total posterior potential U, held-out loss, gradient norm, an average over observables, or a preselected family maximum.

If the logger saved mean training cross-entropy V/n or log likelihood -V, a known nonzero affine transformation leaves normalized autocorrelation unchanged. Those traces can therefore be used after recording the transformation; preferably convert them to V for a single canonical schema. This does not excuse an incorrect posterior target: the chain itself must still have used the specified summed likelihood.

If only U is saved, recover V by subtracting the saved prior quadratic or recomputing that quadratic from the corresponding state and center. A file label `energy` is not enough to decide which quantity it contains.

# 3. Figure 1: construct the spectral-domain data

## 3.1 Normalize exactly once

The plotted variable is

$$
S=\frac{\|W_2\|_{\mathrm{op}}}{2.5\sqrt m}.
$$

Determine which quantity the source contains:

- raw largest singular value: divide by $2.5\sqrt m$;
- largest singular value divided by $\sqrt m$: divide by 2.5;
- the final S variable: use directly.

Persist a `source_normalization` field. Do not infer normalization solely from the magnitude or divide an already normalized value again. As a plausibility check, the submitted results have medians near 0.76--0.79 and upper quantiles below one.

If raw W2 states must be processed, use float64 full singular-value decomposition and take the largest singular value. Do not replace it with the largest eigenvalue magnitude, Frobenius norm, or an unverified power-iteration estimate. Reuse the existing SVD cross-check report when the same calculations and source states are being reused.

## 3.2 Correct sample pool

Use the archived states from the unrestricted reference sampler, not the dynamics chains and not a pool already filtered to lie inside the domain. Within each of the 12 deep targets, the reported archive contains 4 chains times 4,096 saved states, giving 16,384 inspected states.

Pooling the values from the four chains is appropriate for an empirical marginal quantile because they target the same posterior. Preserve chain IDs for diagnostics. Do not pool the three data replicates into one distribution: they correspond to different posterior targets.

The expected count audit, to be verified from source rows, is:

| Group | Targets | Inspected states |
|---|---:|---:|
| Each replicate at a given width | 1 | 16,384 |
| All three replicates at a given width | 3 | 49,152 |
| Entire deep spectral study | 12 | 196,608 |

These count **inspected archived states**. They are not the number of all reference transitions and not independent-sample counts. If the actual rows differ, report the actual count and resolve the discrepancy before reusing the supplied total.

## 3.3 Per-target quantities

For each width m and replicate r compute from all inspected S values:

1. inspected count N;
2. count with S greater than one;
3. count with S at most one;
4. empirical fraction inside;
5. median, 95th percentile, and 99th percentile;
6. maximum S, retained in the result table when the statewise values or an audited maximum are available. If only an otherwise complete quantile/count summary survives, leave this unplotted field null rather than rerunning solely for it.

Use NumPy `quantile(..., method="linear")` throughout. The three posterior quantiles describe the distribution of states; they are not uncertainty bounds for an estimator.

For floating-point membership checks retain the previous numerical guard of $10^{-10}$ around one. A value inside this band is numerically ambiguous and must be resolved with the existing SVD audit procedure or recorded separately. If ambiguities exist, report lower/upper occupancy obtained by assigning them outside/inside. The supplied figure places all samples far from this band, so none are expected.

For plotting, for each quantile q and width m calculate the median of the three replicate quantile estimates. The vertical whisker extends from the smallest to the largest of those three estimates. This is a **replicate range**, not a confidence interval and not posterior probability mass. Retain every replicate value in the source table.

If only the existing per-target summary table is available, it is sufficient to draw this figure provided it contains these quantiles and inspected/exit counts and has the correct provenance. No new posterior samples are then required. A single summary already pooled over replicates is insufficient to reconstruct replicate ranges.

## 3.4 Prior reference

Reuse the 4,096 iid Gaussian W2 matrices per width from the previous prior-control calculation, or its saved quantiles. The W2 prior is identical across the three replicates because its center is zero and sigma=1. Use one prior reference per width; do not invent three prior replicates.

If the prior reference was not retained, regenerate exactly 4,096 independent m-by-m standard Gaussian matrices for each m in {32,64,128,256}. Use separate recorded streams; a seed derived from `20261001|prior_spectral|m=<width>` by SHA-256, first eight bytes modulo $2^{63}-1$, is sufficient. Save the scalar S values. This is direct prior simulation, not posterior MCMC.

Compare prior and posterior q50/q95/q99 only. Similar values support similarity of these norm statistics; they do not establish that all weights or the complete singular-value spectrum remain distributed as the prior.

# 4. Figure 1: exact visual specification

Use a **single panel**, 3.45 inches wide by 2.85 inches high, suitable for a single paper column. The informative content is the distance of spectral quantiles from the boundary. Do not spend half the figure on a second panel containing twelve values equal to one.

## 4.1 Axes and reference lines

- X axis: width, logarithmic base two. Ticks exactly 32, 64, 128, 256; label `Width $m$`.
- Y axis: $S=\|W_2\|_{\mathrm{op}}/(2.5\sqrt m)$; label `$\|W_2\|_{\mathrm{op}}/(2.5\sqrt{m})$`.
- Default y limits: 0.70 to 1.04. Major ticks: 0.7, 0.8, 0.9, 1.0. This is a distribution-position plot, not a bar chart; a zero baseline is unnecessary.
- If any plotted interval or point falls below 0.70, lower the axis to the next lower multiple of 0.05, with 0.02 additional margin. If any point is above 1.02, raise the upper limit to cover it plus 0.02. Never hide an exit or a quantile by clipping.
- Domain boundary: y=1, black dashed line, linewidth 1.1, dash pattern (5,3).
- Boundary label: `Domain boundary`, right-aligned at axes x=0.97 and data y=1.012. The annotation must stay above the boundary and outside the quantile curves.
- Optional asymptotic Gaussian-edge reference is **omitted**. The empirical prior comparator already provides the useful reference and avoids another overlapping label.
- Horizontal major grid only, color #E6E6E6, linewidth 0.5. No minor grid. Draw grid beneath data.

## 4.2 Quantile encoding

| Quantile | Color | Marker |
|---|---|---|
| Median | #0072B2, blue | circle |
| 95th percentile | #E69F00, orange | square |
| 99th percentile | #D55E00, vermilion | triangle up |

For each quantile:

- Posterior: solid line, linewidth 1.8, full opacity; filled marker size 4.2 points at each true width; replicate-range whisker linewidth 0.8 and cap size 2 points in the same color.
- Prior: dashed line with pattern (3,2), linewidth 1.0, same quantile color at opacity 0.65; hollow marker size 6.0 points, edge linewidth 0.8, with **no opaque face fill**.
- Draw posterior lines/whiskers first, prior dashed curves next, posterior filled points next, and hollow prior marker outlines last. When prior and posterior coincide, the larger hollow outline makes the overlap visible.
- Do not vertically or horizontally displace prior values to separate overlapping curves. Agreement should look like agreement.
- The solid posterior line joins the three-replicate median quantiles. It does not connect one selected favorable replicate.

Place two separate compact legends outside the axes below the panel. First row: Median, 95th, 99th using the colored quantile symbols. Second row: Posterior (solid/filled), Prior (dashed/hollow), using neutral black handles. Use a 7-point font and no legend frame. Reserve at least 0.50 inches below the plotting area. If the venue layout cannot accommodate this, place one concise combined legend in the caption; do not cover the data.

## 4.3 Title and occupancy annotation

Panel title: `Deep posterior spectral domain`, 9-point font. Do not use a claim-bearing title such as `Mass concentration verified`.

Put the exact aggregate count in a separate line above the axes, below the title or in the caption:

`0 exits / 196,608 inspected states`

Generate the numerator and denominator from the table. If exits occurred, show the actual count. If numerical membership ambiguities exist, annotate both the exit and unresolved counts and do not state that all samples were inside. Do not write twelve overlapping `0/16384` labels inside the axes. The accompanying table retains all per-target counts.

## 4.4 Caption content

The caption must say, in this order:

1. the variable and event boundary;
2. the fixed n and depth;
3. that samples come from the unrestricted posterior;
4. what solid/dashed curves and whiskers represent;
5. the exact inspected/exit counts;
6. a single sentence stating the empirical scope.

A usable base caption is:

> Posterior coverage of the deep spectral domain. For two-hidden-layer networks with n=128, we plot the median, 95th, and 99th percentiles of S=||W2||op/(2.5 sqrt(m)) from unrestricted posterior samples. The dashed horizontal line S=1 is the boundary of the domain used by the local deep LSI result. Solid curves show the median of the three replicate quantile estimates, with whiskers spanning the replicate range; dashed curves and hollow markers show the iid Gaussian-prior reference. All [N] inspected posterior states satisfied the restriction [replace if needed]. The domain therefore contains all observed samples across the tested widths.

Do not add a claim about the precise exit probability or its exponential rate. Keep the fuller diagnostic explanation in methods or the appendix table.

# 5. Figure 2: select the loss traces

## 5.1 Common step and trace window

Use h=0.01 and retain all rejected transitions as repeated states. A lag of k means **k sampler transitions**, not k accepted proposals and not k archived parameter saves.

Set the primary window to the **last 102,400 retained production transitions per chain**, for every target. This is T=1024 at the chosen h, but the figure's axis remains iterations. The supplied report indicates that each final dynamics chain has at least this many retained transitions; verify that from arrays.

This rule gives every curve the same trace length and avoids choosing windows according to their appearance. For a chain with 204,800 retained observations, use its last 102,400; for a chain with 102,400, use all of them. Exclude the documented discarded initial segment before selecting this window. Do not trim additional pieces because the loss happens to be high or correlated.

Record the original first and last iteration indices of every selected window. If the stored counter resets between chunks, reconstruct the chronological counter using the checkpoint lineage before selecting the window. Do not assume two adjacent files are contiguous merely because their names sort consecutively.

## 5.2 What to do with missing or irregular data

Every selected chain must have one loss value for each consecutive transition. Require iteration differences exactly one. Do not interpolate across gaps, sort by loss value, discard repeated values, or concatenate different chains.

If loss was saved only every s transitions, first look for unthinned scalar logs, complete per-transition states, or replayable checkpoints. A thinned series estimates correlations at multiples of s; it cannot recover the missing lags in the requested unit-stride plot. Because the present request allows recovery runs, obtain the missing unit-stride trace by the procedures in Section 10 rather than silently changing the estimand or interpolating a smooth curve.

If only accepted states were saved, use the initial state and complete acceptance/rejection history to reconstruct the post-transition series exactly. At an accepted transition advance to the new accepted-state loss; at a rejected transition repeat the current loss. Verify the convention with the logger. If the residence times or rejection history are unavailable, accepted-only states are insufficient; replay or run a new unit-stride trajectory.

## 5.3 Minimum integrity and sampling checks

Reuse the existing target/reference diagnostics. Recompute only the checks relevant to the selected V window:

- four chains are present, finite, nonconstant, and aligned with the target metadata;
- no duplicated chunk intervals or missing transitions;
- all four have 102,400 retained observations after selection;
- rank-normalized split/folded R-hat for V is below 1.01 and bulk ESS for V is at least 1,000 across the four selected chains;
- the target's reference posterior run passed the existing diagnostics;
- the actual step, kernel, and loss identity agree across widths.

These are ordinary checks on the data supporting an autocorrelation plot. Do not reapply the old integrated-time bootstrap-precision gate, the entropy-tilt gate, or the continuous-time step-halving criterion as if they tested this different estimand.

Preserve the original `dynamics_unresolved` status in the old audit. Write a separate field such as `fixed_step_loss_acf_status` for this new analysis, with the checks above and an explicit `analysis_scope=fixed_step_discrete_sampler`. A new scope is not a retroactive change to the old outcome.

If V fails the selected-window sampling checks, use the continuation rule in Section 10. If the numerical checks pass but curves are flat or cross, that is a scientific outcome, not a reason for more sampling or a different observable.

# 6. Figure 2: calculate the autocorrelation exactly

## 6.1 Within each chain

For a selected chain $v_0,\ldots,v_{N-1}$, with N=102,400, subtract that chain's own arithmetic mean:

$$
z_t=v_t-\bar v,\qquad \bar v=N^{-1}\sum_{t=0}^{N-1}v_t.
$$

Calculate the conventional unadjusted sample autocorrelation

$$
\widehat\rho_c(k)=
\frac{\sum_{t=0}^{N-k-1} z_tz_{t+k}}
     {\sum_{t=0}^{N-1}z_t^2},\qquad k=0,\ldots,K.
$$

The same N denominator in the two autocovariances cancels. This is the `adjusted=False` convention in statsmodels. Do not add the N/(N-k) finite-lag adjustment in one target but not another. At K=400 and N=102,400 the omitted-pair fraction is less than 0.4%.

Use K=400 initially and retain every integer lag. Lag zero must equal one within floating-point tolerance. Do not rank-transform, smooth, detrend beyond subtracting the mean, fit an exponential, truncate at the first negative value, or force negative estimates to zero. Never normalize by the variance of the sample mean or by the prior variance.

Compute by FFT with enough zero padding to avoid circular correlation. The supplied helper uses a transform length of at least 2N-1 and returns the same result as the direct sum above. It requires only NumPy and accepts the raw one-dimensional scalar array. No full model or gradient is needed.

## 6.2 Aggregate chains and data replicates in that order

For each architecture, width, and data replicate r, compute four separate chain ACFs and take their equal-weight arithmetic mean:

$$
\widehat\rho_{r,m}(k)=\frac14\sum_{c=0}^{3}\widehat\rho_{r,m,c}(k).
$$

The four chains are repeated samples of one posterior. The three data/center replicates are different posterior targets. Do not concatenate chains into a fictitious continuous trajectory, and do not treat all twelve chains as twelve independent data replicates.

At each lag, the central plotted curve is

$$
\widetilde\rho_m(k)=\operatorname{median}_{r=0,1,2}\widehat\rho_{r,m}(k).
$$

The shaded band runs from the smallest to the largest of the three replicate-mean ACFs. Its label is `range across three data/center replicates`. It is neither a 95% confidence interval nor a Monte Carlo error band. Do not recycle the earlier integrated-time bootstrap intervals for these curves.

Save both the chain-level and replicate-level ACF arrays. The pointwise median curve can follow different replicate curves at different lags; call it the pointwise median, not a representative chain.

## 6.3 Full-window display rule

Start both panels at K=400. If the absolute value of any width's median ACF at lag K exceeds 0.10, extend the calculation and the x limit for **both** panels to the next value in {800,1600,3200}. Continue until the endpoint condition is met or K=3200 is reached. Use the same raw 102,400-step windows; this is additional arithmetic, not additional sampling.

This rule prevents the figure from ending while a substantial correlation remains visible. Record the chosen K. Do not shorten the axis to make separation look larger or select different endpoints for the two architectures. If correlation remains substantial at 3200, show it and state that the loss remains correlated over the displayed range.

## 6.4 Numbers to report beside the figure

Create a small result table with the replicate-mean ACF at fixed lags 25, 50, 100, 200, and 400 for every target. For each architecture and replicate also compute

$$
\Delta_r(k)=\widehat\rho_{r,m_{\max}}(k)
-\widehat\rho_{r,m_{\min}}(k).
$$

Negative values mean less correlation at the wider endpoint at that lag. Report differences, not ratios: ratios are unstable when autocorrelations are near zero. These are ordinary observable summaries, not estimates of a functional-inequality constant.

Use this table to check the language in the result paragraph. If the deep curves separate at early/moderate lags but overlap later near zero, describe faster early decorrelation. If shallow improvements level off, describe the plateau. If curves cross substantially, report the crossing or mixed effect; do not select the lag with the strongest favorable difference and describe it as a uniform improvement.

The old integrated-time values are motivation for this reanalysis, not input to the new curves. Do not draw $\exp(-k h/\widehat\tau)$ using them.

# 7. Figure 2: exact visual specification

Use two side-by-side panels in a **6.9 by 3.0 inch** figure. Left: shallow model. Right: deep model. Share both axes. At the default K=400, x is linear; neither axis is logarithmic.

## 7.1 Axes and labels

- Left panel title: `(a) Shallow, $L=2$`.
- Right panel title: `(b) Deep, $L=3$`.
- X label on both panels: `Lag (sampler iterations)`.
- Shared left y label: `Autocorrelation of training loss $V$`.
- Default x limits: 0 to 400. Major ticks: 0,100,200,300,400; minor ticks at 50,150,250,350, without a minor grid.
- If K expands, use five major ticks at 0,K/4,K/2,3K/4,K and one minor midpoint between major ticks.
- Default y limits: -0.08 to 1.02. Major ticks: 0,0.2,0.4,0.6,0.8,1.0.
- If any replicate curve falls below -0.08, lower the shared y minimum to the next lower multiple of 0.05 minus 0.02. Never clip negative ACF estimates.
- Draw y=0 in #777777, solid, linewidth 0.7. No y=1 reference line is needed because normalization already fixes every lag-zero value to one.
- Do not draw a Gaussian-prior decay curve, a PI/LSI bound, an h-versus-h/2 threshold, or an estimated relaxation-rate curve.
- Horizontal major grid only, #E6E6E6, linewidth 0.5; no vertical grid.

## 7.2 Width colors and markers

Colors refer to **actual width**, so m=64 and m=256 have the same appearance in both panels.

| Width | Color | Marker |
|---:|---|---|
| 32 | #0072B2, blue | circle |
| 64 | #56B4E9, sky blue | square |
| 128 | #009E73, green | triangle up |
| 256 | #E69F00, orange | diamond |
| 1024 | #D55E00, vermilion | triangle down |
| 4096 | #CC79A7, purple | filled plus |

All median curves are solid, linewidth 1.8. Use filled markers of size 3.2 points at lags K/8, 2K/8, ..., K, omitting lag zero to avoid a pile of symbols. Draw the translucent replicate-range band first at alpha=0.12, then the central line. Do not add twelve opaque replicate curves on top of four median curves.

Each panel has its own width legend in increasing numerical order, arranged in two columns in the reserved bottom margin beneath its x-axis label. Use four entries: `m=64`, etc. Do not reuse a label such as `small/medium/large`, and do not encode width by rank within a panel because the actual widths differ across architectures.

Reserve a top margin for a shared note:

`Same adjusted sampler; h = 0.01; n = 128`

Keep the note outside the data area. Do not put `numerical validity criteria not met` across the new plot: that was the status of the previous continuous-time interpretation. Retain that status in the old audit and describe the present fixed-step scope accurately in methods.

## 7.3 Caption content

Use the following base, completing the final observation only after inspecting the actual ACF curves:

> Width and loss autocorrelation under a fixed sampling algorithm. We compare the autocorrelation of the summed training cross-entropy V for shallow and deep networks at n=128, using the same Metropolis-adjusted sampler and step size h=0.01 at every width. Each chain contributes its final 102,400 retained transitions, including rejected moves. Curves show pointwise medians across three data/center replicates after averaging four chains within each replicate; shaded regions show the replicate range. [Describe the observed deep and shallow width effects.] Lag is measured in sampler iterations, so the figure compares the implemented discrete kernels.

One short methods sentence is enough to delimit the claim: this comparison concerns loss decorrelation per sampler iteration, rather than a PI/LSI constant or an estimate of continuous-time relaxation. Runtime per iteration grows with model cost, so do not infer faster wall-clock sampling from these axes.

# 8. Shared typography, export, and plotting order

Use the following matplotlib settings or their exact visual equivalent:

```python
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
}
```

Use a white figure and axes background. Use vector PDF for the paper and a 300-dpi PNG for review. Avoid bold axis labels, heavy black frames, translucent text, arbitrary decorative labels, and legends inside the high-correlation part of Figure 2. Mathematical labels should be rendered, not printed as `sigma^2` or `sqrt(m)` text.

The companion `plot_style.json` gives explicit axes rectangles and figure-coordinate title/legend anchors. Use these as the baseline layout: Figure 1 axes [0.18,0.30,0.78,0.54]; Figure 2 axes [0.085,0.28,0.41,0.56] and [0.565,0.28,0.41,0.56]. Coordinates are fractions of figure width/height. All title/count/shared-note anchors use horizontal alignment center and vertical alignment top. Legend anchors use loc="center"; quantile and role legends are separate objects, so preserve the first when adding the second. Set explicit subplot margins after the legends are placed. Figure 1 needs extra bottom margin for its two legend rows and enough top margin for the count annotation. Figure 2 needs a shared-note margin and room below each panel for its width legend. `constrained_layout` or `tight_layout` is acceptable, but inspect the actual saved output. Cropping should not silently change the intended column width; keep explicit margins within the fixed canvas when possible.

The plotting script must load validated, fully computed tables. It should not run samplers, infer step size from filenames, choose observables, compute new validity decisions, or select a preferred replicate. Separate preparation, analysis, and rendering so a cosmetic change cannot change the scientific measurements.

# 9. Canonical exports and calculation tests

## 9.1 Minimal loss export for a plotting collaborator

Export one compressed NPZ file per architecture/width/replicate, 24 files total. Each file contains:

- `V`: float64 array of shape (4,102400), the selected unit-stride windows;
- `chain_ids`: integers [0,1,2,3];
- `iteration_start`: four integers identifying the first original transition in each row;
- `save_stride`: integer 1;
- `h`: float64 0.01;
- `architecture`: `shallow` or `deep`;
- `L`, `m`, `n=128`, `d=32`, `sigma=1`;
- `replicate_id`: 0, 1, or 2;
- `observable`: `summed_training_cross_entropy`;
- `contains_rejected_transitions`: true;
- `target_hash`, `sampler_code_hash`, `source_run_ids`, and source filenames;
- `selection_rule`: `last_102400_retained_production_transitions`.

Use simple numeric/string arrays, no pickled Python objects; load with `allow_pickle=False`. Keep any longer full traces in the original archive. Each selected loss array is approximately 3.28 MB uncompressed; all 24 are approximately 78.6 MB. Transfer in small groups if the upload channel has a per-file size limit. No full parameter archive is needed to compute Figure 2.

The provided helper accepts each V array and computes the replicate-mean ACF. Metadata checks must happen in the adapter before calling it; the helper cannot infer the sampling target from an anonymous numeric array.

## 9.2 Spectral exports

Preferred per-state CSV columns:

```text
architecture,L,m,n,sigma,a,replicate_id,chain_id,
original_draw_index,S,source_normalization,target_hash,source_run_id
```

Required per-replicate summary columns:

```text
m,replicate_id,n_inspected,n_inside,n_outside,n_ambiguous,
inside_fraction_lower,inside_fraction_upper,q50,q95,q99,max_S
```

Prior summary columns:

```text
m,n_iid_prior,seed,q50,q95,q99,normalization
```

If per-target quantiles and counts already exist, preserve and validate their source rather than recalculating from a rounded figure. If only rounded published values survive, recover the underlying table or states; those rounded values are not a replacement for the original plot data.

## 9.3 ACF exports

Save three tables:

1. `acf_chain.csv`: architecture, m, replicate, chain, lag, rho, N, h, start/end indices, source run.
2. `acf_replicate.csv`: architecture, m, replicate, lag, mean-of-four-chain rho.
3. `acf_plot.csv`: architecture, m, lag, pointwise median, replicate minimum, replicate maximum.

The plotted lag index is the actual transition lag. At K=400 there are 401 entries per chain, including lag zero. Save fixed-lag contrasts separately in `acf_width_contrasts.csv`.

## 9.4 Required lightweight tests

Use the existing project environment; do not upgrade dependencies merely to replot. Run `python selfcheck.py` from this package before processing the campaign. The calculation helper needs NumPy; the plot adapter also needs matplotlib and its normal file readers. Record their installed versions. Run these tests on the calculation code before processing the campaign:

- FFT and direct-sum ACF agree to absolute tolerance $10^{-10}$ on a fixed synthetic sequence at every lag through 100.
- Lag zero equals one to $10^{-12}$.
- Adding a constant or multiplying the trace by a nonzero constant leaves ACF unchanged to $10^{-10}$ on a moderately scaled fixture.
- A constant trace raises an explicit error; NaN/Inf or insufficient length raises an error; no value is replaced with zero.
- The replicate ACF equals the mean of four separately computed chain ACFs, including a fixture in which chain means differ. The implementation must not concatenate them.
- Spectral quantiles match `numpy.quantile` using linear interpolation; count totals add up; a value in the boundary guard is flagged ambiguous.
- A negative spectral norm input is rejected.

The bundled `selfcheck.py` implements these deterministic tests and an AR(1) simulation check. Its synthetic inputs are software fixtures only; they must never appear in paper figures or result tables.

Also make an internal 3-by-4 diagnostic sheet for V, one row per replicate and one column per chain, for the narrowest and widest width of each architecture. Plot the selected raw traces and chain ACFs. This is a visual integrity check, not another required paper experiment. Record obvious drift or broken chains and resolve actual integrity/sampling failures under Section 10. Do not smooth a defective trace into a plausible curve.

# 10. Missing-data recovery and targeted runs

Proceed down this list in order. The user has authorized obtaining necessary missing information, including new runs when retrieval cannot supply it. Do not launch a broad new experiment campaign simply because a table filename differs from the assumed one.

## 10.1 Recover files before computing anything new

1. Read the campaign manifest and plotting source to locate the real tables/HDF5 keys.
2. Check the original cluster output and the exported result archive, including checkpoint/scalar-trace files omitted from a lightweight figure export.
3. Export the minimal arrays in Section 9 instead of moving the full 111-GB parameter archive.
4. Verify hashes, counts, target identity, and iteration order after transfer.

If a table is missing but the exact required scalar arrays exist, regenerate the table from those arrays. That is sufficient; no posterior rerun is needed.

## 10.2 Spectral figure recovery

- **Only figure PDF/PNG exists:** retrieve the source quantile/count table. Do not digitize the plot.
- **Summary table lacks replicate detail:** retrieve per-target summaries or recompute quantiles from the statewise norm table.
- **Norm table is absent, W2 archive exists:** calculate S for the archived W2 states by full float64 SVD.
- **Prior comparator is absent:** generate the direct prior draws specified in Section 3.4.
- **Counts cannot be audited:** recover the original inspected-state list or rerun the norm calculation on the saved archive. Do not attach the reported denominator to a different sample pool.
- **All relevant reference states and summaries were lost:** rerun only the missing deep targets using the unchanged reference sampler/target. Use the previous staged reference schedule (4,096 burn-in, then retained totals 8,192,16,384,32,768,65,536 per chain; archive every 16 steps). The old entropy-calibration and separation segments are unnecessary for a new mass-only run. Stop when the existing reference diagnostics on V and S pass (R-hat<1.01, bulk ESS>=1000, tail ESS>=400), without waiting for any entropy-tilt criterion. Report the new actual inspected count. Retain four chains and the original three dataset/center replicates.

Record new source IDs and avoid pooling a rerun with its recovered original as though it were an independent replicate. Choose one complete provenance-consistent pool for each target, or explicitly document an appended continuation.

## 10.3 Loss-trace recovery

Use this priority:

1. Existing full scalar V trace at h=0.01.
2. Existing equivalent affine loss/log-likelihood trace with known semantics.
3. Full per-transition parameter states: recompute V using the original data, centers, and model convention.
4. Acceptance/residence records plus accepted-state losses: reconstruct all rejected repeats exactly.
5. Restart checkpoints containing state, RNG state, code identity, and exact counters: replay the original segment and add scalar V logging. A replay is exact only if the original random streams, backend, and transition logic are reproduced. Added diagnostics must not consume sampler RNG values or change updates. Verify an existing end checkpoint/state or recorded scalar values before calling it recovered original data.
6. If exact replay is unavailable, run a new scalar-recording trajectory for the missing target under the fixed kernel below. Label it a new run, not a recovered trace.

Only occasional parameter checkpoints without RNG/residence history cannot reconstruct the intervening trajectory. Do not linearly interpolate losses between them. If an endpoint half-step run is the only surviving trace, do not substitute it for h=0.01.

## 10.4 Exact new dynamics run, if needed

For each missing target, keep the original dataset and prior-center arrays and use the original tested adjusted sampler at h=0.01. Do not change sigma, architecture, center scaling, or training loss. Start four chains from the four corresponding final reference-chain states with fresh independent recorded RNG streams. Use a new run label `loss_acf_recovery` and derive each stream from a SHA-256 hash of the original master seed, target ID, chain ID, and recovery label.

Discard 2,000 transitions, then retain 102,400 consecutive transitions per chain. This matches the previous discarded duration of 20 at h=0.01. Store V at every transition, the acceptance flags, counters, and periodic restart records. Other expensive probes, gradients of probes, Hessians, entropy estimates, and full-state histories are unnecessary; the sampler still computes its normal likelihood gradients.

Validate the selected window using Section 5.3. If it fails the V sampling checks, continue all four affected chains by another 102,400 retained transitions and analyze the last 102,400. Permit one additional block of the same length if necessary, for at most 307,200 retained transitions per chain in this recovery procedure. Preserve each earlier segment and the continuation reason. If it remains unresolved, report that target as sampling-unresolved; do not change V, h, or the seed to obtain a favorable curve.

This is a completion rule for missing/unusable data, not a request to repeat the whole campaign. For a target with intact, valid existing traces, no new run is needed. For a valid flat or adverse ACF, no new run is authorized by this rule.

## 10.5 When the sampler implementation itself is unavailable

Recover the actual sampler source, commit, and environment used for the completed campaign. The earlier runbook defines its Gaussian-preserving proposal and full forward/reverse Metropolis ratio, but independently reimplementing that kernel is a new software version.

If reimplementation is unavoidable, run its existing known-distribution and acceptance-ratio tests first. Do not mix newly implemented and old kernels across widths in one claimed fixed-kernel comparison without a numerical equivalence check. If equivalence cannot be established, use the new verified implementation consistently across all widths of the affected architecture and identify that rerun. This is the exceptional case where more than one missing target may need recomputation.

# 11. Results text and supporting table

## 11.1 Spectral result to write

Generate a sentence using the exact counts and width range. State that the final spectral restriction contains all or the measured fraction of inspected unrestricted posterior states. Then report the q99 range across widths and whether it remains below the boundary. Describe agreement with prior **norm quantiles**, if present.

Do not say that the entire posterior equals its prior, that zero observed exits establish mass exactly one, or that these observations identify the exponential concentration rate. Those claims are unnecessary to communicate the result.

## 11.2 Width result to write

Describe the actual ACF curves, with the deep result first if it is the clearer pattern. Support the text with fixed-lag differences from Section 6.4. For example, if the curves show it, state that increasing width reduces loss autocorrelation at common iteration lags, with the strongest change occurring over the deep width sweep and a shallow plateau after an initial improvement.

Do not copy that expectation into the final caption before computing the curves. A decrease in the old integral does not uniquely determine the new curve's shape. If the new plot shows crossing curves, describe which lag range improves. If it is flat, say that decorrelation is stable over the tested widths.

The comparison is empirical behavior of V under this sampler. Do not use phrases such as `we measure the PI constant`, `LSI rate recovered`, `the m^{-1/3} law is verified`, or `wall-clock speedup` in these results.

## 11.3 One compact appendix table

Use one row per target, or a clearly labeled aggregation with the full table supplied as a sidecar. Columns:

- architecture, m, replicate;
- reference-diagnostic status;
- V R-hat and bulk ESS on the chosen dynamics window;
- h and selected transitions per chain;
- acceptance fraction;
- inspected spectral-state count and exits for deep targets;
- prior-minus-posterior predictive negative log score from the existing predictive check;
- source/recovery status.

This table establishes provenance, basic sampling quality, and that the posterior predicts better than its prior mixture. The predictive check does not need another main plot. Do not include the discarded entropy-tilt diagnostics or PI/LSI ratios in this table; preserve them in the original reproducibility archive.

# 12. Final production checklist

The plotting collaborator should finish with this sequence:

1. Resolve the logical inputs and write `input_map.csv`.
2. Select the 24 final h=0.01 targets and four chains each; verify loss identity and rejects.
3. Select the same last-102,400 window rule for every chain and check its V diagnostics.
4. Resolve missing data through Section 10 only where necessary.
5. Run the supplied calculation self-checks.
6. Generate spectral per-target quantiles/counts and ACF chain/replicate tables.
7. Generate the exact plotted tables, including all three replicate values/ranges.
8. Draw Figure 1 and Figure 2 with the fixed palette and layout.
9. Fill captions and result text from the computed tables.
10. Inspect both PDF and PNG at intended paper size and at 200% zoom.
11. Save source hashes, metadata, recovery history, tables, captions, and the regeneration command.

Check explicitly that no legend/annotation covers a curve; the spectral boundary is visible; all inspected counts match; width colors agree across ACF panels; every ACF starts at one; no ACF has been clipped at zero; all lag units refer to actual transitions; no rejected move has disappeared; and all outputs contain the same scope and data selection rules.

The required final figure names are new, so the original `main_1_relaxation.pdf`, `main_2_entropy.pdf`, and `main_3_spectral.pdf` remain untouched. The original analysis has not been made to pass retroactively. The paper is using two simpler, explicitly defined empirical measurements.

# 13. References for calculation conventions

- Completed user-supplied campaign summary, `Pasted markdown(20261001-190418).md`, and its five result PDFs. These establish the reported source campaign and current availability; raw traces remain the numerical authority for replotting.
- User-supplied theory slides version (15), slides 3 and 10, for the model and final deep spectral domain.
- Original `execution_runbook.md`, Sections 7--8 and 12, for the sampler, h conventions, archives, and existing SVD checks. This guide changes the paper's empirical measurements, not those sampling targets.
- Statsmodels official ACF documentation, for the unadjusted autocovariance convention, lag indexing, and FFT option: <https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html>.
- NumPy official quantile documentation, for the fixed linear-interpolation quantile convention: <https://numpy.org/doc/stable/reference/generated/numpy.quantile.html>.

The reference calculation module implements the explicit formulas above without depending on a particular current statsmodels return type. Plotting decisions, displayed lags, trace window, colors, and recovery budgets are design choices fixed in this guide.
