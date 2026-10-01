---
title: "Final figure cleanup guide"
subtitle: "Two main figures for BNN posterior sampling"
date: "1 October 2026"
---

# 1. Final design decision: show the theorem's threshold and the whole coverage curve

**Revision 2 supersedes the fixed-2.5 design and the Posterior/Prior main-panel layout.** The main spectral figure now has two complementary posterior panels: **(a) Spectral quantiles versus width** and **(b) Coverage versus cutoff**. Both mark the theorem-derived threshold $a_*=r_0+2\sigma=2$. A companion table gives the exact number of inspected states above 2 for every width and replicate. No extra positive margin is selected for the main analysis.

The prior comparison moves to a small appendix table. The existing prior and posterior curves nearly overlap; their agreement remains relevant context but does not warrant using half the main figure for a duplicate visual. Figure 2 retains Shallow/Deep loss-autocorrelation panels.

## 1.1 The mathematical statement the figure illustrates

For this deep architecture ($L=3$), define

$$T_m(\theta)=\frac{\|W_2\|_{\mathrm{op}}}{\sqrt m},\qquad G_a=\{T_m\le a\}.$$

The later hidden-layer prior center is exactly zero, so its uniform normalized bound is $r_0=0$. The configured prior scale is $\sigma=1$. Consequently, **the strict admissibility threshold is $a_*=2$**. Verify those facts from the configuration and center matrices; do not infer them from posterior draws.

The mass statement says: **choose any fixed $a>2$, keep that cutoff and the dataset fixed, and increase width; the posterior mass of $G_a$ tends to one.** Equivalently, for every fixed $\varepsilon>0$, $\pi_m(T_m>2+\varepsilon)\to0$ as $m\to\infty$. The asymptotic variable is width, not $\varepsilon$.

The reference line at 2 is therefore labelled **admissibility threshold**, never an admissible local-domain cutoff, a posterior support boundary, or an established posterior limit. The theorem does not assert that $\pi_m(T_m>2)\to0$, nor that every empirical quantile is monotone in width. Finite-width empirical curves need not be perfectly ordered.

In general depth, the same domain statistic is $T_m=\max_{2\le\ell\le L-1}\|W_\ell\|_{\mathrm{op}}/\sqrt m$. The present campaign has just one such matrix, $W_2$. The head and first layer remain unrestricted.

## 1.2 What each output answers

| Output | Exact question | Interpretation |
|---|---|---|
| Figure 1a | Where are the median, 95th, and 99th posterior quantiles relative to 2, as width changes? | Direct measurement of the domain-defining statistic |
| Figure 1b | How much posterior mass is empirically below every cutoff $a$ in the displayed range? | Any fixed admissible cutoff can be read without choosing a favorable margin |
| Threshold-count table | How many archived states satisfy $T_m>2$? | Boundary-case diagnostic; nonzero counts do not contradict the theorem |
| Figure 2 | How does loss decorrelation change with width under the fixed kernel? | Practical observation compatible with the width story, not a PI/LSI estimate |

The two spectral panels use the same unrestricted reference samples. Re-evaluating an event does not change the posterior or require another chain. Old counts at 2.5 remain in the historical audit; they are no longer the main concentration evidence.

## 1.3 Do we need more widths?

**No additional widths are recommended for the current illustrative claims.** Four widths and three data/prior-center replicates already show a resolved upper-quantile movement. First use the existing samples to recover the full coverage function and threshold counts. Another deep width would require a new posterior target and reference sampling, while still not establishing an asymptotic theorem from finitely many points.

Do not fit a concentration exponent, claim the tested widths satisfy the quantitative large-width regime, or describe this fixed-$n$ experiment as verifying $m\gg n^3$. More widths should answer a separately stated scientific question, not be added until the plot looks more like a desired limit.

# 2. Published visual references and what to borrow

The following papers were checked in their published or author-hosted PDFs. They are examples of readable scientific presentation, not a claim that a conference mandates one plotting style. The dimensions and colors below are design choices for this paper.

1. **Zhang et al., ICLR 2020, _Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning_, Figure 3, PDF page 7.** The panels each answer one question; short panel labels identify the comparison, and the caption carries the explanation. Borrow the compact panel structure and redundant color/marker encoding. Use a white background rather than copying its tinted background. [R1]
2. **Sharma et al., AISTATS 2023, _Do Bayesian Neural Networks Need To Be Fully Stochastic?_, Figure 2, PDF page 5.** Two aligned panels use a common scale for a direct comparison, with a detailed caption explaining what distinguishes them. Borrow the separation of questions into clearly labelled panels; the revised spectral panels use different y scales because they measure different quantities. Its function-space mixing question also motivates the first supplementary analysis. [R2]
3. **Izmailov et al., ICML 2021, _What Are Bayesian Neural Network Posteriors Really Like?_, Figures 2 and 3, PDF page 5.** Aligned small multiples separate conditions, and the legends identify the measured series without repeating the experimental setup inside every panel. Borrow that division of labor between axes, legend, and caption. Do not imitate the tiny type required by its many-panel displays. [R3]

These sources motivate presentation choices only. They do not establish the user's width results, and their algorithms and datasets are not experimental baselines for this study.

# 3. One shared visual specification

## 3.1 Size and typography

Render at the physical width used in the manuscript. The renderer defaults to **5.5 inches wide** and **2.65 inches high** for each two-panel figure. This is a convenient narrow full-width layout, not a universal venue requirement. Read the actual template's `\textwidth`; if necessary, print `\the\textwidth` in a temporary TeX build and divide its value in TeX points by 72.27 to obtain inches. For a wider two-column layout, pass `--width-in 6.75`. Keep the height and point sizes fixed when changing width.

Do not create a 10-inch figure and shrink it until its labels become unreadable. Include the final full-width PDF with `\includegraphics[width=\textwidth]{...}` only when its original width matches the manuscript's text width. Use the appropriate `figure` or `figure*` environment for the actual template.

| Element | Exact setting |
|---|---|
| Font | DejaVu Sans; mathematical text in DejaVu Sans |
| Axis-label size | 9 pt |
| Tick, legend, threshold-label size | 8 pt |
| Panel-title size | 9 pt, regular weight |
| Curve width | 1.6 pt |
| Quantile markers | 4 pt, filled, edge width 0.5 pt |
| Autocorrelation markers | 3.2 pt, filled, edge width 0.5 pt |
| Axis spines | Bottom and left only, 0.65 pt, `#444444` |
| Tick direction and length | Outward, 3 pt, width 0.6 pt |
| Grid | Major horizontal only, `#E5E5E5`, 0.45 pt |
| Figure / axes background | White |
| Export | Vector PDF; 300-dpi PNG; no transparent background |
| Font embedding | `pdf.fonttype = 42`, `ps.fonttype = 42` |

Use manual axes rectangles so the legend space is deterministic. Figure 1 uses left `(0.11, 0.29, 0.36, 0.57)` and right `(0.62, 0.29, 0.36, 0.57)` in figure coordinates; both y axes have their own labels and ticks. Figure 2 retains left `(0.105, 0.29, 0.405, 0.57)` and right `(0.570, 0.29, 0.405, 0.57)`, with a shared y scale and y labels on the left only. Panel titles sit above the axes with a 6-pt pad.

Use no overall title or subtitle in the figure. A figure caption is already its title and explanation. Remove the current lines “Deep posterior spectral domain,” “0 exits / 196,608 inspected states,” and “Same adjusted sampler; h = 0.01; n = 128.”

## 3.2 Color and marker dictionaries

Color identifies the quantity named in each panel legend. Figure 1a colors identify quantiles; Figure 1b and all autocorrelation panels use the common width dictionary. Each panel has its own explicit legend. The prior is not overlaid in the main figure.

**Spectral quantiles**, used in Figure 1a:

| Quantile | Color | Marker | Legend text |
|---|---|---|---|
| 0.50 | `#56B4E9` | Circle `o` | Median |
| 0.95 | `#0072B2` | Square `s` | 95th |
| 0.99 | `#253494` | Up-triangle `^` | 99th |

This ordered blue palette emphasizes increasingly high quantiles. Marker shape makes the curves distinguishable in grayscale. Quantile lines are solid and markers filled. The coverage curves use step functions and the width palette. There is no prior/posterior line-style legend.

**Network width**, shared by the coverage panel and every autocorrelation figure:

| Width | Color | Marker |
|---:|---|---|
| 32 | `#0072B2` | `o` |
| 64 | `#56B4E9` | `s` |
| 128 | `#009E73` | `^` |
| 256 | `#E69F00` | `D` |
| 1024 | `#D55E00` | `v` |
| 4096 | `#CC79A7` | `P` |

The same numerical width must have the same color and marker in shallow, deep, predictive, and step-size plots. Do not recolor a panel by its local width rank. Quantile and width palettes describe different variables; their legend labels make that distinction explicit.

# 4. Figure 1: spectral quantiles and coverage without a chosen margin

## 4.1 Required inputs and source conversion

Use every archived unrestricted posterior state on which the spectral norm was inspected. The supplied report lists four widths, three targets per width, four chains per target, and 4,096 archived states per chain: 196,608 states total. Verify the actual source counts and original draw IDs.

The input is the unrounded statewise value $T=\|W_2\|_{\mathrm{op}}/\sqrt m$. If only the old normalized values $S=T/2.5$ were stored, multiply each by the recorded historical normalizer 2.5 exactly once. That operation recovers $T$; it does not set a new cutoff. Record the source units and conversion in the adapter manifest.

A table of only three quantiles is insufficient for the coverage panel or exact counts above 2. Retrieve the statewise norms, or recompute full float64 SVD norms from the existing reference parameter archive. Do not infer a CDF by fitting a Gaussian to the three quantiles, digitizing curves, or linearly interpolating between quantiles. No new posterior sampling is needed if the archived norms or matrices remain available.

## 4.2 Panel (a): spectral quantiles versus width

Title: **“(a) Spectral quantiles”**. The x axis is **“Width $m$”**, log base 2, with integer ticks 32, 64, 128, 256 and 0.12 log-base-2 padding. The y label is **“$T=\|W_2\|_{\mathrm{op}}/\sqrt m$”**. Use a linear y axis with default limits 1.85--2.16 and ticks 1.9, 2.0, 2.1. Expand the limits if needed to include every displayed quantile and replicate whisker; never clip them. The sample maximum is reported in the table and is not plotted as a quantile.

For each width and replicate, pool its four balanced chains and calculate the 0.50, 0.95, and 0.99 quantiles using linear interpolation. For each quantile, plot the median of the three replicate-specific quantiles, with whiskers from their minimum to maximum. Use the quantile palette from Section 3, 1.6-pt lines, 4-pt markers, 0.75-pt whiskers, and 2-pt caps. Whiskers describe between-replicate variation, not a confidence interval or a posterior credible interval.

Draw the only horizontal reference at **$T=2$**, gray `#555555`, 0.9 pt, dash pattern `(4,3)`. Label it **“$a_*=2$”** near the left edge, just below the line. Define it as the admissibility threshold in the caption; do not label it “domain boundary” or shade the upper half as a failed-theorem region. The median may be below 2 while upper quantiles remain above it.

Use a three-entry legend below this panel: Median, 95th, 99th. Put it at figure coordinates `(0.29,0.085)` in one row. Percentile meaning is explicit in the caption. Straight segments between observed widths are visual guides, not a fitted scaling law.

## 4.3 Panel (b): empirical coverage versus cutoff

Title: **“(b) Coverage vs. cutoff”**. The x label is **“Cutoff $a$”** and the y label is **“Empirical coverage”**. Both axes are linear. The y range is 0--1.02 with ticks 0, 0.25, 0.5, 0.75, 1. The default x range is 1.75--2.30, with ticks 1.8, 2.0, 2.2. Expand it to include the minimum and maximum recorded $T$ across all targets, with 0.01 padding, if those fall outside the default range. Axis limits are display choices, not domain definitions.

For each width $m$ and replicate $r$, calculate the empirical CDF

$$\widehat F_{m,r}(a)=\frac1{N_{m,r}}\sum_{i=1}^{N_{m,r}}\mathbf1\{T_{m,r,i}\le a\}.$$

The comparison uses unrestricted draws and includes every archived state. It does not resample or condition the posterior separately at each cutoff.

For each width, plot the pointwise median of the three CDFs as a right-continuous step curve, with same-color 0.10-alpha shading between their pointwise minimum and maximum. Use the established width palette. Evaluate on the sorted union of that width's three replicate sample supports, plus $a_*=2$ and the plotting endpoints. This aligns the evaluation grid; it does **not** pool three different posterior targets into one empirical distribution. Do not smooth the CDF.

Draw a vertical dashed gray line at **$a=2$** and label it **“$a_*=2$”** to its right near the bottom (at 8% of axes height). Place one marker per width at its median coverage at exactly 2. The marker identifies an empirical boundary-case measurement, not an admissible theorem cutoff. Keep any overlapping markers at their true coordinates; the numerical table resolves overlaps. Use a two-column width legend below the panel at `(0.80,0.075)`.

Do not add selected lines at 2.05, 2.1, or 2.5. The full function lets the reader examine any fixed cutoff. For a fixed $a>2$, the theorem predicts coverage tending to one with width; it does not require the four finite-width CDFs to be ordered at every $a$, especially at or below 2.

## 4.4 Exact threshold counts and numerical audit

For each width and replicate report $N$, $N_{>2}=\#\{T>2\}$, $N_{>2}/N$, $N_{\le2}$, and the sample maximum. Equality belongs to $G_2$ because the domain is defined with $\le$. The complementary count is a useful diagnostic even though the stated theorem requires a strictly larger cutoff.

Use unrounded float64 values. Flag values within $10^{-10}$ of 2. For such states, inspect the original matrix and cross-check the norm with an independent SVD implementation before treating the sign as resolved. Export the nominal count plus the count bounds implied by this numerical guard. The guard is a computation check, not an added domain margin: it must never replace 2 by $2+10^{-10}$ in the scientific definition.

The strict-above count is `sum(T > 2.0)`. An old $S$ export gives the same comparison through `S > 0.8`, subject to roundoff near the boundary. Tests must check below, equal, and above cases and reject double normalization. Use $\sum\mathbf1\{T\le2\}+\sum\mathbf1\{T>2\}=N$ as a bookkeeping check.

Create a 12-row table, one row per posterior target. A compact four-row paper table may show the three per-replicate counts separately and the median [minimum, maximum] percentage above 2. Do not present counts pooled across datasets as a probability for one posterior; if a descriptive grand total is supplied, explicitly label it as an aggregate across distinct targets. No iid-binomial interval is justified from the nominal correlated-state count.

The existing summary quantiles and maxima show that there are values above 2, but do not determine these counts. A median across replicate-specific quantiles cannot be used to reconstruct a pooled tail percentage. If the underlying arrays cannot be recovered, deliver the quantile panel alone and mark the coverage/count analysis unavailable; do not invent it from the PDF.

## 4.5 Prior reference and retained historical output

Keep the existing posterior/prior comparison as an appendix table: width, posterior median-across-replicates $q_{0.50},q_{0.95},q_{0.99}$, and the three prior quantiles from the existing 4,096 iid matrices per width. All values use $T$ units. The prior pool is shared across the three posterior replicates and is not three independent experiments. No new prior draws are required.

The agreement explains why a hidden spectral norm can remain prior-like while predictive quantities still learn from data. Do not turn this agreement into a theorem that the entire posterior or singular-value spectrum equals its prior. The old statement “all inspected states are inside $G_{2.5}$” stays in the historical audit, without a main-figure reference line or badge.

## 4.6 Caption and manuscript sentence

**Caption:**

> **Width dependence of the posterior spectral statistic and domain coverage.** Here $T=\|W_2\|_{\mathrm{op}}/\sqrt m$, the hidden-layer prior center is zero, and $\sigma=1$, so the theorem permits any fixed cutoff $a>a_*=r_0+2\sigma=2$. (a) Median, 95th, and 99th posterior percentiles of $T$ versus width. Points summarize three dataset/prior-center replicates by their median, with whiskers showing their range. (b) Empirical coverage $\widehat\pi(G_a)$ versus cutoff $a$, with one curve per width; curves and shading show the pointwise median and range across replicates. Both panels use unrestricted posterior samples and mark the threshold $a_*=2$. For every fixed $a>2$, the theorem predicts coverage tending to one as width increases; it makes no such assertion at $a=2$. Exact sample counts above 2 are reported in the accompanying table. Here $n=128$ and each target uses four chains.

**Result sentence supported by the current quantile report:** “The reported 99th-percentile curve decreases from approximately 2.113 at width 32 to 2.030 at width 256, illustrating a narrowing upper tail above the theorem's admissibility threshold.” These are rounded median-across-replicate quantile summaries. Insert any coverage percentages only after calculating the statewise CDF and count table.

Do not write “all posterior draws lie in the theorem domain” after replacing 2.5 by 2. The new line represents a different mathematical object: the lower threshold for choosing an admissible cutoff.

# 5. Figure 2: training-loss autocorrelation

## 5.1 Keep the accepted analysis unchanged

Use the existing fixed-step $h=0.01$ results, with four chains and three replicates per width. Each chain contributes its last 102,400 retained transitions, including rejected moves as repeated values. Keep the current per-chain centering and normalization, average the four chain curves within each replicate, and plot the pointwise median of the three replicate curves with their pointwise range.

Do not recompute these curves from integrated-time tables, refit an exponential, smooth a curve, or switch to rank-transformed loss. The observable remains summed training cross-entropy $V$.

## 5.2 Final display

Titles are **“(a) Shallow”** and **“(b) Deep”**. Put $L=2$ and $L=3$, $n=128$, and $h=0.01$ in the caption. The x label is **“Lag (sampler iterations)”**. The left y label is **“Loss autocorrelation”**. Repeat neither the y label nor y tick labels on the right panel.

Display **lags 0 through 200** in the main paper. The reported curves already have little structure beyond this range; concentrating on the first 200 lags makes the visible separation readable without a log axis or an inset. Keep the complete audited 0-through-400 arrays in the exported CSV and plotting package. This is a display crop only, not a shorter chain or a change in the estimator. The caption explicitly identifies the displayed lag range; it must not claim decorrelation is complete by lag 200.

| Setting | Specification |
|---|---|
| x scale and ticks | Linear, 0 to 200; ticks 0, 50, 100, 150, 200 |
| y scale and ticks | Linear, -0.08 to 1.02; ticks 0, 0.2, 0.4, 0.6, 0.8, 1.0 |
| Curves | Solid, width 1.6 pt; exact width colors and markers from Section 3 |
| Marker positions | Lags 25, 50, 75, 100, 125, 150, 175, 200; none at lag 0 |
| Replicate range | Same-color fill, alpha 0.10, no border, behind the curve |
| Zero reference | `#777777`, 0.65 pt, solid |
| Other references | None; no $1/e$, 0.1, Gaussian-time, PI, or LSI line |

If any displayed replicate bound is below -0.08, expand both y axes to contain it with at least 0.02 padding. Do not truncate negative values. The renderer uses the smallest needed lower limit rounded down to a multiple of 0.05. Its upper bound stays at 1.02. Values outside [-1,1] beyond numerical tolerance indicate a calculation error.

Retain every width in its established order: shallow 64, 256, 1024, 4096; deep 32, 64, 128, 256. Near-overlap in the shallow panel is a result and must stay visible. Do not offset curves vertically or omit a width because it overlaps another.

Put a separate two-column legend below each panel, centered at the panel center and figure y=0.075. The intended visual reading order is increasing width across the top row, then increasing width across the bottom row. Matplotlib fills legend columns; reorder the handles explicitly to obtain that reading order. The renderer implements this reorder. No legend frame, title, or background patch is needed.

## 5.3 Caption structure and ready-to-use draft

Use: result sentence; observable and algorithm; trace window; aggregation; architecture-specific finding; scope of the measurement.

> **Width improves loss decorrelation, with diminishing returns in shallow networks.** Autocorrelation of the summed training cross-entropy $V$ is shown over the first 200 sampler lags for shallow ($L=2$) and deep ($L=3$) networks. All runs use the same Metropolis-adjusted Gaussian-preserving Langevin kernel with step size $h=0.01$ and $n=128$. Each of four chains contributes its last 102,400 retained transitions, including repeats after rejection. Curves are the pointwise median across three dataset/prior-center replicates after averaging the four chain autocorrelations within each replicate; shading shows the replicate range. The deep curves show a pronounced width benefit, while the shallow improvement largely levels off after $m=256$. The comparison concerns decorrelation per sampler iteration.

“Gaussian-preserving” refers to the proposal's treatment of the Gaussian prior, not a claim that the posterior is Gaussian. Use the exact kernel name from the methods; do not silently rename it standard MALA if its proposal differs.

One methods sentence can explain that observable autocorrelation is not an estimate of an optimal PI/LSI constant. The caption needs no account of the old integrated-time analysis or its unresolved continuous-time checks.

# 6. Implementation contract

## 6.1 Files and renderer commands

Use the existing project environment. The renderer requires Python, NumPy, and Matplotlib; record the versions actually used. The CSV paths below are proposed adapter outputs, not assertions about current project filenames.

```bash
python clean_figures.py spectral \
  --input exports/spectral_states.csv \
  --context exports/theory_context.json \
  --out paper_figures/figure_1_spectral_domain \
  --width-in 5.5

python clean_figures.py acf \
  --input exports/loss_acf_replicates.csv \
  --out paper_figures/figure_2_loss_acf \
  --width-in 5.5 --lag-limit 200
```

Each command writes a PDF, PNG, and JSON sidecar with source SHA-256, plotting settings, package versions, and figure dimensions. The spectral command additionally writes per-target threshold counts, per-replicate quantiles, and the exact per-replicate CDF values behind the plotted coverage summary. Keep the existing full-resolution summary arrays and provenance alongside the adapter CSVs. Never replace the source campaign files.

## 6.2 Statewise spectral CSV

Columns: `width,replicate,chain,draw,T`.

- Widths: 32, 64, 128, 256. Replicates: 0, 1, 2. Chains: 0, 1, 2, 3.
- `draw` is the original archived transition ID, unique within each target/chain, with a consistent archive stride of 16 for this campaign. It is not an independent-draw index.
- `T` is the unrounded normalized operator norm $\|W_2\|_{\mathrm{op}}/\sqrt m$, finite and nonnegative.
- Expected production export: 4,096 rows per chain and 196,608 rows in total. The renderer's default `--expected-per-chain 4096` enforces this. Change it only to match an explicitly audited different archive, not to bypass missing rows.
- The renderer rejects missing target/chain combinations, duplicate draw IDs, wrong counts, and irregular archive spacing.

The adapter must check units against at least one directly recomputed matrix norm per target before setting `T`. Values alone cannot reliably reveal a normalization error. Record `r0=0`, `sigma=1`, the exact center/configuration hashes, and any historical $S$-to-$T$ conversion. The plotting command accepts only this campaign's theorem threshold; it provides no user-tunable positive margin.

The required `theory_context.json` records `weight_layers: 3`, `n: 128`, `sigma: 1.0`, `r0: 0.0`, `hidden_center_max_normalized_opnorm: 0.0`, and `statistic: "opnorm_W2_over_sqrt_m"`, plus the actual SHA-256 of the audited target manifest in `target_manifest_sha256`. The adapter obtains these facts from the original model/configuration and center matrices. The renderer checks this campaign context and computes `r0 + 2*sigma`; it does not accept a freely chosen cutoff. A JSON declaration alone does not replace checking the saved matrices and hashes.

Spectral output suffixes are `_threshold_counts.csv` (12 target rows), `_quantiles.csv` (36 target/quantile rows), and `_coverage.npz` (one cutoff grid and a 3-by-grid CDF array per width; load with `allow_pickle=False`). The PDF/PNG and JSON sidecar retain the chosen output prefix. The compressed NPZ avoids duplicating large support grids in text tables.

The statewise CSV replaces the old quantile-only renderer input. The previous spectral quantiles remain useful for cross-checking the new computation, but cannot supply panel (b).

## 6.3 Autocorrelation CSV

Columns: `architecture,width,replicate,lag,acf`.

- `architecture` is `shallow` or `deep`; width sets are as above.
- Replicate IDs are 0, 1, 2.
- One row contains the already averaged four-chain curve for one replicate at one integer lag.
- Every combination must have the same consecutive lag grid starting at zero and extending to at least 400.
- Store the ordinary unrounded float64 value, including negative values.
- There are 24 replicate curves: four widths times three replicates times two architectures.

The renderer recomputes only the pointwise median and min/max across the three replicate curves. It does not estimate autocorrelation or read chains. If only a final median curve is available, retrieve the replicate curves; do not invent the band.

## 6.4 Duplicate-legend correction

Create figure legends with explicit handles once. A legend produced by `fig.legend(...)` is already registered in `fig.legends`; do not also call `fig.add_artist(legend)` on that same object. The supplied script uses two figure legends for Figure 1 and two figure legends for Figure 2, and checks that their artist identities are unique. [R4]

The older advice about preserving an axes legend when replacing it with a second axes legend does not apply to adding multiple figure legends. The submitted PDFs show duplicate rendering consistent with this registration error; also check that the same legend-creation code is not being called twice.

## 6.5 No data reconstruction from screenshots

The attached plotting code has been exercised on clearly labelled synthetic layout fixtures. Those fixtures are not campaign results and are not included as paper figures. Render the final publication figures only after building the two CSVs from audited unrounded outputs. Do not use the report's rounded values to recreate error bars or continuous ACF curves.

# 7. Audit of theorem relevance across all results

Every quantity is classified below. An empirical design necessarily chooses widths, percentiles, observation functions, and computation budgets. Those choices must be explicit and justified; they must not be presented as theorem constants.

| Quantity or choice | Origin and purpose | Required interpretation |
|---|---|---|
| $T$, $G_a$, $r_0+2\sigma$ | Definition and condition in the deep local result | Directly tied to the domain in the theorem |
| $r_0=0$, $\sigma=1$ | Existing prior configuration | Verify from centers and metadata, not fit from samples |
| Reference line 2 | Computed admissibility threshold | Strict condition is $a>2$; values above 2 are not counterexamples |
| Former 2.5 | Previously selected admissible margin | Retained only to invert old normalization and in the historical audit |
| 50th, 95th, 99th percentiles | Descriptive summaries chosen for readability | Not theorem confidence levels or bounds; full CDF removes dependence on one percentile |
| Widths and fixed $n=128$ | Computational experimental design | Illustrate a width trend; do not verify joint sample-size scaling |
| Loss ACF and held-out probability ACF | Practical finite-kernel observables | Compatible sampling evidence; not direct PI/LSI validation |
| $h=0.01$, $h=0.005$, $kh$ | Existing kernel settings and accumulated algorithmic lag | Algorithm controls, not theorem constants or wall-clock time |
| ACF reference at zero | Definition of zero linear correlation | No Gaussian rate or functional-inequality coefficient is inferred |
| Lag display and chain budgets | Rendering/computation choices | Preserve full source arrays and label displayed range |
| $\widehat R$/ESS thresholds | Numerical adequacy rules | Neither mathematical hypotheses nor empirical estimates of theorem constants |
| Prior/posterior predictive score | Evidence that data affect predictions | Not a theorem about improved accuracy with width |

The slide's mass envelope contains setup-dependent constants in an expression of the form $(L-2)\exp(c_0n-c_1m)$. Do not draw a numerical envelope with guessed constants, fit those constants to the same coverage curve and call it verification, or assume uniform control as $a\downarrow2$. The present sources do not supply audited numerical values of those constants.

Retain the main claim boundaries: high-coverage domains and finite-lag width effects are measurable; optimal PI/LSI constants, exact asymptotic rates, uniform exploration of all posterior directions, and the joint $m,n$ threshold are not established by these figures. The reader should recognize the theorem's objects in Figure 1 and the practical motivation for Figure 2 without confusing the two kinds of evidence.

# 8. Validation and paper integration

Validate numerical preservation before reviewing appearance:

1. Match all target IDs, replicate IDs, widths, and counts to the accepted export. Check the source and adapter hashes.
2. Verify each recovered $T$ equals the directly computed norm divided by $\sqrt m$, and equals 2.5 times an old $S$ value when that historical normalization was used. Preserve source states and counts; recompute membership at every displayed cutoff instead of reusing old membership at 2.5.
3. Verify quantile ordering and agreement with the accepted unrounded target summaries. Check each empirical CDF is nondecreasing and lies in [0,1]; compare searchsorted counts to direct boolean counts at every test threshold.
4. Verify the Figure 2 input curves match the accepted replicate curves at every exported lag. The visible crop does not authorize recalculation.
5. Check that each ACF equals 1 at lag zero and that replicate ranges bracket their median. No NaN, missing width, or missing replicate is allowed.

Then inspect the output at **the manuscript's actual printed size**:

- Every title, tick, axis label, and legend is readable without zooming. Neither Figure 1 panel calls the threshold at 2 an admissible domain boundary.
- The horizontal and vertical lines in Figure 1 both mark 2 and are identified as the admissibility threshold. The two panels have intentionally different axis meanings.
- Each figure contains exactly two axes and the intended number of legends.
- PDF text extraction does not show duplicate legend labels at the same bounding boxes.
- Quantile markers and ACF marker shapes remain distinguishable in grayscale.
- ACF shading remains subordinate to the lines. No band is advertised as a confidence interval.
- No labels, whisker caps, or markers are clipped. The zero line does not cover nearby colored curves.
- The caption defines every encoding and reports the correct analysis window and count.

Render the final PDF at 150 dpi for a page-size inspection and at 300 dpi for a clipping check. Inspect the compiled paper as well as standalone PDFs. An export that looks good at 200% zoom can still fail at its final manuscript width.

Final deliverables are two main PDF figures, PNG previews, source CSVs, JSON provenance/style sidecars, per-target counts above 2, per-replicate quantiles/CDFs, the compact prior-reference table, plotting code, and captions. The loss-analysis diagnostics are unchanged. Figure 1 requires additional statewise postprocessing, not new sampling. Missing arrays are retrieved or reconstructed from archived matrices; quantile-only fallbacks are labelled incomplete.

# References

[R1] Ruqi Zhang et al. _Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning_. ICLR 2020. Figure 3, p. 7. <https://arxiv.org/pdf/1902.03932>

[R2] Mrinank Sharma, Sebastian Farquhar, Eric Nalisnick, and Tom Rainforth. _Do Bayesian Neural Networks Need To Be Fully Stochastic?_ AISTATS 2023, PMLR 206. Figure 2, p. 5. <https://proceedings.mlr.press/v206/sharma23a.html>

[R3] Pavel Izmailov, Sharad Vikram, Matthew D. Hoffman, and Andrew Gordon Wilson. _What Are Bayesian Neural Network Posteriors Really Like?_ ICML 2021, PMLR 139. Figures 2 and 3, p. 5. <https://proceedings.mlr.press/v139/izmailov21a.html>

[R4] Matplotlib official documentation, `Figure.legend`. Consulted for explicit-handle and figure-legend behavior; use the API supported by the recorded project version. <https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.legend.html>

[T] Supplied theory slides, `_AISTATS_27_Wide_BNN_Posterior_Geometry (15).pdf`, slides 3, 5, and 10; existing campaign configuration and prior-center definitions. These establish the model, spectral domain, and strict condition used in this audit.
