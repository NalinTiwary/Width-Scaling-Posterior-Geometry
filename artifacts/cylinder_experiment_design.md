# A minimal experimental study of the high-mass cylinder

**Design document · 24 September 2026**

**Status:** a proposed protocol, not a report of completed posterior experiments. Values labeled “analytic” are evaluations of the manuscript’s formulas. Expected outcomes are hypotheses.

## 1. Recommendation and paper structure

Run one fixed-data, two-layer width sweep: **three widths and three prior-center seeds**, with four independent chains per target. Reuse those chains for every scientific figure.

The two main figures should establish the following joint statement:

> As width increases, the theorem’s cylinder continues to contain substantial posterior probability, while the negative likelihood curvature that competes with Gaussian prior curvature becomes smaller.

Neither half is sufficient alone. Excellent geometry on a negligible-mass set would be uninformative; high mass without improving geometry would leave the proposed width mechanism untested.

| Priority and placement | Figure | Question answered | Additional posterior targets |
|---|---|---|---|
| Essential, main text | **1. Cylinder coverage versus width** | Does the unrestricted posterior occupy the theorem’s domain as parameter dimension grows? | The shared nine targets |
| Essential, main text | **2. Negative curvature versus width** | Does the likelihood’s adverse curvature diminish relative to prior precision? | None; reuse Figure 1 states |
| Essential appendix figure | **S1. Cylinder cutoff calibration** | Is near-unit coverage hiding a conservative boundary? How much of the allowed head radius do draws use? | None |
| Useful appendix figure; first to cut | **S2. Sampling efficiency versus width** | Does geometric improvement coincide with greater efficiency for the chosen sampler? | None |
| Essential appendix table | **Sampling and numerical validation** | Are estimates credible and curvature computations correct? | One central-target sampler cross-check |

This is **two main plots and at most two appendix plots**. There is no separate dataset, depth, prior-scale, temperature, or sample-size sweep. The central-target cross-check changes the sampling algorithm, not the scientific target.

The main improvement to measure is geometric. Improved wall-clock sampling is a stronger, algorithm-dependent claim. It is useful to examine, but the paper should not depend on obtaining it.

## 2. What the current theory predicts

### 2.1 Authoritative model and domain

Use the cylinder model in Sections 3–4 of the uploaded 23-page manuscript, *AISTATS 27 Wide BNN Posterior Geometry (3)*, and Appendix A. The accompanying 37-slide document provides context; the manuscript supplies the operative constants.

Here \(L=2\) means **one hidden layer and two linear maps**. With \(k=C-1\),

\[
f_\theta(x)=\frac{1}{\sqrt m}\sum_{j=1}^m a_j\tanh(w_j^\top x),
\qquad \theta=(a_1,w_1,\ldots,a_m,w_m).
\]

The logits are \(Qf_\theta(x)\), where \(Q:\mathbb R^k\to\mathbf1_C^\perp\) is an isometry. The posterior is

\[
\pi(d\theta)\propto e^{-V(\theta)}
N(\theta_0,\sigma^2I_p)(d\theta),\qquad
V(\theta)=\sum_{i=1}^n\ell_i(f_\theta(x_i)).
\]

All parameters are sampled. The likelihood is the **sum** of cross-entropies. There is no \(1/\sqrt d\) input normalization in this cylinder model.

The relevant two-layer cylinder is

\[
G_B=\left\{\theta:\max_{1\le j\le m}\|a_j\|_2\le B\right\}.
\]

These head balls are centered at zero. Hidden weights are unrestricted. Do not replace this with a parameter ball, a bound on \(\|a_j-a_{j,0}\|\), or the deep network’s spectral cylinder.

### 2.2 Mass cutoff to implement literally

Under the paired-center assumptions, set

\[
b_0=\max_j\|a_{j,0}\|_2,\qquad
\mathcal A_n=n\left(\log C+2\sqrt{b_0^2+(C-1)\sigma^2}\right).
\]

Lemma 4.1 gives

\[
B_\delta=b_0+\sigma\left[
\sqrt{C-1}+\sqrt{2\{\mathcal A_n+\log(m/\delta)\}}
\right],\qquad \pi(G_{B_\delta}^c)\le\delta.
\]

Fix \(s=1\), set \(\delta=m^{-1}\), and use

\[
\boxed{B_m=b_0+\sigma\left[
\sqrt{C-1}+\sqrt{2\{\mathcal A_n+2\log m\}}
\right].}
\]

Compute this cutoff before examining posterior draws. Estimating a posterior quantile and calling that the theorem’s cylinder would make the coverage experiment circular.

### 2.3 Width improvement and its observable

Let \(X\) have rows \(x_i^\top\), \(M_2=\|X\|_{2\to2}\), \(M_4=\|X\|_{2\to4}\), and \(c_2=4/(3\sqrt3)\). Theorem 2 gives, throughout \(G_{B_m}\),

\[
\nabla^2U(\theta)\succeq\kappa_m I,\qquad
\kappa_m=\sigma^{-2}
-\sqrt{\frac{2n}{m}}(M_2+c_2B_mM_4^2).
\]

Define the dimensionless theoretical deficit

\[
\boxed{D_{\rm th}(m)=
\sigma^2\sqrt{\frac{2n}{m}}(M_2+c_2B_mM_4^2).}
\]

Then \(\sigma^2\kappa_m=1-D_{\rm th}(m)\). If \(D_{\rm th}<1\), the conditional PI and LSI coefficients satisfy

\[
C_P(\pi_{G_{B_m}}),\,C_{\rm LS}(\pi_{G_{B_m}})
\le\frac{\sigma^2}{1-D_{\rm th}(m)}.
\]

The natural statewise measurement is the quantity proposed in the manuscript’s placeholder Figure 2:

\[
\boxed{d_H(\theta)=
\sigma^2[-\lambda_{\min}(\nabla^2V(\theta))]_+.}
\]

It measures negative likelihood curvature relative to the Gaussian prior’s precision. On the cylinder,

\[
0\le d_H(\theta)\le D_{\rm th}(m),\qquad
\nabla^2U(\theta)\succeq\sigma^{-2}(1-d_H(\theta))I.
\]

For fixed data and prior scale, \(D_{\rm th}=O(\sqrt{\log m/m})\). This is an **upper envelope**, not a prediction that a posterior quantile must have exactly that exponent or lie close to the bound. Faster decay or a much smaller observed deficit is compatible with the theorem.

Use the exact finite-width \(D_{\rm th}(m)\) as the primary reference curve. Over the proposed widths, \(\mathcal A_n\) dominates \(2\log m\), so an arbitrarily normalized \(\sqrt{\log m/m}\) curve would be less faithful.

### 2.4 Scope

The core study tests the two-layer result: its domain, mass cutoff, and uniform curvature mechanism are explicit and accessible.

The deep result uses additional spectral restrictions and different center assumptions. The manuscript explains why the same uniform pointwise curvature argument can fail at depth three. A falling two-layer \(d_H\) curve therefore does not test the deep theorem’s curvature-moment argument.

This protocol also does not estimate the optimal PI/LSI constants, entropy decay, or the global \(m^{-1/3}\) correction. It tests the high-mass domain and a central mechanism behind the conditional result.

## 3. Exact shared setup

### 3.1 Frozen configuration

| Quantity | Choice | Reason |
|---|---|---|
| Architecture | One hidden layer, tanh, all weights random | Exactly the two-layer cylinder theorem |
| Widths | \(256,1024,4096\) | Factor-16 range with three targets per seed |
| Data size and input dimension | \(n=d=32\) | Small full-batch likelihood and controlled data norms |
| Classes | \(C=2,\ k=1\) | Nontrivial classification with a scalar head constraint |
| Prior standard deviation | \(\sigma=0.5\) | Gives informative finite-width bounds at affordable widths |
| Head center magnitude | \(b_0=1\) | Nonzero paired centers satisfy the assumptions |
| Missing-mass exponent | \(s=1\) | One concrete theorem cutoff |
| Prior-center seeds | \(0,1,2\) | Replication without a large grid |
| Chains per target | 4 | Between-chain diagnostics and independent starts |
| Initial retained draws | 4,000 per chain after 2,000 burn-in updates | Initial allocation with a capped extension rule |
| Curvature states | 128 per target, 32 per chain | Expensive postprocessing is restricted to a prespecified subset |

There are **nine posterior targets and 36 primary chains**. The seeds vary the prior center, not the dataset. Do not present them as independent datasets.

Three widths permit a trend assessment, not a reliable asymptotic exponent estimate. Replication is more valuable here than densely interpolating widths with one chain.

### 3.2 Data generation

Use a named random-number generator, such as NumPy PCG64, and record its version.

1. With data seed 2027, draw a \(32\times32\) matrix \(G\) of independent standard normals.
2. Compute \(G=QR\), fixing column signs so the diagonal of \(R\) is positive. Set \(X=Q^\top\).
3. Draw a teacher vector \(v_*\sim N(0,I_{32})\) from the same generator.
4. Assign \(y_i=1\) to the 16 largest \(x_i^\top v_*\), and \(y_i=0\) to the rest. Save arrays and hashes.
5. Generate eight additional unit-norm Gaussian inputs with seed 2028. These are fixed predictive probes, not an accuracy benchmark.

Then \(\|x_i\|_2=1\), \(M_2=M_4=1\). Orthogonality gives \(\|Xu\|_4\le\|Xu\|_2=\|u\|_2\), with equality when \(Xu\) is a coordinate vector.

**Justification:** the finite-width theorem curve is exactly computable and conditioning stays fixed. Otherwise estimating a \(2\to4\) norm, or changing conditioning alongside \(n\), introduces another problem.

**Limitation:** this is a favorable controlled geometry experiment. It does not establish robustness to arbitrary datasets or predictive superiority. Orthogonality is an experimental convenience, not a theorem assumption. A real-data benchmark is outside the first pass.

### 3.3 Paired prior centers without tying sampled parameters

For each center seed, generate 2,048 independent hidden-row centers \(u_j\sim N(0,I_{32})\). Width \(m\) uses the first \(m/2\) pairs:

\[
(a_{j,0},w_{j,0})=(+1,u_j),\qquad
(a_{j+m/2,0},w_{j+m/2,0})=(-1,u_j).
\]

This gives \(f_{\theta_0}(x)=0\) for every \(x\) and satisfies Assumptions 3.2–3.3. Assemble each width from complete pairs; truncating a flattened parameter array can destroy pairing.

Conditional on the center, draw **every coordinate independently** from its Gaussian prior. Pairing centers does not tie neurons or posterior perturbations.

Nested pairs reduce arbitrary between-width changes in centers. They do not make the finite-width posteriors identical or produce paired posterior draws.

No training is required. The cylinder theorem does not require a trained center or tangent-kernel nondegeneracy; adding optimization would introduce cost and a confound.

These centers have \(\|A_0\|_{\rm op}=\sqrt m\). Use the paired-center branch, not the alternative width-uniform head-operator-norm assumption.

### 3.4 Binary likelihood and coordinate conventions

Take \(Q=(-1,1)^\top/\sqrt2\). For \(y_i\in\{0,1\}\),

\[
p_i=\operatorname{sigmoid}(\sqrt2 f_i),\qquad
V(\theta)=\sum_i[
\operatorname{softplus}(\sqrt2 f_i)-\sqrt2 y_if_i].
\]

Use stable softplus and summed loss. An accidental average changes the posterior and curvature scale.

Sampling can use \(z=(\theta-\theta_0)/\sigma\), but reported curvature is with respect to original Euclidean \(\theta\). Since \(\nabla_z^2V=\sigma^2\nabla_\theta^2V\), applying the \(\sigma^2\) normalization twice is an error.

### 3.5 Analytic preflight: expected scales

Here \(\mathcal A_n=93.7348851\). These are **theorem calculations, not measurements**:

| \(m\) | \(p=33m\) | \(B_m\) | Coverage floor \(1-1/m\) | \(D_{\rm th}\) | Normalized curvature floor \(1-D_{\rm th}\) |
|---:|---:|---:|---:|---:|---:|
| 256 | 8,448 | 8.7397 | 0.996094 | 0.96597 | 0.03403 |
| 1,024 | 33,792 | 8.8348 | 0.999023 | 0.48756 | 0.51244 |
| 4,096 | 135,168 | 8.9287 | 0.999756 | 0.24604 | 0.75396 |

The adverse-curvature envelope roughly halves at each width step. The normalized conditional gap lower bound improves from 0.034 to 0.754. This is the concrete theoretical scale the measurements should accompany.

Choosing \(\sigma=0.5\) uses these formulas before sampling. It does not establish the same behavior for larger prior scales; make the choice visible.

## 4. Shared sampling protocol

### 4.1 Unrestricted elliptical slice sampling

Use full-parameter elliptical slice sampling (Murray, Adams and MacKay, 2010 [R1]). It fits the Gaussian-prior-times-likelihood target, needs no step-size sweep, and preserves the posterior in exact arithmetic.

In standardized coordinates:

1. Draw \(\nu\sim N(0,I_p)\) and \(u\sim{\rm Uniform}(0,1)\).
2. Set threshold \(h=-V(\theta_0+\sigma z)+\log u\).
3. Draw \(\alpha\sim{\rm Uniform}(0,2\pi)\), with bracket \([\alpha-2\pi,\alpha]\).
4. Propose \(z'=z\cos\alpha+\nu\sin\alpha\).
5. Accept if \(-V(\theta_0+\sigma z')\ge h\).
6. Otherwise shrink the bracket around zero according to the sign of \(\alpha\), redraw uniformly inside it, and repeat.

Use the original algorithm’s full-angle bracket. Initialize each chain independently from the prior. Record every completed update and every likelihood evaluation, including unsuccessful angle proposals.

**Never reject proposals for being outside the cylinder.** That would sample a conditional target and make coverage equal to one by construction. Membership is postprocessing.

This algorithm generates estimates; it is not the theorem’s Langevin dynamics. Iterations are not physical Langevin time. Exact invariance does not make finite-chain draws independent or exactly stationary.

### 4.2 Initial budget and bounded extension

Start with 2,000 burn-in and 4,000 retained updates per chain. Each target initially has 16,000 retained states for cheap statistics.

Assess these prespecified operational targets:

- Rank-normalized split and folded \(\widehat R<1.01\) for the continuous observables below.
- Bulk ESS at least 1,000 and tail ESS at least 400, using the four chains with chain-aware diagnostics.
- For the head maximum, localized quantile ESS at 0.95 and 0.99 of at least 400 where estimable.
- Stability between the first and second halves of retained draws, assessed with Monte Carlo errors.

These are practical diagnostic targets, not sufficient mathematical conditions for convergence. The diagnostic methods follow Vehtari et al. [R2].

If needed, extend retained draws to 8,000 per chain, then at most 16,000. Stop earlier if diagnostics are satisfactory. Also cap work at **one million likelihood evaluations per target**, including burn-in. Mark a target unresolved if the cap is reached first.

Do not repeatedly change datasets, priors, seeds, or widths to obtain the desired trend. Persistent problems call for revisiting the estimator explicitly.

### 4.3 Record these observables at every retained update

| Observable | Purpose |
|---|---|
| \(H=\max_j|a_j|\), \(H/B_m\) | Domain margin and tail mixing |
| \(\mathbf1\{H\le B_m\}\) | Coverage |
| Summed loss \(V\) | Target exploration |
| All 32 training outputs \(f_i\) | Function-space mixing |
| Probabilities on eight fixed probe inputs | Interpretable predictive observables |
| \(\|z\|^2/p\), mean squared standardized head coordinate | Radial and head exploration |
| Eight fixed unit-vector projections of \(z\) | Weight-space checks beyond predictions |
| Likelihood counts and compute time | Cost |

Generate projection vectors with seed 2029 for each dimension, save them before sampling, and do not select them based on ESS.

Use float64 for the reference implementation and curvature calculations. Record software, device, thread count, and precision. Synchronize GPU work before timing.

### 4.4 One independent-kernel cross-check

At \(m=1024\), center seed 0 only, run four independent pCN chains:

\[
z'=\sqrt{1-\beta^2}\,z+\beta\xi,\quad \xi\sim N(0,I_p),
\quad
\alpha_{\rm MH}=\min\{1,\exp[V(\theta)-V(\theta')]\}.
\]

This Gaussian-preserving proposal and likelihood-only acceptance ratio follow Cotter et al. [R3]. Start \(\beta=0.2\). During 2,000 burn-in steps, update

\[
\log\beta_{t+1}=\log\beta_t+
0.05(t+10)^{-1/2}(I_{\rm accept}-0.3),
\]

clipping \(\beta\) to \([10^{-3},0.999]\); freeze afterward. Use the same retained-length and likelihood-budget caps. Retain repeated states after rejections.

Compare \(E[V]\), \(E[H]\), \(q_{0.95}(H)\), and the eight predictive means with elliptical slice estimates. Report differences relative to combined Monte Carlo errors where estimable. Persistent discrepancies above three combined standard errors are diagnostic flags, not a multiple-testing guarantee.

Do not pool discrepant estimates. If pCN remains poorly mixed at its cap, label the cross-check unresolved; this alone does not establish that the primary estimate is wrong.

An independent long “gold” chain is unnecessary because no convergence-to-reference curve is claimed. This cross-check catches target and implementation errors.

## 5. Experiment 1 — Does the cylinder retain posterior mass?

### 5.1 Estimand and calculation

For each width and center seed,

\[
p_m=\pi(G_{B_m}),\qquad
\widehat p_m=\frac1{4T}\sum_{c=1}^4\sum_{t=1}^T
\mathbf1\{H(\theta_{c,t})\le B_m\}.
\]

Use all retained states, including those outside. Save chainwise coverage, exit counts, separate exit episodes, \(H\) quantiles, and maximum observed \(H\).

The fraction estimates probability. Raw retained-sample counts reflect computational allocation and must not be the vertical axis.

### 5.2 Uncertainty and the likely all-inside case

If repeated exits and returns permit dependence estimation, use chainwise batch means. For batch length \(b=\lfloor\sqrt T\rfloor\), \(a=\lfloor T/b\rfloor\), \(T'=ab\), and batch averages \(\bar I_{c,j}\), set

\[
\widehat v_c=\frac{b}{a-1}\sum_{j=1}^a
(\bar I_{c,j}-\bar I_c)^2,\qquad
\widehat{\operatorname{Var}}(\widehat p_m)
\approx\frac1{16}\sum_{c=1}^4\frac{\widehat v_c}{T}.
\]

Here \(\bar I_c\) is the average over complete batches; the reported coverage uses all draws. Adequate mixing and asymptotic conditions are needed for batch-means errors [R4].

If all draws are inside, report **“no observed exits among \(4T\) retained correlated draws.”** Zero indicator variance is not zero uncertainty. Do not insert an ESS into an independent-binomial interval or assert an exit bound of \(3/N_{\rm eff}\). A constant indicator does not provide an estimable exit-event ESS.

With only a few exit episodes, report counts and chain breakdown rather than a spurious narrow interval. Continuous \(H\) diagnostics cannot resolve an unobserved extreme tail.

Even independent draws require roughly \(3/\delta\) zero-exit observations for a 95% upper bound near \(\delta\). At \(m=4096\), \(\delta=1/m\) means about 12,288 independent observations. This study is not designed to verify the rare-exit exponent.

### 5.3 Exact plot

**Figure 1: one axis.**

- X: width, logarithmic; ticks 256, 1,024, 4,096.
- Y: unrestricted posterior fraction inside \(G_{B_m}\).
- One thin line and markers per center seed.
- Dashed analytic floor \(1-1/m\), labeled “theorem lower bound.”
- Default Y range \([0.99,1.0005]\), with the truncated range explicit. Expand it if observations fall below 0.99.
- Draw chain-aware error bars only where estimable. Use open markers labeled “0 observed exits” for all-inside results.
- Report \(p=33m\) in the caption or secondary tick annotations.

The positive expected result is nearly flat high coverage as parameter dimension increases sixteenfold. Actual coverage need not be monotone; a visibly rising empirical curve is not required.

### 5.4 Inference and caption

High coverage with satisfactory diagnostics supports the domain’s relevance to the sampled posterior. It does not empirically prove the exact missing-mass bound.

A material, well-resolved deficit below the theorem floor requires checking the target, centers, domain, and sampler. Do not treat it as harmless variation or change parameters afterward.

**Draft caption:** “Posterior coverage of the theorem-defined cylinder as network width grows. Each curve represents one paired prior-center construction, using four unrestricted elliptical slice chains per target. The horizontal axis is width on a logarithmic scale; the vertical axis is the retained fraction satisfying \(\max_j|a_j|\le B_m\), with \(B_m\) fixed from Lemma 4.1 using \(s=1\). Hidden weights remain unrestricted. The dashed curve is the analytic lower bound \(1-1/m\). [Describe observed results here.] Open markers denote no observed exits and do not imply zero Monte Carlo uncertainty. High occupancy supports the domain’s relevance despite increasing parameter dimension; it does not resolve the rate of rare exits.”

## 6. Experiment 2 — What improves with width?

### 6.1 Primary estimand

For each target, estimate

\[
q_m=Q_{0.95}(d_H(\theta)\mid\theta\in G_{B_m}).
\]

An upper quantile emphasizes adverse states without claiming a cylinder-wide supremum. The 95th percentile matches the manuscript placeholder. With this state budget, treat it as descriptive, not a precise tail-rate estimate.

Save the median and maximum sampled deficit as diagnostics without adding figures.

After the final chain length is set, choose 32 evenly spaced indices per chain:

\[
t_j=\left\lfloor(j+\tfrac12)T/32\right\rfloor,\quad j=0,\ldots,31.
\]

Use only inside states for the conditional quantile. Do not replace outside states with selected favorable draws. If fewer than 100 of the 128 states are inside, report the reduced count and withhold a precise 95th-percentile conclusion.

### 6.2 Efficiently measuring the actual Hessian deficit

The Hessian decomposes as

\[
\nabla^2V=J^\top\operatorname{diag}(h_i)J+\mathcal R,
\quad q_i=\sqrt2(p_i-y_i),\quad h_i=2p_i(1-p_i)\ge0,
\]

where \(J\) is the Jacobian of the scalar outputs \(f_i\). The first term is positive semidefinite. The residual is block diagonal over neurons:

\[
\mathcal R=\operatorname{blockdiag}(R_1,\ldots,R_m),\qquad
R_j=\frac1{\sqrt m}
\begin{pmatrix}0&c_j^\top\\c_j&D_j\end{pmatrix},
\]

\[
c_j=X^\top(q\odot\phi'_j),\qquad
D_j=a_jX^\top\operatorname{diag}(q\odot\phi''_j)X,
\]

with \(\phi'_{ij}=1-\tanh^2(w_j^\top x_i)\) and
\(\phi''_{ij}=-2\tanh(w_j^\top x_i)\phi'_{ij}\). Each block is just \(33\times33\). Its Jacobian block is

\[
J_j=\frac1{\sqrt m}
[\phi_j,\ a_j\operatorname{diag}(\phi'_j)X].
\]

There is no need to build a \(135{,}168\times135{,}168\) Hessian.

**Default: bracket \(d_H\) instead of treating an unconverged eigenvalue approximation as exact.**

1. Diagonalize the small symmetric blocks, in chunks if needed.
2. Let \(\ell\) be their smallest eigenvalue.
3. Select the \(K=n+1=33\) lowest eigenvectors of \(\mathcal R\), forming an orthonormal embedded matrix \(E\). Store block indices and local vectors instead of allocating dense \(p\times K\) arrays.
4. Compute \(JE\) from the selected blocks.
5. Form
   \[
   H_E=\operatorname{diag}(\lambda_1(\mathcal R),\ldots,\lambda_K(\mathcal R))
   +(JE)^\top\operatorname{diag}(h_i)(JE),\quad
   u=\lambda_{\min}(H_E).
   \]
6. Positive semidefiniteness and the Rayleigh principle give
   \[
   \ell\le\lambda_{\min}(\nabla^2V)\le u.
   \]
7. Record
   \[
   \boxed{d_-=\sigma^2[-u]_+\le d_H\le
   d_+=\sigma^2[-\ell]_+.}
   \]

These algebraic bounds are evaluated numerically in float64; check eigensolver residuals and allow numerical tolerances. \(K=n+1\) exploits the rank-at-most-\(n\) positive update at little extra cost.

A simpler check uses a unit minimum-eigenvalue vector \(v\) from the worst block \(R_j\):

\[
\ell\le\lambda_{\min}(\nabla^2V)
\le\ell+\sum_i h_i(J_jv)_i^2.
\]

The projected estimate must be at least as tight when its subspace contains that direction.

**Why bracket rather than just measure \(\mathcal R\)?** The residual deficit \(d_+\) is a conservative diagnostic, not the actual likelihood-Hessian deficit. The interval accounts for the stabilizing positive term and preserves the current figure’s intended meaning.

The interval can be loose. Its width is a numerical-information limitation, not a confidence interval. If it cannot resolve a decrease, report that rather than relabeling the residual as the true deficit.

### 6.3 Quantiles and effect size

Compute both endpoint quantiles with the order statistic \(x_{(\lceil0.95N\rceil)}\), without interpolation. Then

\[
\widehat Q_{0.95}(d_-)\le
\widehat Q_{0.95}(d_H)\le
\widehat Q_{0.95}(d_+).
\]

This brackets the quantile of evaluated states, not the population quantile. Report each seed separately. Preserve chainwise summaries; do not bootstrap correlated states as independent observations. Three center seeds do not support a precise population confidence interval.

The primary effect size is the width-256 to width-4096 reduction. When denominator endpoints are positive,

\[
\frac{q_{-,4096}}{q_{+,256}}\le
\frac{q_{H,4096}}{q_{H,256}}\le
\frac{q_{+,4096}}{q_{-,256}}.
\]

An upper endpoint below one resolves a decrease in the evaluated-state quantiles. A zero lower denominator leaves the upper ratio unresolved.

Every inside-state \(d_+\) should satisfy \(d_+\le D_{\rm th}(m)\), up to numerical tolerance. A persistent violation indicates an implementation or theorem-alignment problem.

### 6.4 Exact plot

**Figure 2: one axis, logarithmic in width and positive deficit.**

- X: width, with the same three ticks.
- Y: normalized negative likelihood curvature.
- For each seed and width, a vertical interval from the empirical 95th percentile of \(d_-\) to that of \(d_+\). Slight horizontal offsets can separate seeds.
- Connect upper endpoints within a seed with thin lines, explicitly labeled as upper endpoints.
- Overlay the exact \(D_{\rm th}(m)\) as a thick dashed “uniform theorem envelope.”
- Mark \(d=1\), where adverse likelihood curvature can exhaust prior precision.
- If the lower endpoint is zero, use a downward arrow labeled “includes zero.” If both are zero, use an off-axis zero marker; never invent a positive value for logarithmic plotting.

The favorable pattern is falling empirical intervals below the falling envelope. The envelope decreases about fourfold over the range; the empirical curve can decrease faster.

Do not fit a formal scaling exponent to three widths. A descriptive endpoint slope, if recorded, is exploratory.

### 6.5 Inference and caption

The measured improvement is that nonlinear likelihood curvature consumes less of the prior’s positive curvature. Combined with Figure 1, this occurs on a domain with meaningful posterior occupancy.

It does not establish Gaussianity in every direction, equality of optimal PI/LSI constants with theorem bounds, or faster sampling by every algorithm.

**Draft caption:** “Negative likelihood curvature versus width on the high-mass cylinder. For each paired prior-center seed, vertical intervals bracket the empirical 95th percentile of \(d_H=\sigma^2[-\lambda_{\min}(\nabla^2V)]_+\) over 128 prespecified posterior states, restricted to the cylinder. Endpoints use the block-diagonal residual Hessian and a Rayleigh–Ritz projection of the full Hessian; they are numerical bounds, not confidence intervals. Axes are logarithmic for positive values. The dashed curve is the finite-width uniform envelope from Theorem 2. [Describe observed results here.] A decrease indicates less adverse likelihood curvature relative to Gaussian prior precision. Together with coverage, this tests the proposed mechanism on a region containing substantial posterior probability.”

## 7. Appendix S1 — Is the cutoff excessively conservative?

This is the most informative supporting plot because Figure 1 may show no exits anywhere.

### 7.1 Estimand and exact plot

Using all retained states, define

\[
Z_m=H/B_m,\qquad
\widehat F_m(z)=\frac1{4T}\sum_{c,t}
\mathbf1\{H(\theta_{c,t})/B_m\le z\}.
\]

Plot its empirical CDF:

- X: \(z=H/B_m\), from zero to at least one.
- Y: empirical cumulative probability.
- Three width colors, with a line per seed in that color.
- Vertical boundary at \(z=1\).
- Mark the 0.95 and 0.99 quantiles or report them in the accompanying table.

This exposes the continuous margin to the boundary. If the CDF reaches nearly one well before \(z=1\), the theorem cylinder has considerable slack. That is useful information, not evidence that the cutoff is tight.

### 7.2 Exact prior reference

For the binary paired centers, the \(|a_j|\) are identically distributed and independent under the prior. For \(b\ge0\),

\[
P_\gamma(H\le b)=
\left[
\Phi\!\left(\frac{b-1}{\sigma}\right)
-\Phi\!\left(\frac{-b-1}{\sigma}\right)
\right]^m.
\]

Evaluate in log space. Overlay the exact prior CDF of \(H/B_m\) as dashed curves, clearly labeled prior.

The **analytic prior**, not posterior, 99th percentiles are:

| Width | \(Q_{0.99,\gamma}(H)\) | \(Q_{0.99,\gamma}(H)/B_m\) |
|---:|---:|---:|
| 256 | 2.9744 | 0.3403 |
| 1,024 | 3.1345 | 0.3548 |
| 4,096 | 3.2844 | 0.3678 |

These values make all-inside results plausible without assuming the posterior equals the prior. The theorem transfers a Gaussian tail using a conservative evidence bound; close empirical agreement with the cutoff is not expected.

A posterior head distribution near its prior does not by itself mean the likelihood was ignored. Individual parameters can be weakly affected while predictions change. For context, report prior-versus-posterior mean loss and the fixed predictive means in the validation table.

Use 2,000 independent prior draws per target for those comparison summaries. These need forward passes only and are not additional posterior chains.

**Draft caption:** “How much of the permitted head radius is used by posterior draws? The horizontal axis is the maximum absolute head coefficient divided by the theorem cutoff; the vertical axis is its empirical CDF under unrestricted posterior sampling. The vertical line at one is the cylinder boundary. Solid curves show posterior draws; dashed curves show the exact Gaussian-prior maximum distribution. Concentration to the left of one explains near-unit coverage and quantifies cutoff conservatism. The prior reference is not substituted for posterior mass.”

## 8. Appendix S2 — Does the chosen sampler become more efficient?

This helps connect the geometry to ML practice and needs no new chains. It is the first figure to omit if space or analysis bandwidth is tight.

### 8.1 Quantity and cost normalization

For each of the eight fixed predictive probabilities, calculate chain-aware bulk ESS. Let \(N_\ell\) be likelihood evaluations during retained sampling, summed across the same four chains, and \(t_{\rm ret}\) the corresponding compute time:

\[
E_{\rm eval}=\operatorname{median}_{j=1,\ldots,8}
\frac{\operatorname{ESS}(p_j)}{N_\ell},\qquad
E_{\rm sec}=\operatorname{median}_{j=1,\ldots,8}
\frac{\operatorname{ESS}(p_j)}{t_{\rm ret}}.
\]

Produce one figure with two panels:

1. Median predictive ESS per 1,000 likelihood evaluations versus width.
2. Median predictive ESS per second versus width.

Show individual center seeds. Report the minimum predictive ESS in the validation table so the median cannot hide a poorly explored probe. Count unsuccessful slice-angle proposals. Report burn-in costs separately.

Use consistent hardware and execution schedules. If chains run concurrently, distinguish end-to-end elapsed time from summed device time and identify the denominator used. Avoid comparing a sequential small-width run with a parallel large-width run as though the difference came from geometry.

### 8.2 Why this is supplementary

Hron et al. [R5] connect width, a data-dependent reparameterization, and sampling improvement. Pezzetti et al. [R6] study Gaussian-reference samplers for a reparameterized wide-network posterior. They motivate function-space and cost-normalized diagnostics, but their reparameterization-specific guarantees do not predict the same speedup for this original-coordinate target.

A plausible result is stable or improving ESS per likelihood evaluation but worse ESS per second, since each evaluation costs more with width. That remains informative.

**Draft caption:** “Practical sampling efficiency of elliptical slice sampling across widths. Left: median bulk ESS across eight fixed predictive probabilities per 1,000 likelihood evaluations. Right: the same ESS per second on fixed hardware. Each point represents one prior-center seed with four chains. Unsuccessful slice-angle proposals count toward cost. These measurements concern this algorithm, implementation, and observable set; they do not estimate the PI/LSI constants or follow automatically from the continuous-time theorem.”

## 9. Required validation without an ablation campaign

### 9.1 Numerical and target checks

These are implementation checks, not additional scientific figures.

1. **Paired center:** check \(f_{\theta_0}=0\) on training and probe inputs and \(b_0=1\). Verify independent perturbations across the paired neurons.
2. **Likelihood:** compare the implementation with the explicit binary formula. Check summed loss and the \(\sqrt2\) output-coordinate factor.
3. **Prior-only sampler:** set \(V=0\), then compare Gaussian marginal moments and the head maximum with the exact prior distribution.
4. **Hessian decomposition:** on a tiny network, compare a dense automatic-differentiation Hessian with \(J^\top\operatorname{diag}(h)J+\mathcal R\). Verify its minimum eigenvalue lies inside \([\ell,u]\). Repeat in standardized coordinates to catch normalization errors.
5. **Small-block eigensolvers:** verify symmetry and eigenpair residuals in float64. Target relative residual \(10^{-8}\); do not plot reversed intervals beyond numerical tolerance.
6. **Theorem alignment:** recompute \(B_m,D_{\rm th},p\) from configuration and compare with the analytic preflight table.
7. **MCMC and cross-kernel checks:** use the already specified diagnostics and report their outcomes.

During preparation of this design, the block formulas were checked on a tiny generic binary network using central differences of the analytic gradient. The largest Hessian-entry discrepancy was approximately \(2.5\times10^{-10}\), and the exact small-system minimum eigenvalue was inside the proposed bracket. This checks the displayed algebra, not the future sampling implementation or any posterior claim.

### 9.2 Required appendix table

Each target row should contain:

- Width, parameter count, prior scale, cutoff, theoretical deficit, center seed, data hash.
- Chain count, burn-in, retained length, likelihood evaluations, hardware and timing.
- Maximum reported \(\widehat R\), minimum bulk/tail ESS, head-quantile diagnostic status.
- Coverage, exit count, exit episodes, and whether coverage MCSE is estimable.
- The 95th and 99th percentiles of \(H/B_m\).
- Curvature-state count, quantile bracket endpoints, typical bracket width, maximum eigenpair residual.
- Mean posterior loss and prior reference loss, with Monte Carlo errors.
- Any unresolved diagnostic or sampler cross-check.

The table is essential even if S2 is omitted. It should expose limitations instead of forcing every target into a “passed” label.

## 10. What to reuse from sin-geoopt

The supplied repository’s experiment instructions and audit connect theorem quantities to occupancy, statewise geometry, and transparent plotting. Borrow that discipline.

| Repository component or pattern | Transfer |
|---|---|
| PLOTS.md and explicit run specifications | One frozen configuration and short execution sequence |
| checkpoint_validation.py and metadata audits | Validate target identity, arrays, normalization, seeds, and resumed lengths |
| Statewise minimum-curvature diagnostics | Use the two-layer block formula and actual-Hessian brackets |
| Separation of analytic probability and occupancy | Keep theorem floor, prior CDF, and posterior estimates distinct |
| Chain-aware ESS | Preserve chain boundaries and observable identities |
| CSV/JSON plot sidecars | Save the exact numbers and provenance for each curve |
| Explicit claim boundaries | Avoid treating sampled occupancy, ESS, or curvature as a proof |

Do not transfer its spherical Langevin transition, vMF priors, or averaged-loss target. The present model is Euclidean with a Gaussian prior and summed cross-entropy.

The repository’s step-halving and physical-time comparisons address discretized Langevin sampling. Elliptical slice and Metropolis-corrected pCN avoid that Euler step-size bias; target correctness, exploration, and kernel agreement are the relevant checks here.

Its short one-dimensional sweeps are useful exploratory patterns, not substitutes for this replicated multi-chain study. Test outcomes reported in its audit apply to its own code, not a future adaptation.

## 11. Execution order, budget, and artifact contract

### 11.1 Exact order

1. Implement model, target, cutoffs, sampler, and block curvature. Complete the small numerical checks.
2. Save data, paired centers, configuration, and hashes. Compute the analytic preflight table.
3. Run the central target \(m=1024\), seed 0. Profile cost and diagnose exploration. Run its pCN cross-check.
4. Fix implementation problems before launching the grid. If mixing fails at the cap, revisit the estimator explicitly before spending on the other targets.
5. Run the other eight targets with the frozen protocol; extend only under the diagnostic rule.
6. Compute coverage and cutoff CDFs from all retained scalar records.
7. Evaluate curvature at the prespecified 128 states per target and calculate quantile brackets.
8. Produce Figures 1–2, S1, and the validation table. Produce S2 from existing records if useful.
9. Audit captions against measured quantities and unresolved limitations.

Do not add widths merely because the plot is noisy. First identify whether uncertainty comes from chains, center variation, or curvature brackets.

### 11.2 Concrete cost

The default primary allocation is

\[
9\times4\times(2000+4000)=216{,}000
\quad\text{elliptical slice updates}.
\]

Each update can require multiple likelihood evaluations. The cap and reported cost therefore use evaluations, not just updates. The central pCN validation adds four chains on one existing target.

There are at most \(9\times128=1152\) primary curvature-state evaluations, each using many small eigensystems rather than a dense network Hessian.

Forward cost is approximately \(O(nmd)\). Do not promise a wall-clock duration before profiling on the actual hardware.

If resources tighten, omit S2 first, then optional postprocessing. Do not drop diagnostics or use conditional draws to estimate coverage. Two center seeds may serve as an explicitly labeled pilot, but the proposed paper run uses three.

### 11.3 What the future implementation must output

These are specified future artifacts, not files already generated by this design.

| File | Contents |
|---|---|
| config.yaml | Frozen model, widths, data, seeds, prior, sampler, and stopping rules |
| data.npz | Training and probe arrays, teacher, hashes |
| centers_seed*.npz | Nested center-pair banks and checksums |
| runs/*/metadata.json | Target identity, versions, device, precision, seeds, counts, timings |
| runs/*/observables.npz | Observables with separate chain and draw axes |
| runs/*/curvature_states.npz | Prespecified states or exact replay information and indices |
| curvature.csv | Statewise membership, \(\ell,u,d_-,d_+\), numerical residuals |
| target_summary.csv | Fields in the required validation table |
| figure1_coverage.csv | Seedwise occupancy, exits, estimable errors, analytic floor |
| figure2_curvature.csv | Quantile endpoints, counts, theorem envelope |
| figureS1_cutoff.csv | Posterior ECDF, prior curves, quantiles |
| figureS2_efficiency.csv | Observable ESS and cost denominators, if included |
| figures/*.pdf and figures/*.png | Publication figures and previews |
| captions.md | Observed-result captions with interval meanings |

A resumed run must fail validation if the loss normalization, center array, dataset, or prior changed. Never append draws from different targets.

## 12. What to omit and why

### 12.1 No \(m\times n\) sweep in this first pass

An \(m\)-versus-\(n\) plot is scientifically reasonable but answers an additional joint-scaling question. The current width-asymptotic statements hold data fixed. Changing \(n\) changes the target, the evidence-bound term \(\mathcal A_n\), and possibly \(M_2,M_4\).

With controlled data norms and fixed prior and class count, the displayed bound has the rough structure

\[
D_{\rm th}\lesssim
\sqrt{\frac nm}\left[1+\sqrt{n+\log m}\right].
\]

Thus the sample count and cutoff growth both matter. A sufficient scale can involve \(n^2+n\log m\). Do not import the old ball-based \(r/\sqrt m\) collapse unchanged.

Establish coverage and geometric improvement at fixed data first. Add an \(m\times n\) experiment only if an explicit joint-scaling claim becomes central; it should replace a weaker plot rather than automatically expanding the plan.

### 12.2 No ball, training, NTK, or linearization benchmark

Those controls addressed the earlier local-ball theorem. They are unnecessary for the cylinder’s explicit mass cutoff and direct two-layer curvature mechanism, and would add substantial work.

### 12.3 No direct high-dimensional LSI or KL estimate

ESS and empirical curvature quantiles do not estimate entropy relative to an unknown posterior or the optimal LSI constant. Direct entropy estimation would need separate reference machinery. These experiments accompany the theorem rather than imitate a numerical proof.

### 12.4 No rare-event simulation solely to reproduce \(m^{-1}\)

Continuous cutoff calibration makes an all-inside result interpretable. Importance sampling or rare-event methods are justified only if sharp exit probabilities themselves become a central claim.

### 12.5 No deep panel without a different protocol

The deep cylinder constrains additional matrices and uses different assumptions and mechanisms. Adding a hidden layer does not preserve this experiment’s interpretation.

## 13. Decision rules for writing the results

| Observed outcome | Defensible interpretation |
|---|---|
| High coverage and falling curvature brackets | The domain is relevant to sampled mass, and the finite-width curvature mechanism is visible |
| High coverage, tiny curvature at every width | Geometry is favorable throughout; no improvement rate is resolved |
| High coverage, wide overlapping curvature brackets | Domain relevance is supported; actual-curvature comparison remains unresolved |
| Falling residual upper bounds, unresolved actual-Hessian trend | The measured adverse-curvature envelope improves; do not assert the true deficit decreases |
| No observed exits | No empirical rare-exit exponent is inferred |
| Better ESS per evaluation but worse ESS per second | Statistical efficiency improves while forward-pass cost dominates runtime |
| No sampler speedup despite falling curvature | The geometric claim survives; a speedup claim does not |
| Poor chain diagnostics | Posterior estimates remain provisional |
| Well-resolved violation of a theorem bound | Investigate assumptions, target, numerics, and theorem alignment before claiming consistency |

If supported, the concise paper message is:

> The high-probability cylinder remains relevant as ambient parameter dimension grows, and negative likelihood curvature within it decreases with width relative to Gaussian prior precision.

This is precise, close to the current theorem, and achievable using a small reusable set of runs.

## 14. References and the decisions they support

**[M] Current supplied manuscript.** *AISTATS 27 Wide BNN Posterior Geometry (3)*: Section 3 for model and centers; Lemma 4.1 for mass; Theorem 2 and Appendix A.3.2 for curvature and conditional inequalities; Section 5 and the depth obstruction for limits of the two-layer interpretation. The accompanying *(2)* document is the theory deck. These supply all manuscript-specific constants.

**[R1] Murray, I., Adams, R. P., and MacKay, D. J. C. (2010).** *Elliptical slice sampling.* AISTATS, PMLR 9:541–548. Supports the Gaussian-prior sampling algorithm and its parameter-free ellipse/bracket construction, not a width-specific mixing theorem for this BNN.

<https://proceedings.mlr.press/v9/murray10a.html>

**[R2] Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Bürkner, P.-C. (2021).** *Rank-normalization, folding, and localization: An improved \(\widehat R\) for assessing convergence of MCMC.* Bayesian Analysis 16(2):667–718. Supports rank-based multi-chain diagnostics, localized ESS, and quantile error assessment.

<https://arxiv.org/abs/1903.08008>

**[R3] Cotter, S. L., Roberts, G. O., Stuart, A. M., and White, D. (2013).** *MCMC Methods for Functions: Modifying Old Algorithms to Make Them Faster.* Statistical Science 28(3):424–446. Supports Gaussian-reference-preserving proposals and pCN. Mesh-robustness results are not automatically width-uniform guarantees for every neural-network target.

<https://arxiv.org/abs/1202.0709>

**[R4] Flegal, J. M., and Jones, G. L. (2010).** *Batch means and spectral variance estimators in Markov chain Monte Carlo.* Annals of Statistics 38(2):1034–1070. Supports dependence-aware Monte Carlo variance estimates under stated conditions; it does not justify zero uncertainty from an unobserved rare event.

<https://arxiv.org/abs/0811.1729>

**[R5] Hron, J., Novak, R., Pennington, J., and Sohl-Dickstein, J. (2022).** *Wide Bayesian neural networks have a simple weight posterior: theory and accelerated sampling.* ICML, PMLR 162:8926–8945. Motivates the width/sampling connection. Its simplification and acceleration use a data-dependent reparameterization that this protocol does not apply.

<https://proceedings.mlr.press/v162/hron22a.html>

**[R6] Pezzetti, L., Favaro, S., and Peluchetti, S. (2025).** *Function-Space MCMC for Bayesian Wide Neural Networks.* AISTATS, PMLR 258:478–486. Motivates function-space observables and explicit sampling-cost diagnostics. Its acceptance results concern a reparameterized posterior and are not imported as predictions here.

<https://proceedings.mlr.press/v258/pezzetti25a.html>

**[R7] Supplied sin-geoopt repository archive.** PLOTS.md, EXPERIMENT_ALIGNMENT_AUDIT.md, and checkpoint/figure-sidecar conventions supply the reproducibility template. Repository-specific statements refer to the supplied snapshot, not an independently verified current public revision.
