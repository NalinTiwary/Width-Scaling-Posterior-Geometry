---
title: "Width and posterior geometry: final experiment execution runbook"
subtitle: "Global deep PI, global shallow LSI, and local deep LSI"
author: "Execution specification | theory slides, version (15)"
date: "30 September 2026"
---

# 1. Read this before implementing anything

This is a complete specification for one experimental campaign. Its deliverable is three main figures, two appendix figures, diagnostic tables, and the machine-readable results behind them. The implementation team should not need a scientific follow-up decision. All continuation, step-size refinement, exclusion, and reporting rules are fixed below. A failed diagnostic produces an explicitly unresolved result; it never authorizes changing the dataset, prior, observable, or plotting convention until the result looks favorable.

**What is being delivered now:** a build-and-run specification, not an implemented or tested sampler package. The `bnn_geometry` commands below define the interface the implementer must build. They do not refer to commands already present in `sin-geoopt`. The attached YAML records the scientific and numerical settings. The only numerical work performed while preparing this document was checking the algebraic equivalence of the two Metropolis ratios in Section 8; no proposed posterior experiment has been run.

The authoritative mathematical source is the supplied *Width Improves BNN Posterior Geometry*, file `_AISTATS_27_Wide_BNN_Posterior_Geometry (15).pdf`, especially slides 3 and 8--10 [T]. The slides specify the results and assumptions; this document does not independently establish their proofs. Earlier ball-localization protocols and head-cutoff plots are not the execution specification for this campaign.

**The question is:** as width increases, does the unrestricted posterior exhibit favorable relaxation, do prespecified entropy--Fisher probes show favorable geometry on the appropriate final domains, and does the final deep spectral domain retain substantial posterior mass?

An experiment cannot guarantee a positive result in advance. This protocol makes a positive, flat, adverse, or numerically unresolved result interpretable without requesting another experiment. It deliberately does not attempt to prove an inequality by finite sampling, recover an optimal PI/LSI constant, or estimate an extremely small exit probability from zero observed exits.

## 1.1 Fixed scope and deliverables

| Item | Required output | Scientific purpose |
|---|---|---|
| Main Figure 1 | Two panels: shallow and deep, relaxation versus width | Practical consequence for unrestricted Langevin-type dynamics |
| Main Figure 2 | Two panels: shallow global and deep conditional entropy--Fisher ratios | Direct finite-probe diagnostic of the LSI functional inequality |
| Main Figure 3 | Two panels: deep spectral occupancy and normalized spectral-norm quantiles | Scope and non-vacuity of the final deep LSI domain |
| Appendix Figure S1 | Step-size comparison and Gaussian calibration | Numerical interpretation of Figures 1 and 2 |
| Appendix Figure S2 | Prior/posterior predictive log score and static PI probes | Evidence of learning and an independent geometry diagnostic from existing samples |
| Appendix tables | Target audit, convergence, counts, cost, exclusions | Establish exactly which conclusions the data support |

There are **eight model settings and three independent replicates: 24 posterior targets**. Each target has four reference chains and four dynamics chains. Endpoint step-halving checks add trajectories for existing targets, not new scientific settings. No real-data run, extra depth, sample-size sweep, Hessian sweep, or sampler competition is included. Do not append any of those after looking at the results.

The three existing data replicates are the independent units for variation across datasets and prior centers. Thousands of correlated posterior draws are not thousands of independent experimental replicates.

## 1.2 What this campaign will and will not identify

At fixed data and depth, the slides give an upper correction of order $m^{-1/3}$ for the deep global PI coefficient. The shallow global LSI correction is $O(n/\sqrt m+n^2/m)$; the deep conditional LSI correction is $O(n^{3/2}/\sqrt m+n^2/m)$ [T].

These are sufficient upper bounds, not predictions that every observable's relaxation time follows those powers. Consequently:

- The study tests width dependence at **fixed $n=128$** and the relevance of the final domains.
- It does **not** validate the joint $m\gg n^2$ or $m\gg n^3$ regimes. A paper must not describe these figures as a sample-size scaling study.
- A quantity already below the Gaussian benchmark may stay flat or decrease further. It need not approach the benchmark from above.
- An empirical constant inferred from a finite probe family is a lower diagnostic of the optimal inequality coefficient, not an upper estimate of it.
- Global deep PI does not require conditioning the sampling trajectory. Conditional deep LSI does not become global LSI merely because the conditioning event was observed frequently.

The fixed-width ranges below are computational choices. They are not claimed to enter the asymptotic regime guaranteed by the unspecified constants in the slides. This is an intentional limit of a small experimental section.

# 2. Exact scientific configuration

## 2.1 Target matrix

| Architecture | Weight layers $L$ | Hidden layers | Widths $m$ | Parameter count $p$ |
|---|---:|---:|---|---|
| Shallow | 2 | 1 | 64, 256, 1024, 4096 | $33m$ |
| Deep | 3 | 2 | 32, 64, 128, 256 | $m^2+33m$ |

For both architectures: $n=128$, input dimension $d=32$, two classes, activation tanh, no biases, prior standard deviation $\sigma=1$, and summed cross-entropy. Replicate IDs are 0, 1, 2; chain IDs are 0, 1, 2, 3. Replicate identity includes both the dataset and the first-layer prior-center bank. Use the same replicate's data and center bank at every width and in both architectures.

The shallow/deep widths differ because deep likelihood evaluation contains an $m\times m$ multiplication. Compare width trends **within architecture**. Do not interpret a shallow/deep difference at unmatched widths as an isolated causal effect of depth.

## 2.2 Generate data exactly once per replicate

Use NumPy's `Generator(PCG64(seed))` for persisted data, centers, and probe directions. Make separate random streams with the seed scheme in Section 4. Do not use the same stream for data generation and chain noise.

1. Draw a teacher vector $v_*\sim N(0,I_{32})$. Do not normalize this vector.
2. Draw 128 independent $z_i\sim N(0,I_{32})$ and set $x_i=z_i/\|z_i\|_2$. Redraw only if a norm is exactly zero in floating point.
3. Set teacher probabilities $p_i^*=\operatorname{sigmoid}(2v_*^\top x_i)$ and sample $y_i\sim\operatorname{Bernoulli}(p_i^*)$.
4. Independently generate 1,024 held-out inputs and labels by the identical teacher rule.
5. Save inputs, labels, teacher probabilities, and the teacher vector. Never select a new replicate because labels, accuracy, conditioning, or posterior behavior are inconvenient.
6. Record class counts, $\|X\|_{\mathrm{op}}$, input-norm extrema, and the nonzero eigenvalues of $XX^\top$. Rank at most 32 is expected. Do not whiten the rows into an orthogonal design.

Why this design: the inputs meet the fixed norm assumption exactly, the signal and label noise are controlled, and samples are genuinely nonorthogonal. This avoids repeating the special orthogonal-input geometry of the old experiments. The teacher norm convention keeps the distribution of $v_*^\top x$ of order one rather than reducing it by $1/\sqrt d$.

## 2.3 Implement the network and binary logits without ambiguity

Store data as rows. Let $A\in\mathbb R^{1\times m}$, $W_1\in\mathbb R^{m\times32}$, and, for $L=3$, $W_2\in\mathbb R^{m\times m}$. Compute

$$
H_1=\tanh(XW_1^\top),\qquad
H_2=\tanh(H_1W_2^\top/\sqrt m)\quad(L=3),
$$

$$
f=H_{L-1}A^\top/\sqrt m.
$$

Use the orthonormal centered-logit convention

$$
\ell_i=(-f_i/\sqrt2,\ f_i/\sqrt2),\qquad
p_i=\operatorname{sigmoid}(\sqrt2 f_i).
$$

The label 1 is the second class. The exact likelihood potential is

$$
V(\theta)=\sum_{i=1}^{128}\left[\operatorname{softplus}(\sqrt2 f_i)-y_i\sqrt2 f_i\right].
$$

This formula fixes a factor of $\sqrt2$ that otherwise silently changes the model. Check it against a two-class `log_softmax` implementation. Do not use `sigmoid(f)` in one component and `sigmoid(sqrt(2)*f)` elsewhere.

No input $1/\sqrt d$ factor, bias, normalization layer, minibatching, dropout, weight decay optimizer, temperature, or averaged-loss posterior is allowed. The full potential is

$$
U(\theta)=V(\theta)+\frac{\|\theta-\theta_0\|_2^2}{2\sigma^2}.
$$

All coordinates are sampled. There is no MAP training stage or trained checkpoint defining the prior.

## 2.4 Prior centers: the required correction to the old experiments

For each replicate, generate a $4096\times32$ bank of independent standard Gaussian entries. Use its first $m$ rows for $W_{1,0}$. For the deep architecture set $W_{2,0}=0$. Set

$$
(A_0)_j=\frac{(-1)^j}{\sqrt m},\quad j=0,\ldots,m-1.
$$

Thus $\|A_0\|_{\mathrm{op}}=1$ exactly, and $\|W_{2,0}\|_{\mathrm{op}}/\sqrt m=0$. This supplies width-independent constants $a_0=1$ and $r_0=0$.

The prior for every coordinate is still $N(\theta_{0,k},1)$. **Only the head means shrink with width. Neither the prior variance nor the sampled head weights are divided by $\sqrt m$.** The network already has its output normalization.

The older means $a_{j,0}=\pm1$ have operator norm $\sqrt m$ and do not test the current quantitative bounded-center results. Old posterior draws cannot be relabeled as draws from the corrected target. Do not pool old and new measurements in these figures.

## 2.5 Final deep conditioning event

Fix $a=2.5$ before any run. With $\sigma=1$ and $r_0=0$, it satisfies the slide's strict condition $a>r_0+2\sigma$.

$$
G_a=\left\{\|W_2\|_{\mathrm{op}}/\sqrt m\le2.5\right\},
\qquad \nu_a=\pi(\cdot\mid G_a).
$$

There is **no head cutoff** and **no first-layer cutoff**. All chains sample $\pi$ on the full Euclidean parameter space. The conditional expectations needed for Figure 2 are computed by the event weighting in Section 11. No reflection, projection, clipping, rejection at $G_a$, or post-update rescaling is used by either sampler.

# 3. Build requirements and module boundaries

Implement in a clean directory or branch separate from the old cylinder experiment outputs. Use Python 3.11, PyTorch with float64 support, NumPy, SciPy, an MCMC diagnostics package implementing rank-normalized split $\widehat R$ and bulk/tail ESS, h5py, pandas, PyYAML, matplotlib, and pytest. ArviZ or its maintained statistics component is suitable; pin the exact installed API after the compatibility tests. A version lock produced on the execution machine is authoritative. This document does not assert an untested set of package versions is compatible.

Install a PyTorch wheel appropriate for the execution machine using the official installation instructions [D1], then install the remaining dependencies. Use one shared environment for the whole campaign. Export the exact environment, package hashes where supported, Python version, GPU driver, accelerator model, BLAS implementation, and operating system before sampling.

Disable automatic mixed precision and TF32; all state, potential, gradient, acceptance, covariance, and spectral calculations are float64. No inference graph may retain every past state. Detach states and gradients at each transition. Evaluate the full training set in each likelihood call.

The required package interface is:

```text
bnn_geometry/
  config.py          strict schema; target enumeration; immutable hashes
  randomness.py      independent seeded streams and checkpoint RNG state
  data.py            teacher datasets; center banks; probe directions
  model.py           pure forward, V, grad_V, parameter packing
  probes.py          fixed observables, gradients, calibration constants
  ellipse.py         centered full-state elliptical slice reference sampler
  pcnl.py            adjusted Langevin proposal and reverse density
  spectral.py        full SVD, error margin, event classification
  storage.py         chunked arrays, atomic checkpoints, resume validation
  diagnostics.py     rank diagnostics; raw-mean ESS; MCSE; status rules
  entropy.py         event-weighted tilt ratios; held-out selection
  relaxation.py      physical-time autocorrelation and uncertainty
  analyze.py         tables and per-target metrics
  figures.py         main and appendix figures from saved tables only
  audit.py           end-to-end count, target, and figure-source checks
  __main__.py        staged CLI and single campaign runner
configs/campaign.yaml
tests/
```

The CLI shown in Section 15 is an implementation contract. Implement `--dry-run` for every command. A dry run prints target IDs, array dimensions, projected storage, numerical work limits, and output paths without sampling.

Borrow from `sin-geoopt` its metadata validation, separate chain identities, physical-time step checks, resumability, and figure sidecars [R]. Write new Euclidean model and sampler code. Its spherical state space, priors, averaged-loss target, and model-specific Hessians do not transfer here.

## 3.1 Pure functions and dimensions

`potential_and_grad(theta, target)` returns the likelihood-only scalar $V$, its gradient in original parameter coordinates, and optional cached predictions. `full_potential` separately adds the Gaussian quadratic. Both functions must give the same answer regardless of whether the state came from an ellipse, Langevin, or prior run.

Flatten parameters in the immutable order `A, W1, W2`, with row-major flattening within blocks; omit W2 for shallow models. Store block shapes, offsets, and a schema version. Packing followed by unpacking must be bitwise exact. A chain dimension must never be treated as a neuron or data dimension.

Euclidean gradient norms always mean the sum over **all original parameter coordinates**. If an internal standardized coordinate $z=(\theta-\theta_0)/\sigma$ is used, convert gradients by $\nabla_\theta f=\nabla_z f/\sigma$. Tests must include $\sigma\ne1$ so that a missing factor cannot hide in the production setting.

## 3.2 Scientific identity versus execution identity

A scientific target hash includes architecture, widths, data-array hashes, prior-center hashes, $\sigma$, activation, logit convention, loss normalization, and spectral event definition. It excludes chain seed, output path, and number of transitions.

An execution hash additionally includes sampler, step size, dtype, code revision, numerical options, probe definitions, and diagnostic settings. A change to either hash must never silently resume an incompatible checkpoint. A valid continuation changes only the requested final count and appends contiguous transitions using the saved RNG state.

# 4. Randomness, reproducibility, and storage

Set master seed `20260930`. For every stream form the UTF-8 string

```text
20260930|schema=1|rep=<r>|arch=<arch>|width=<m>|chain=<c>|stage=<stage>|stream=<name>
```

Use the first eight bytes of its SHA-256 digest, interpreted as an unsigned big-endian integer, reduced modulo $2^{63}-1$, as the seed. Encode the exact step size (float.hex) and calibration/validation role in the stage string for trajectories at different steps. For numerical fixtures use `rep=0`, `arch=fixture`, `width=shared`, and the explicitly named test as stage; include sigma in that stage when it varies. Explicitly replace irrelevant fields with `shared`: data, teacher, and center-bank streams are shared across architecture and width within a replicate. Never use Python's process-randomized `hash()`.

Keep a separate RNG for each chain and each stage. A proposal's Gaussian vector and its uniform accept/slice variables may use that chain's sampler stream. Data, prior predictive draws, calibration simulations, bootstrap indices, and probe construction must have distinct stream names. GPU noise may use a dedicated `torch.Generator`; record its device and entire RNG state. Exact cross-device bitwise replay is not promised; exact same-backend continuation is required.

Store scalar trace arrays at every retained transition. Do not thin scalar traces for convenience. Store reference parameter states every 16 production transitions; this is a storage schedule, not a claim of independence. Save the draw indices with every archived state. At the initial 8,192-draw reference stage this yields 512 states per chain, or 2,048 per target. At the maximum stage it yields 16,384 per target.

Use one HDF5 file per target, chain, and stage, with one writer. Store float64 arrays in chunks, optionally with lossless compression. Do not save full dynamics parameter histories. Dynamics require scalar traces and periodic restart states only. Save reference calibration states every eight steps. Never substitute lossy state compression for exact saved samples.

At the maximal reference archive, the 24 targets require approximately 111 GB of uncompressed parameter arrays alone. Reserve at least 160 GB for this campaign, or implement a checked streaming archive with the same retained state indices and exact float64 values. The initial reference stage uses one eighth of the maximal production archive. Compression savings are not part of the capacity guarantee.

Every 1,024 reference transitions or 2,048 dynamics transitions, atomically write a restart record containing current state, cached likelihood/gradient where used, RNG state, exact completed counts, stage, hashes, call counters, and elapsed timing. Write a temporary file, flush it, then rename. Resume must neither repeat nor omit an already saved transition.

# 5. Fixed observables and probe functions

An observable is specified by formula and index before results are seen. For training indices use zero-based

$$
I_{\mathrm{train}}=(0,16,32,48,64,80,96,112).
$$

For held-out indices use $(0,128,256,384,512,640,768,896)$. No selection by label, confidence, posterior variance, or autocorrelation is allowed.

## 5.1 Dynamics observables

Record at each dynamics transition:

1. Likelihood potential $V$ and full potential $U$.
2. The eight training centered logits $f_i$.
3. The eight held-out probabilities $\operatorname{sigmoid}(\sqrt2 f_i)$.
4. Four interactions $b_j$, defined below, with $j=0,1,2,3$.
5. Two linear calibration projections $q_A,q_W$, defined below.
6. Squared displacement of each parameter block divided by that block's dimension.
7. Proposal accepted/rejected, potential/gradient evaluations, and time counters.

The main relaxation families are `loss` (V alone), `train_logit` (eight), `test_probability` (eight), and `interaction` (four). U, block norms, and linear projections are diagnostic controls. Do not include controls in the main family maximum: nearly unaffected Gaussian coordinates can otherwise force a curve toward one by construction.

## 5.2 Four interactions

For shallow models, use

$$
b_j=(A_j-A_{0,j})\tanh\left[(W_{1,j}-W_{1,0,j})^\top x_{16j}\right].
$$

For deep models, construct a fixed unit vector $u_m$ with entries $(-1)^k/\sqrt m$ and use

$$
b_j=(A_j-A_{0,j})\tanh\left[(W_{2,j}-W_{2,0,j})^\top u_m\right].
$$

These are smooth, defined on the whole state space, and couple a head coordinate with a hidden-weight coordinate. They need not be the slowest posterior functions. Their exponential tilts are integrable: their absolute values are bounded by the absolute head displacement, and the posterior likelihood factor is at most one relative to its Gaussian prior.

## 5.3 Linear controls

Use $q_A=u_m^\top(A-A_0)$ and

$$
q_W=\frac{1}{\sqrt{32m}}\sum_{j=0}^{m-1}\sum_{k=0}^{31}(-1)^{j+k}(W_1-W_{1,0})_{jk}.
$$

Each direction has Euclidean norm one. Under the prior, both are centered Gaussians with variance $\sigma^2$. They calibrate the normalization and gradient metric. They are not proof that the posterior's worst directions are Gaussian.

## 5.4 Entropy probes and frozen standardization

Use nine substantive raw probes: V; the training logits at indices 0, 32, 64, 96; and the four $b_j$. Also calculate the two linear controls separately. This requires 11 gradient-norm calculations per selected reference state, not a Hessian.

Using only the reference calibration segment from Section 7, compute the ordinary pooled arithmetic mean $\mu_g$ and sample standard deviation $s_g$ (denominator N-1) across the four chains. For both shallow and deep models use the full-posterior calibration segment; standardization need not be conditional for the conditional inequality to be valid. Freeze

$$
f_g=(g-\mu_g)/s_g,
\qquad \|\nabla f_g\|^2=\|\nabla g\|^2/s_g^2.
$$

If $s_g<10^{-8}\max(1,|\mu_g|)$, mark that probe degenerate and do not replace it. Record its status. Its absence can make its family incomplete. The calibration segment is never used in the final expectations or plotted estimates.

Use tilts $t\in\{-1,-0.5,0.5,1\}$. These are fixed amplitudes in units of the probe's calibration standard deviation. Do not tune t to manufacture an excess above one. Compute all four for every eligible probe, including results that are flat, small, or noisy.

# 6. Testing: complete these gates before production

Tests below are requirements for the implementation team, not tests already executed. A failed test is a code or numerical issue to resolve before launching production. Debugging against known answers is allowed; changing scientific settings in response to desired outcomes is not.

## 6.1 Deterministic unit tests

**A. Model and gradients.** For L=2 and L=3, m=4, d=3, n=5, use fixed random arrays. Compare the vectorized forward pass with a literal loop implementation. Require absolute/relative error below $10^{-11}/10^{-10}$. Compare binary loss with centered two-class `log_softmax`. At f=0 require V=$n\log2$. Duplicating the data must double V and grad V exactly to numerical tolerance.

Run double-precision autograd gradient checks on V and every probe. Also compare central directional differences with gradient dot products for five fixed unit directions, using increments $10^{-4},10^{-5},10^{-6}$. Require at least one increment to achieve relative error below $10^{-5}$, using absolute error $10^{-7}$ when the derivative is close to zero. Include nonzero prior means and sigma=0.7 in the full-potential check.

**B. Architecture and centers.** Assert parameter counts, trainability of every block, $\|A_0\|=1$ to $10^{-12}$, exact zero W2 center, nested W1 bank, all input norms equal one to $10^{-12}$, and no hidden input normalization. Changing m must not change the variance used to draw any prior coordinate.

**C. Sampler algebra.** At ten fixed random pairs of states and step sizes, compare the stable Metropolis log ratio in Section 8 with the full log-target plus forward/reverse Gaussian-density ratio. Require relative/absolute agreement $10^{-9}/10^{-9}$ on the small test. Repeat with V=0; the log ratio must vanish to $10^{-10}$. Deliberately omitting the reverse proposal should fail a test.

**D. Spectral membership.** Test diagonal matrices with known largest singular value below, at, and above $a\sqrt m$. Test nonnormal matrices whose eigenvalue radius differs from their singular norm. Membership uses the largest singular value, not the largest eigenvalue magnitude. Near-boundary numerical cases must follow Section 12 rather than being assumed inside.

**E. Estimators.** Test the entropy formula against an independent direct weighted-sum implementation. A constant probe is degenerate, not a division-by-zero workaround. Test conditional weighting on a one-dimensional truncated Gaussian against numerical quadrature; keep the zero weights at outside states in the time series. Test the raw-time IAT formula on synthetic AR(1) sequences with known coefficient. A test using rank-transformed data in place of raw observables must detect the resulting change for a nonlinear transform.

**F. Storage and bookkeeping.** Compare a small uninterrupted run with the identical run stopped/resumed twice. Require bitwise equality on the same backend. Reject a checkpoint after changing data, sigma, logit normalization, or step size. Detect duplicate draw IDs, missing chunks, truncated files, incorrect chain concatenation, and shuffled time indices. A rejection in the dynamics sampler must retain an exact repeated state in its scalar trace.

## 6.2 Known-distribution integration tests

Run these on CPU in float64 before GPU production. Repeat the deterministic forward and gradient checks on the actual production backend.

1. **Gaussian OU calibration:** V=0, sigma in {0.7,1}, dimension 16, four independent chains of 65,536 retained transitions at h=$0.02\sigma^2$, initialized from the exact Gaussian. No burn-in is needed for this analytic test. Acceptance must be one except roundoff-level discrepancies; mean log-acceptance errors larger than $10^{-10}$ fail. Check linear means and variances against known values using MCSE, and linear IAT against the exact discrete expression in Section 10. Require agreement within the larger of 5% and three estimated standard errors. Record failures, do not reroll seeds.
2. **Gaussian entropy calibration:** use 65,536 iid draws in dimension 8 for sigma in {0.7,1}; evaluate a unit linear probe with all four tilts. The exact ratio is $\sigma^2$ for every nonzero t. Require error within the larger of 3% and three bootstrap standard errors. This detects factors of two and incorrect gradient coordinates.
3. **Quadratic likelihood:** set V=$\|D\theta-b\|^2/2$ with fixed non-diagonal D in dimension 8 and a nonzero prior mean. Generate D as G/sqrt(8) + diag(1/8,2/8,...,1), with iid standard Gaussian G; draw b as an independent standard Gaussian vector; set the prior mean to (0.1,0.2,...,0.8) and sigma=0.7. Use the named `quadratic_fixture` stream in the master seed scheme. The exact posterior covariance is $(\sigma^{-2}I+D^\top D)^{-1}$ and its mean follows the corresponding linear system. Run both samplers for four chains, 4,096 burn-in and 32,768 retained transitions. Compare eight means and covariance diagonals within four MCSEs; examine every failure. This is a more informative target-correctness test than V=0 alone.
4. **Conditional entropy:** for iid N(0,1) states, condition on $|x|\le1.5$, use f=x, and compare numerator, denominator, and ratio with adaptive one-dimensional quadrature to the larger of 2% and four MCSEs. This checks the normalizing probability and conditional-event implementation.

Statistical tests have nonzero false-failure probability. A failure may be accepted only if a written audit identifies ordinary Monte Carlo fluctuation with an independently computed error estimate and all deterministic checks pass. Do not repeatedly simulate until the test passes. Preserve the first results.

## 6.3 End-to-end fixture and hardware benchmark

Run a reduced fixture with n=8, d=3, m=4, both architectures, two replicates, four chains, short fixed traces, and the entire export pipeline. This fixture is never a scientific result. It must create the same schemas, statuses, figure sidecars, and counts as production. Include a deliberately failed target and an all-inside event to exercise those reporting paths.

Benchmark 200 full V evaluations, 200 V-and-gradient evaluations, 20 sets of 11 probe gradients, and 20 full SVDs for each largest production architecture. Synchronize the accelerator around timing. Separately time I/O and scalar trace recording. Print a resource estimate using these timings and the planned transition counts. A count of model settings is not a compute estimate.

Resolve backend or memory failures before production. Do not reduce n, change precision, or substitute a power-iteration norm solely to fit an unreported resource limit. If the required campaign is unaffordable on the available machine, export `resource_blocked` with the benchmark and estimated demand; do not produce an apparently complete paper figure from an unannounced reduced study.

# 7. Reference sampling: one unrestricted posterior pool per target

Use full-state elliptical slice sampling around the nonzero Gaussian prior center [1]. It supplies reference expectations and states. Its iteration count is not a Langevin clock, so its ESS is not the empirical PI experiment.

Write $x=\theta-\theta_0$. At a transition, draw $\zeta\sim N(0,\sigma^2I)$, $u\sim U(0,1)$, and a uniform initial angle $\alpha\in[0,2\pi)$. The slice level is $\log y=-V(\theta)+\log u$. Set the bracket to $[\alpha-2\pi,\alpha]$. Propose

$$
\theta'=\theta_0+x\cos\alpha+\zeta\sin\alpha.
$$

Accept if $-V(\theta')\ge\log y$. Otherwise shrink the bracket on the side of zero containing the rejected angle, resample an angle uniformly within the bracket, and repeat. The prior quadratic is not included again in this likelihood slice test. The ellipse includes every parameter block jointly.

Do not cap the bracket search and treat an exhausted search as a successful unchanged transition. A numerical guard at 10,000 bracket evaluations triggers an error checkpoint and a sampler failure. Count every attempted likelihood evaluation, including rejected angles. Save bracket evaluation counts.

## 7.1 Initialization and segments

Initialize four chains from independent $N(\theta_0,\sigma^2 I)$ prior draws. For each chain run these disjoint segments:

| Segment | Transitions per chain | Use |
|---|---:|---|
| Burn-in | 4,096 | Discard from every estimate |
| Calibration | 2,048 | Freeze probe means/scales; retain calibration archive |
| Separation segment | 1,024 | Discard; reduces immediate calibration/production dependence |
| Initial production | 8,192 | First reference diagnostic checkpoint |
| Continuation stages | totals 16,384; 32,768; 65,536 | Append only if required gates fail |

The separation segment is not a mathematical guarantee of independence. The data separation prevents direct reuse of calibration draws; subsequent MCSE still respects chain dependence.

The stage values are **cumulative production counts**, not extra batches. For example, reaching 32,768 means adding 16,384 after the 16,384 checkpoint. Per target, 65,536 production draws per chain means 262,144 total retained transitions, not a per-target count of 65,536.

## 7.2 Reference stopping rule

At each stage, check the fixed scalar observables from Section 5, including V, U, all logits/probabilities/interactions, controls, and block norms. Require, across the four chains:

- maximum rank-normalized split/folded $\widehat R<1.01$;
- bulk ESS at least 1,000 for each nondegenerate monitored scalar;
- tail ESS at least 400 for each nondegenerate monitored scalar;
- no NaN/Inf, no missing transitions, and no incompatible metadata;
- agreement of first-half and second-half scalar means within three combined batch-means MCSEs plus $0.05$ pooled standard deviations.

The last check is a drift screen, not a stationarity proof. A truly constant quantity gets status `constant`, not an infinite ESS. A nearly constant but numerically nonconstant quantity must not be automatically excused.

For deep targets, compute SVD diagnostics on archived states at each candidate stopping stage. Apply $\widehat R<1.01$, bulk ESS at least 1,000, and tail ESS at least 400 to the normalized spectral norm S. If the event indicator has no variation, mark its ESS/MCSE `not_estimable_zero_exits`; that alone is not a reason to keep sampling indefinitely.

Once these gates pass, attempt the static estimates in Section 11 using the fixed archive budgets. Continue the reference chains to the next predefined stage if a required nondegenerate tilt lacks precision or if the conditional pool is inadequate. Stop when both reference and static-estimate gates pass, or at 65,536 production transitions. A failed strong tilt may therefore consume a continuation stage, but can never cause a new tilt or dataset to be chosen.

Hard cap: at most 1,500,000 likelihood evaluations per reference chain, including burn-in, calibration, separation, and bracket proposals. If reached before the scheduled count, stop after completing the current transition, preserve the partial segment, and mark `reference_budget_exhausted`. Do not combine unequal unfinished chains into a nominally complete balanced reference pool; use their common completed prefix for an explicitly flagged partial analysis.

These thresholds are protocol choices, not theorem hypotheses or universal convergence tests. Modern rank diagnostics motivate monitoring both location and tails [3]; passage does not prove all modes have been explored.

## 7.3 Reference outputs that must exist before dynamics

Save an unambiguous status for each target: `reference_pass`, `reference_unresolved`, or `reference_error`. Also save per-observable diagnostic rows, actual counts, chain-specific likelihood calls, calibration constants, the selected archive indices, and the last state of every chain.

Use the final state of reference chain c to initialize dynamics chain c. This is an approximately stationary start justified by the reported reference diagnostics, not an exact posterior sample. Dynamics use fresh random streams. If reference diagnostics fail at the hard cap, the dynamics may still be run within the fixed budget for diagnostic completeness, but their results cannot receive a publication-valid equilibrium interpretation. Label them accordingly.

# 8. Dynamics sampler and physical-time convention

For Figure 1 use a Metropolis-adjusted proposal that preserves the Gaussian prior exactly when V=0. It is equivalent to a pCNL parameterization [2]. Its small-step limit is the Euclidean Langevin diffusion

$$
d\theta_t=-\nabla U(\theta_t)\,dt+\sqrt2\,dB_t.
$$

The adjustment preserves the intended posterior in exact arithmetic. Finite-step trajectory relaxation is still the relaxation of a discrete kernel; it is interpreted as a diffusion diagnostic only with the checks below. An LSI for the posterior alone is not an automatic mixing theorem for this implementation [4].

## 8.1 Proposal and exact acceptance ratio

For x=$\theta-\theta_0$, g=$\nabla V(\theta)$, step parameter h, define

$$
\eta=e^{-h/\sigma^2},\qquad
b=\sigma^2(1-\eta),\qquad
q_v=\sigma^2(1-\eta^2).
$$

Propose

$$
y=\eta x-bg+\sqrt{q_v}\,\xi,\quad \xi\sim N(0,I),
\qquad \theta'=\theta_0+y.
$$

Use `expm1` to calculate $1-\eta$ and $1-\eta^2$ accurately at small h. Evaluate $V'=V(\theta')$ and $g'=\nabla V(\theta')$.

The full log acceptance ratio is

$$
\Lambda=-U(\theta')+U(\theta)
+\log q(\theta\mid\theta')-\log q(\theta'\mid\theta),
$$

where q is the Gaussian proposal above. A numerically preferable equivalent expression, obtained by canceling the prior-reversible Gaussian terms, is

$$
\begin{split}
\Lambda={}&V-V'
+\frac{(y-\eta x)^\top g-(x-\eta y)^\top g'}{1+\eta}\\
&+\frac{\sigma^2(1-\eta)}{2(1+\eta)}
\left(\|g\|^2-\|g'\|^2\right).
\end{split}
$$

Accept if $\log u<\min(0,\Lambda)$ for $u\sim U(0,1)$. On rejection retain x, V, and g exactly. Cache the accepted gradient; one new candidate gradient per transition then suffices. Count the initial gradient too. No gradient clipping or diagonal preconditioning is permitted.

A likelihood-only acceptance ratio would be valid for the prior-preserving proposal without the likelihood drift, but is **incorrect for this proposal**. The reverse-density term is mandatory.

## 8.2 Choose one step size per architecture before production

Use candidates $h/\sigma^2\in\{0.02,0.01,0.005,0.0025\}$ in descending order. For each target, initialize one calibration trajectory from reference chain 0, run physical time $10\sigma^2$ to discard, then $20\sigma^2$ to assess the candidate. Restart from the same reference state for each candidate, using distinct recorded noise streams. Calibration traces never appear in results.

A candidate is admissible if it has no numerical failures, acceptance at least 0.85, and no rejection streak longer than 500 transitions. For each target select the largest admissible candidate. Set the architecture's common production h to the minimum of its 12 target choices. This holds the kernel's time resolution fixed across widths within each architecture. Do not tune production h separately to make wide models appear faster.

If a target has no admissible candidate, mark its dynamics calibration failed. Use the minimum selected h for the remaining targets of that architecture, keeping the missing cell visibly unresolved. Do not introduce another proposal or a smaller unplanned calibration candidate.

Acceptance is only a screening criterion. High acceptance does not establish time-discretization accuracy; endpoint comparisons remain required.

## 8.3 Dynamics trajectory lengths and stopping

For every target and each of its four chains:

1. Start from the designated reference state.
2. Run and discard $20\sigma^2$ of dynamics time.
3. Retain an initial physical duration $T=512\sigma^2$ per chain, saving every scalar transition.
4. If the diagnostics in Section 10 fail, extend cumulatively to $T=1024\sigma^2$ and then $2048\sigma^2$.
5. Stop at the first passing stage or the hard cap. Do not use an arbitrary iteration cap without converting it to physical duration.

Use $N=\lceil T/h\rceil$ transitions and record the realized duration Nh. All rejected moves consume one h of the discrete time convention. Burn-in, step calibration, and reference transitions are excluded from retained T.

For V=0, the update is exact OU at spacing h. For nonzero V, kh is the small-step diffusion-time convention, not an identity between the adjusted chain and the continuous diffusion.

## 8.4 Mandatory step-halving matrix and one bounded refinement

For the narrowest and widest width of **each architecture and each replicate**, run four additional dynamics chains at h/2 from the same four reference starting states, with new noise. There are 12 endpoint target checks. Use equal retained physical duration to the corresponding base run, and the same discarded physical time.

Compare each main observable family between h and h/2. Require both estimates to meet their ordinary precision gates and the absolute difference of family estimates to be at most 20% of their average. Also compare V and the eight training-logit stationary means: differences must be within three combined MCSEs plus $0.05$ reference standard deviations. Show the actual differences and uncertainty, not only a pass/fail label.

If any endpoint fails, or any target has production acceptance below 0.80 or a rejection streak longer than 1,000 transitions, perform **one** predefined refinement for that entire architecture: use h/2 as its production step, generate the missing interior-width trajectories at h/2, reuse passing endpoint h/2 trajectories, and check its endpoints against h/4. Retain every original trace as validation material. Do not keep halving beyond this second comparison.

If the refined endpoint comparison still fails, publish Figure 1 with a visibly unresolved dynamics panel for that architecture; do not describe kh as experimentally validated diffusion time there. Static figures remain eligible if their own gates pass. This rule prevents an unending sampler-tuning project.

Hard cap: 8,000,000 new V-and-gradient evaluations per scientific target across all dynamics calibration, production, validation, and refinement trajectories combined. Count probe-gradient evaluations separately. Reaching this cap means `dynamics_budget_exhausted`, not permission to drop difficult observables or halve the requested precision.

# 9. Static and dynamic Monte Carlo uncertainty

Keep two sources of uncertainty separate:

- **Within-target Monte Carlo uncertainty:** correlated finite chains used to estimate an expectation or correlation time.
- **Across-replicate variation:** the three independent data/center realizations.

Do not put a single standard error around millions of pooled states and label it uncertainty across datasets. Main figures show all three replicate estimates with small horizontal offsets and light lines joining matched replicates. A thicker line shows the median of the three. Per-replicate error bars show within-target Monte Carlo uncertainty; the range of the three points shows observed replicate variation. Three replicates do not support a precise population confidence interval or a precise fitted scaling exponent.

For expectations use chain-aware batch means or a moving-block bootstrap [5]. For bootstrap intervals use 400 resamples, generated from fixed bootstrap seeds, resampling blocks within each chain and never crossing a chain boundary. Recompute the complete nonlinear statistic in each resample, including its normalizing means. For each chain of N saved values and block length b, draw block starts uniformly from 0 through N-b, concatenate contiguous length-b blocks until at least N values are obtained, and truncate to N. Never wrap across chains. Define bootstrap MCSE as the sample standard deviation of the 400 recomputed estimates. Use percentile 2.5% and 97.5% intervals as descriptive Monte Carlo intervals. The automatic stopping decisions and finite number of chains mean these are not presented as exact sequential-coverage guarantees.

Choose block length from preliminary raw-mean correlation times: at least ten estimated integrated autocorrelation times in saved-step units, rounded upward to the next power of two. Require at least 20 nonoverlapping blocks per chain at this length. Repeat the MCSE calculation at twice the block length when at least ten such blocks per chain remain; require MCSEs to agree within 30%. If not, the relevant precision gate fails and the predefined continuation rule applies. This is a practical stability diagnostic, not a proof of the asymptotic variance assumptions.

The raw-mean ESS used for a physical autocorrelation time must not be replaced with bulk rank ESS. Rank diagnostics assess sampling quality; raw autocorrelations describe the observable's time evolution.

# 10. Experiment 1: unrestricted relaxation

## 10.1 Quantity being measured

For a nonconstant observable g under stationary continuous Langevin dynamics, define

$$
\tau_g=\int_0^\infty
\frac{\operatorname{Cov}_\pi(g(\theta_0),g(\theta_t))}
{\operatorname{Var}_\pi(g)}\,dt.
$$

For the reversible diffusion, PI implies $\tau_g\le C_P(\pi)$. Thus a large $\tau_g$ witnesses a slow tested direction. A small finite collection of $\tau_g$ values cannot exclude an unmeasured slower direction.

The normalized quantity is $\tau_g/\sigma^2$. A linear observable under the Gaussian prior has $\tau_g/\sigma^2=1$. Nonlinear Gaussian observables can have smaller values; one is a reference timescale, not a lower bound for all plotted probes.

## 10.2 Discrete estimator and factor of two

Use unranked scalar traces and a chain-aware initial-positive/monotone sequence estimator for the statistical inefficiency

$$
\widehat s_g=1+2\sum_{k\ge1}\widehat\rho_g(k).
$$

Estimate the trapezoidal physical-time integral as

$$
\widehat\tau_g=\frac h2\widehat s_g
=\frac h2\frac{N_{\mathrm{total}}}{\widehat{\mathrm{ESS}}_{\mathrm{mean},g}}.
$$

Here $N_{\mathrm{total}}$ is the number of retained scalar observations across the chains supplied to that estimator. Do not use h times statistical inefficiency; that is a factor-of-two error for the continuous-time integral. Validate any package's `mean` ESS behavior on the Gaussian tests; do not infer its convention from its name alone.

For exact OU at step h, the corresponding discrete trapezoidal value is

$$
\tau_{\mathrm{OU},h}=\frac h2\frac{1+e^{-h/\sigma^2}}{1-e^{-h/\sigma^2}}.
$$

It tends to $\sigma^2$ as h tends to zero. Use this expression in calibration tests and record its very small offset from the continuous benchmark.

## 10.3 Prespecified family maximum with held-out evaluation

The main plot summarizes the most persistent tested member of each family, while reducing noise from selecting the largest estimated correlation time:

1. Divide dynamics chains into folds A={0,1} and B={2,3}.
2. Within a family, use fold A to select the observable with the largest estimated $\tau_g$; evaluate that selected observable using fold B only.
3. Reverse the folds and average the two held-out estimates.
4. Apply the same selection and evaluation algorithm in bootstrap resamples. Record both selected probe IDs and selection frequencies.

For the loss family, the only probe is V. The statistic is called the **held-out family relaxation diagnostic**, not the exact maximum autocorrelation time. Fold-specific estimates may select different probes. Selection protects against a simple winner's-noise effect but does not recover the optimal PI coefficient.

## 10.4 Validity and precision gates

Require the reference target to pass. Require production acceptance at least 0.80 in every chain and no rejection streak longer than 1,000; failure invokes the single step-refinement rule rather than merely a longer trajectory. For the four-chain dynamics trace require all monitored nondegenerate scalars to have $\widehat R<1.01$, bulk ESS at least 1,000, and tail ESS at least 400. For each family estimate require relative bootstrap MCSE at most 15%, the block-length stability check, and enough time to resolve the correlation tail: retained duration per chain at least 100 times the largest relevant estimated $\tau_g$.

Also compare dynamics stationary V, training logits, and held-out probabilities with the independent reference pool, using three combined MCSEs plus $0.05$ reference standard deviations as a discrepancy screen. Reference and dynamics share initial states, so this is a practical consistency screen, not a formally independent hypothesis test; the independent transition noise and long retained trajectories reduce the influence of that start.

Failure triggers only the duration stages already specified. If still unresolved at the cap, preserve the estimate with an open marker in the diagnostic export and omit its connecting main-result line. Do not silently drop the slowest probe.

## 10.5 Required results and inference

Save every raw probe estimate, selected probe IDs, the family estimate and MC interval, raw-mean ESS, rank diagnostics, chosen lag window, block length, time duration, h, acceptance, longest rejection streak, gradient count, likelihood count, and wall-clock time.

For each replicate and family also calculate the widest/narrowest ratio, using the same bootstrap procedure. A ratio below one indicates faster measured relaxation over the observed width range. State improvement only when the corresponding intervals support it and the direction is not driven by a single replicate. Do not fit or draw an $m^{-1/3}$ decay line for this statistic.

Possible outcomes are all meaningful:

- A decrease toward the Gaussian timescale supports a favorable width effect for these observables.
- A flat curve near or below one supports stability over the tested widths, not acceleration.
- An increase indicates a limitation of the tested geometry or dynamics; report it if numerically resolved.
- A failure of the time-step or mixing checks leaves the dynamics question unresolved within the allocated budget.

# 11. Experiment 2: entropy--Fisher diagnostics

## 11.1 Target and exact mathematical quantity

For shallow models use $\nu=\pi$. For deep models use $\nu=\nu_a=\pi(\cdot\mid G_a)$. For a frozen standardized probe f and nonzero tilt t, define a probability density relative to this target:

$$
r_{f,t}(\theta)=\frac{e^{t f(\theta)}}{E_\nu[e^{t f}]},\qquad
\rho_{f,t}(d\theta)=r_{f,t}(\theta)\nu(d\theta).
$$

Define

$$
D_{f,t}=E_\nu[r_{f,t}\log r_{f,t}],\qquad
I_{f,t}=t^2E_\nu[r_{f,t}\|\nabla f\|^2],
$$

$$
R_{f,t}=\frac{2D_{f,t}}{I_{f,t}}.
$$

With the convention $\operatorname{Ent}_\nu(g^2)\le2C_{\rm LS}E_\nu\|\nabla g\|^2$, LSI implies $R_{f,t}\le C_{\rm LS}(\nu)$ [4]. This follows by substituting $g=\sqrt r$. The factor of two is essential. The Fisher quantity uses the relative log-density gradient $\nabla\log r=t\nabla f$; it does not include $\nabla U$.

No posterior density estimator, partition function, or simulated KL-decay curve is required. The expectations are measured directly from the reference posterior pool. For a Gaussian target and a unit linear raw probe, $R_{f,t}=\sigma^2$ at all nonzero t, even after fixed affine standardization.

This is a static finite-probe diagnostic of entropy relative to gradient energy. It is not a measurement of the entire entropy-decay trajectory or an empirical upper bound on the optimal LSI coefficient.

## 11.2 Stable empirical calculation, including conditioning

Work on an archived reference sequence with chain and original draw indices preserved. Let $J_s=1$ for shallow targets and $J_s=\mathbf1_{G_a}(\theta_s)$ for deep targets. Let $u_s=t f_s-c$, with $c=\max_{s:J_s=1}t f_s$, and $w_s=e^{u_s}$. The numerical shift c cancels from the final answer.

Use averages over **all selected reference states**, including states outside $G_a$:

$$
P=\overline J,\quad Z=\overline{Jw},\quad
B=\overline{Jwu},\quad C=\overline{Jw\|\nabla f\|^2}.
$$

Then compute

$$
\widehat D=B/Z-\log(Z/P),\qquad
\widehat I=t^2C/Z,\qquad
\widehat R=2\widehat D/\widehat I.
$$

For the shallow target P=1. These expressions include the conditional normalizer correctly. Do not set P=1 merely because the outside states were discarded from a convenience array.

For uncertainty, resample the original time-indexed vectors $(J,Jw,Jwu,Jw\|\nabla f\|^2)$ within chains and recompute the complete ratio. **Do not concatenate the inside states into an apparently regularly spaced chain.** Conditional-event gaps retain temporal dependence.

The indicator is an expectation weight, not a differentiable part of f. Compute gradients of the smooth original probe; do not differentiate the hard event boundary, project the gradient, or add a boundary penalty.

If P=0, Z=0, I=0, or a nonfinite value occurs, mark the ratio invalid with its reason. Never add a denominator floor that changes the statistic. Tiny negative D within $10^{-12}$ of zero may be classified as numerically zero and degenerate, with the raw value logged; a larger negative D is an implementation failure. Record exponential underflows and maximum normalized weight. Do not clip importance weights.

## 11.3 Archive selection and a bounded gradient budget

At a candidate reference stopping stage, select 512 regularly spaced archived states per chain, for 2,048 total. Because reference stages and archive lengths are powers of two, use an integer stride through each archive, starting with its first saved state. Save the exact chosen indices. Evaluate the 11 probe values and full Euclidean gradient norms on these states.

If the static precision gates fail, increase to 1,024 states per chain and then 2,048 per chain if the current archive contains that many, retaining all earlier computed values by draw ID. The maximum is 8,192 selected states per target. If an increase is unavailable, the reference protocol may proceed to its next prescribed production stage. Never exceed 8,192 static states per target, even when the full reference archive is larger.

No autograd graph is retained after a state is processed. A loop over 11 probes is acceptable. Batched Jacobian-vector tools are acceptable only after agreement with the simple loop on the fixture. A single final analysis uses at most 90,112 probe-gradient evaluations per target. Earlier continuation stages can select states absent from the final subset; caching by original draw ID bounds cumulative work by 180,224 evaluations (11 probes times the maximal 16,384-state reference archive), plus numerical tests. Record cumulative and final-subset counts separately. No full Hessian is computed.

## 11.4 Precision gates for each probe and tilt

Apply gates on each selection/evaluation fold separately, with folds {0,1} and {2,3}:

- At least 256 inside states per chain for conditional analysis. Shallow states are all inside.
- The diagnostic reference gates have passed, including S for deep targets; the nonconstant normalizer and Fisher integrand also have four-chain rank R-hat below 1.01 on the selected archived sequence.
- No single state contributes more than 2% of the total normalized tilt weight within the fold.
- The weight-concentration quantity $(\sum Jw)^2/\sum(Jw)^2$ is at least 200 per fold. This is an iid weight-concentration screen, **not** chain-aware ESS.
- Chain-aware raw-mean ESS for the nonconstant normalizer and Fisher integrand is at least 200 per fold. For the entropy integrand, use its batch-means error and final-ratio precision even if cancellation makes a relative error to its own mean unhelpful.
- Relative bootstrap MCSE of D and I is at most 15%, and of R at most 15%; block-length stability passes.

For quantities exactly constant by construction, use their known value rather than manufacturing ESS. If a nondegenerate candidate fails at the final archive/reference cap, its family is marked incomplete. List the failing probe and tilt. Do not choose a smaller t after seeing failure.

## 11.5 Family selection, controls, and output

Use three substantive families: loss (four tilts of V), train logits (four logits times four tilts), and interactions (four interactions times four tilts). Within each family, select the largest ratio on reference chains 0 and 1 and evaluate it on chains 2 and 3. Reverse the folds and average the held-out ratios. Recompute selection during the bootstrap and report selected IDs and frequencies.

Call the result the **held-out family entropy--Fisher diagnostic**. If all candidates are valid, it estimates the performance of this fixed selection procedure. It is not exactly the largest population ratio. An incomplete family gets an open marker and an explicit incomplete status, without a connected confirmatory line. All candidate ratios still appear in the machine-readable table.

Report the two linear controls separately in Appendix S1; do not insert them into a substantive family to push its maximum toward the Gaussian benchmark.

Required per-probe output includes target law, raw probe ID, calibration mean/SD, t, P, Z, B, C, D, I, R, all MCSEs, normalized maximum weight, weight-concentration screen, raw-mean ESS, draw counts, selected indices, and validity reason. Save the standardized gradient norm before multiplying by weights so that the metric can be audited independently.

## 11.6 Interpretation that is allowed

A resolved finite-width ratio above $\sigma^2$, followed by smaller ratios at larger widths, is informative evidence that these tested entropy/gradient tradeoffs improve with width. A curve below $\sigma^2$ throughout means the tested tilts already satisfy a favorable tradeoff. Neither outcome estimates the optimal LSI coefficient from above.

The current theory supplies sufficient bounds and does not require these particular ratios to decrease monotonically. Flat or nonmonotone results must not be disguised by drawing theoretical slope guides or fitting an offset power law. Deep results always retain the conditional-law label.

# 12. Experiment 3: mass and geometry of the final deep domain

## 12.1 Spectral calculation

For every archived deep reference state, calculate

$$
S(\theta)=\frac{\|W_2\|_{\mathrm{op}}}{2.5\sqrt m}.
$$

Use a full float64 singular-value decomposition and take its largest singular value. At these widths (at most 256) this is feasible. Unconverged power iteration supplies a lower approximation to the norm and can incorrectly classify a state as inside; it is not an allowed substitute.

Numerical classification uses a guard band of $10^{-10}$ in S: inside if $S<1-10^{-10}$, outside if $S>1+10^{-10}$. For a state in the guard band, recompute with an independent CPU float64 SVD implementation and inspect reconstruction/orthogonality residuals. If the result remains within the guard band, label membership numerically unresolved. This is floating-point error handling, not a rigorous interval-arithmetic singular-value bound.

Store all singular-norm values, original state indices, backend, and near-boundary flags. Check 20 fixed states per target against the independent SVD implementation; require relative agreement below $10^{-9}$. A matrix eigenvalue computation is not a valid substitute for a singular-value norm.

## 12.2 Empirical occupancy and quantiles

On the archived unrestricted posterior sequence compute the occupancy, exit count, and median, 95th, and 99th percentiles of S. Do this per chain and per target. The denominator is the number of archived states actually inspected, not the number of all retained scalar transitions.

If there are numerical membership ambiguities, report an occupancy interval from counting every ambiguous state outside versus inside. Conditional entropy estimates cannot use those states as if their membership were known; mark them unresolved rather than assigning a convenient event label.

Use chain-aware block-bootstrap uncertainty for the nonconstant S quantiles. When at least 20 outside states occur overall and outside states are observed in at least two chains, also provide a block-bootstrap interval for occupancy, together with indicator diagnostics. With fewer observed exits, give the observed fraction, per-chain counts, and the label `rare-event uncertainty unresolved`; do not present a precise ordinary normal interval.

If there are zero exits, report `0 exits / N inspected archived states`, with N exact. Do not report a zero-width uncertainty interval, the iid rule of three, or an exponential exit-rate fit. Passing diagnostics for S supports sampled mass relevance but does not establish the probability of a much rarer event.

## 12.3 Prior comparator without new posterior runs

The W2 prior is the same iid standard Gaussian matrix law for all three replicates. For each of the four deep widths, generate 4,096 independent prior matrices from a separate, width-specific stream. Compute S by the same SVD code. Save the prior quantiles and their iid bootstrap intervals. Use one common prior reference per width, explicitly labeled; do not count it three times as independent experimental replication.

This is a cheap comparator explaining whether the event's high mass is largely inherited from the prior. It is not an additional posterior target. It is acceptable for posterior and prior curves to be almost indistinguishable.

For iid Gaussian matrices, the normalized spectral norm is expected to concentrate near $2\sigma/a=0.8$ as width grows. The median can move upward toward this reference while an upper quantile contracts; S is **not** expected to tend to zero. This is a Gaussian-matrix reference heuristic, not a measured posterior limit. The protocol measures the finite-width prior comparator directly and does not need an asymptotic fit.

## 12.4 What the figure can demonstrate

High observed occupancy, resolved spectral-norm diagnostics, and S quantiles comfortably below one support that the final conditional deep LSI concerns a substantial part of the sampled posterior. If only the indicator is plotted and every value is one, the result is visually uninformative; the norm-quantile panel supplies the required context.

The theorem's bound $(L-2)e^{c_0n-c_1m}$ must not be drawn with invented or fitted constants. Unless the paper independently provides explicit applicable constants, the figure shows empirical occupancy and the exact event boundary only.

If occupancy is materially lower at a width, report it. If reference chains disagree on S or event occupancy, the mass claim for that target is unresolved even if V mixes well. If prior and posterior S distributions agree, say that the domain's observed coverage is largely consistent with prior concentration; do not claim the likelihood actively confines W2.

# 13. Appendix computations using the same targets

## 13.1 Appendix S1: required numerical support

Create a four-panel appendix figure:

- **S1(a):** h versus h/2 held-out family relaxation estimates at endpoints, normalized by $\sigma^2$, with the identity line. If refinement occurred, show the final h/2 versus h/4 comparison and label the original failure in the table.
- **S1(b):** relative endpoint differences for the same quantities, with the 20% protocol threshold; retain all three replicates and both architectures.
- **S1(c):** Gaussian OU measured linear-probe IAT versus the exact discrete value for sigma=0.7 and 1, including MC intervals.
- **S1(d):** Gaussian linear entropy--Fisher ratios at t=-1,-0.5,0.5,1, normalized by $\sigma^2$, and the value-one reference.

Place acceptance, ESS, and computational costs in tables rather than adding more scientific plots. The diagnostics in this appendix are mandatory support for the interpretation of the main figures, not optional aesthetic material.

## 13.2 Appendix S2(a): verify that the posterior learns

For each target, evaluate predictions on all 1,024 held-out points using 2,048 equally spaced archived posterior states, 512 per chain. If fewer are available because the target failed, show it as unresolved. Independently generate 2,048 full prior states for that target and evaluate the same points, streaming states so full prior parameters need not be archived.

Calculate the Bayesian predictive negative log score

$$
\mathrm{NLS}(\mathcal Q)=-\frac1{1024}\sum_{i=1}^{1024}
\log E_{\theta\sim\mathcal Q}[p_\theta(y_i\mid x_i)],
\qquad \mathcal Q\in\{\gamma,\pi\}.
$$

Average probabilities before taking the logarithm. This differs from averaging each draw's negative log likelihood. Compute with stable log-sum-exp of log-probabilities, not probability clipping. Save the predicted probability matrix or sufficient bootstrap inputs. Plot improvement $\mathrm{NLS}(\gamma)-\mathrm{NLS}(\pi)$ versus width, with zero reference. Positive means the posterior predicts the held-out labels better than the prior mixture.

Use iid resampling for the prior MC contribution and chain-aware blocks for the posterior contribution; keep dataset variation separate. Accuracy and teacher predictive log score may be reported in the companion table, but no extra accuracy figure is required. Do not redraw data when improvement is small or negative.

## 13.3 Appendix S2(b): static PI ratios at no additional sampling cost

Using the same nine substantive probes and gradient values as Experiment 2, calculate for the **unrestricted posterior in both architectures**

$$
Q_g=\frac{\operatorname{Var}_\pi(g)}{E_\pi\|\nabla g\|^2}\le C_P(\pi).
$$

Use all selected states, including deep states outside $G_a$. Compute variance as the empirical second central moment with denominator N; compute the denominator as the arithmetic mean of the saved squared gradient norms. Recompute both in each bootstrap resample. Standardization cancels from the ratio. Select and evaluate a family maximum using the two independent-chain folds exactly as for entropy, with block-bootstrap uncertainty. No tilt optimization is involved. Apply the same 15% relative MCSE gate and reference-validity requirement.

Plot the three substantive family diagnostics, divided by $\sigma^2$, versus width in small shallow/deep subpanels. This is a static check against sampler-dependent explanations of Figure 1. A decreasing Q is informative; a small Q is not an upper bound on $C_P$. If dynamics fail but these ratios are resolved, report the static evidence without using it to replace the missing relaxation claim.

# 14. Exact figure construction and caption instructions

Figures must be generated exclusively from validated tables, not by reloading samples and performing undocumented calculations inside plotting code. Export each figure as vector PDF, 300-dpi PNG, and its exact plotted CSV/JSON sidecar. Use a white background, consistent fonts, colorblind-safe colors, no 3D, no smoothing spline, and no omitted adverse replicate. Use symbols as well as color.

Keep logarithmic x axes with ticks at the actual widths. Join only measurements from the same architecture, target law, observable family, and replicate. Do not join across missing or invalid cells. Error bars denote Monte Carlo uncertainty unless the caption explicitly says otherwise. Thick lines are medians of the three replicate estimates; they do not have a pooled-draw confidence interval.

A point for an unresolved estimate is open/hollow and its status is listed in the companion table. Main text must not use it as confirmatory evidence. A whole failed panel should remain visible with the sentence `Numerical validity criteria not met within the prescribed budget`, rather than being replaced by an attractive old experiment.

## 14.1 Main Figure 1: relaxation versus width

Layout: two side-by-side panels, shallow full posterior and deep full posterior. X axis: `Width m (log scale)`. Y axis: `Held-out family relaxation diagnostic / sigma^2`. Use a log y axis if the full range spans more than a factor of ten; otherwise a linear y axis including zero. Use the same y scale for both panels whenever readable; if scales differ, state it explicitly. Four colors identify V, training logits, held-out probabilities, and interactions. Gaussian reference is a thin horizontal dashed line at one.

Show all three replicate points lightly and the across-replicate median prominently. Put h and the retained duration range in the caption or a small panel note. Do not label the y axis `PI constant`, `spectral gap`, or `mixing time` without the observable qualification.

**Caption template (replace bracketed fields with computed facts):**

> Width dependence of observable relaxation under unrestricted posterior dynamics. The horizontal axes give network width for the one-hidden-layer and two-hidden-layer models, at fixed n=128. Vertical values are cross-evaluated family autocorrelation integrals in the physical-time convention of the adjusted Gaussian-preserving Langevin proposal, normalized by the prior variance. Families contain the likelihood potential, eight training logits, eight held-out probabilities, and four fixed head--hidden interactions. Thin points and lines show three independent data/center replicates; the thick line is their median, and error bars show within-target Monte Carlo uncertainty. The horizontal line at one is the continuous Gaussian linear-observable timescale. The production steps were [values]; endpoint step-halving checks [passed/status]. Over the tested width range, [state measured decrease, stability, increase, or unresolved behavior]. These finite-observable diagnostics do not estimate the optimal PI coefficient, and no global head or hidden-layer restriction was imposed on the chains.

## 14.2 Main Figure 2: entropy--Fisher tradeoff

Layout: shallow full posterior on the left, deep posterior conditioned on $G_{2.5}$ on the right. X axis: width on log scale. Y axis: `Held-out entropy-Fisher diagnostic / sigma^2`. Three family colors: loss, training logits, interactions. Gaussian linear-tilt reference is one. Keep the linear controls out of the main maximum and main legend.

Use a linear y axis starting at zero unless a factor-of-ten range requires log scale. If a ratio is indistinguishable from zero, retain its linear-scale value; do not apply an arbitrary floor to enable a log plot. Add an explicit target-law label inside each panel. A deep panel labeled merely `posterior LSI` is insufficient.

**Caption template:**

> Finite-probe entropy--Fisher diagnostics for the final LSI domains. For each standardized smooth probe f and fixed tilt t in {-1,-0.5,0.5,1}, we form the normalized density r proportional to exp(tf) relative to the shallow posterior or the deep posterior conditioned on the later-hidden-layer spectral event. The vertical axis reports held-out family estimates of 2 Ent(r)/E[r||grad log r||^2], divided by the prior variance; the horizontal axis is width. The head and first hidden layer remain unrestricted in both panels. Probe selection and evaluation use separate chain folds. The value-one line is attained by Gaussian linear tilts and is a reference, not a required lower limit for these probes. [Describe measured trend and incomplete candidates.] LSI bounds these population ratios above by its coefficient, so the plotted tests are finite-probe lower diagnostics of that coefficient; they neither determine its optimum nor establish global deep LSI.

## 14.3 Main Figure 3: scope of the final deep LSI domain

Left panel: x=width, y=`Observed posterior fraction in G_2.5`, with fixed y range [0,1.02]. Show per-replicate points and exact exit counts in a small annotation or adjacent table. If all observations are inside, the caption says so without a zero-width interval. Do not use an expanded axis near one to suggest a resolved rare-event trend.

Right panel: x=width, y=`||W2||op / (2.5 sqrt(m))`. Plot posterior median, q95, and q99; use distinct line styles and all replicate markers. Add the matching prior quantiles as thin gray lines with the same styles. Boundary S=1 is dashed. If desired, annotate the Gaussian-matrix reference 0.8 as a light dotted line labeled `prior large-width reference`; it is not a fitted posterior limit. The y axis must include one and all displayed uncertainty ranges.

**Caption template:**

> Posterior relevance of the final deep spectral domain. Left: empirical fractions of archived draws from the unrestricted two-hidden-layer posterior satisfying ||W2||op/sqrt(m) <= 2.5, at fixed n=128. [N] states were inspected, with [exit counts] observed exits; counts retain chain identity and are not treated as iid rare-event trials. Right: the median, 95th, and 99th percentiles of the normalized spectral norm, with matching iid Gaussian-prior reference quantiles. The boundary is one. [Describe actual posterior/prior agreement and margins.] This assesses whether the conditional LSI domain contains a substantial part of the sampled posterior. It does not estimate the theorem's exponential exit-rate constants, and high observed mass does not turn conditional deep LSI into a global statement.

## 14.4 Scaling statements and presentation limits

Do not add fitted exponent labels, an $m^{-1/3}$ guide to Figure 1, or $n^2/n^3$ phase boundaries. Do not claim dimensional independence from these fixed-n sweeps. If any descriptive slope is computed for internal inspection, place it only in the raw analysis export and label it descriptive; it is not a headline result.

No conclusion should rely on the old two-layer negative-Hessian plot to stand in for the new deep results. The deep proof uses integrated control and cutoff removal, and does not require uniform positive curvature everywhere. No additional Hessian figure is requested in this campaign.

# 15. Execution order and command contract

Once the implementation exists, the operator should be able to run the following sequence. The commands are required interfaces to implement, not claims that this document ships their source code.

```bash
python -m venv .venv
source .venv/bin/activate
mkdir -p results
# Install the official PyTorch wheel matching this machine first.
python -m pip install numpy scipy h5py pandas pyyaml matplotlib pytest arviz
python -m pip install -e .
python -m pip freeze > results/environment.freeze.txt

python -m bnn_geometry validate-config --config configs/campaign.yaml
python -m pytest -q tests/unit
python -m pytest -q tests/integration
python -m bnn_geometry fixture --out results/fixture
python -m bnn_geometry benchmark --config configs/campaign.yaml
python -m bnn_geometry plan --config configs/campaign.yaml --dry-run
python -m bnn_geometry freeze --config configs/campaign.yaml
python -m bnn_geometry run --config configs/campaign.yaml --resume
python -m bnn_geometry analyze --config configs/campaign.yaml
python -m bnn_geometry figures --config configs/campaign.yaml
python -m bnn_geometry audit --config configs/campaign.yaml --strict
python -m bnn_geometry export --config configs/campaign.yaml
```

The build process must create `results/` before the environment export. The `freeze` command writes the immutable manifest, validates all hashes, and records the successful test reports. It is not an interactive approval prompt. Installing a package version after freeze is an environment change requiring a new recorded execution identity.

## 15.1 Runner state machine

The `run` command must implement this order:

1. Generate/freeze all three datasets, center banks, targets, and probe specifications. Verify 24 unique target IDs.
2. Run all reference targets through the prescribed stages until each passes or reaches its cap. Compute archive-based spectral and entropy checks at the candidate stopping stages.
3. Run dynamics step calibration on the existing reference states. Select the common step per architecture.
4. Run production dynamics for all targets, extending physical duration only under the fixed criteria.
5. Run endpoint half-step validation. If needed, perform the one allowed architecture-wide refinement and final endpoint checks.
6. Generate prior spectral and prior predictive controls, and complete all static analyses from the saved reference pools.
7. Write status tables for every target and every experiment, including unresolved ones.
8. Generate figures and captions from those tables, then perform the strict export audit.

Hardware failures are retried by resuming the exact checkpoint and execution identity, up to three job restarts per stage. A deterministic numerical exception is not retried with a different seed. Fixing an identified code bug is allowed, but the affected outputs must be invalidated and rerun from the appropriate earlier stage with a new code hash. Document the bug and the affected targets. Never combine pre-fix and post-fix results without demonstrating that the fix cannot change the former.

## 15.2 Parallel scheduling

Independent targets and chains may run concurrently if memory and I/O allow. One process owns each chain output. Fix quantile estimation throughout to linear interpolation between sorted order statistics (NumPy method="linear"), and record this convention in the analysis metadata. Start with one target's four chains per accelerator; vectorize independent chains only after the fixture verifies exact per-chain randomness and likelihood separation. Increase concurrent jobs based on the hardware benchmark, not on whether a target appears easy.

Do not compare wall-clock times from shared and exclusive accelerator jobs as though they were the same hardware allocation. Record concurrency, allocated device, and CPU thread count with each timing. The paper's primary time axis is physical time; practical cost is a separately labeled table of gradient evaluations and measured wall-clock time.

## 15.3 Counts and limits to display before launch

Initial reference production: $24\times4\times8192=786{,}432$ retained transitions. At the maximum reference stage: $24\times4\times65{,}536=6{,}291{,}456$. These counts exclude burn-in, calibration, and the separation segment. Their inclusion brings total initial reference transitions to 1,474,560 and maximum reference transitions to 6,979,584. Likelihood evaluations exceed transitions because ellipse proposals can be rejected within each transition.

Initial dynamics work at h=0.02 and T=512 is 25,600 retained transitions per chain, or 2,457,600 across all 24 targets and four chains. Discarded dynamics time and calibration are additional. Endpoint checks at h/2 and equal physical duration add 2,457,600 retained transitions at these initial durations. A smaller selected h increases counts proportionally. These are workload examples, not a runtime promise or a prescription to stop before a required gate.

Maximum reference likelihood work is 1.5 million evaluations per chain. Maximum total dynamics work is 8 million candidate gradient evaluations per scientific target. Maximum selected static archive is 8,192 states per target. Maximum dynamics refinement is one architecture-wide halving. These caps bound automatic escalation; they do not ensure every target passes.

# 16. Required schemas and complete output tree

A reader must be able to reconstruct every plotted number without asking which run produced it. Use this minimum result structure:

```text
results/final_geometry/
  campaign.yaml
  manifest.json
  environment.freeze.txt
  environment.json
  source_revision.txt
  tests/unit_report.xml
  tests/integration_report.xml
  tests/calibration_results.csv
  benchmark.csv
  data/rep_0.npz                    # also rep_1, rep_2
  centers/rep_0.npz                 # also rep_1, rep_2
  targets/<target_id>/spec.json
  targets/<target_id>/calibration.json
  targets/<target_id>/reference/chain_0.h5
  targets/<target_id>/dynamics/<step_id>/chain_0.h5
  targets/<target_id>/status.json
  controls/prior_spectral.csv
  controls/prior_predictive.h5
  tables/target_audit.csv
  tables/reference_diagnostics.csv
  tables/dynamics_diagnostics.csv
  tables/relaxation_probes.csv
  tables/relaxation_families.csv
  tables/entropy_probes.csv
  tables/entropy_families.csv
  tables/static_pi.csv
  tables/spectral_states.csv
  tables/spectral_summary.csv
  tables/predictive_scores.csv
  tables/cost_and_counts.csv
  tables/exclusions.csv
  figures/main_1_relaxation.pdf      # plus PNG, CSV, JSON
  figures/main_2_entropy.pdf        # plus PNG, CSV, JSON
  figures/main_3_spectral.pdf       # plus PNG, CSV, JSON
  figures/appendix_s1_validation.pdf
  figures/appendix_s2_static_checks.pdf
  captions.tex
  SUMMARY.md
  audit.json
  checksums.sha256
```

Repeat chain_0 files for chains 1--3. Include all required stage files or a manifest mapping each logical stage to an HDF5 group. Do not move outputs by hand after export without updating checksums.

## 16.1 Common identifier columns

Every measurement row contains `campaign_id, target_id, target_hash, execution_hash, replicate, architecture, L, n, d, m, p, sigma, law, stage, code_revision`. Chain-specific rows additionally contain `chain_id, first_draw, last_draw, saved_stride`. Deep conditional rows include `a=2.5` and their exact event mask/source hash. A column called `samples` is prohibited without specifying whether it counts transitions, inspected archived states, inside states, or effective sample size.

Every estimate contains `estimate, mcse, mc_low, mc_high, n_draws_used, validity, failure_reason`. Use null with an explanatory status for undefined uncertainty, not zero. For numerical brackets use separate columns; never reuse a confidence-interval column for a numerical approximation interval.

## 16.2 Figure sidecar

Each figure JSON lists source table filenames and hashes, filtering rules, y-axis normalization, uncertainty definition, target-law labels, selected probe IDs, all omitted/flagged rows, plotting-code revision, and the complete generated caption. The plotted CSV is exactly the subset used in the figure, including status flags.

The caption generator fills bracketed fields from tables and fails if a placeholder remains. Handwritten prose about trends must be checked against the widest/narrowest ratios and validity statuses. A figure audit must detect a deep conditional quantity mislabeled as unconditional.

## 16.3 Summary to return with the figures

`SUMMARY.md` should be no more than three pages and contain:

1. Completion counts: 24 intended targets, counts passing each reference, dynamics, entropy, and spectral gate.
2. Exact widths, n, sigma, head-center normalization, and final conditional domain.
3. A factual sentence for each main figure, including flat/adverse/unresolved outcomes.
4. Widest/narrowest family ratios by replicate, with uncertainty and target law.
5. Exact spectral inspected-state/exit counts and posterior/prior quantile comparisons.
6. Numerical-check outcomes, any step refinement, budget extensions, and code fixes.
7. Posterior-versus-prior predictive log-score change.
8. Total compute and storage, and links to diagnostic/source tables.

The figures should carry enough labels and caption information to stand alone, but this short summary prevents interpretation from depending on invisible implementation details.

# 17. Status rules: no follow-up decisions needed

| Situation at the prescribed cap | Required action | Permitted paper statement |
|---|---|---|
| Reference diagnostics fail | Retain data; mark dependent experiments unresolved | Finite-run exploratory behavior only |
| Dynamics passes but width curve is flat | Plot all values and widest/narrowest ratios | Stability over tested widths; no measured acceleration |
| Dynamics changes under final step check | Keep diagnostic comparison; flag main dynamics panel | Discrete-kernel results only; diffusion interpretation unresolved |
| Entropy family includes an unresolved candidate | Show open family point; identify candidate | Resolved individual probes only; family maximum incomplete |
| All valid entropy probes are below one | Present them without adding controls to the maximum | Tested tilts have favorable entropy--Fisher tradeoffs |
| Spectral occupancy is one everywhere | Report zero exits and S quantiles/prior reference | High observed occupancy; rare exit probability unresolved |
| Spectral occupancy is lower than expected | Show the measured fraction and margins | Conditional domain has the measured empirical scope |
| Deep conditional pool is too small | Flag entropy estimates; retain global PI/static evidence | Deep conditional LSI diagnostic unresolved |
| Prior/posterior predictions barely differ | Report small or absent log-score gain | Geometry measured in a weakly updated predictive regime |
| One replicate is adverse but valid | Keep it in all plots and medians | Heterogeneity across the three realizations |
| Numerical precision or resource cap prevents completion | Export statuses, partial tables, and budget accounting | The planned result was not established within this campaign |

No row authorizes another scientific experiment. If the completed campaign is mixed, the paper should make a narrower empirical claim. This is preferable to a sequence of unreported changes to the target or selection of only favorable runs.

# 18. Final audit checklist

The audit command must fail publication export if any of these invariants is violated:

- Exactly eight intended settings and 24 target IDs are represented, including failed targets.
- Every target has the correct summed loss, centered-logit factor, prior variance, center norm, parameter count, and data/center pairing.
- No production sample is reused as a calibration observation or silently treated as a prior sample.
- Reference and dynamics chain boundaries and draw indices are intact.
- Reported retained counts equal actual array lengths; inspection counts equal actual SVD rows.
- Scalar ESS is not confused with saved parameter-state count or importance-weight concentration.
- Every deep Figure 2 point uses the specified final spectral event, with head and W1 unrestricted.
- Every Figure 1 point uses unrestricted dynamics and the raw-time IAT convention with the factor one half.
- Every main result has an experiment-specific validity status, not just a global green check for the target.
- Step comparisons use equal physical durations and record every rejected transition.
- No plot quietly replaces a failed candidate, replicate, or width.
- No zero-exit row carries a zero uncertainty claim or an iid rare-event bound.
- Static and dynamic ratios are described as finite-observable diagnostics, not optimal constants.
- All five figure PDFs/PNGs, exact source tables, captions, software identity, and checksums are present.
- Every caption is supported by its figure/table data and contains no placeholder text.
- The final package includes a deterministic command that regenerates all figures from exported tables without resampling.

Publication status and archival completeness are different. An incomplete scientific result can still have a complete and honest export. `audit --strict` should return a machine-readable archival pass plus a separate publication-readiness status; it must not force an implementer to delete failed rows in order to export the study.

# 19. Why these choices are fixed

**Eight settings rather than a broad Cartesian sweep.** Four widths per architecture give three width contrasts while keeping the deep matrix cost bounded. Three matched replicates reveal whether a trend is driven by one realization. This study sacrifices identification of joint n-scaling and robustness to many datasets; those limitations are explicit.

**n=128 and nonorthogonal inputs.** This is a larger sample than the original n=32 pilot and avoids special orthogonal-row simplifications. More samples are not a substitute for independent replicates. The fixed-n design matches the scope of the deep global PI width result.

**Corrected prior means and fixed sigma.** Width-independent head-center operator norm is an assumption of the quantitative slides. Altering variance to compensate would change the posterior and its Gaussian benchmark. The protocol therefore changes only the means required for alignment and freezes the variance across all widths.

**Reference sampler plus dynamics sampler.** The reference pool supports posterior expectations without assigning ellipse iterations an invented Langevin time. The adjusted Langevin proposal supplies trajectories close to the theorem's dynamics at small h. They answer different measurement needs; they are not a sampler leaderboard.

**Entropy--Fisher probes rather than density estimation.** The variational form tests an LSI-relevant quantity using expectations and gradients. It is feasible in high parameter dimension and has a clear inferential limitation. A high-dimensional histogram or learned density would introduce an additional unvalidated estimation problem.

**Final spectral mass rather than old head coverage.** The current deep local LSI conditions only later hidden matrices. Measuring a head cutoff would not quantify the mass of that final domain. The paired norm-quantile plot keeps all-inside outcomes informative and exposes prior-driven coverage.

**No forced favorable scaling curve.** Sufficient inequality bounds may be loose, and fixed probe families may miss the worst direction. The section is strongest when it distinguishes what was measured from what the theorem guarantees. The result can support improvement, stability, or a limitation without changing the protocol.

# 20. Sources and provenance

[T] User-supplied theory slides, *Width Improves BNN Posterior Geometry: Cylinder-based global PI with sharper LSI bounds*, version (15), 12 slides. Model and conventions: slide 3. Deep unrestricted PI: slides 5--8. Shallow global LSI: slide 9. Deep conditional LSI and mass: slide 10. These are the statement-level source for this protocol.

[R] User-supplied snapshot of `mjhajharia/sin-geoopt`, especially `EXPERIMENT_ALIGNMENT_AUDIT.md`, `checkpoint_validation.py`, `PLOTS.md`, and the experiment scripts. Reused here as an engineering reference for metadata, chain identity, step comparisons, and artifact provenance. Its scientific target differs from this one. Its historical test reports are not test results for the new implementation.

[1] Murray, I., Adams, R. P., and MacKay, D. J. C. (2010). *Elliptical slice sampling*. AISTATS, PMLR 9, 541--548. Original Gaussian-prior slice method. <https://proceedings.mlr.press/v9/murray10a.html>

[2] Cotter, S. L., Roberts, G. O., Stuart, A. M., and White, D. (2013). *MCMC Methods for Functions: Modifying Old Algorithms to Make Them Faster*. Statistical Science 28(3), 424--446. Gaussian-reference-preserving proposals and pCNL. The step parameterization and stable Metropolis-ratio algebra in this runbook are written explicitly to fix implementation conventions. <https://arxiv.org/abs/1202.0709>

[3] Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Bürkner, P.-C. (2021). *Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC*. Bayesian Analysis 16(2), 667--718. Motivation for monitoring rank, scale, and tail diagnostics. The numerical pass thresholds and continuation caps here are design choices. <https://arxiv.org/abs/1903.08008>

[4] Vempala, S. S., and Wibisono, A. (2019). *Rapid Convergence of the Unadjusted Langevin Algorithm: Isoperimetry Suffices*. NeurIPS. See Section 2 for entropy, Fisher information, and LSI conventions, and the distinction between continuous and discrete dynamics. This protocol does not assume their separate ULA smoothness hypotheses hold for the BNN or apply their ULA theorem to the adjusted proposal. <https://www.cs.yale.edu/homes/wibisono/VW19.pdf>

[5] Flegal, J. M., and Jones, G. L. (2010). *Batch means and spectral variance estimators in Markov chain Monte Carlo*. Annals of Statistics 38(2), 1034--1070. Background for dependence-aware Monte Carlo variance estimation; finite-run diagnostic gates do not prove its asymptotic assumptions. <https://arxiv.org/abs/0811.1729>

[D1] PyTorch official installation selector. Choose the wheel for the actual execution platform, then freeze and test the resulting environment. <https://pytorch.org/get-started/locally/>

[D2] ArviZ official documentation, or the maintained statistics component to which it redirects. Check the installed API for rank R-hat, bulk/tail ESS, and raw-mean ESS; validate conventions against the known-distribution tests. <https://python.arviz.org/>

All experimental thresholds, sample counts, probe families, and resource limits in this document are prespecified engineering and design decisions. The citations justify the methods and inferential distinctions; they do not guarantee that these particular settings will produce a positive trend or pass every numerical gate.
