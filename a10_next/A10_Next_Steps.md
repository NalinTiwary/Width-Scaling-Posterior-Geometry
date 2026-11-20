# A10 follow-up experiments

**Runbook for the revision of *Width Improves Posterior Sampling in Bayesian Neural Networks***  
Nalin Tiwary · November 2026 · implementation pinned to `308a9e5`

The fixed-step cost question already has an answer. Across the three datasets, widening the deep network from 32 to 256 improved archived training-loss bulk ESS per A10 second by **1.59–1.85×**. Widening the shallow net from 64 to 4,096 gave **0.71–0.79×**. Every one of the eight held-out predictions has the same endpoint direction within each architecture and dataset. These are comparisons at one step size, under four-chain batching, with timing measured separately from the long trajectories.

**What I would put in the paper.** Lead with the fixed-step cost decomposition. A second main figure should show mean loss from independent prior starts against an independently estimated posterior mean, even if the experiment reaches its precision limit. Give the controlled input-geometry comparison a figure or appendix spread with the certificate and empirical measurements shown separately. Include ReLU and the tuned comparison in the reviewer response. Move the broad perturbation grid to the appendix.

| Order | Work | What becomes reportable |
|---|---|---|
| Now | Use the frozen exports and deterministic bound code. | Per-transition efficiency, cost per four-chain step, replay ESS/s, the preliminary tuning curve, and the certificate grid. |
| 1 | Inventory saved cluster scalars and recover the prior-start histories. | Acceptance and prediction diagnostics by tuning candidate, prior-start bias, and descriptive robustness diagnostics. |
| 2 | Time three finalist steps, freeze four selections, confirm each against 0.01 using fresh trajectories. | Whether the deep and shallow endpoint conclusions survive step selection under a common efficiency objective. |
| 3 | Qualify the six geometry pilots and the ReLU pair using independent references. Expand the matched designs. | An observable width-by-geometry interaction, if present, and a replicated activation comparison. |
| Optional | Add same-target prior discrepancy, broader variants and slow-probe summaries. | Which aspects of equilibrium geometry or robustness explain the observations within this design. |

**What is in the companion.** The ZIP contains the editable runbook, four current figure PDFs, their plot scripts, the supplied A10 export, frozen target identities, the independent reference values, and cluster scripts. Run all commands from the experiment repository root after extracting the ZIP there. `a10_next/` is the companion folder. New sampling writes to `results/revision_next/`. No new cluster results are assumed anywhere below.

# 1. Figures you can use today

![Fixed-step cost decomposition](figures/cost_decomposition.pdf)

**Suggested caption.** Each line follows one dataset at $h/\sigma^2=0.01$. Top: pooled loss bulk ESS per 1,000 individual chain-transitions. Middle: milliseconds per joint step advancing all four chains on one NVIDIA A10. Bottom: archived loss bulk ESS divided by 102,400 times the corresponding A10 joint-step duration. All four widths are shown. The point at width 256 in the deep panel does not imply it is the fastest width for every probe.

For a four-chain window of $N$ steps, use $1000E/(4N)$ on the top axis and $E/(N c)$ on the bottom, where $c$ is seconds per joint step. With raw-mean ESS $E_\mu$, the associated integrated autocorrelation estimate is $4N/E_\mu$ steps. Rank bulk ESS uses transformed draws. Use raw-mean ESS when labeling an axis as a physical IAT. Example: short-run `deep_m0256_r0` gives $410.26/26.08=15.73$ bulk ESS/s. The stored 3.886 uses a different ESS and counts the batch four times.

# 2. Predictions, robustness of the timing, and step sensitivity

| Wide / narrow replay bulk ESS/s | Dataset 0 | Dataset 1 | Dataset 2 |
|---|---:|---:|---:|
| Deep, loss | 1.85 | 1.59 | 1.65 |
| Deep, median of eight paired prediction ratios | 1.26 | 1.27 | 1.24 |
| Shallow, loss | 0.79 | 0.72 | 0.71 |
| Shallow, median of eight paired prediction ratios | 0.63 | 0.68 | 0.63 |

Use `fixed_efficiency.pdf` for the full prediction curves. Its prediction points are medians of eight absolute rates at each width. The table above takes the median **after** dividing at matched input and dataset. Keep these two summaries identifiable. For the separate interleaved deep dataset-0 pilot, $2.0684/1.135=1.822$. The five block-matched efficiency ratios span 1.818–1.826. Combining the extreme block costs independently gives the wider 1.813–1.830 sensitivity envelope. Both calculations hold the archived ESS fixed. Neither range quantifies ESS uncertainty across datasets.

![Step-size pilot](figures/step_sensitivity.pdf)

**Suggested caption.** Each curve uses 2,048 post-warmup transitions per chain. Open rings show the selected step under the stored loss-only rule. Nine of twelve selections reach 0.16, the largest tested step. Ten of twelve raw maxima occur there. The difference comes from the 5% tie rule. This figure motivates longer finalists and does not identify the best step per second.

Use the 24 short fixed-step jobs as a diagnostic table. Include acceptance, rank $\widehat R$, bulk/tail ESS, kernel cost, and actual sampler elapsed time. Nine loss $\widehat R$ values exceed 1.01. These runs started at the prior **center**. Keep the archived reference-initialized window as the source of the main efficiency figure. High acceptance alone supplies neither stationarity nor a step-size recommendation.

# 3. Start on the cluster with trace recovery

The project already records loss, held-out probabilities, and parameter probes at stride one. Tuning summaries dropped all but one score. Prior-start summaries retained only timing, acceptance and length. Read the underlying HDF5 before scheduling replacement trajectories. The original 68 scheduled runs and the new analysis have distinct output roots.

```bash
cd "$(git rev-parse --show-toplevel)"
unzip A10_Next_Steps_Kit.zip
source scripts/revision/common.sh
mkdir -p logs/revision results/revision_next
git rev-parse --short=7 HEAD
python3 -m bnn_geometry.revision preflight --config configs/revision.yaml
python3 a10_next/plot_current.py
python3 a10_next/recover.py
python3 a10_next/cold_plot.py
python3 -m bnn_geometry.revision prediction-acf \
  --config configs/revision.yaml --device cpu
```

**Where to look.** Paths below start at `results/revision/runs/<cell>/`. Replace the config path if the campaign used a different root. The supplied adapters target commit `308a9e5`; compare local changes to the runner or schema before using them on another checkout.

| Stage | Example suffix | What to extract |
|---|---|---|
| Prior starts | `prior-starts/batch_0/chain_0/from_prior.h5` | 32 global chain IDs, batches 0, 4, …, 28. Retain complete independent start histories. |
| Tuning | `tune/pilot_0.16/chain_0/pilot.h5` | Acceptance and per-probe ESS after the configured discard, four chains per candidate. |
| Fixed or changed target | `relu/fixed/chain_0/production.h5` | 8,192 steps from the center. The adapter exports the original window and a common nominal warmup sensitivity. |

Inspect `recovery/inventory.csv` under the new root. It gives an explicit reason per missing or invalid group. At full coverage expect 12 complete prior-start targets, 72 tuning target/step groups, 24 fixed-step targets, six geometry pilots and two per completed robustness stage. Retain accept/reject repeats. For the raw prediction audit inspect the JSON status; the project command can exit successfully when raw states are absent. Accept the replication only for `status=ok` (ACF tolerance $10^{-8}$, bulk ESS tolerance $10^{-6}$). Restore the original stages and unrounded prediction export if absent. The archived per-probe summary remains identifiable as an export in the meantime.

**If files are missing.** Recover the named HDF5, original dataset/center banks and target identity files from cluster storage first. If a prior-start cell truly needs regeneration, copy `configs/revision.yaml`, change only `output_root` to a new path, list the `prior-starts` tasks with the project CLI, and submit only the missing IDs with `scripts/revision/target_stage.sbatch`, `STAGE=prior-starts` and the new `CONFIG`. Run recovery on that config into a separate recovery folder. Merge only whole, validated cells by cell, chain and transition. Preserve the master seed.

# 4. Measure convergence from the prior

**Question for iKyw Q3/W3.** At matched step and width, after how many transitions does mean training loss agree with its independent posterior reference from a fresh prior start? The reference CSV already supplies mean, SD and mean MCSE for loss, eight probabilities, $q_A$ and $q_W$ on all 12 baseline targets. The full reference is four elliptical-slice chains with 65,536 production observations each. Original and revision target IDs use different hash schemas. Verify $X,y$, held-out inputs, center bank, dimensions, activation and prior across schemas. Within a trajectory enforce its recorded hash.

For each of the 32 independent starts per cell extract $t=1,2,4,\ldots,8192$. The sampler advances and then writes, so transition $t$ is HDF5 row $t-1$. Reconstruct $t=0$ with the exact original RNG and device. The provided GPU job replays one proposal and checks the first saved acceptance flag and all eleven scalar values before accepting each reconstruction.

```bash
sbatch --array=0-11 --export=ALL,JOB=cold-zero a10_next/next.sbatch
# After the array finishes and original reference files are mounted:
python3 a10_next/postprocess.py cold-band
python3 a10_next/plot_distribution.py cold-band
```

The immediate `recovery/prior_start_loss.pdf` shows $[\bar V_t-\bar V_{\rm ref}]/s_{\rm ref}$, one panel per architecture/dataset and one line per endpoint width, starting at transition one. Its approximate pointwise bands combine independent-start standard errors and reference mean MCSE, treating the reference SD as fixed. The formal table `cold-band.csv` uses 2,000 resamples of entire start histories and independent blocks within reference chains, recomputes the reference SD, and uses the 95th percentile of the maximum absolute error over the checkpoint grid. Reference stages `production_s1.h5` through `production_s4.h5` are concatenated within chain. The adapter verifies their pooled mean and SD against the supplied reference.

**Decision and plot.** Draw the signed mean with simultaneous bands and horizontal lines at ±0.25 posterior SD. Mark the first of **three consecutive checkpoints** whose complete intervals lie inside the margin. Otherwise show a right-censoring arrow at the last checkpoint. Bands cover times within one target. Give the primary decision for loss and label the other observables exploratory. If time zero is unavailable, report the observed grid beginning at one. Relabel $t c$ as estimated batched kernel time if a seconds axis is useful. The original logs give total batch time only.

**When to buy additional runs.** Near equilibrium, 32 independent starts give a nominal pointwise half-width around $1.96/\sqrt{32}=0.35$ reference SD before reference error. A narrow residual with a wide band calls for **more independent starts**. Target 128, then 256 if required, retaining new global chain IDs. If residuals remain clearly displaced at 8,192, prioritize a longer horizon to 32,768. To extend existing paths, open the original checkpoint and append a newly named 24,576-step segment in a separate work copy. Each new segment resets its local draw ID; offset it by 8,192. Increasing the requested length under an already completed segment name returns the completed segment. For a larger ensemble at the old horizon, the stock prior-start stage supports `prior_starts.n_starts`; use a work copy of the original root and preserve the original streams. The supplied recovery adapter is for the completed 32-start pilot: extend its chain range and checkpoint grid before analyzing an expansion.

# 5. Time finalists and confirm the selected steps

**Question for Wj1c Q1 and iKyw Q2.** Does widening still help loss and predictions if each endpoint uses a step chosen by the same rule? Use shallow 64/4,096 and deep 32/256 on each of the three frozen datasets. Keep float64, four chains in one batch, scalar logging and an A10. The new scripts call the existing target builder, PCNL, RNG, trajectory/checkpoint and timing code. The stock `tune` command uses a loss-per-transition score; changing `finalist_transitions` alone does not launch a timed confirmation.

1. **Screen.** Run $h/\sigma^2\in\{0.08,0.16,0.32\}$. One array task runs all three steps on one target. Use matched independent prior draws across steps within target, separate proposal streams, discard $\lceil 20/(h/\sigma^2)\rceil$ transitions and retain 8,192. Time five synchronized blocks starting at 512 four-chain steps, doubling until every block lasts at least one second. The disposable timing stream leaves the production chain untouched.
2. **Choose.** For each target/step take the smaller of loss raw-mean ESS/s and the median of eight held-out raw-mean ESS/s values. For each architecture/width take the median of those scores across the three datasets. Choose the smallest tested step within 5% of the best. Store all 36 scores and freeze four winners. A top-boundary winner is the best tested under this rule. Report stickiness, drift, and acceptance by candidate. Investigate numerical failures before selection.
3. **Confirm.** For each of 12 targets, compare the frozen step to 0.01 on fresh RNG streams, using the same initial prior draws within each pair and a separate burn-in segment. This produces 24 trajectories. Report per-replicate cost, raw/bulk ESS per second, loss and prediction MCSE, acceptance, and warmup and tuning costs. This new comparison has its own initialization and window. The archived fixed-step efficiency figure remains a separate analysis.

```bash
python3 a10_next/cluster.py list --scope endpoint
sbatch --array=6 --export=ALL,JOB=finalists a10_next/next.sbatch
# Inspect deep_m0032_r0, then run the remaining eleven cells:
sbatch --array=0-5,7-11 --export=ALL,JOB=finalists a10_next/next.sbatch
# Wait for complete finalist tables before choosing and confirming:
python3 a10_next/choose.py
sbatch --array=0-11 --export=ALL,JOB=confirm a10_next/next.sbatch
# After confirmation completes:
python3 a10_next/postprocess.py gates --scope endpoint
python3 a10_next/plot_next.py endpoint
```

Start at 8,192 retained steps per chain. For unresolved pairs rerun the same task IDs with `LEVEL=2` in the export, then levels 3, 4 or 5 if needed. Cumulative retained counts are 16,384, 32,768, 65,536 and 131,072. Each level appends a named segment. Compare paired endpoints at the same retained length as a sensitivity. If $\widehat R$, ESS, drift or reference agreement remains unresolved at the cap, preserve the open point and give the attained window. The two-hour job limit is an initial allocation; price larger blocks on the actual A10 first.

# 6. Separate the geometry certificate from the posterior experiment

**Question for iKyw Q4/W2.** Does the explicit sufficient width depend meaningfully on the measured input geometry, and what do the tested posterior observables do on matched data? Use $n=16,d=32$, 1,024 held-out inputs, widths 64, 256, 1,024 and 4,096, and datasets 0–2. Within each dataset the generator reuses latent examples, teacher scores, Bernoulli uniforms, labels and center bank. With $r=0,0.2,0.8$, the row Gram matrix is $(1-r)I+r ss^T$ and $\chi=\|X\|_{\rm op}^2=1+15r$ takes values 1, 4 and 13.

![Deterministic certificate](figures/geometry_certificate.pdf)

**Caption and readout.** Left: the shallow curvature quantity $\rho$. Right: the resulting upper bound relative to the Gaussian prior constant wherever $\rho>0$. Equation (72) gives sufficient widths of 260, 1,040 and 10,816. At $\chi=4,m=1024$, $\rho=0.496$ already yields a finite bound despite narrowly missing 1,040. At width 4,096 the upper bounds are approximately $1.18,1.51,3.49$ times the prior constant. These are deterministic calculations for the controlled design. No posterior simulation enters this figure.

```bash
python3 a10_next/postprocess.py geometry
python3 a10_next/plot_certificate.py
python3 a10_next/cluster.py list --scope geometry
GEO=0,9,12,21,24,33
sbatch --array="$GEO" --export=ALL,JOB=reuse,SCOPE=geometry a10_next/next.sbatch
sbatch --array="$GEO" --export=ALL,JOB=reference,SCOPE=geometry a10_next/next.sbatch
```

The six IDs are the already-sampled dataset-0 endpoints at all three $\chi$. `reuse` reads their trajectories and measures only the missing four-chain timing. It discards 2,000 center-start steps at 0.01 and retains 6,192. A separate reference job samples each newly defined posterior with elliptical slice. Check row norms, full Gram, SVD vs. Gram $\chi$, matching labels and held-out teacher scores across $r$. Review cost, per-probe errors and reference gates on these six before the remaining 30 cells.

# 7. Complete the paired geometry and ReLU comparisons

```bash
# Remaining geometry targets only, after inspecting the pilot:
REST=1-8,10-11,13-20,22-23,25-32,34-35
sbatch --array="$REST" --export=ALL,JOB=compare,SCOPE=geometry a10_next/next.sbatch
sbatch --array="$REST" --export=ALL,JOB=reference,SCOPE=geometry a10_next/next.sbatch
python3 a10_next/postprocess.py gates --scope geometry
python3 a10_next/plot_next.py geometry
```

Run the final two commands after the arrays finish. `compare` uses the original center start, $h/\sigma^2=0.01$, an explicit nominal burn, and 8,192 retained transitions initially. Normalize by the actual retained count. Extend an unresolved reused cell by running `JOB=compare` at the required level in the new root. This starts a new, longer run. The gate table records whether the source is reused or new. Use the same window length for a paired sensitivity before attributing a small difference to geometry. For an exact matched-window check, trim each comparable four-chain trajectory to the first 6,192 post-discard draws, recompute per-probe metrics, and keep the same timing denominator per joint step.

**Geometry outputs.** Plot loss and median prediction efficiency against width in three $\chi$ series, with replicate identities visible. Report the range across the eight fixed prediction inputs and the raw paired values. Add a marginal panel with the held-out mean prediction or predictive loss if explaining changes in task difficulty; the eight monitored inputs alone do not give a full test-set score. A positive, flat, or reversed empirical ordering answers the same design question. The theorem orders explicit bounds. These observables do not estimate an optimal functional-inequality constant.

**ReLU.** Use deep 32 and 256 first. Hold the dataset, centers, prior, hidden-layer scaling, held-out probes, step and initial-law protocol fixed. The original ReLU acceptance near 0.936 at the common step is worth reporting, but does not settle the width comparison. Reuse both dataset-0 traces, then add the missing four target/dataset combinations. The adapter's ReLU IDs are width-major: 0–2 at width 32, 3–5 at width 256.

```bash
python3 a10_next/cluster.py list --scope relu
sbatch --array=0,3 --export=ALL,JOB=reuse,SCOPE=relu a10_next/next.sbatch
sbatch --array=1,2,4,5 --export=ALL,JOB=compare,SCOPE=relu a10_next/next.sbatch
sbatch --array=0-5 --export=ALL,JOB=reference,SCOPE=relu a10_next/next.sbatch
# After the arrays finish:
python3 a10_next/postprocess.py gates --scope relu
python3 a10_next/plot_next.py relu
```

Use each target's own reference. Pair ReLU with the baseline tanh $0.01$ endpoint in the same dataset; preserve ESS definition, cost clock and retained window. For an activation interaction, report the wide/narrow efficiency ratio within each activation, then their ratio. Recover tanh short-run post-discard rows from `recovered_diagnostics.csv` and join the baseline A10 clock by cell; trim ReLU to the same window for the direct interaction. Report logit or prediction dispersion to expose changes in scale. Zero-activation fraction or activation RMS would require additional forward evaluations on retained states. ReLU probes the empirical reach of the observation beyond the smooth-activation theory.

For likelihood concentration, depth four, sine and nonlinear labels, begin with the recovered dataset-0 pair. If a reviewer-facing conclusion needs replication, use the ReLU commands with `SCOPE=concentrated`, `depth4`, `sin` or `nonlinear` and the same IDs. Preselect the perturbation for the scientific question. Report all screened pairs in a small table even if only ReLU or concentration is expanded.

# 8. Equilibrium covariance, prior discrepancy and optional runs

**Decorrelated observables.** If the original trajectory stages are available, compute chain-centered ACFs of loss and a fixed prediction at lags 0–400, together with raw-mean IAT, rank bulk/tail ESS and acceptance. Use the same 102,400 retained S1 steps at each width. Plot an ACF against transitions and a companion against lag times the matched seconds per joint step. Mark the ACF truncation scheme for IAT; the area of the first 400 displayed lags alone is a different quantity. If only the archived per-probe exports survive, plot those exports and label their provenance.

**Normalized covariances.** To answer whether the monitored directions relax similarly at equilibrium, fix a modest vector $g=(V,p_0,q_A,q_W)$ and the existing frozen interaction probes if present. Center and scale each coordinate using independent reference calibration. Form $C_0=\operatorname{Cov}(g_t)$ and the symmetrized lagged covariance $C_k$ using within-chain products in the retained window. Show a $C_0$ heatmap per endpoint and lagged correlations for the preselected probes. An optional slow linear combination can be fit from $C_k v=\lambda C_0 v$ on two chains and evaluated on the other two, swapping folds. Remove tiny-variance or numerically redundant directions with a declared tolerance. Run `python3 a10_next/equilibrium.py --arch deep --rep 0` from the project root. It produces the two ACF clocks, IAT table, covariance PDF and a two-fold slow-direction summary at lag 50 for the four named probes. Repeat for shallow and replicates 1–2. The width comparison is about this finite probe family. A transient prior-start series cannot supply a stationary covariance.

**Prior versus posterior discrepancy.** For a single width and matching Gaussian prior $\gamma$, $d\pi/d\gamma=e^{-V}/Z$ makes the maximizing event a training-loss threshold. Hence $\|\pi-\gamma\|_{\rm TV}=\sup_c[F_\pi^V(c)-F_\gamma^V(c)]$. This compares two laws on the same parameter space. It is distinct from the distance of a running chain to stationarity.

```bash
# Per endpoint: 4,096 iid full-parameter prior draws, forward evaluations only.
sbatch --array=0-11 --export=ALL,JOB=iid a10_next/next.sbatch
# Requires original reference production_s1 ... production_s4 for each cell:
python3 a10_next/postprocess.py tv
python3 a10_next/plot_distribution.py tv
```

Use the loss from matched independent priors and elliptical-slice posterior production. Plot both empirical CDFs at each endpoint, then report the plug-in supremum and the held-out signed event gap. Fit the maximizing threshold on two chains and half the prior draws; evaluate on the others and swap. The implementation uses sorted empirical CDFs to avoid a quadratic search. It supplies fold-specific conditional MCSEs; the two folds reuse data and do not yield independent error bars. If the archive lacks the reference loss series, the reference mean and SD are insufficient for this analysis. Restore those trajectories or run a target-matched reference, validate it, and read its new named segments explicitly. For a time-dependent chain, a loss-CDF gap is an observable lower bound only.

**Where I would stop.** The baseline efficiency, tuned endpoints, prior starts, paired geometry and ReLU cover the core experimental objections. A second benchmark of elliptical slice vs. PCNL is optional. For that comparison, time both samplers with the same batched production/I/O rules, record bracket likelihood calls and gradient evaluations, and compare each sampler's own per-probe ESS/s. The current reference job is designed to estimate posterior moments; it does not produce a matched kernel-cost benchmark. A static Poincaré witness $\mathrm{Var}_\pi(g)/\mathbb E_\pi\|\nabla g\|^2$ would need gradients at retained states or sparse parameter archives. The supplied scalar-only follow-ups do not record that denominator. Add it only if this becomes a specific reviewer claim.

# 9. Test, release and estimate the GPU budget

**Local verification.** The editable scripts reproduce the current figures directly from the A10 ZIP and the pinned bound evaluator. The supplied self-check exercises the CDF direction, known finite TV, bootstrap shapes and deterministic cutoff logic. The paper-window loss, Gaussian posterior fixture and 12+12 resume checks were previously verified in the project. The new CUDA adapters were checked against the pinned source. Their first real HDF5/CUDA execution is the cluster canary.

```bash
source scripts/revision/common.sh
python3 a10_next/selfcheck.py
python3 -m bnn_geometry.revision smoke --config configs/revision_smoke.yaml
python3 -m compileall -q a10_next
bash -n a10_next/next.sbatch
```

**Canary acceptance.** For finalist task 6, expect three `h_*/metrics_8192.csv` files of eleven probe rows each, four chains at unit stride, zero numerical failures, all five timing blocks at least one second, and five elapsed-time records per step size. Verify one raw ESS and one bulk ESS rate as $E/(8192c)$. Rerun the same completed task and check unchanged draw counts, counters and timing file. For the reference canary, expect burn/calibration/separation lengths of 4,096/2,048/1,024, four independent chains, and all eleven probe rows. For ReLU, retain the project's known-posterior and rejection checks and add a small forward/gradient comparison away from kinks and a zero-input convention check if the activation implementation has changed.

**Gate for stationary claims.** For every primary observable, require finite values, rank $\widehat R<1.01$, bulk ESS at least 1,000, tail ESS at least 400, a passing split-window drift screen and zero numerical failures. Compare means to the independent target reference using the project criterion $|\bar g-\bar g_R|\le3\sqrt{\mathrm{MCSE}_g^2+\mathrm{MCSE}_R^2}+0.05s_R$. Check the reference itself. Failure is an unresolved observable. Do not hide it in the median over predictions. `gates_<scope>.csv` records probe-level flags and `efficiency_<scope>.pdf` uses open symbols for unresolved points. Check the expected target count explicitly after every array; the gate table contains only available trajectories. Record all failures, missing targets and capped comparisons in the manuscript's supplementary table.

**Budget.** Trace recovery, existing-figure generation and deterministic geometry use no GPU sampling. The finalist grid costs 36 four-chain production windows plus warmups and disposable timing. Confirmation initially adds 24 windows. The six geometry pilot targets need timing and independent references; the remaining 30 require a sampler plus a reference if reporting equilibrium comparisons over the full grid. ReLU initially requires timing on two saved cells, four new PCNL trajectories, and six references. Use 19–38 sampler seconds per old 8,192-step cell as an order-of-magnitude guide only. Measure reference time and peak memory on both endpoints; elliptical slice makes a variable number of likelihood calls. The adapter also records synchronized production-only wall time including checkpoint I/O and separate ESS/s on that clock. Interrupted segments carry an unavailable production-wall rate. Record A10 model, dtype, filesystem, git revision and config with the tables. No timing result here prices the largest new reference job.

**Stop order under a deadline.** Freeze the fixed-step panels first. Then recover and publish the initialization curve, including any censoring. Finish independent tuned endpoints and the paired ReLU comparison. Show the deterministic geometry result even if only the six geometry pilot cells have qualified independent references; identify them as dataset 0. Defer extra architectures, static witnesses and second-sampler efficiency if their references or denominators remain incomplete.

**Source and execution scope.** The project snapshot is [Width-Scaling-Posterior-Geometry at 308a9e5](https://github.com/NalinTiwary/Width-Scaling-Posterior-Geometry/tree/308a9e53eab01af549d0d288ba957779e2ea9964). The implementation reuses its revision runner, trajectory/storage schema, sampler, diagnostic estimators, and SLURM environment. Proposed choices specific to this follow-up include the 0.32 finalist, independent prior initialization for finalist/confirmation, matched fixed-step confirmation, and prioritizing ReLU before the secondary variants.
