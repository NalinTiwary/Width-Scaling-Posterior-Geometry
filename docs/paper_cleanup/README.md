# Paper cleanup and selective supplements: revision 2

Read the two documents independently:

- `figure_cleanup_guide.pdf` / `.md`: theorem-relevance audit and final design for the main figures. Figure 1 now shows quantiles and full empirical coverage, with threshold 2 and exact per-target counts above 2; the arbitrary-margin main display is retired.
- `supplementary_experiments.pdf` / `.md`: mandatory spectral reanalysis first, then predictive ACF and an optional reuse-only step-size check. Includes recovery, formulas, diagnostics, and a capped S1 endpoint fallback.

Implementation aids:

- `clean_figures.py`: runnable main-figure renderer. The spectral command now requires statewise `width,replicate,chain,draw,T` rows and an audited `theory_context.json`, and exports quantiles, threshold counts, and CDFs; the loss-ACF input is unchanged.
- `supplement_metrics.py`: array-only finite-lag calculations for predictive ACF and the step-size comparison.
- `captions.tex`: copyable main-figure captions and neutral supplementary caption templates.
- `selfcheck.py`: calculation/schema/rendering checks, including exact ECDF and strict-threshold counts. Run `python selfcheck.py` in this directory.
- `selfcheck_result.txt`: check results from preparation.

The scripts require NumPy and Matplotlib. Reuse and record the existing project environment. Convergence diagnostics remain in the project's validated diagnostic implementation. No archive adapter or sampler is supplied or implied.

No new scientific experiment was run for this package. No raw campaign traces were supplied, and no synthetic layout fixture is a publication result. The templates must consume the exact audited outputs rather than rounded numbers from the written report. Historical experiment artifacts are preserved; these guides supersede the previous presentation instructions.

The fixed-2.5 and Posterior/Prior layouts in earlier figure-production guides are superseded. Historical campaign configurations and scientific results remain unchanged. No new widths are recommended; the new coverage panel is postprocessing of the existing reference archive.
