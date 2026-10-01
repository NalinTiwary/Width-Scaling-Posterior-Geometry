# Two-figure production package

Read `figure_production_guide.md` or the companion PDF. This replaces the paper's previous PI/LSI proxy plotting plan with spectral-domain coverage and fixed-step loss autocorrelation.

Files:
- `figure_production_guide.md`: complete input, calculation, visual design, recovery, and reporting instructions.
- `plot_style.json`: exact colors, markers, axes, geometry, and aggregation settings.
- `figure_metrics.py`: implemented array-only calculations; no project-specific loader or sampler.
- `selfcheck.py`: deterministic calculation tests and a synthetic AR(1) check.
- `selfcheck_result.txt`: test output from preparation of this package.

Run `python selfcheck.py` in this directory. Then implement a small adapter for the actual project's table and scalar-trace locations using the schemas in the guide. The raw campaign traces were not supplied with the figure PDFs, so this package does not claim to contain regenerated scientific figures. It explains exactly how to retrieve or regenerate missing inputs if needed.

Use the existing project environment. Do not generate paper curves from the old integrated-time summaries or synthetic self-check sequences.
