"""Two-figure paper package (docs/paper_figures/figure_production_guide.pdf).

``figure_metrics.py``, ``selfcheck.py`` and ``plot_style.json`` are vendored unchanged from the production package
(checksums in docs/paper_figures/checksums.sha256). The project adapter is split so that a cosmetic change cannot
change a measurement: ``prepare`` (campaign files -> validated source tables), ``analysis`` (tables/NPZ -> computed
tables), ``render`` (computed tables -> figures) and ``text`` (computed tables -> captions/results).
"""
