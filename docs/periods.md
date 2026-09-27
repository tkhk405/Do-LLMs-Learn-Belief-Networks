# Period comparisons and S12/S13A tables

`code/period_analysis.py` contains three operations. Use Python 3.12 with the data requirements. No model fitting, generation or API calls are needed.

## Three survey periods

```sh
python code/period_analysis.py three-periods --config config/paths.local.json --transfer /path/to/gemma_transfer.csv --cosine /path/to/gemma_cosine.csv --output output/three_periods
```

The periods remain 2003–2007, 2009–2012 and 2013–2025. The configured elected-member and all-candidate UTAS files are analyzed separately. Invalid responses are removed using the original coding. The 2004 public-safety item remains structurally missing, with completeness required on the other five items; other survey years require all six items.

The operation writes six period-specific correlation matrices and pairwise counts, sample counts, issue-pair changes, six between-period comparisons and 12 Gemma-to-period comparisons. Transfer matrices are symmetrized before comparison. Between-period comparisons retain the original floating-point tolerance (rho minus 1e-12) in counting exact-permutation extremes. The Gemma comparisons retain the existing shared exact Mantel function. These conventions are preserved rather than silently changed during consolidation.

## Model-release periods

```sh
python code/period_analysis.py release-periods --config config/paths.local.json --utas-2026 /path/to/2026_utas.csv --transfer /path/to/gemma_transfer.csv --cosine /path/to/gemma_cosine.csv --output output/release_periods
```

The comparison uses 2003–2024 and 2025–2026, without overlapping years. The 2026 source file is supplied locally; response flags, elected-member flags, six variable mappings, encoding and complete-six filtering remain as in the existing implementation. The 2025 and 2026 response records are pooled before calculating their issue correlations. Elected members and all candidates are separate analyses.

Outputs include four UTAS correlation matrices, pairwise counts, survey-year sample counts and eight Gemma comparisons. The periods refer to model release timing, not verified individual training-data inclusion. Neither temporal operation tests the statistical significance of differences between reported correlation coefficients.

## Export saved numerical tables

```sh
python code/period_analysis.py tables --periods output/three_periods --release output/release_periods --output output/period_tables
```

This reads the three comparison CSVs and writes S12 Table A, S12 Table B and S13 Table A in CSV and Markdown, with unrounded source summaries. It requires no raw UTAS data when the saved summaries are available. S13 B/C belong to the recall analysis and are not generated here. Existing output directories are rejected. The manuscript and captions are not changed.

Local input manifests may contain private paths. This consolidation does not change which files have been approved for public distribution.
