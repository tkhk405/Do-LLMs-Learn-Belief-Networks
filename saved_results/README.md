# Saved numerical results

For each model:

- `scores/` contains six full head-performance arrays and six coefficient arrays. These support head selection, S3–S9 Tables, and continuous-projection transfer analysis. They are not complete serialized ordinal classifiers: thresholds and out-of-fold predictions are not included in this directory. Recomputing cross-issue matrices also requires activation arrays and statement labels.
- `sensitivity/replicates/` contains 1,000 saved replicates. Each retains four 6-by-6 matrices: transfer performance, cosine similarity, elected-member UTAS correlations, and all-candidate UTAS correlations. These support sensitivity aggregation without refitting probes. They do not contain respondent-level answers or resampling indices.
- Each sensitivity run manifest retains design settings and a hash of the original manifest. Private input-cache paths and UTAS path/hash mappings are omitted. The `saved_matrix_aggregation_only` field identifies the distributed files as inputs for aggregation, rather than full training checkpoints.

## Reproduce the probing tables

From the repository root, using the saved-result dependencies in the main README and a fresh output directory:

```bash
python code/probing.py tables --gemma-scores saved_results/gemma/scores --llama-scores saved_results/llama/scores --output results/S3_S9
```

This exports S3–S9 Tables as CSV and Markdown files.

## Recalculate the sensitivity summaries

Run one command for each model, using fresh output directories:

```bash
python code/sensitivity_analysis.py summarize --source saved_results/gemma/sensitivity --output results/sensitivity_gemma
python code/sensitivity_analysis.py summarize --source saved_results/llama/sensitivity --output results/sensitivity_llama
```

Each command calculates 4,000 matrix correlations (1,000 replicates × two measures × two UTAS populations) and writes `replicate_correlations.csv`, `sensitivity_summary.csv`, and `aggregation_manifest.json`. The summaries include the 2.5th and 97.5th percentiles used for S10 Table.

These commands use saved numerical results and do not refit probes or call model APIs. For figure rendering and the formatted S10 Table export, see the [repository README](../README.md). For the local inputs required to rerun extraction, training, or generation, see [local inputs](../docs/local_inputs.md).
