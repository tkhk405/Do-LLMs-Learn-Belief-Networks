# Fixed-head refitted-probe sensitivity analysis

`code/sensitivity_analysis.py` provides `prepare`, `run`, `summarize` and `table`. Use Python 3.12. Training requires the pinned versions in `requirements/requirements-sensitivity.txt`. Saved-result aggregation uses only NumPy, pandas and SciPy.

## Local inputs and preparation

```sh
python code/sensitivity_analysis.py prepare --model gemma --config config/paths.local.json --heads /path/to/gemma_heads --output output/gemma_sensitivity_inputs
```

The head directory contains `common_top20_heads.csv` and `union_heads.csv`. Preparation extracts the selected heads from local float16 activation arrays and records their order and checksums. Use `--model llama` with separate Llama inputs/output for that model. The synthetic statements and respondent-level UTAS data must be supplied locally.

## Refitting with resumable checkpoints

```sh
python code/sensitivity_analysis.py run --config config/paths.local.json --inputs output/gemma_sensitivity_inputs --design output/gemma_sensitivity_inputs/design.json --output output/gemma_sensitivity_run --reps 1000 --jobs 1
```

Synthetic statements are resampled as 288 context groups, with replacement. The same sampled group weights are applied across issues. UTAS respondents are independently resampled with replacement within each survey year, separately for elected Diet members and all candidates. The seed is 20260913. The selected heads remain fixed, but the ordinal probes are refitted in each replicate. Cross-validation fold assignment precedes resampling so copies of an original row stay in the same fold. Transfer uses target-wise standardization, and cosine uses standardized high-minus-low mean activation directions. Gemma and Llama retain their original arithmetic dtypes.

`--jobs` controls parallel probe fits. `--max-new 1` stops after saving one new replicate. A file named `STOP` in the run output directory stops execution before the next replicate; remove it before resuming. Rerun the same command to reuse completed probe/replicate checkpoints. The run manifest validates the data, selected heads, seed, repetition count, code and package versions. Different settings require a new directory. File locks prevent two runners from writing to the same directory.

This consolidated script has a new code signature: old run checkpoints are not silently resumed. Completed historical replicate matrices can still be summarized. `progress.json` records the number of completed replicates. Intermediate summaries include a `complete` column and are distinct from the final 1,000-replicate table.

## Saved results and S10 Table

```sh
python code/sensitivity_analysis.py summarize --source /path/to/gemma/sensitivity --output output/gemma_sensitivity_summary
python code/sensitivity_analysis.py summarize --source /path/to/llama/sensitivity --output output/llama_sensitivity_summary
python code/sensitivity_analysis.py table --gemma-summary output/gemma_sensitivity_summary/sensitivity_summary.csv --llama-summary output/llama_sensitivity_summary/sensitivity_summary.csv --gemma-main /path/to/gemma/matrix_comparisons.csv --llama-main /path/to/llama/matrix_comparisons.csv --output output/S10
```

The source contains `run_manifest.json` and exactly 1,000 `replicates/0000.npz`–`0999.npz` files. Each replicate contains transfer, cosine, elected and all_candidates matrices. Summarization calculates the four matrix correlations for each replicate and their 2.5th/97.5th percentiles. Table export requires both complete 1,000-replicate summaries and the main-analysis coefficients. It writes `S10_Table_values.csv` and `S10_Table.md`, without changing the manuscript or caption.

Run manifests can include local input provenance. The aggregation manifest is a local record and should not be published without checking its contents.
