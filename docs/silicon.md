# Silicon sampling

`code/silicon_sampling.py` combines Gemma/Llama joint-response generation, saved-response analysis and Fig1/S1 rendering. Use Python 3.12.

```sh
python code/silicon_sampling.py generate --model gemma --raw-dir /local/annual_utas --mapping-workbook /local/mappings.xlsx --output /local/gemma_generation
```

This saves an offline plan by default. Supply the six original Japanese question strings in `config/utas_questions.local.json`, keyed by canonical issue name. These are local inputs and are not included in public code. Add `--execute` only when intending to load the model and generate responses on CUDA. Use `--model llama` and a separate output directory for Llama.

The procedure retains the original plain Japanese prompt, without a chat template, temperature 0.8, seed 42 and batch size 16. The constrained continuation fixes JSON punctuation and A–F keys and samples the six integer responses in the same continuation. The existing model loader and dtype are preserved. Generation records include eligible candidate profiles; the main reported comparison selects elected respondents with complete UTAS responses during analysis. Generation count is not the paper's final comparison sample size.

The plan checks input and prompt hashes. The runner checks the execution environment, response IDs, valid answer values and complete ordered batch prefixes before resuming. No model checkpoint revision that was absent from the historical generation code is inferred during consolidation.

```sh
python code/silicon_sampling.py summarize --model gemma --responses /local/joint_candidate_responses.csv --raw-dir /local/annual_utas --mapping-workbook /local/mappings.xlsx --output output/gemma_silicon
python code/silicon_sampling.py figures --inputs /path/to/comparison_inputs/inputs.json --output output/silicon_figures
```

`summarize` rebuilds UTAS records from the original annual sources, validates response identifiers and metadata, and computes the original complete-respondent/elected-respondent matrices and comparisons. The 2004 public-safety missingness rule is preserved. `figures` needs only `gemma` and `llama` entries with `output` and `elected` matrix paths. Existing comparison-input JSON files with additional fields remain usable. It renders only Fig1 and S1 Fig; internal-representation figures belong to `mantel_test.py`.

Official text, local UTAS and respondent-linked generated responses remain outside the public distribution. Saved numerical matrices allow figure reproduction without those inputs.
