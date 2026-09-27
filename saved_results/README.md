# Saved numerical results — local distribution candidate

For each model:

- `scores/` contains six full head-performance arrays and six coefficient arrays. They support head selection and the coefficients used by continuous-projection transfer analysis. They are not complete serialized ordinal classifiers: thresholds and out-of-fold predictions are not included. Activation arrays are separate inputs for recomputing cross-issue matrices.
- `sensitivity/replicates/` contains 1,000 saved replicates, each retaining only four 6-by-6 matrices: transfer, cosine, elected-member UTAS and all-candidate UTAS. They support final sensitivity aggregation without refitting probes. They do not contain respondent-level answers or the resampling indices.
- The sensitivity run manifest retains design settings and a hash of the original manifest. Private input-cache paths and UTAS path/hash mappings are omitted. It is marked `saved_matrix_aggregation_only`, and is not advertised as a full training checkpoint.

Validation reproduced S3–S9's seven probing tables and both models' complete 4,000-correlation sensitivity results and interval summaries. Score/coefficients were copied byte-for-byte. Replicate matrices were rewritten into compressed NPZ files with only the named numeric arrays.

Use `export_probing_tables.py` with the two scores directories. Use `summarize_sensitivity.py --source MODEL/sensitivity --output NEW_DIRECTORY` for each model. Full activation extraction, ordinal-probe refitting, prediction-distribution figures and the held-out experiment require additional inputs documented in the code package.

This is a local candidate, not a public release. Aggregated UTAS-derived matrices are included and their distribution status remains part of final review. No fresh bootstrap fits or API calls were made.
