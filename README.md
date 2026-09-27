# Policy stance representations in language models: analysis code

This repository organizes the analysis into 12 Python entry points with shared
utilities. Saved numerical inputs are included for reproducing the figures and
S3–S13 Tables without API calls, model downloads or access to individual UTAS records.

## Setup

Use Python 3.12. From this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-saved-results-verified.txt
```

These saved-result dependencies were verified on macOS arm64. Arial is required
for identical figure typography and is not bundled. Other fonts/platforms may
change the layout. Training and GPU work use separate requirements; see
[environments](docs/environments.md).

## Reproduce figures and tables

Run the following from the repository root, using fresh output directories:

```sh
python code/silicon_sampling.py figures --inputs comparison_inputs/inputs.json --output results/silicon
python code/mantel_test.py figures --inputs comparison_inputs/inputs.json --output results/comparison
python code/probing.py figures --inputs remaining_figure_inputs/probing/inputs.json --output results/probing
python code/generator_heldout.py figures --results remaining_figure_inputs/heldout --output results/heldout
python code/probing.py tables --gemma-scores saved_results/gemma/scores --llama-scores saved_results/llama/scores --output results/S3_S9
python code/sensitivity_analysis.py table --gemma-summary table_inputs/gemma/sensitivity_summary.csv --llama-summary table_inputs/llama/sensitivity_summary.csv --gemma-main table_inputs/gemma/matrix_comparisons.csv --llama-main table_inputs/llama/matrix_comparisons.csv --output results/S10
python code/party_analysis.py table --party table_inputs/party --output results/S11
python code/period_analysis.py tables --periods table_inputs/periods --release table_inputs/release --output results/period_tables
python code/utas_recall.py tables --recall table_inputs/recall --output results/recall_tables
python code/utas_recall.py assemble --a results/period_tables/S13_Table_A.csv --b results/recall_tables/S13_Table_B.csv --c results/recall_tables/S13_Table_C.csv --output results/S13
```

The first four commands reproduce Fig1–6 and S1–S10 Fig. The remaining commands
export S3–S13 Tables as CSV/Markdown, including the combined S13 panels.
Original Table1–3 and S1–S2 Table manuscript sources and captions are not bundled.

## Analysis files

| File in code/ | Analysis | Guide |
|---|---|---|
| generate_synthetic_data.py | Requests, batch submission, collection and corpus assembly | [Generation](docs/synthetic.md) |
| extract_activations.py | Model activation extraction and resumable checkpoints | [Extraction](docs/extraction.md) |
| probing.py | Probes, head selection, tables and figures | [Probing](docs/probing.md) |
| transfer_analysis.py | Cross-issue transfer performance | [Internal comparisons](docs/internal_comparisons.md) |
| cosine_similarity.py | Stance-direction cosine similarity | [Internal comparisons](docs/internal_comparisons.md) |
| mantel_test.py | Matrix comparisons and figures | [Internal comparisons](docs/internal_comparisons.md) |
| generator_heldout.py | Generator-held-out training and evaluation | [Held-out analysis](docs/generator_heldout.md) |
| sensitivity_analysis.py | Resampling with fixed heads and probe refitting | [Sensitivity](docs/sensitivity.md) |
| party_analysis.py | Party/year means and residual comparisons | [Party analysis](docs/party.md) |
| period_analysis.py | Survey-period comparisons | [Periods](docs/periods.md) |
| utas_recall.py | Recall generation, judging, scoring and tables | [Recall](docs/recall.md) |
| silicon_sampling.py | Response generation, matrix comparison and figures | [Silicon sampling](docs/silicon.md) |

Run `python code/FILE.py --help` and the relevant operation's `--help` for arguments.
Model configurations and recorded generation profiles are in config/. Use the
exact configuration for the relevant experiment. Explicit model/revision arguments
must be supplied where required; the code does not choose a substitute model.

## Included data and local inputs

- saved_results/: saved probe scores and sensitivity results.
- comparison_inputs/: numerical matrices for comparison figures.
- remaining_figure_inputs/: probing and held-out numerical figure inputs.
- table_inputs/: aggregate numerical summaries for table export.
- config/: settings, examples and the study's attribute mappings.

Official UTAS records, question wording/translations, target counts,
respondent-linked generated answers, synthetic statement text, manuscript sources,
API credentials and model weights are excluded. Full recalculation requires locally
supplied inputs; see [local inputs](docs/local_inputs.md). Saved-result reproduction
does not constitute a new full training or generation run.

Code licensing is stated in LICENSE. Third-party materials are not relicensed;
see NOTICE.md. FILE_LIST.txt and FILE_MANIFEST.json list the distributed files and
SHA-256 hashes (the manifest excludes itself).

## Earlier repository version

The existing data/ directory is retained unchanged from the earlier repository.
It is not part of the newly assembled numerical-input distribution described above
and is not used by the saved-result commands in this README. The earlier analysis
code remains accessible in Git history. The previous baseline_analysis.py is
replaced by silicon_sampling.py for the revised analysis.

FILE_LIST.txt and FILE_MANIFEST.json describe the revised distribution files,
not the unchanged legacy data/ directory or Git metadata.
