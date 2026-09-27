# Environments and configuration

Use Python 3.12. Run commands from the package root. The analysis stages use different environments; do not combine their NumPy/GPU requirements into one installation.

| Work | Dependencies | Local input |
|---|---|---|
| Saved numerical aggregation, party/period/recall tables | `requirements/requirements-data.txt` | Saved numerical inputs; raw UTAS only for recalculation |
| Saved figures | `requirements/requirements-figures.txt` | Saved matrices, scores or prediction arrays |
| Probe fitting and transfer evaluation | `requirements/requirements-probing.txt` | Statements, activations and selected heads |
| Sensitivity refitting | `requirements/requirements-sensitivity.txt` | Prepared activations, statements and UTAS |
| GPU activation extraction | `requirements/requirements-extraction.txt` plus CUDA PyTorch | Local statements; accessible model weights |
| Silicon/recall generation | CUDA PyTorch, Transformers, accelerate, NumPy, pandas, tqdm; openpyxl for annual UTAS mappings | Local questions/targets and source data |
| Explicit API transport or judging | The selected provider's `openai`, `anthropic` or `google-genai` SDK | Private local keys and prepared requests |

The GPU extraction requirements describe the recorded Llama extraction environment. They do not establish an identical environment for every generation experiment. API/GPU availability is not validated by saved-result tests. Model-loading and API execution paths are explicit; help and offline saved-result operations do not require these SDKs.

## Configuration examples

- Copy `config/paths.example.json` to `config/paths.local.json` and fill only paths used by the selected operation. Relative entries in this JSON are resolved against the package root, not the JSON file's directory.
- Use `config/generation_questions.example.json` for synthetic-generation topic keys. Supply original wording locally using `--questions`.
- Copy `config/utas_questions.example.json` to `config/utas_questions.local.json` for silicon sampling. Replace all placeholders with locally obtained wording.
- `config/api_keys.example.json` illustrates a private key file supplied with `--keys`; no environment variables are required. Never include that populated file in a public package.
- `config/corpus_sources.example.json` lists local final-workbook inputs for historical corpus reconstruction.
- `config/probing_figures.example.json` and `config/comparison_figures.example.json` show figure input structures. Paths in figure-input JSON files are resolved relative to that JSON file.

## Verified boundary

All 12 entry points and their help paths were checked in an isolated copy of code/config, with no private local configuration or old scripts. This verifies command startup and removal of legacy imports. It does not replace the saved-result end-to-end reproduction check or establish that a new full GPU/API run has completed.
