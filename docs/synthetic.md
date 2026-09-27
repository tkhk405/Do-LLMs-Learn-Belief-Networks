# Synthetic political statements

`code/generate_synthetic_data.py` contains request construction, batch transport, collection, text reconciliation and reconstruction of the analyzed corpus. Use Python 3.12. Offline operations need the data requirements; actual transport additionally needs the corresponding provider SDK. Each operation exposes `--help`.

```sh
python code/generate_synthetic_data.py requests --profile gpt_initial --questions /local/generation_questions.json --output /local/requests.jsonl
python code/generate_synthetic_data.py batch --provider gpt --requests /local/requests.jsonl --output /local/batch_run
python code/generate_synthetic_data.py gemini-batch --requests /local/gemini_requests.jsonl --model EXACT_API_MODEL_ID --output /local/gemini_run
```

The question JSON maps the topic keys in `config/generation_profiles.json` to their original Japanese wording. It is supplied locally, not embedded in the distributed configuration. Request construction preserves each profile's original prompt and generation parameters. Gemini row IDs are identifiers, not evidence of the API model used: Gemini transport requires an explicit model ID and does not infer one from them. Historical profile names and templates are retained without settling provenance questions by renaming them.

Batch operations are offline plans unless `--execute --keys /local/api_keys.json` is supplied. The local key file uses `openai`, `anthropic` or `gemini` as appropriate and must not be committed. Use `--action status` or `--action download` with the same request file/output directory after submission. The transport records a submission attempt before remote mutation and refuses blind resubmission after an uncertain attempt. It preserves batch IDs and verifies downloaded-file hashes before reusing them. An unresolved submission must be reconciled with the provider, not retried as a new job.

`collect` reads downloaded responses in request order, retaining the original first-choice/block/text extraction. Its `--historical-suffix` option supports the existing historical ID convention; do not use it to infer model identity. Inspect the operation's help for provider, request and response paths.

`reconcile export` compares collected text with a local final workbook and writes a hash-guarded patch. `reconcile apply` applies that patch after validating the base file and old/new text hashes. These are recorded final-workbook differences, not newly regenerated responses or proof of per-request API retry provenance.

```sh
python code/generate_synthetic_data.py assemble --sources /local/corpus_sources.json --output /local/reconstructed_corpus
```

`assemble` reconstructs the historical analyzed corpus from final workbooks using `config/corpus_row_map.csv`. It requires matching text hashes and preserves all 25,920 rows and their order. It is not a collector for an arbitrary newly generated corpus. A sources example is supplied in `config/corpus_sources.example.json`.

All generated text, official local question strings and local workbooks remain subject to the established public-scope separation; this consolidation does not add them to the distribution.
