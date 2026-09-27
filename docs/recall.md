# UTAS recall: preparation, execution, scoring and tables

`code/utas_recall.py` consolidates prompt preparation, GPU generation, API judging, saved-output scoring, judge-log validation, consensus, reported-target selection and S13 table export. Generation and API judging require the explicit `--execute` flag. Without it, those operations print a plan without loading models or sending requests.

Use Python 3.12 and the data requirements. Each operation provides `--help`. All output directories must be new.

```sh
python code/utas_recall.py score-distributions --raw-outputs /local/distribution/raw_outputs.jsonl --prompt-manifest /local/distribution/prompt_manifest.csv --truth /local/truth.json --output output/distribution_scores
python code/utas_recall.py score-questions --raw-outputs /local/question/raw_outputs.jsonl --prompt-manifest /local/question/prompt_manifest.csv --output output/question_scores
python code/utas_recall.py validate-judges --scored-texts output/question_scores/scored_outputs.csv --raw-judgments /local/judge_raw_outputs.jsonl --output output/validated_judges
python code/utas_recall.py consensus --scored-texts output/question_scores/scored_outputs.csv --judge-results output/validated_judges/judge_results_long.csv --output output/consensus
python code/utas_recall.py summarize --scored-distributions output/distribution_scores/scored_outputs.csv --judge-consensus output/consensus/judge_consensus.csv --output output/recall_summary
python code/utas_recall.py tables --recall output/recall_summary --output output/S13_B_C
```

Distribution scoring reparses raw generated text, checks generation metadata and computes scores using locally supplied official counts. Question scoring verifies the historical 202 successful records and scores text against its original prompt manifest. Judge validation selects successful records with matching configured models and prompt hashes; consensus preserves the original voting rules and manual-review flags.

The historical experiments contain more conditions and years than the reported table. Summarization selects percentage-recall variant C and correct-identifier question completions for 2003–2024, requiring the same 89 distinct year/issue items per Gemma variant in both analyses. It recomputes normalized TVD, threshold percentages and classification counts. Prefix-only completions do not enter the reported table.

`tables` can be run using only saved `distribution_summary.csv` and `question_summary.csv`. It writes S13 Table B/C as CSV and Markdown. S13 A is produced by `period_analysis.py`; the `assemble` operation combines its CSV with the B/C CSVs, preserving display precision and without recalculation. No manuscript or captions are edited.

Official wording, official counts and full response/judgment logs are local inputs, not added to the public distribution by this consolidation. Local scoring outputs may contain that text and should not be included in the public package automatically.

## Prepare prompts and execute explicitly

```sh
python code/utas_recall.py prepare --targets /local/recall_targets.json --output output/recall_inputs
python code/utas_recall.py generate --kind question --model pt --manifest output/recall_inputs/question/prompt_manifest.csv --output output/question_pt
python code/utas_recall.py judge --scored-texts output/question_scores/scored_outputs.csv --output output/judges
```

`prepare` requires a local targets file containing 95 historical items, question prefixes/continuations and official counts. These are not embedded in public code. The historical manifests contain 855 distribution prompts and 101 question prompts per model. The reported 89-item filtering happens later at summarization.

For actual generation, append `--execute`. The operation requires CUDA, PyTorch, Transformers and accelerate. Model IDs and exact revisions remain those recorded in the original experiment; pretrained uses plain completion and instruction-tuned uses its chat template. Both use bfloat16, no quantization, greedy generation and seed 20260825. Maximum new tokens remain 192 for distribution and 96 for question completion. There is no automatic model replacement.

For actual judging, append `--execute --keys /local/api_keys.json`. This local JSON contains `openai`, `anthropic` and `gemini` keys supplied by the user and must not be committed. Only the selected providers' SDKs are loaded. Judge models remain gpt-5.4, claude-opus-4-6 and gemini-3.1-pro-preview, with their original request parameters. Availability and access to those historical models were not checked by sending requests in this consolidation.

Both execution operations support `--max-new`, a `STOP` file and atomic per-item checkpoints. The same command resumes completed records without repeating them. Input/code/configuration mismatches reject resumption; old-run checkpoints are not silently accepted with the new code signature. API execution records successful calls for restart, but cannot guarantee exactly-once billing if a process dies after the provider accepts a request and before the local record is saved.

```sh
python code/utas_recall.py assemble --a output/period_tables/S13_Table_A.csv --b output/S13_B_C/S13_Table_B.csv --c output/S13_B_C/S13_Table_C.csv --output output/S13_all
```
