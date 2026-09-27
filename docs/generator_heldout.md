# Generator-held-out analysis

`code/generator_heldout.py` contains the Gemma analysis used for S8/S9 Fig. It provides six explicit operations. Use Python 3.12 and the existing probing requirements for training/evaluation; saved-result aggregation and figures do not require mord or scikit-learn.

## Training and selection

For each held-out generator (GPT, Claude, Gemini), train probes using only statements from the other two generators. Head selection uses training cross-validation scores. Evaluation statements do not enter probe fitting or head selection.

```sh
python code/generator_heldout.py train --config config/paths.local.json --heldout GPT --output output/heldout_training
python code/generator_heldout.py select --scores output/heldout_training/heldout_GPT/checkpoints --checkpoints --output output/heldout_GPT/heads
python code/generator_heldout.py evaluate --config config/paths.local.json --heldout GPT --scores output/heldout_GPT/heads/probing_full --heads output/heldout_GPT/heads --output output/heldout_GPT/evaluation
```

Repeat with Claude and Gemini. Training can be restricted with `--issues`, `--layers`, and `--heads`. Selection requires complete scores for all six issues, 42 layers and 16 heads. Existing complete score arrays can instead be selected without `--checkpoints`.

Transfer uses 20 common heads and target-issue standardization of held-out activations. Cosine uses the union of issue-specific top-20 heads, comparing training-generator directions with held-out-generator directions. The original float32 conversion in cosine direction calculation is retained. These are distinct from the all-generator analysis.

Training saves per-head checkpoints; evaluation saves per-layer checkpoints. Inputs, code and package versions must match to resume. New code does not silently resume checkpoints created by a different script. Completed numerical outputs remain usable. Select a new output directory when changing inputs.

## Saved results and figures

`summarize --results DIR --output DIR` reads `heldout_{GPT,Claude,Gemini}/transfer/transfer_per_head.csv` and `heldout_{GPT,Claude,Gemini}/cosine/cross_split_cosine_per_head.csv`. The root results directory also requires `pooled_original_transfer_matrix.csv`, `pooled_original_cosine_matrix.csv`, `utas_elected_spearman_matrix.csv`, and `utas_all_candidates_spearman_matrix.csv`. It writes directional/symmetric matrices, 18 comparisons and their 720-permutation distributions. It does not refit probes or reselect heads.

```sh
python code/generator_heldout.py summarize --results /path/to/saved_results --output output/heldout_summary
python code/generator_heldout.py prepare-figures --aggregated output/heldout_summary --original-transfer /path/to/pooled_original_transfer_matrix.csv --original-cosine /path/to/pooled_original_cosine_matrix.csv --elected /path/to/utas_elected_spearman_matrix.csv --all-candidates /path/to/utas_all_candidates_spearman_matrix.csv --output output/heldout_figure_inputs
python code/generator_heldout.py figures --results output/heldout_figure_inputs --output output/heldout_figures
```

The adapter validates all 18 fold comparisons before creating the figure inputs. The figures operation writes S8/S9 Fig in PDF, PNG and flattened TIFF formats. Original label aliases, rendering and statistical display rules are preserved. Saved matrix/summary figure inputs can be passed directly to `figures` without training data.
