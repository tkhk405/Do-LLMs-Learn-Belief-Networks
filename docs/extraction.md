# Activation extraction

`code/extract_activations.py` integrates the extraction runner and numerical readout functions. It requires local synthetic statements and uses `config/models.json` and `config/extraction.json`.

```sh
python code/extract_activations.py --model llama --config config/paths.local.json --plan-only
python code/extract_activations.py --model gemma --config config/paths.local.json --revision EXACT_HF_COMMIT --plan-only
```

Plan-only mode validates inputs and prints settings without importing PyTorch or loading models. Gemma requires an explicit commit because the original extraction commit was not recorded. The supplied Llama commit is retained. An explicit Gemma commit makes a new run identifiable; it does not retrospectively establish the original checkpoint.

Remove `--plan-only` to perform extraction on a suitable CUDA environment. Default `--mode smoke` processes 16 rows per issue at two layers. Use `--mode full` explicitly for all 4,320 rows per issue and every layer. The Llama protocol requires an A100 GPU and has no CPU fallback.

## Preserved protocol

- Gemma: 42 layers, 16 heads, 256 values per head. Llama: 32 layers, 32 heads, 128 values per head.
- Four-bit model loading, double quantization and float16 compute, using the original BitsAndBytesConfig arguments. Effective quantization settings are saved.
- Left padding, maximum token length 4,096 and batch size eight.
- A forward hook reads the input of each attention output projection (`o_proj`). The final non-padding token is selected and reshaped into heads.
- Activation files use float16 and shape `(number of statements, heads, head dimension)`.

The kernel rejects empty attention masks, wrong head dimensions or an unexpected readout position, verifies hook-call counts, and removes the hook even if extraction fails.

## Outputs and resume

Files are written under the configured `output_dir/activations/{model}/{smoke|full}`. Each completed issue/layer is saved atomically, with its checksum recorded in `extraction_manifest.json`. Repeating the same command reuses only files whose checksums match. Inputs, code, revision, versions, GPU and effective model/tokenizer settings must match. Changing settings requires a new output directory. Consolidated code has a new signature and does not silently resume old-script checkpoints.

Use `requirements/requirements-extraction.txt` and `config/recorded_llama_extraction_environment.json` as the existing environment records. Install the appropriate CUDA PyTorch build separately. No model installation or download is performed by plan-only mode.
