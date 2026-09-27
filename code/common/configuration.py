"""Shared local configuration. Loading never calls an API or prints credentials."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[2]
PATH_KEYS = {"synthetic_statements_dir", "utas_elected_csv", "utas_all_candidates_csv",
             "gemma_activations_dir", "llama_activations_dir", "gemma_results_dir",
             "llama_results_dir", "output_dir"}

def load_paths(config=None):
    source = Path(config) if config else ROOT / "config/paths.local.json"
    values = json.loads(source.read_text())
    if not isinstance(values, dict) or set(values) - PATH_KEYS:
        raise ValueError("Invalid path configuration keys")
    paths = {}
    for key, value in values.items():
        if not isinstance(value, str):
            raise ValueError("Paths must be strings")
        if value.strip():
            p = Path(value).expanduser()
            paths[key] = p if p.is_absolute() else ROOT / p
    return paths

def load_api_key(provider):
    if provider not in {"openai", "anthropic", "gemini"}:
        raise ValueError("Unknown provider")
    source = ROOT / "config/api_keys.local.json"
    values = json.loads(source.read_text())
    key = values.get(provider, "")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("API key is not configured for the requested provider")
    return key.strip()
