"""Silicon sampling: explicit generation, saved-response analysis and Fig1/S1."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import argparse,hashlib,itertools,json,time,importlib.metadata
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common.utas_records import build_utas_records,YEAR_CONFIGS,OPTION_TEXT
from common.constants import ISSUES
from common.data_loading import load_matrix
ISSUE_TEXTS={}


def persona_sentence(row: pd.Series | dict) -> str:
    return f'あなたは{int(row['year'])}年の{row['chamber']}選挙に立候補した日本の政治家です。所属政党は{row['party_prompt']}です。候補者区分は{row['incumbency_prompt']}です。'

MODEL_ID = 'tokyotech-llm/Gemma-2-Llama-Swallow-9b-pt-v0.1'

PROMPT_VERSION = 'joint-six-issue-v1'

LETTERS = tuple('ABCDEF')

def load_model(model_id: str=MODEL_ID):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map='auto' if torch.cuda.is_available() else None, low_cpu_mem_usage=True)
    model.eval()
    return (tokenizer, model)

def respondent_id(row: pd.Series | dict) -> str:
    """Stable identifier for one actual candidate-election record."""
    return f'{int(row['year'])}-{int(row['raw_row'])}'

def build_joint_prompt(row: pd.Series | dict) -> str:
    """Build the persona-conditioned counterpart of the paper's API prompt."""
    if set(ISSUE_TEXTS) != set(ISSUES):
        raise ValueError('Create config/utas_questions.local.json from your locally obtained UTAS materials before generation')
    options = '  '.join((f'{number}={text}' for number, text in OPTION_TEXT.items()))
    questions = '\n'.join((f'{letter}. {ISSUE_TEXTS[issue]}' for letter, issue in zip(LETTERS, ISSUES)))
    schema = '{' + ', '.join((f'"{letter}": ?' for letter in LETTERS)) + '}'
    return persona_sentence(row) + 'この人物になりきって、以下の6つの政策すべてに回答してください。\n\n' + options + '\n\n' + questions + '\n\n以下のJSON形式で回答してください。各値は1から5の整数です。\n' + schema + '\n回答：\n'

def json_suffix(choices: Sequence[int]) -> str:
    if len(choices) != len(LETTERS):
        raise ValueError(f'Expected {len(LETTERS)} choices, received {len(choices)}')
    if any((int(choice) not in range(1, 6) for choice in choices)):
        raise ValueError(f'Choices must be integers from 1 to 5: {choices}')
    return '{' + ', '.join((f'"{letter}": {int(choice)}' for letter, choice in zip(LETTERS, choices))) + '}'

def compile_constrained_suffix(tokenizer, prompt: str) -> tuple[list[list[int]], list[int]]:
    """Compile a token plan that permits variation only at six JSON values.

    Returns one allowed-token list per generated position and the positions at
    which substantive choices are sampled.  Every non-choice JSON token is
    forced.  The plan is derived from complete tokenizations, so it does not
    assume that punctuation, whitespace, or digits are tokenized separately.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    reference_choices = [1] * len(LETTERS)
    reference_ids = tokenizer(prompt + json_suffix(reference_choices), add_special_tokens=True).input_ids
    if reference_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError('Prompt/JSON boundary retokenized. End the prompt with a stable boundary before using constrained generation.')
    continuation = reference_ids[len(prompt_ids):]
    plans = [[token_id] for token_id in continuation]
    choice_positions: list[int] = []
    for position in range(len(LETTERS)):
        allowed_ids = []
        changed_offsets = []
        for choice in range(1, 6):
            variant = reference_choices.copy()
            variant[position] = choice
            variant_ids = tokenizer(prompt + json_suffix(variant), add_special_tokens=True).input_ids
            if len(variant_ids) != len(reference_ids):
                raise ValueError(f'Choice {position}/{choice} changes continuation length; the constrained plan is not safe.')
            differences = [index for index, (left, right) in enumerate(zip(reference_ids, variant_ids)) if left != right]
            if len(differences) > 1:
                raise ValueError(f'Choice {position}/{choice} changes multiple tokens: {differences}')
            if differences:
                offset = differences[0] - len(prompt_ids)
                if offset < 0:
                    raise ValueError('A choice changed a prompt token.')
                changed_offsets.append(offset)
                allowed_ids.append(variant_ids[differences[0]])
            else:
                allowed_ids.append(None)
        unique_offsets = sorted(set(changed_offsets))
        if len(unique_offsets) != 1:
            raise ValueError(f'Could not isolate one token position for answer {LETTERS[position]}: {unique_offsets}')
        offset = unique_offsets[0]
        allowed_ids[0] = continuation[offset]
        if len(set(allowed_ids)) != 5:
            raise ValueError(f'The five answers do not map to five distinct tokens at {LETTERS[position]}.')
        plans[offset] = [int(token_id) for token_id in allowed_ids]
        choice_positions.append(offset)
    if len(set(choice_positions)) != len(LETTERS):
        raise ValueError(f'Choice positions are not unique: {choice_positions}')
    return (plans, choice_positions)

def validate_plan_across_prompts(tokenizer, prompts: Sequence[str]) -> tuple[list[list[int]], list[int]]:
    """Assert that the same constrained suffix plan works for every persona."""
    if not prompts:
        raise ValueError('No prompts supplied')
    reference_plan, reference_positions = compile_constrained_suffix(tokenizer, prompts[0])
    for prompt in prompts[1:]:
        plan, positions = compile_constrained_suffix(tokenizer, prompt)
        if plan != reference_plan or positions != reference_positions:
            raise ValueError('JSON token plan differs across persona prompts')
    return (reference_plan, reference_positions)

def decode_choices_from_tokens(generated_ids: Sequence[int], plan: Sequence[Sequence[int]], choice_positions: Sequence[int]) -> list[int]:
    if len(generated_ids) != len(plan):
        raise ValueError(f'Expected {len(plan)} generated tokens, received {len(generated_ids)}')
    answers = []
    for offset in choice_positions:
        token_id = int(generated_ids[offset])
        allowed = list(plan[offset])
        if token_id not in allowed:
            raise ValueError(f'Unexpected token {token_id} at answer offset {offset}')
        answers.append(allowed.index(token_id) + 1)
    return answers

def prepare_candidate_frame(records: pd.DataFrame) -> pd.DataFrame:
    """Keep actual candidate records and attach stable inference identifiers."""
    required = ['year', 'chamber', 'raw_row', 'party', 'party_prompt', 'incumbency', 'incumbency_prompt', 'elected', 'complete_response', 'persona_complete']
    missing = [column for column in required if column not in records]
    if missing:
        raise ValueError(f'Missing candidate columns: {missing}')
    candidates = records.loc[records['persona_complete'], required].copy()
    candidates['respondent_id'] = candidates.apply(respondent_id, axis=1)
    if candidates['respondent_id'].duplicated().any():
        raise ValueError('respondent_id is not unique')
    return candidates.sort_values(['year', 'raw_row'], kind='stable').reset_index(drop=True)

def config_fingerprint(model_id: str, temperature: float, seed: int) -> tuple[str, dict]:
    payload = {'model_id': model_id, 'temperature': float(temperature), 'seed': int(seed), 'prompt_version': PROMPT_VERSION, 'issues': ISSUES, 'issue_texts': ISSUE_TEXTS, 'option_text': OPTION_TEXT, 'generation': 'single shared constrained JSON continuation'}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return (hashlib.sha256(encoded).hexdigest(), payload)

def _write_metadata_once(path: Path, fingerprint: str, payload: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing.get('config_sha256') != fingerprint:
            raise RuntimeError('Existing joint responses were produced by a different prompt/model/config. Use a new output directory instead of mixing cached results.')
        return
    path.write_text(json.dumps({'config_sha256': fingerprint, **payload}, ensure_ascii=False, indent=2), encoding='utf-8')

def run_joint_inference(candidates: pd.DataFrame, output_dir: Path, model_id: str=MODEL_ID, temperature: float=0.8, batch_size: int=16, seed: int=42, limit_respondents: int | None=None) -> pd.DataFrame:
    """Run or resume one joint six-answer sample for each candidate record."""
    import torch
    from transformers import set_seed
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / 'joint_candidate_responses.csv'
    progress_path = output_dir / 'progress.json'
    log_path = output_dir / 'progress.log'
    metadata_path = output_dir / 'run_metadata.json'
    fingerprint, payload = config_fingerprint(model_id, temperature, seed)
    _write_metadata_once(metadata_path, fingerprint, payload)
    selected = candidates.head(limit_respondents).copy() if limit_respondents else candidates.copy()
    existing = pd.read_csv(response_path) if response_path.exists() else pd.DataFrame()
    if not existing.empty:
        existing = existing.drop_duplicates('respondent_id', keep='last')
    completed = set(existing['respondent_id'].astype(str)) if not existing.empty else set()
    pending = selected.loc[~selected['respondent_id'].astype(str).isin(completed)].copy()
    if pending.empty:
        return existing
    tokenizer, model = load_model(model_id)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    validation_rows = selected.iloc[np.linspace(0, len(selected) - 1, min(20, len(selected)), dtype=int)]
    validation_prompts = [build_joint_prompt(row) for _, row in validation_rows.iterrows()]
    plan, choice_positions = validate_plan_across_prompts(tokenizer, validation_prompts)
    max_new_tokens = len(plan)
    started = time.time()
    completed_this_run = 0
    for start in range(0, len(selected), batch_size):
        manifest_batch = selected.iloc[start:start + batch_size].copy()
        batch = manifest_batch.loc[~manifest_batch['respondent_id'].astype(str).isin(completed)].copy()
        if batch.empty:
            continue
        prompts = [build_joint_prompt(row) for _, row in batch.iterrows()]
        encoded = tokenizer(prompts, add_special_tokens=True, padding=True, return_tensors='pt')
        input_length = int(encoded['input_ids'].shape[1])
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}

        def allowed_tokens_fn(_batch_id, input_ids):
            offset = int(input_ids.shape[-1]) - input_length
            if 0 <= offset < len(plan):
                return plan[offset]
            return [tokenizer.eos_token_id]
        batch_number = start // batch_size
        set_seed(seed + batch_number)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, prefix_allowed_tokens_fn=allowed_tokens_fn, renormalize_logits=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
        suffix_ids = generated[:, input_length:].detach().cpu().tolist()
        new_rows = []
        for (_, candidate), token_ids in zip(batch.iterrows(), suffix_ids):
            answers = decode_choices_from_tokens(token_ids, plan, choice_positions)
            raw_suffix = tokenizer.decode(token_ids, skip_special_tokens=True)
            new_rows.append({'respondent_id': candidate['respondent_id'], 'year': int(candidate['year']), 'chamber': candidate['chamber'], 'raw_row': int(candidate['raw_row']), 'party': candidate['party'], 'incumbency': candidate['incumbency'], 'elected': bool(candidate['elected']), 'complete_response': bool(candidate['complete_response']), 'raw_generated_json': raw_suffix, **dict(zip(ISSUES, answers))})
        new_frame = pd.DataFrame(new_rows)
        new_frame.to_csv(response_path, mode='a', header=not response_path.exists(), index=False, encoding='utf-8-sig')
        existing = pd.concat([existing, new_frame], ignore_index=True)
        completed.update((str(value) for value in batch['respondent_id']))
        completed_this_run += len(new_rows)
        elapsed = time.time() - started
        state = {'config_sha256': fingerprint, 'total_pending_at_start': len(pending), 'completed_this_run': completed_this_run, 'remaining': len(pending) - completed_this_run, 'elapsed_seconds': elapsed, 'estimated_remaining_seconds': elapsed / completed_this_run * (len(pending) - completed_this_run) if completed_this_run else None, 'last_update': pd.Timestamp.now(tz='Asia/Tokyo').isoformat()}
        progress_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        message = f'[{state['last_update']}] {completed_this_run}/{len(pending)} respondents; remaining={state['remaining']}; elapsed={elapsed / 60:.1f} min'
        print(message, flush=True)
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(message + '\n')
    return pd.read_csv(response_path)

def correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[ISSUES].corr(method='spearman').loc[ISSUES, ISSUES]

def matrix_rho(left: pd.DataFrame, right: pd.DataFrame) -> float:
    indices = np.triu_indices(len(ISSUES), k=1)
    left_values = left.loc[ISSUES, ISSUES].to_numpy(float)[indices]
    right_values = right.loc[ISSUES, ISSUES].to_numpy(float)[indices]
    keep = np.isfinite(left_values) & np.isfinite(right_values)
    return float(spearmanr(left_values[keep], right_values[keep]).statistic)

def exact_label_p(left: pd.DataFrame, right: pd.DataFrame) -> tuple[float, float]:
    """Exact one-sided 6! label-permutation test, without a +1 correction."""
    observed = matrix_rho(left, right)
    right_array = right.loc[ISSUES, ISSUES].to_numpy(float)
    permuted_values = []
    for permutation in itertools.permutations(range(len(ISSUES))):
        permuted = pd.DataFrame(right_array[np.ix_(permutation, permutation)], index=ISSUES, columns=ISSUES)
        permuted_values.append(matrix_rho(left, permuted))
    values = np.asarray(permuted_values, dtype=float)
    return (observed, float(np.mean(values >= observed)))

def summarize_joint_outputs(responses: pd.DataFrame, records: pd.DataFrame, output_dir: Path, model_slug: str='gemma') -> pd.DataFrame:
    """Compare matched respondent-level model and UTAS correlation matrices.

    ``model_slug`` affects output filenames only.  Keeping the statistical
    implementation here shared ensures that Gemma and Llama are summarized
    with exactly the same sample definitions, missing-item rule, matrix
    construction, and exact label-permutation test.
    """
    if not model_slug or any((char not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for char in model_slug)):
        raise ValueError('model_slug must contain only lowercase ASCII letters, digits, or underscores')
    output_dir.mkdir(parents=True, exist_ok=True)
    model_frame = responses.copy()
    model_frame.loc[model_frame['year'].eq(2004), 'Public Safety'] = np.nan
    record_lookup = records.copy()
    record_lookup['respondent_id'] = record_lookup.apply(respondent_id, axis=1)
    utas_columns = ['respondent_id', 'elected', 'complete_response', *ISSUES]
    utas_frame = record_lookup[utas_columns].copy()
    definitions = {'complete_respondents': lambda frame: frame['complete_response'].astype(bool), 'elected_respondents': lambda frame: frame['complete_response'].astype(bool) & frame['elected'].astype(bool)}
    rows = []
    for sample, selector in definitions.items():
        model_subset = model_frame.loc[selector(model_frame)].copy()
        ids = set(model_subset['respondent_id'].astype(str))
        utas_subset = utas_frame.loc[utas_frame['respondent_id'].astype(str).isin(ids)].copy()
        if len(model_subset) != len(utas_subset):
            raise ValueError(f'Matched row count differs for {sample}: model={len(model_subset)}, UTAS={len(utas_subset)}')
        model_matrix = correlation_matrix(model_subset)
        utas_matrix = correlation_matrix(utas_subset)
        model_matrix.to_csv(output_dir / f'joint_{model_slug}_{sample}_spearman_matrix.csv', encoding='utf-8-sig')
        utas_matrix.to_csv(output_dir / f'matched_utas_{sample}_spearman_matrix.csv', encoding='utf-8-sig')
        rho, p = exact_label_p(model_matrix, utas_matrix)
        rows.append({'sample': sample, 'n_records': len(model_subset), 'n_public_safety_pairs': int(model_subset['Public Safety'].notna().sum()), 'matrix_spearman_rho': rho, 'exact_one_sided_p': p})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / f'joint_{model_slug}_vs_utas_summary.csv', index=False, encoding='utf-8-sig')
    return summary

def generate(argv=None):
    global ISSUE_TEXTS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=['gemma', 'llama'])
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--mapping-workbook', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true', help='Load weights and generate; otherwise only save an offline plan')
    a = p.parse_args(argv)
    question_path = Path(__file__).resolve().parents[1] / 'config/utas_questions.local.json'
    ISSUE_TEXTS = json.loads(question_path.read_text())
    if set(ISSUE_TEXTS) != set(ISSUES) or any((not isinstance(v, str) or not v.strip() or v.startswith('<') for v in ISSUE_TEXTS.values())):
        raise ValueError('Supply six original local UTAS question strings')
    records, _ = build_utas_records(a.raw_dir, a.mapping_workbook)
    candidates = prepare_candidate_frame(records)
    models = json.loads((Path(__file__).resolve().parents[1] / 'config/models.json').read_text())
    prompts = [build_joint_prompt(row) for _, row in candidates.iterrows()]
    payload = {'model_id': models[a.model], 'temperature': 0.8, 'seed': 42, 'batch_size': 16, 'candidate_records': len(candidates), 'elected_complete_records': int((candidates.elected & candidates.complete_response).sum()), 'ordered_candidates_sha256': hashlib.sha256(candidates.to_csv(index=False).encode()).hexdigest(), 'ordered_prompts_sha256': hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest(), 'script_hashes': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).parent/'common/utas_records.py']}}
    a.output.mkdir(parents=True, exist_ok=True)
    guard = a.output / 'generation_plan.json'
    if guard.exists():
        if json.loads(guard.read_text()) != payload:
            raise ValueError('Inputs or settings changed: use a new output directory')
    elif (a.output / 'joint_candidate_responses.csv').exists():
        raise ValueError("Existing responses without this runner's plan: use a new directory")
    else:
        guard.write_text(json.dumps(payload, indent=2))
    if not a.execute:
        print(json.dumps(payload, indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this runner; CPU fallback would change numerical precision')
    env = {'versions': {k: importlib.metadata.version(k) for k in ['torch', 'transformers', 'tokenizers', 'accelerate', 'numpy', 'pandas']}, 'gpu': torch.cuda.get_device_name(0)}
    ep = a.output / 'generation_environment.json'
    if ep.exists() and json.loads(ep.read_text()) != env:
        raise ValueError('Execution environment changed: use a new output directory')
    ep.write_text(json.dumps(env, indent=2))
    response = a.output / 'joint_candidate_responses.csv'
    if response.exists():
        old = pd.read_csv(response)
        ids = old.respondent_id.astype(str).tolist()
        expected = candidates.respondent_id.astype(str).tolist()
        if ids != expected[:len(ids)] or (len(ids) != len(expected) and len(ids) % 16):
            raise ValueError('Checkpoint is not a complete ordered batch prefix; preserve it and recover before resuming')
        if not old[['Defense', 'Social Welfare', 'Public Works', 'Fiscal Stimulus', 'North Korea', 'Public Safety']].isin([1, 2, 3, 4, 5]).all().all():
            raise ValueError('Invalid checkpoint answers')
    run_joint_inference(candidates, a.output, model_id=models[a.model], temperature=0.8, batch_size=16, seed=42)

def validate_responses(responses, records):
    required = ['respondent_id', 'year', 'raw_row', 'elected', 'complete_response', *ISSUES]
    if set(required) - set(responses):
        raise ValueError('Missing required response columns')
    if responses.respondent_id.duplicated().any():
        raise ValueError('Duplicate respondent identifiers')
    expected_ids = responses.apply(respondent_id, axis=1)
    if not expected_ids.equals(responses.respondent_id):
        raise ValueError('Identifier does not match survey year and raw row')
    lookup = records.copy()
    lookup['respondent_id'] = lookup.apply(respondent_id, axis=1)
    lookup = lookup.set_index('respondent_id', verify_integrity=True)
    if not set(responses.respondent_id) <= set(lookup.index):
        raise ValueError('Responses contain unknown UTAS records')
    matched = lookup.loc[responses.respondent_id]
    for col in ['elected', 'complete_response']:
        if not responses[col].isin([True, False]).all():
            raise ValueError(f'{col} must contain booleans')
        if not np.array_equal(responses[col].to_numpy(), matched[col].to_numpy()):
            raise ValueError(f'{col} differs from source UTAS records')
    if not matched.persona_complete.all():
        raise ValueError('Incomplete persona attributes')
    if not responses[ISSUES].isin([1, 2, 3, 4, 5]).all().all():
        raise ValueError('Expected six integer choices from 1 to 5 for every response')
    eligible = set(lookup.index[lookup.persona_complete & lookup.complete_response])
    if not eligible <= set(responses.respondent_id):
        raise ValueError('Missing responses for eligible complete UTAS records')

def summarize(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=['gemma', 'llama'])
    p.add_argument('--responses', type=Path, required=True)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--mapping-workbook', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError('Choose a new output directory')
    records, qc = build_utas_records(a.raw_dir, a.mapping_workbook)
    responses = pd.read_csv(a.responses, dtype={'respondent_id': str})
    validate_responses(responses, records)
    summary = summarize_joint_outputs(responses, records, a.output, 'gemma' if a.model == 'gemma' else 'llama31')
    qc.to_csv(a.output / 'year_sample_qc.csv', index=False)
    inputs = [a.responses, a.mapping_workbook] + [a.raw_dir / c.filename for c in YEAR_CONFIGS.values()]
    scripts = [Path(__file__), Path(__file__), Path(__file__).parent/'common/utas_records.py']
    manifest = {'model': a.model, 'versions': {k: importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scipy', 'openpyxl']}, 'input_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, 'script_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts}, 'generation_performed': False}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))

def figures(argv=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from common import plotting as layout
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError('Choose a fresh output directory')
    config=json.loads(a.inputs.read_text());data={};sources={}
    for model in ['gemma','llama']:
        data[model]={}
        for field in ['elected','output']:
            path=Path(config[model][field]).expanduser()
            if not path.is_absolute():path=a.inputs.resolve().parent/path
            data[model][field]=load_matrix(path);sources[model+'.'+field]=hashlib.sha256(path.read_bytes()).hexdigest()
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold','axes.unicode_minus':True})
    a.output.mkdir(parents=True);records={}
    for model,name,expected in [('gemma','Fig1',(.529,.015)),('llama','S1_Fig',(.364,.078))]:
        o=data[model]['output'];u=data[model]['elected']
        fig=plt.figure(figsize=(7.5,3.3));layout.matrix(fig,[.085,.29,.23,.55],o,'A',True);layout.matrix(fig,[.415,.29,.23,.55],u,'B',True);layout.scatter(fig,[.765,.29,.22,.55],o,u,'Output correlation','UTAS correlation','C',expected,True)
        fig.savefig(a.output/(name+'.pdf'));fig.savefig(a.output/(name+'.png'),dpi=600)
        with Image.open(a.output/(name+'.png')) as image:image.convert('RGB').save(a.output/(name+'.tif'),compression='tiff_lzw',dpi=(600,600))
        records[name]=list(layout.statistics);layout.statistics.clear();plt.close(fig)
    (a.output/'verification.json').write_text(json.dumps({'source_sha256':sources,'statistics':records,'annotation_overlap_assertions':'passed'},indent=2))

def main():
    import sys
    ops={'generate':generate,'summarize':summarize,'figures':figures}
    if len(sys.argv)<2 or sys.argv[1] not in ops:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=list(ops));p.parse_args()
    else:ops[sys.argv[1]](sys.argv[2:])
if __name__=='__main__':main()
