"""UTAS recall preparation, generation, judging, scoring and table export.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import argparse,json,hashlib,re,unicodedata
import numpy as np
import pandas as pd


recall_distribution_kernel_COUNT_KEYS = ['1', '2', '3', '4', '5', 'missing']

recall_distribution_kernel_PERCENT_KEYS = list('12345')

recall_distribution_kernel_PERCENT_SUM_TOLERANCE = 0.2

recall_distribution_kernel_count_columns = ['c1', 'c2', 'c3', 'c4', 'c5', 'missing']

recall_distribution_kernel_valid_count_columns = recall_distribution_kernel_count_columns[:5]

def recall_distribution_kernel__to_int(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    match = re.fullmatch('\\s*([0-9][0-9,]*)\\s*', str(value))
    return int(match.group(1).replace(',', '')) if match else None

def recall_distribution_kernel__to_float(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    match = re.fullmatch('\\s*([0-9]+(?:\\.[0-9]+)?)\\s*[%％]?\\s*', str(value))
    return float(match.group(1)) if match else None

def recall_distribution_kernel__json_objects(text):
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != '{':
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj

def recall_distribution_kernel_parse_count_prediction(text):
    if re.fullmatch('\\s*(UNKNOWN|不明|わかりません)[。\\s]*', text, flags=re.I):
        return {'parse_status': 'abstained', 'total_n': None, 'counts': None, 'percentages': None}
    for obj in recall_distribution_kernel__json_objects(text):
        counts_obj = obj.get('counts', obj)
        if not isinstance(counts_obj, dict):
            continue
        aliases = {'1': ('1', 'c1', 'C1'), '2': ('2', 'c2', 'C2'), '3': ('3', 'c3', 'C3'), '4': ('4', 'c4', 'C4'), '5': ('5', 'c5', 'C5'), 'missing': ('missing', 'MISSING', 'na', 'NA', '99')}
        counts = {}
        for key, candidates in aliases.items():
            value = next((counts_obj[c] for c in candidates if c in counts_obj), None)
            counts[key] = recall_distribution_kernel__to_int(value)
        total_n = recall_distribution_kernel__to_int(obj.get('total_n', obj.get('TOTAL_N')))
        if all((counts[key] is not None for key in recall_distribution_kernel_COUNT_KEYS)):
            return {'parse_status': 'parsed_json', 'total_n': total_n, 'counts': counts, 'percentages': None}
    patterns = {'total_n': 'TOTAL_N\\s*[:=]\\s*([0-9][0-9,]*)', '1': 'C1\\s*[:=]\\s*([0-9][0-9,]*)', '2': 'C2\\s*[:=]\\s*([0-9][0-9,]*)', '3': 'C3\\s*[:=]\\s*([0-9][0-9,]*)', '4': 'C4\\s*[:=]\\s*([0-9][0-9,]*)', '5': 'C5\\s*[:=]\\s*([0-9][0-9,]*)', 'missing': 'MISSING\\s*[:=]\\s*([0-9][0-9,]*)'}
    found = {key: re.search(pattern, text, flags=re.I) for key, pattern in patterns.items()}
    if all((found[key] is not None for key in recall_distribution_kernel_COUNT_KEYS)):
        counts = {key: recall_distribution_kernel__to_int(found[key].group(1)) for key in recall_distribution_kernel_COUNT_KEYS}
        total_n = recall_distribution_kernel__to_int(found['total_n'].group(1)) if found['total_n'] else None
        return {'parse_status': 'parsed_key_value', 'total_n': total_n, 'counts': counts, 'percentages': None}
    return {'parse_status': 'invalid', 'total_n': None, 'counts': None, 'percentages': None}

def recall_distribution_kernel_parse_percentage_prediction(text):
    if re.fullmatch('\\s*(UNKNOWN|不明|わかりません)[。\\s]*', text, flags=re.I):
        return {'parse_status': 'abstained', 'total_n': None, 'counts': None, 'percentages': None}
    aliases = {'1': ('1', 'p1', 'P1'), '2': ('2', 'p2', 'P2'), '3': ('3', 'p3', 'P3'), '4': ('4', 'p4', 'P4'), '5': ('5', 'p5', 'P5')}
    for obj in recall_distribution_kernel__json_objects(text):
        percentages_obj = obj.get('percentages', obj.get('percentage', obj.get('proportions', obj)))
        if not isinstance(percentages_obj, dict):
            continue
        percentages = {}
        for key, candidates in aliases.items():
            value = next((percentages_obj[c] for c in candidates if c in percentages_obj), None)
            percentages[key] = recall_distribution_kernel__to_float(value)
        if all((percentages[key] is not None for key in recall_distribution_kernel_PERCENT_KEYS)):
            return {'parse_status': 'parsed_percentage_json', 'total_n': None, 'counts': None, 'percentages': percentages}
    patterns = {key: f'(?:P{key}|^\\s*{key}(?:（[^）]*）|\\([^)]*\\))?)\\s*[:=]\\s*([0-9]+(?:\\.[0-9]+)?)\\s*[%％]?' for key in recall_distribution_kernel_PERCENT_KEYS}
    found = {key: re.search(pattern, text, flags=re.I | re.M) for key, pattern in patterns.items()}
    if all((found[key] is not None for key in recall_distribution_kernel_PERCENT_KEYS)):
        percentages = {key: recall_distribution_kernel__to_float(found[key].group(1)) for key in recall_distribution_kernel_PERCENT_KEYS}
        return {'parse_status': 'parsed_percentage_key_value', 'total_n': None, 'counts': None, 'percentages': percentages}
    return {'parse_status': 'invalid', 'total_n': None, 'counts': None, 'percentages': None}

def recall_distribution_kernel_parse_prediction(text, test_mode):
    if test_mode == 'percentage_recall':
        return recall_distribution_kernel_parse_percentage_prediction(text)
    return recall_distribution_kernel_parse_count_prediction(text)

def recall_distribution_kernel_baseline_distribution(item_id):
    row = recall_distribution_kernel_truth_map[item_id]
    peers = recall_distribution_kernel_truth_df[(recall_distribution_kernel_truth_df['issue'] == row['issue']) & (recall_distribution_kernel_truth_df['item_id'] != item_id) & (recall_distribution_kernel_truth_df['analysis_group'] == 'historical_recall_target')]
    assert not peers.empty, item_id
    peer_props = peers[recall_distribution_kernel_count_columns].div(peers['total_n'], axis=0)
    return peer_props.mean(axis=0).to_numpy(dtype=float)

def recall_distribution_kernel_baseline_valid_distribution(item_id):
    row = recall_distribution_kernel_truth_map[item_id]
    peers = recall_distribution_kernel_truth_df[(recall_distribution_kernel_truth_df['issue'] == row['issue']) & (recall_distribution_kernel_truth_df['item_id'] != item_id) & (recall_distribution_kernel_truth_df['analysis_group'] == 'historical_recall_target')]
    assert not peers.empty, item_id
    peer_valid_n = peers[recall_distribution_kernel_valid_count_columns].sum(axis=1)
    peer_props = peers[recall_distribution_kernel_valid_count_columns].div(peer_valid_n, axis=0)
    return peer_props.mean(axis=0).to_numpy(dtype=float)

def recall_distribution_kernel_score_record(record):
    item_id = record['item_id']
    truth = recall_distribution_kernel_truth_map[item_id]
    true_counts = np.array([truth[col] for col in recall_distribution_kernel_count_columns], dtype=int)
    true_n = int(truth['total_n'])
    true_props = true_counts / true_n
    true_valid_counts = np.array([truth[col] for col in recall_distribution_kernel_valid_count_columns], dtype=int)
    true_valid_n = int(true_valid_counts.sum())
    true_valid_props = true_valid_counts / true_valid_n
    baseline_full_tvd = float(0.5 * np.abs(recall_distribution_kernel_baseline_distribution(item_id) - true_props).sum())
    baseline_valid_tvd = float(0.5 * np.abs(recall_distribution_kernel_baseline_valid_distribution(item_id) - true_valid_props).sum())
    baseline_tvd = baseline_valid_tvd if record['test_mode'] == 'percentage_recall' else baseline_full_tvd
    parsed = record.get('parsed') or recall_distribution_kernel_parse_prediction(record.get('raw_output', ''), record['test_mode'])
    parse_status = parsed.get('parse_status', 'invalid')
    result = {'run_key': record['run_key'], 'model_key': record['model_key'], 'model_id': record['model_id'], 'prompt_format': record['prompt_format'], 'item_id': item_id, 'year': truth['year'], 'issue': truth['issue'], 'question_code': truth['question_code'], 'analysis_group': truth['analysis_group'], 'test_mode': record['test_mode'], 'prompt_variant': record['prompt_variant'], 'run_index': record['run_index'], 'parse_status': parse_status, 'true_n': true_n, 'true_valid_n': true_valid_n, 'predicted_total_n': parsed.get('total_n'), 'baseline_tvd': baseline_tvd, 'baseline_full_tvd': baseline_full_tvd, 'baseline_valid_tvd': baseline_valid_tvd, 'valid_output': False, 'valid_counts': False, 'exact_counts': False, 'exact': False, 'near_exact_or_better': False, 'mae_pp': None, 'max_abs_pp': None, 'tvd': None, 'tvd_normalized': None, 'beats_other_year_mean_baseline': False, 'raw_output': record.get('raw_output', '')}
    if parse_status == 'abstained':
        result.update({'classification': 'abstained', 'valid_output': False, 'valid_counts': False, 'exact': False, 'near_exact_or_better': False})
        return result
    if record['test_mode'] == 'percentage_recall':
        percentages_dict = parsed.get('percentages')
        if not percentages_dict:
            result.update({'classification': 'invalid', 'valid_output': False, 'valid_counts': False, 'exact': False, 'near_exact_or_better': False})
            return result
        pred_percentages = np.array([percentages_dict[key] for key in recall_distribution_kernel_PERCENT_KEYS], dtype=float)
        finite_nonnegative = bool(np.isfinite(pred_percentages).all() and (pred_percentages >= 0).all())
        percentage_sum = float(pred_percentages.sum())
        sum_correct = bool(finite_nonnegative and abs(percentage_sum - 100.0) <= recall_distribution_kernel_PERCENT_SUM_TOLERANCE)
        valid_output = finite_nonnegative and sum_correct
        tvd_normalized = None
        if finite_nonnegative and percentage_sum > 0:
            normalized_props = pred_percentages / percentage_sum
            tvd_normalized = float(0.5 * np.abs(normalized_props - true_valid_props).sum())
        result.update({'valid_output': valid_output, 'valid_counts': valid_output, 'sum_correct': sum_correct, 'percentage_sum': percentage_sum, 'percentage_sum_tolerance': recall_distribution_kernel_PERCENT_SUM_TOLERANCE, 'tvd_normalized': tvd_normalized, 'exact_counts': False, 'exact': False})
        for index, key in enumerate(recall_distribution_kernel_PERCENT_KEYS):
            result[f'pred_pct_{key}'] = float(pred_percentages[index])
            result[f'true_pct_{key}'] = float(true_valid_props[index] * 100)
        if not valid_output:
            result.update({'classification': 'invalid', 'near_exact_or_better': False})
            return result
        pred_props = pred_percentages / 100.0
        abs_pp = np.abs(pred_percentages - true_valid_props * 100)
        tvd = float(0.5 * np.abs(pred_props - true_valid_props).sum())
        near_exact = float(abs_pp.max()) <= 1.0 and tvd <= 0.02
        distribution_close = tvd <= 0.05
        if near_exact:
            classification = 'near_exact'
        elif distribution_close:
            classification = 'distribution_close'
        else:
            classification = 'no_match'
        result.update({'classification': classification, 'near_exact_or_better': near_exact, 'mae_pp': float(abs_pp.mean()), 'max_abs_pp': float(abs_pp.max()), 'tvd': tvd, 'beats_other_year_mean_baseline': tvd < baseline_valid_tvd})
        return result
    counts_dict = parsed.get('counts')
    if not counts_dict:
        result.update({'classification': 'invalid', 'valid_output': False, 'valid_counts': False, 'exact': False, 'near_exact_or_better': False})
        return result
    pred_counts = np.array([counts_dict[key] for key in recall_distribution_kernel_COUNT_KEYS], dtype=int)
    nonnegative = bool((pred_counts >= 0).all())
    predicted_sum = int(pred_counts.sum())
    sum_correct = predicted_sum == true_n
    n_correct = parsed.get('total_n') == true_n
    strict_n_requirement = record['test_mode'] != 'strict_recall' or n_correct
    valid_counts = nonnegative and sum_correct
    tvd_normalized = None
    if nonnegative and predicted_sum > 0:
        normalized_pred_props = pred_counts / predicted_sum
        tvd_normalized = float(0.5 * np.abs(normalized_pred_props - true_props).sum())
    result.update({'valid_output': valid_counts, 'valid_counts': valid_counts, 'sum_correct': sum_correct, 'n_correct': n_correct, 'predicted_sum': predicted_sum, 'tvd_normalized': tvd_normalized})
    for index, key in enumerate(recall_distribution_kernel_COUNT_KEYS):
        result[f'pred_{key}'] = int(pred_counts[index])
        result[f'true_{key}'] = int(true_counts[index])
    if not valid_counts:
        result.update({'classification': 'invalid', 'exact': False, 'near_exact_or_better': False})
        return result
    pred_props = pred_counts / true_n
    abs_pp = np.abs(pred_props - true_props) * 100
    tvd = 0.5 * np.abs(pred_props - true_props).sum()
    exact_counts = bool(np.array_equal(pred_counts, true_counts))
    exact = exact_counts and strict_n_requirement
    near_exact = float(abs_pp.max()) <= 1.0 and float(tvd) <= 0.02
    distribution_close = float(tvd) <= 0.05
    if exact:
        classification = 'exact'
    elif near_exact:
        classification = 'near_exact'
    elif distribution_close:
        classification = 'distribution_close'
    else:
        classification = 'no_match'
    result.update({'classification': classification, 'exact_counts': exact_counts, 'exact': exact, 'near_exact_or_better': exact or near_exact, 'mae_count': float(np.abs(pred_counts - true_counts).mean()), 'max_abs_count_error': int(np.abs(pred_counts - true_counts).max()), 'mae_pp': float(abs_pp.mean()), 'max_abs_pp': float(abs_pp.max()), 'tvd': float(tvd), 'beats_other_year_mean_baseline': float(tvd) < float(baseline_tvd)})
    return result

def recall_distribution_kernel_assert_resume_compatible(record, expected, run_key):
    """Refuse to reuse a success produced by a different prompt/config/model."""
    mismatches = {key: {'saved': record.get(key), 'current': value} for key, value in expected.items() if record.get(key) != value}
    if mismatches:
        raise RuntimeError(f'Stale cached result for {run_key}: ' + json.dumps(mismatches, ensure_ascii=False, default=str))

score_question_recall_MODEL_CONFIGS = {'pt': {'model_id': 'tokyotech-llm/Gemma-2-Llama-Swallow-9b-pt-v0.1', 'prompt_format': 'plain_completion'}, 'instruct': {'model_id': 'tokyotech-llm/Gemma-2-Llama-Swallow-9b-it-v0.1', 'prompt_format': 'chat_template'}}

def score_question_recall_normalize_text(text: object) -> str:
    value = unicodedata.normalize('NFKC', str(text))
    value = value.replace('，', '、').replace(',', '、')
    value = re.sub('\\s+', '', value)
    value = value.strip('"\'`「」『』【】()[]{}')
    return re.sub('[。．.]+$', '', value)

def score_question_recall_extract_first_completion(raw_output: object, fixed_prefix: str) -> str:
    text = unicodedata.normalize('NFKC', str(raw_output)).strip()
    text = re.sub('^```(?:text)?\\s*', '', text, flags=re.I)
    text = re.sub('\\s*```$', '', text)
    text = re.sub('^(?:続き|回答|設問文の続き|completion)\\s*[:：]\\s*', '', text, flags=re.I)
    text = text.lstrip('"\'`「『 ')
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate = nonempty_lines[0] if nonempty_lines else ''
    if '。' in candidate:
        candidate = candidate.split('。', 1)[0]
    normalized_candidate = score_question_recall_normalize_text(candidate)
    normalized_prefix = score_question_recall_normalize_text(fixed_prefix)
    if normalized_candidate.startswith(normalized_prefix):
        normalized_candidate = normalized_candidate[len(normalized_prefix):]
    return normalized_candidate

def score_question_recall_levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = (right, left)
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_left != char_right)))
        previous = current
    return previous[-1]

def score_question_recall_score_record(record: dict) -> dict:
    prediction = score_question_recall_extract_first_completion(record.get('raw_output', ''), record['fixed_prefix'])
    target = score_question_recall_normalize_text(record['expected_continuation'])
    distance = score_question_recall_levenshtein_distance(prediction, target)
    denominator = max(len(prediction), len(target), 1)
    return {'run_key': record['run_key'], 'model_key': record['model_key'], 'model_id': record['model_id'], 'model_revision': record.get('model_revision'), 'condition': record['condition'], 'item_id': record['item_id'], 'year': record.get('year'), 'issue': record['issue'], 'identifier_shown': record.get('identifier_shown', ''), 'fixed_prefix': record['fixed_prefix'], 'expected_continuation': record['expected_continuation'], 'raw_output': record.get('raw_output', ''), 'extracted_prediction': prediction, 'normalized_target': target, 'official_full_text': score_question_recall_normalize_text(record['fixed_prefix'] + record['expected_continuation']), 'generated_full_text': score_question_recall_normalize_text(record['fixed_prefix'] + prediction), 'exact_match': prediction == target, 'edit_distance': distance, 'edit_similarity': 1 - distance / denominator}

def score_question_recall_load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'Invalid JSONL at {path}:{line_number}') from exc
    return records

def score_question_recall_main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-outputs', type=Path, required=True)
    parser.add_argument('--prompt-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    results_dir = args.output.resolve()
    if results_dir.exists():
        raise FileExistsError('Choose a fresh output directory')
    raw_path = args.raw_outputs.resolve()
    manifest_path = args.prompt_manifest.resolve()
    records = [row for row in score_question_recall_load_jsonl(raw_path) if row.get('record_type') == 'substantive' and row.get('status') == 'ok']
    if len(records) != 202:
        raise RuntimeError(f'Expected 202 successful records, found {len(records)}')
    if len({row['run_key'] for row in records}) != len(records):
        raise RuntimeError('Duplicate successful run_key entries')
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 101 or not manifest['prompt_id'].is_unique:
        raise RuntimeError('Prompt manifest must contain 101 unique prompts')
    expected_keys = {f'{model_key}::{prompt_id}' for model_key in score_question_recall_MODEL_CONFIGS for prompt_id in manifest['prompt_id']}
    actual_keys = {row['run_key'] for row in records}
    if actual_keys != expected_keys:
        raise RuntimeError(f'Run-key mismatch: missing={len(expected_keys - actual_keys)}, unexpected={len(actual_keys - expected_keys)}')
    for row in records:
        config = score_question_recall_MODEL_CONFIGS[row['model_key']]
        expected_metadata = {'model_id': config['model_id'], 'prompt_format': config['prompt_format'], 'dtype': 'bfloat16', 'quantization': 'none', 'do_sample': False, 'max_new_tokens': 96, 'random_seed': 20260825}
        mismatches = {key: {'saved': row.get(key), 'expected': expected} for key, expected in expected_metadata.items() if row.get(key) != expected}
        if mismatches:
            raise RuntimeError(f'Metadata mismatch for {row['run_key']}: {mismatches}')
    scored_df = pd.DataFrame([score_question_recall_score_record(row) for row in records])
    scored_df['exact_int'] = scored_df['exact_match'].astype(int)
    results_dir.mkdir(parents=True)
    scored_df.to_csv(results_dir / 'scored_outputs.csv', index=False, encoding='utf-8-sig')
    condition_summary_df = scored_df.groupby(['model_key', 'model_id', 'condition']).agg(n=('run_key', 'count'), exact_rate=('exact_int', 'mean'), mean_edit_similarity=('edit_similarity', 'mean'), median_edit_similarity=('edit_similarity', 'median')).reset_index()
    condition_summary_df.to_csv(results_dir / 'condition_summary.csv', index=False, encoding='utf-8-sig')
    issue_summary_df = scored_df.groupby(['model_key', 'condition', 'issue']).agg(n=('run_key', 'count'), exact_rate=('exact_int', 'mean'), mean_edit_similarity=('edit_similarity', 'mean'), median_edit_similarity=('edit_similarity', 'median')).reset_index()
    issue_summary_df.to_csv(results_dir / 'issue_summary.csv', index=False, encoding='utf-8-sig')
    audit = {'successful_records': len(records), 'manifest_prompts': len(manifest), 'input_sha256': {'raw_outputs': hashlib.sha256(raw_path.read_bytes()).hexdigest(), 'prompt_manifest': hashlib.sha256(manifest_path.read_bytes()).hexdigest(), 'scoring_script': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    (results_dir / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(condition_summary_df.to_string(index=False))
    print(f'Scored rows: {len(scored_df)}')

recall_judge_protocol_OPENAI_JUDGE_MODEL = 'gpt-5.4'

recall_judge_protocol_ANTHROPIC_JUDGE_MODEL = 'claude-opus-4-6'

recall_judge_protocol_GEMINI_JUDGE_MODEL = 'gemini-3.1-pro-preview'

recall_judge_protocol_JUDGE_PROMPT_VERSION = 'utas-question-text-judge-v1'

def recall_judge_protocol_as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'true', '1', 'yes'}

recall_judge_protocol_JUDGE_CLASSES = ['near_verbatim', 'semantic_equivalent', 'partial_or_related', 'non_matching']

recall_judge_protocol_JUDGE_SCHEMA = {'type': 'object', 'properties': {'classification': {'type': 'string', 'enum': recall_judge_protocol_JUDGE_CLASSES}, 'confidence': {'type': 'number'}, 'reason': {'type': 'string'}}, 'required': ['classification', 'confidence', 'reason'], 'additionalProperties': False}

recall_judge_protocol_JUDGE_INSTRUCTIONS = '\n'.join(['You are a strict evaluator of Japanese survey-question text.', 'Compare only the official completed sentence and the generated completed sentence.', 'Do not infer whether any model memorized a survey, training data, or a codebook.', 'Do not use outside knowledge. Evaluate only the relationship between the two supplied sentences.', '', 'Classification rules:', '- near_verbatim: The main wording and word order are almost the same. Differences are limited to orthography, particles, inflection, punctuation, or a very small number of words. Do not use this label for substantial paraphrases.', '- semantic_equivalent: The sentences express substantively the same policy proposition and stance, but the generated sentence is a clear paraphrase rather than near-verbatim reproduction.', '- partial_or_related: The generated sentence is on the same topic but is incomplete, more general, more specific, ambiguous, or does not fully preserve the official proposition and stance.', '- non_matching: The generated sentence differs in meaning or stance, is unrelated, uninterpretable, or empty.', '', 'Set confidence from 0 to 1. Give a short reason in Japanese. Return only the required structured result.'])

def recall_judge_protocol_build_blind_prompt(row):
    return f'公式の完成文:\n{row['official_full_text']}\n\n生成された完成文:\n{row['generated_full_text']}'

def recall_judge_protocol_prompt_hash(prompt):
    payload = {'version': recall_judge_protocol_JUDGE_PROMPT_VERSION, 'instructions': recall_judge_protocol_JUDGE_INSTRUCTIONS, 'schema': recall_judge_protocol_JUDGE_SCHEMA, 'prompt': prompt}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()

def recall_judge_protocol_load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'Invalid JSONL at {path}:{line_number}') from exc
    return rows

def score_distribution_recall_main(argv=None):
    global recall_distribution_kernel_truth_df, recall_distribution_kernel_truth_map
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['raw-outputs', 'prompt-manifest', 'truth', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    truth = pd.DataFrame(json.loads(args.truth.read_text(encoding='utf-8')))
    if len(truth) != 95 or not truth.item_id.is_unique:
        raise ValueError('Expected 95 unique historical truth items')
    counts = truth[recall_distribution_kernel_count_columns]
    if not ((counts >= 0).all().all() and (counts % 1 == 0).all().all() and counts.sum(axis=1).eq(truth.total_n).all() and truth[recall_distribution_kernel_valid_count_columns].sum(axis=1).gt(0).all()):
        raise ValueError('Invalid official counts')
    recall_distribution_kernel_truth_df = truth
    recall_distribution_kernel_truth_map = truth.set_index('item_id').to_dict('index')
    manifest = pd.read_csv(args.prompt_manifest)
    expected = {}
    for model_key, config in score_question_recall_MODEL_CONFIGS.items():
        for row in manifest.to_dict('records'):
            key = f'{model_key}::{row['item_id']}::{row['test_mode']}::{row['prompt_variant']}::0'
            if key in expected:
                raise ValueError('Duplicate manifest key')
            expected[key] = dict(model_key=model_key, **config, dtype='bfloat16', quantization='none', do_sample=False, max_new_tokens=192, random_seed=20260825, prompt_sha256=row['prompt_sha256'])
    records = [r for r in score_question_recall_load_jsonl(args.raw_outputs) if r.get('record_type') == 'substantive' and r.get('status') == 'ok']
    keys = [r['run_key'] for r in records]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError('Duplicate, missing or unexpected successful run keys')
    scored = []
    for record in records:
        recall_distribution_kernel_assert_resume_compatible(record, expected[record['run_key']], record['run_key'])
        parsed = recall_distribution_kernel_parse_prediction(record.get('raw_output', ''), record['test_mode'])
        if record.get('parsed') and record['parsed'] != parsed:
            raise ValueError('Cached parsing differs from raw text: ' + record['run_key'])
        scored.append(recall_distribution_kernel_score_record(dict(record, parsed=parsed)))
    frame = pd.DataFrame(scored)
    args.output.mkdir(parents=True)
    frame.to_csv(args.output / 'scored_outputs.csv', index=False, encoding='utf-8-sig')
    sources = dict(raw_outputs=args.raw_outputs, prompt_manifest=args.prompt_manifest, truth=args.truth, runner=Path(__file__), kernel=Path(__file__))
    audit = dict(rows=len(frame), reparsed=True, input_sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in sources.items()})
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(f'Scored {len(frame)} saved outputs')

validate_judge_logs_MODELS = {'openai': recall_judge_protocol_OPENAI_JUDGE_MODEL, 'anthropic': recall_judge_protocol_ANTHROPIC_JUDGE_MODEL, 'gemini': recall_judge_protocol_GEMINI_JUDGE_MODEL}

def validate_judge_logs_select(scored, raw):
    f = scored.copy()
    f['exact_match'] = f.exact_match.map(recall_judge_protocol_as_bool)
    if f.run_key.duplicated().any():
        raise ValueError('Duplicate run keys')
    for target, suffix in [('official_full_text', 'expected_continuation'), ('generated_full_text', 'extracted_prediction')]:
        if target not in f:
            f[target] = f.fixed_prefix.fillna('') + f[suffix].fillna('')
    valid = {}
    for _, row in f[~f.exact_match].iterrows():
        sha = recall_judge_protocol_prompt_hash(recall_judge_protocol_build_blind_prompt(row))
        for provider, model in validate_judge_logs_MODELS.items():
            valid[provider, row.run_key] = (model, sha)
    latest = {}
    counts = Counter()
    for record in raw:
        if record.get('status') != 'ok':
            counts['non_success'] += 1
            continue
        key = (record.get('provider'), record.get('run_key'))
        expected = valid.get(key)
        if expected is None:
            counts['outside_current_targets'] += 1
            continue
        if (record.get('configured_model'), record.get('judge_prompt_sha256')) != expected:
            counts['model_or_prompt_mismatch'] += 1
            continue
        if record.get('classification') not in recall_judge_protocol_JUDGE_CLASSES:
            raise ValueError('Invalid classification in matching successful record')
        if key in latest:
            counts['superseded_success'] += 1
        latest[key] = record
    counts['selected'] = len(latest)
    counts['expected'] = len(valid)
    return (pd.DataFrame(list(latest.values())), dict(counts))

def validate_judge_logs_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scored-texts', type=Path, required=True)
    p.add_argument('--raw-judgments', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    result, counts = validate_judge_logs_select(pd.read_csv(a.scored_texts), recall_judge_protocol_load_jsonl(a.raw_judgments))
    a.output.mkdir(parents=True)
    result.to_csv(a.output / 'judge_results_long.csv', index=False)
    (a.output / 'audit.json').write_text(json.dumps({'counts': counts, 'configured_models': validate_judge_logs_MODELS, 'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.scored_texts, a.raw_judgments, Path(__file__), Path(__file__)]}}, indent=2))
    print(counts)

build_recall_consensus_PROVIDERS = ['openai', 'anthropic', 'gemini']

def build_recall_consensus_build(scored_df, long_df):
    if scored_df.run_key.duplicated().any() or long_df.duplicated(['provider', 'run_key']).any():
        raise ValueError('Duplicate records')
    if not long_df.provider.isin(build_recall_consensus_PROVIDERS).all():
        raise ValueError('Unknown provider')
    if not long_df.classification.isin(['exact', 'near_verbatim', 'semantic_equivalent', 'partial_or_related', 'non_matching']).all():
        raise ValueError('Unknown classification')
    latest = {(r['provider'], r['run_key']): r for r in long_df.to_dict('records')}
    consensus_rows = []
    manual_rows = []
    for _, row in scored_df.iterrows():
        base = {'run_key': row['run_key'], 'model_key': row['model_key'], 'condition': row['condition'], 'year': row.get('year'), 'issue': row['issue'], 'official_full_text': row['official_full_text'], 'generated_full_text': row['generated_full_text'], 'exact_match': bool(row['exact_match']), 'edit_similarity': row['edit_similarity']}
        if row['exact_match']:
            consensus_rows.append({**base, 'openai_classification': 'exact', 'anthropic_classification': 'exact', 'gemini_classification': 'exact', 'n_judges_available': 3, 'judge_unanimous': True, 'majority_count': 3, 'consensus_type': 'exact', 'final_classification': 'exact', 'manual_review_required': False})
            continue
        judge_rows = {provider: latest.get((provider, row['run_key'])) for provider in build_recall_consensus_PROVIDERS}
        judge_classes = {provider: judge_row.get('classification') if judge_row else None for provider, judge_row in judge_rows.items()}
        available_classes = [value for value in judge_classes.values() if value is not None]
        class_counts = pd.Series(available_classes).value_counts()
        majority_count = int(class_counts.max()) if len(class_counts) else 0
        all_three_available = len(available_classes) == 3
        unanimous = bool(all_three_available and majority_count == 3)
        majority_available = bool(all_three_available and majority_count >= 2)
        if unanimous:
            consensus_type = 'unanimous'
        elif majority_available:
            consensus_type = 'majority_2_of_3'
        else:
            consensus_type = 'manual_review'
        final_class = str(class_counts.index[0]) if majority_available else 'manual_review'
        result = {**base, 'openai_classification': judge_classes['openai'], 'openai_confidence': judge_rows['openai'].get('confidence') if judge_rows['openai'] else None, 'openai_reason': judge_rows['openai'].get('reason') if judge_rows['openai'] else None, 'anthropic_classification': judge_classes['anthropic'], 'anthropic_confidence': judge_rows['anthropic'].get('confidence') if judge_rows['anthropic'] else None, 'anthropic_reason': judge_rows['anthropic'].get('reason') if judge_rows['anthropic'] else None, 'gemini_classification': judge_classes['gemini'], 'gemini_confidence': judge_rows['gemini'].get('confidence') if judge_rows['gemini'] else None, 'gemini_reason': judge_rows['gemini'].get('reason') if judge_rows['gemini'] else None, 'n_judges_available': len(available_classes), 'judge_unanimous': unanimous, 'majority_count': majority_count, 'consensus_type': consensus_type, 'final_classification': final_class, 'manual_review_required': not majority_available}
        consensus_rows.append(result)
        if not majority_available:
            manual_rows.append(result)
    return pd.DataFrame(consensus_rows)

def build_recall_consensus_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scored-texts', type=Path, required=True)
    p.add_argument('--judge-results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    result = build_recall_consensus_build(pd.read_csv(a.scored_texts), pd.read_csv(a.judge_results))
    a.output.mkdir(parents=True)
    result.to_csv(a.output / 'judge_consensus.csv', index=False)
    result[result.manual_review_required].to_csv(a.output / 'manual_review_queue.csv', index=False)
    (a.output / 'manifest.json').write_text(json.dumps({'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.scored_texts, a.judge_results, Path(__file__)]}, 'api_calls': False}, indent=2))

summarize_utas_recall_CLASSES = ['exact', 'near_verbatim', 'semantic_equivalent', 'partial_or_related', 'non_matching']

def summarize_utas_recall_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scored-distributions', type=Path, required=True)
    p.add_argument('--judge-consensus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    scored = pd.read_csv(a.scored_distributions)
    judges = pd.read_csv(a.judge_consensus)
    b = scored[(scored.test_mode == 'percentage_recall') & (scored.prompt_variant == 'C') & scored.year.between(2003, 2024)].copy()
    c = judges[(judges.condition == 'correct_identifier') & judges.year.between(2003, 2024)].copy()
    distributions = []
    classifications = []
    details = []
    for model, label in [('pt', 'Pretrained'), ('instruct', 'Instruction-tuned')]:
        f = b[b.model_key == model].copy()
        g = c[c.model_key == model].copy()
        for x in [f, g]:
            if len(x) != 89 or x.duplicated(['year', 'issue']).any():
                raise ValueError('Expected 89 distinct year/issue items per model')
        if set(zip(f.year, f.issue)) != set(zip(g.year, g.issue)):
            raise ValueError('Distribution and wording targets differ')
        pred = f[[f'pred_pct_{i}' for i in range(1, 6)]].to_numpy(float)
        truth = f[[f'true_pct_{i}' for i in range(1, 6)]].to_numpy(float)
        if not np.isfinite(pred).all() or (pred < 0).any() or (pred.sum(1) <= 0).any():
            raise ValueError('Unparseable or invalid distribution')
        if not np.isfinite(truth).all() or (truth < 0).any() or (not np.allclose(truth.sum(1), 100)):
            raise ValueError('Invalid official proportions')
        tvd = 0.5 * np.abs(pred / pred.sum(1)[:, None] - truth / 100).sum(1)
        if not np.allclose(tvd, f.tvd_normalized, atol=1e-12):
            raise ValueError('Recomputed TVD differs from saved scores')
        uniform = 0.5 * np.abs(0.2 - truth / 100).sum(1)
        distributions.append({'Gemma variant': label, 'n': 89, 'mean_tvd': float(tvd.mean()), **{f'percent_le_{t:.2f}': float(100 * np.mean(tvd <= t)) for t in [0.05, 0.1, 0.2]}, 'uniform_mean_tvd': float(uniform.mean())})
        details.append(f[['model_key', 'year', 'issue', 'item_id']].assign(recomputed_tvd=tvd, uniform_tvd=uniform))
        if not g.final_classification.isin(summarize_utas_recall_CLASSES).all():
            raise ValueError('Unknown judge classification')
        counts = g.final_classification.value_counts()
        classifications.append({'Gemma variant': label, 'n': 89, **{k: int(counts.get(k, 0)) for k in summarize_utas_recall_CLASSES}})
    a.output.mkdir(parents=True)
    pd.DataFrame(distributions).to_csv(a.output / 'distribution_summary.csv', index=False)
    pd.concat(details).to_csv(a.output / 'distribution_items.csv', index=False)
    pd.DataFrame(classifications).to_csv(a.output / 'question_summary.csv', index=False)
    c.to_csv(a.output / 'included_judgments.csv', index=False)
    (a.output / 'manifest.json').write_text(json.dumps({'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.scored_distributions, a.judge_consensus, Path(__file__)]}, 'generation_rerun': False, 'judging_rerun': False}, indent=2))
    print(pd.DataFrame(distributions).to_string(index=False))
    print(pd.DataFrame(classifications).to_string(index=False))

def one(frame, **criteria):
    selected = frame
    for key, value in criteria.items():
        selected = selected[selected[key] == value]
    if len(selected) != 1:
        raise ValueError('Expected one row: ' + str(criteria))
    return selected.iloc[0]

def tables(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--recall',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError('Choose a fresh output directory')
    dist=pd.read_csv(a.recall/'distribution_summary.csv');question=pd.read_csv(a.recall/'question_summary.csv')
    if len(dist)!=2 or len(question)!=2:raise ValueError('Expected two variants')
    rows=[];qrows=[]
    for variant in ['Pretrained', 'Instruction-tuned']:
        v = one(dist, **{'Gemma variant': variant})
        q = one(question, **{'Gemma variant': variant})
        if v.n != 89 or q.n != 89:
            raise ValueError('Expected 89 targets per variant')
        if not 0 <= v.mean_tvd <= 1 or not 0 <= v.uniform_mean_tvd <= 1:
            raise ValueError('Invalid TVD')
        row = {'Gemma variant': variant, 'Mean TVD': f'{v.mean_tvd:.3f}'}
        for threshold in ['0.05', '0.10', '0.20']:
            value = v['percent_le_' + threshold]
            if not 0 <= value <= 100:
                raise ValueError('Invalid percentage')
            row['Outputs with TVD ≤ ' + threshold + ' (%)'] = f'{value:.1f}'
        row['Uniform reference mean TVD'] = f'{v.uniform_mean_tvd:.3f}'
        rows.append(row)
        row = {'Gemma variant': variant, 'n': 89}
        counts = [q[c] for c in ['exact', 'near_verbatim', 'semantic_equivalent', 'partial_or_related', 'non_matching']]
        if any((c < 0 or int(c) != c for c in counts)) or sum(counts) != 89:
            raise ValueError('Invalid class counts')
        row.update(dict(zip(['Exact', 'Near-verbatim', 'Semantic equivalent', 'Partial or related', 'Non-matching'], map(int, counts))))
        qrows.append(row)
    a.output.mkdir(parents=True)
    for name,f in [('S13_Table_B',pd.DataFrame(rows)),('S13_Table_C',pd.DataFrame(qrows))]:
        f.to_csv(a.output/(name+'.csv'),index=False)
        lines=['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']
        lines+=['| '+' | '.join(map(str,row))+' |' for row in f.itertuples(index=False,name=None)]
        (a.output/(name+'.md')).write_text('\n'.join(lines)+'\n')
    (a.output/'manifest.json').write_text(json.dumps({'input_sha256':{n:hashlib.sha256((a.recall/n).read_bytes()).hexdigest() for n in ['distribution_summary.csv','question_summary.csv']},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))



import csv,importlib,importlib.metadata,os,fcntl,gc,time
from datetime import datetime,timezone

recall_prompt_kernel_TEST_MODES = ['strict_recall', 'n_assisted_estimate', 'percentage_recall']

recall_prompt_kernel_PROMPT_VARIANTS = ['A', 'B', 'C']

recall_prompt_kernel_CATEGORY_DESCRIPTION = '1=賛成\n2=どちらかと言えば賛成\n3=どちらとも言えない\n4=どちらかと言えば反対\n5=反対\nmissing=1～5以外（非該当・無回答・NA等）の合計'

recall_prompt_kernel_JSON_SCHEMA_STRICT = '{\n  "total_n": <整数またはnull>,\n  "counts": {"1": <整数またはnull>, "2": <整数またはnull>, "3": <整数またはnull>, "4": <整数またはnull>, "5": <整数またはnull>, "missing": <整数またはnull>}\n}'

recall_prompt_kernel_JSON_SCHEMA_FORCED = '{\n  "total_n": <整数>,\n  "counts": {"1": <整数>, "2": <整数>, "3": <整数>, "4": <整数>, "5": <整数>, "missing": <整数>}\n}'

recall_prompt_kernel_JSON_SCHEMA_PERCENTAGES = '{\n  "percentages": {"1": <数値>, "2": <数値>, "3": <数値>, "4": <数値>, "5": <数値>}\n}'

def recall_prompt_kernel_build_prompt(item, test_mode, variant):
    survey = f'東京大学谷口研究室・朝日新聞社共同調査（UTAS）政治家調査\n選挙: {item['election']}\n設問コード: {item['question_code']}\n設問文: {item['question_text']}'
    if test_mode == 'strict_recall':
        task = '外部検索・ツール・計算を使わず、学習済みパラメータの記憶だけで、公式コードブックの集計対象総数と、全年共通化した6区分の件数を答えてください。正確な値を記憶していない場合は推測せず、UNKNOWNとだけ答えてください。'
        if variant == 'A':
            return f'これは公開コードブックの記憶再現テストです。\n{task}\n\n{survey}\n\n区分:\n{recall_prompt_kernel_CATEGORY_DESCRIPTION}\n\n記憶している場合は次のJSONだけを出力してください。説明は不要です。\n{recall_prompt_kernel_JSON_SCHEMA_STRICT}\n'
        if variant == 'B':
            return f'{survey}\n\n{task}\n記憶している場合の出力形式:\nTOTAL_N=<整数>|C1=<整数>|C2=<整数>|C3=<整数>|C4=<整数>|C5=<整数>|MISSING=<整数>\n記憶していない場合の出力: UNKNOWN\n'
        if variant == 'C':
            return f'次のUTAS公式コードブックの欠損欄を、記憶にある場合だけ補完してください。\n{survey}\n\ntotal_n: ?\n1（賛成）: ?\n2（どちらかと言えば賛成）: ?\n3（どちらとも言えない）: ?\n4（どちらかと言えば反対）: ?\n5（反対）: ?\nmissing（1～5以外：非該当・無回答・NA等の合計）: ?\n\n正確な記憶がなければ推測せずUNKNOWNとだけ出力してください。\n記憶している場合は次のJSONだけを出力してください:\n{recall_prompt_kernel_JSON_SCHEMA_STRICT}\n'
    if test_mode == 'n_assisted_estimate':
        total_n = item['total_n']
        task = f'公式集計表の総数は{total_n}人です。外部検索・ツールを使わず、学習済みパラメータに含まれる情報と最善の推定に基づいて、全年共通化した6区分の件数を整数で回答してください。件数の合計は必ず{total_n}にしてください。'
        if variant == 'A':
            return f'以下の回答分布を推定してください。\n{survey}\n{task}\n\n区分:\n{recall_prompt_kernel_CATEGORY_DESCRIPTION}\n\n次のJSONだけを出力してください。説明は不要です。\n{recall_prompt_kernel_JSON_SCHEMA_FORCED}\n'
        if variant == 'B':
            return f'{survey}\n{task}\n出力形式:\nTOTAL_N={total_n}|C1=<整数>|C2=<整数>|C3=<整数>|C4=<整数>|C5=<整数>|MISSING=<整数>\n数字以外の説明は出力しないでください。\n'
        if variant == 'C':
            return f'UTAS政治家調査の集計表を完成させてください。\n{survey}\n総数: {total_n}\n\n1（賛成）: ?\n2（どちらかと言えば賛成）: ?\n3（どちらとも言えない）: ?\n4（どちらかと言えば反対）: ?\n5（反対）: ?\nmissing（1～5以外：非該当・無回答・NA等の合計）: ?\n\n6件数の合計を{total_n}にし、次のJSONだけを出力してください:\n{recall_prompt_kernel_JSON_SCHEMA_FORCED}\n'
    if test_mode == 'percentage_recall':
        task = '外部検索・ツール・計算を使わず、学習済みパラメータに含まれる情報だけに基づいて、公式集計における有効回答（選択肢1～5）内の各選択肢の割合（%）を答えてください。非該当・無回答・NA・system missingは分母にも分子にも含めないでください。5つの割合の合計は100.0%にしてください。正確な割合を記憶していない場合は、推測せずUNKNOWNとだけ答えてください。'
        if variant == 'A':
            return f'これは公開コードブックの割合の記憶再現テストです。\n{task}\n\n{survey}\n\n選択肢:\n1=賛成\n2=どちらかと言えば賛成\n3=どちらとも言えない\n4=どちらかと言えば反対\n5=反対\n\n記憶している場合は次のJSONだけを出力してください。数値の単位は%です。説明は不要です。\n{recall_prompt_kernel_JSON_SCHEMA_PERCENTAGES}\n'
        if variant == 'B':
            return f'{survey}\n\n{task}\n記憶している場合の出力形式（単位は%）:\nP1=<数値>|P2=<数値>|P3=<数値>|P4=<数値>|P5=<数値>\n記憶していない場合の出力: UNKNOWN\n'
        if variant == 'C':
            return f'次のUTAS公式コードブックの有効回答割合を、記憶にある場合だけ補完してください。\n{survey}\n\n1（賛成）: ? %\n2（どちらかと言えば賛成）: ? %\n3（どちらとも言えない）: ? %\n4（どちらかと言えば反対）: ? %\n5（反対）: ? %\n\n{task}\n記憶している場合は次のJSONだけを出力してください:\n{recall_prompt_kernel_JSON_SCHEMA_PERCENTAGES}\n'
    raise ValueError((test_mode, variant))

def recall_prompt_kernel_build_distribution_manifest(ITEMS):
    prompt_rows = []
    for item in ITEMS:
        for test_mode in recall_prompt_kernel_TEST_MODES:
            for variant in recall_prompt_kernel_PROMPT_VARIANTS:
                prompt = recall_prompt_kernel_build_prompt(item, test_mode, variant)
                if test_mode in {'strict_recall', 'percentage_recall'}:
                    assert f'候補者総数は{item['total_n']}' not in prompt
                    assert f'総数は{item['total_n']}' not in prompt
                    assert f'総数: {item['total_n']}' not in prompt
                    total_n_pattern = f'(?<!\\d){re.escape(str(item['total_n']))}(?!\\d)'
                    assert not re.search(total_n_pattern, prompt), f'total_n leaked into {test_mode} prompt: {item['item_id']}'
                prompt_rows.append({'item_id': item['item_id'], 'year': item['year'], 'issue': item['issue'], 'analysis_group': item['analysis_group'], 'question_code': item['question_code'], 'test_mode': test_mode, 'prompt_variant': variant, 'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(), 'prompt': prompt})
    return prompt_rows

def recall_prompt_kernel_build_context_prompt(item, condition):
    prefix = item['fixed_prefix']
    if condition == 'prefix_only':
        return prefix
    if condition != 'correct_identifier':
        raise ValueError(condition)
    identifier = item['question_identifier']
    identifier_label = '設問番号' if item['identifier_type'] == 'question_code' else '変数名'
    return f'東京大学谷口研究室・朝日新聞社共同調査（UTAS）政治家調査\n調査年: {item['year']}年\n{identifier_label}: {identifier}\n設問文: {prefix}'

def recall_prompt_kernel_build_question_manifest(ITEMS):
    prompt_rows = []
    seen_prefixes = set()
    for item in ITEMS:
        if item['fixed_prefix'] not in seen_prefixes:
            seen_prefixes.add(item['fixed_prefix'])
            context_prompt = recall_prompt_kernel_build_context_prompt(item, 'prefix_only')
            prompt_rows.append({'prompt_id': f'prefix_only::{item['issue']}', 'item_id': f'prefix_only_{item['issue'].lower().replace(' ', '_')}', 'condition': 'prefix_only', 'year': None, 'issue': item['issue'], 'identifier_shown': '', 'correct_identifier': '', 'fixed_prefix': item['fixed_prefix'], 'expected_continuation': item['expected_continuation'], 'context_prompt': context_prompt, 'prompt_sha256': hashlib.sha256(context_prompt.encode('utf-8')).hexdigest()})
        for condition in ('correct_identifier',):
            context_prompt = recall_prompt_kernel_build_context_prompt(item, condition)
            identifier_shown = item['question_identifier']
            prompt_rows.append({'prompt_id': f'{item['item_id']}::{condition}', 'item_id': item['item_id'], 'condition': condition, 'year': item['year'], 'issue': item['issue'], 'identifier_shown': identifier_shown, 'correct_identifier': item['question_identifier'], 'fixed_prefix': item['fixed_prefix'], 'expected_continuation': item['expected_continuation'], 'context_prompt': context_prompt, 'prompt_sha256': hashlib.sha256(context_prompt.encode('utf-8')).hexdigest()})
    return prompt_rows

def prepare_recall_inputs_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', type=Path, required=True, help='Locally prepared official target data; see docs/local_inputs.md')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    data = json.loads(args.targets.read_text(encoding='utf-8'))
    for name in ['distribution_items', 'question_items', 'truth']:
        rows = data[name]
        if len(rows) != 95 or len({r['item_id'] for r in rows}) != 95:
            raise ValueError(f'Expected 95 unique items in {name}')
    if not set((r['item_id'] for r in data['truth'])) == set((r['item_id'] for r in data['distribution_items'])) == set((r['item_id'] for r in data['question_items'])):
        raise ValueError('Item sets differ')
    for row in data['truth']:
        counts = [row[k] for k in ['c1', 'c2', 'c3', 'c4', 'c5', 'missing']]
        if any((not isinstance(v, int) or v < 0 for v in counts)) or sum(counts) != row['total_n'] or sum(counts[:5]) <= 0:
            raise ValueError('Invalid official counts')
    for row in data['question_items']:
        if row['fixed_prefix'] + row['expected_continuation'] != row['canonical_question_text']:
            raise ValueError('Question prefix and continuation do not reconstruct question')
    distribution = pd.DataFrame(recall_prompt_kernel_build_distribution_manifest(data['distribution_items']))
    question = pd.DataFrame(recall_prompt_kernel_build_question_manifest(data['question_items']))
    if len(distribution) != 855 or len(question) != 101:
        raise ValueError('Unexpected number of prompts')
    args.output.mkdir(parents=True)
    for name, frame in [('distribution', distribution), ('question', question)]:
        folder = args.output / name
        folder.mkdir()
        frame.to_csv(folder / 'prompt_manifest.csv', index=False, encoding='utf-8-sig')
    (args.output / 'distribution' / 'truth.json').write_text(json.dumps(data['truth'], ensure_ascii=False, indent=2) + '\n')
    sources = {'targets': args.targets, 'runner': Path(__file__), 'kernel': Path(__file__)}
    (args.output / 'audit.json').write_text(json.dumps({'distribution_prompts': len(distribution), 'question_prompts': len(question), 'input_sha256': {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in sources.items()}}, indent=2))
    print(f'Prepared {len(distribution)} distribution prompts and {len(question)} question prompts')

recall_distribution_generation_DO_SAMPLE = False

recall_distribution_generation_MAX_NEW_TOKENS = 192

def recall_distribution_generation_prepare_inputs(prompt, loaded_tokenizer, prompt_format, device):
    import torch
    if prompt_format == 'plain_completion':
        return loaded_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=4096).to(device)
    if prompt_format == 'chat_template':
        input_ids = loaded_tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, return_tensors='pt').to(device)
        return {'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids)}
    raise ValueError(f'Unknown prompt_format: {prompt_format}')

def recall_distribution_generation_generate_one(prompt, loaded_model, loaded_tokenizer, prompt_format):
    import torch
    if loaded_model is None or loaded_tokenizer is None:
        raise RuntimeError('Model is not loaded')
    inputs = recall_distribution_generation_prepare_inputs(prompt, loaded_tokenizer, prompt_format, loaded_model.device)
    input_length = inputs['input_ids'].shape[1]
    eos_token_ids = [loaded_tokenizer.eos_token_id]
    if prompt_format == 'chat_template':
        end_of_turn_id = loaded_tokenizer.convert_tokens_to_ids('<end_of_turn>')
        if end_of_turn_id is not None and end_of_turn_id != loaded_tokenizer.unk_token_id and (end_of_turn_id not in eos_token_ids):
            eos_token_ids.append(end_of_turn_id)
    with torch.inference_mode():
        output_ids = loaded_model.generate(**inputs, max_new_tokens=recall_distribution_generation_MAX_NEW_TOKENS, do_sample=recall_distribution_generation_DO_SAMPLE, pad_token_id=loaded_tokenizer.pad_token_id, eos_token_id=eos_token_ids, use_cache=True)
    new_ids = output_ids[0, input_length:]
    text = loaded_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    del inputs, output_ids, new_ids
    gc.collect()
    torch.cuda.empty_cache()
    return text

recall_question_generation_DO_SAMPLE = False

recall_question_generation_MAX_NEW_TOKENS = 96

recall_question_generation_IT_INSTRUCTION = '以下に示す文の続きを、説明を加えず、続きの文字列だけで答えてください。\n\n'

def recall_question_generation_prepare_inputs(context_prompt, tokenizer, prompt_format, device):
    import torch
    if prompt_format == 'plain_completion':
        encoded = tokenizer(context_prompt, return_tensors='pt', truncation=True, max_length=4096)
        return {key: value.to(device) for key, value in encoded.items()}
    if prompt_format == 'chat_template':
        user_prompt = recall_question_generation_IT_INSTRUCTION + context_prompt
        input_ids = tokenizer.apply_chat_template([{'role': 'user', 'content': user_prompt}], add_generation_prompt=True, return_tensors='pt').to(device)
        return {'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids)}
    raise ValueError(prompt_format)

def recall_question_generation_generate_one(context_prompt, model, tokenizer, prompt_format):
    import torch
    inputs = recall_question_generation_prepare_inputs(context_prompt, tokenizer, prompt_format, model.device)
    input_length = inputs['input_ids'].shape[1]
    eos_ids = [tokenizer.eos_token_id]
    if prompt_format == 'chat_template':
        end_turn_id = tokenizer.convert_tokens_to_ids('<end_of_turn>')
        if end_turn_id is not None and end_turn_id != tokenizer.unk_token_id and (end_turn_id not in eos_ids):
            eos_ids.append(end_turn_id)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=recall_question_generation_MAX_NEW_TOKENS, do_sample=recall_question_generation_DO_SAMPLE, pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids, use_cache=True)
    new_ids = output_ids[0, input_length:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    del inputs, output_ids, new_ids
    return text

generate_recall_MODELS = {'pt': ('tokyotech-llm/Gemma-2-Llama-Swallow-9b-pt-v0.1', '7858b8cd43769fc63a4b635340ffc3e3dbc77d2b', 'plain_completion'), 'instruct': ('tokyotech-llm/Gemma-2-Llama-Swallow-9b-it-v0.1', '9d643c2f9808d44d3151531d527c0eb480fb6417', 'chat_template')}

def generate_recall_atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)

def generate_recall_make_records(kind, model_key, manifest):
    model_id, revision, prompt_format = generate_recall_MODELS[model_key]
    records = []
    for original in csv.DictReader(manifest.open(encoding='utf-8-sig', newline='')):
        row = dict(original)
        prompt = row['prompt' if kind == 'distribution' else 'context_prompt']
        if hashlib.sha256(prompt.encode()).hexdigest() != row['prompt_sha256']:
            raise ValueError('Prompt hash mismatch')
        row['year'] = int(float(row['year'])) if row.get('year') else None
        if kind == 'distribution':
            key = f'{model_key}::{row['item_id']}::{row['test_mode']}::{row['prompt_variant']}::0'
            row['run_index'] = 0
        else:
            key = f'{model_key}::{row['prompt_id']}'
        row.update(run_key=key, record_type='substantive', model_key=model_key, model_id=model_id, model_revision=revision, prompt_format=prompt_format, dtype='bfloat16', quantization='none', do_sample=False, max_new_tokens=192 if kind == 'distribution' else 96, random_seed=20260825)
        records.append(row)
    if len({r['run_key'] for r in records}) != len(records):
        raise ValueError('Duplicate run keys')
    if len(records) != (855 if kind == 'distribution' else 101):
        raise ValueError('Use the complete historical manifest')
    return records

def generate_recall_run(args):
    records = generate_recall_make_records(args.kind, args.model, args.manifest)
    kernel_path = Path(__file__)
    config = {'kind': args.kind, 'model': args.model, 'model_config': generate_recall_MODELS[args.model], 'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(), 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'kernel_sha256': hashlib.sha256(kernel_path.read_bytes()).hexdigest()}
    config = json.loads(json.dumps(config))
    if not args.execute:
        print(json.dumps(dict(config, records=len(records), execution='plan only'), indent=2))
        return
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; no CPU inference fallback')
    config['environment'] = {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'numpy', 'accelerate']}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard = args.output / 'run_manifest.json'
        if guard.exists():
            if json.loads(guard.read_text()) != config:
                raise ValueError('Configuration/code/environment changed; use a new output directory')
        else:
            if any((path.name != '.lock' for path in args.output.iterdir())):
                raise ValueError('Nonempty output directory lacks run manifest')
            generate_recall_atomic_json(guard, config)
        cache = args.output / 'records'
        cache.mkdir(exist_ok=True)
        completed = {}
        expected = {hashlib.sha256(r['run_key'].encode()).hexdigest(): r for r in records}
        for path in cache.glob('*.json'):
            saved = json.loads(path.read_text())
            base = expected.get(path.stem)
            if base is None or any((saved.get(k) != v for k, v in base.items())) or saved.get('status') != 'ok':
                raise ValueError('Incompatible checkpoint')
            completed[saved['run_key']] = saved
        pending = [r for r in records if r['run_key'] not in completed]
        print(f'{len(completed)}/{len(records)} complete; {len(pending)} remaining', flush=True)
        if pending:
            torch.manual_seed(20260825)
            np.random.seed(20260825)
            model_id, revision, fmt = generate_recall_MODELS[args.model]
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True, low_cpu_mem_usage=True, attn_implementation='sdpa')
            model.eval()
            model.config.use_cache = True
            generate_one = {'distribution': recall_distribution_generation_generate_one, 'question': recall_question_generation_generate_one}[args.kind]
            for index, row in enumerate(pending):
                if (args.output / 'STOP').exists() or (args.max_new is not None and index >= args.max_new):
                    break
                prompt = row['prompt' if args.kind == 'distribution' else 'context_prompt']
                text = generate_one(prompt, model, tokenizer, fmt)
                saved = dict(row, status='ok', raw_output=text, created_at=datetime.now(timezone.utc).isoformat(), transformers_version=config['environment']['transformers'], torch_version=config['environment']['torch'])
                generate_recall_atomic_json(cache / (hashlib.sha256(row['run_key'].encode()).hexdigest() + '.json'), saved)
                completed[row['run_key']] = saved
                print(f'{len(completed)}/{len(records)} complete', flush=True)
        temp = args.output / 'raw_outputs.tmp'
        with temp.open('w', encoding='utf-8') as handle:
            for row in records:
                if row['run_key'] in completed:
                    handle.write(json.dumps(completed[row['run_key']], ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, args.output / 'raw_outputs.jsonl')

def generate_recall_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['distribution', 'question'], required=True)
    p.add_argument('--model', choices=list(generate_recall_MODELS), required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--max-new', type=int)
    a = p.parse_args(argv)
    if a.max_new is not None and a.max_new < 1:
        p.error('--max-new must be positive')
    generate_recall_run(a)

recall_judge_api_MAX_OUTPUT_TOKENS = 300

recall_judge_api_GEMINI_MAX_OUTPUT_TOKENS = 1024

recall_judge_api_GEMINI_THINKING_LEVEL = 'low'

def recall_judge_api_validate_judgment(result):
    if result.get('classification') not in recall_judge_protocol_JUDGE_CLASSES:
        raise ValueError(f'Unexpected classification: {result}')
    confidence = float(result.get('confidence'))
    if not 0 <= confidence <= 1:
        raise ValueError(f'Confidence must be in [0, 1]: {result}')
    reason = str(result.get('reason', '')).strip()
    if not reason:
        raise ValueError(f'Reason is empty: {result}')
    return {'classification': result['classification'], 'confidence': confidence, 'reason': reason}

def recall_judge_api_call_openai(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(model=recall_judge_protocol_OPENAI_JUDGE_MODEL, instructions=recall_judge_protocol_JUDGE_INSTRUCTIONS, input=prompt, temperature=0, max_output_tokens=recall_judge_api_MAX_OUTPUT_TOKENS, store=False, text={'format': {'type': 'json_schema', 'name': 'utas_question_text_judgment', 'strict': True, 'schema': recall_judge_protocol_JUDGE_SCHEMA}})
    parsed = recall_judge_api_validate_judgment(json.loads(response.output_text))
    return (parsed, {'response_id': response.id, 'returned_model': response.model, 'raw_response_text': response.output_text, 'usage': response.usage.model_dump() if response.usage else None})

def recall_judge_api_call_anthropic(prompt):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(model=recall_judge_protocol_ANTHROPIC_JUDGE_MODEL, max_tokens=recall_judge_api_MAX_OUTPUT_TOKENS, system=recall_judge_protocol_JUDGE_INSTRUCTIONS, messages=[{'role': 'user', 'content': prompt}], output_config={'format': {'type': 'json_schema', 'schema': recall_judge_protocol_JUDGE_SCHEMA}})
    if response.stop_reason in {'max_tokens', 'refusal'}:
        raise RuntimeError(f'Anthropic stop_reason={response.stop_reason}')
    text_blocks = [block.text for block in response.content if block.type == 'text']
    if not text_blocks:
        raise RuntimeError('Anthropic response has no text block')
    parsed = recall_judge_api_validate_judgment(json.loads(text_blocks[0]))
    return (parsed, {'response_id': response.id, 'returned_model': response.model, 'raw_response_text': text_blocks[0], 'stop_reason': response.stop_reason, 'usage': {'input_tokens': response.usage.input_tokens, 'output_tokens': response.usage.output_tokens}})

def recall_judge_api_model_dump_if_available(value):
    if value is None:
        return None
    if hasattr(value, 'model_dump'):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return str(value)

def recall_judge_api_call_gemini(prompt):
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(model=recall_judge_protocol_GEMINI_JUDGE_MODEL, contents=prompt, config=types.GenerateContentConfig(system_instruction=recall_judge_protocol_JUDGE_INSTRUCTIONS, temperature=0, max_output_tokens=recall_judge_api_GEMINI_MAX_OUTPUT_TOKENS, thinking_config=types.ThinkingConfig(thinking_level=recall_judge_api_GEMINI_THINKING_LEVEL), response_mime_type='application/json', response_json_schema=recall_judge_protocol_JUDGE_SCHEMA))
    raw_text = response.text
    if not raw_text:
        raise RuntimeError('Gemini response has no text')
    parsed = recall_judge_api_validate_judgment(json.loads(raw_text))
    returned_model = getattr(response, 'model_version', None)
    return (parsed, {'response_id': getattr(response, 'response_id', None), 'returned_model': str(returned_model or recall_judge_protocol_GEMINI_JUDGE_MODEL), 'thinking_level': recall_judge_api_GEMINI_THINKING_LEVEL, 'raw_response_text': raw_text, 'usage': recall_judge_api_model_dump_if_available(getattr(response, 'usage_metadata', None))})

def recall_judge_api_configure(provider, key):
    global OpenAI, anthropic, genai, types
    global OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY
    if provider == 'openai':
        from openai import OpenAI
        OPENAI_API_KEY = key
    elif provider == 'anthropic':
        import anthropic
        ANTHROPIC_API_KEY = key
    elif provider == 'gemini':
        from google import genai
        from google.genai import types
        GEMINI_API_KEY = key
    else:
        raise ValueError('Unknown provider')
    return globals()['recall_judge_api_call_' + provider]

run_recall_judges_MODELS = {'openai': recall_judge_protocol_OPENAI_JUDGE_MODEL, 'anthropic': recall_judge_protocol_ANTHROPIC_JUDGE_MODEL, 'gemini': recall_judge_protocol_GEMINI_JUDGE_MODEL}

def run_recall_judges_targets(path, providers):
    frame = pd.read_csv(path)
    if frame.run_key.duplicated().any():
        raise ValueError('Duplicate run keys')
    for target, suffix in [('official_full_text', 'expected_continuation'), ('generated_full_text', 'extracted_prediction')]:
        if target not in frame:
            frame[target] = frame.fixed_prefix.fillna('') + frame[suffix].fillna('')
    rows = []
    for provider in providers:
        for _, row in frame[~frame.exact_match.map(recall_judge_protocol_as_bool)].iterrows():
            prompt = recall_judge_protocol_build_blind_prompt(row)
            rows.append((dict(record_type='judge', provider=provider, configured_model=run_recall_judges_MODELS[provider], run_key=row.run_key, judge_prompt_version=recall_judge_protocol_JUDGE_PROMPT_VERSION, judge_prompt_sha256=recall_judge_protocol_prompt_hash(prompt)), prompt))
    return rows

def run_recall_judges_execute(rows, output, guard, callers, max_new=None):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / 'run_manifest.json'
        if path.exists():
            if json.loads(path.read_text()) != guard:
                raise ValueError('Inputs/code/configuration changed')
        else:
            if any((p.name != '.lock' for p in output.iterdir())):
                raise ValueError('Use an empty output directory')
            generate_recall_atomic_json(path, guard)
        cache = output / 'records'
        cache.mkdir(exist_ok=True)
        done = []
        new = 0
        for base, prompt in rows:
            key = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
            path = cache / (key + '.json')
            if path.exists():
                record = json.loads(path.read_text())
                if any((record.get(k) != v for k, v in base.items())) or record.get('status') != 'ok':
                    raise ValueError('Invalid checkpoint')
                done.append(record)
                continue
            if (output / 'STOP').exists() or (max_new is not None and new >= max_new):
                continue
            for attempt in range(1, 5):
                try:
                    result, metadata = callers[base['provider']](prompt)
                    record = dict(base, status='ok', timestamp_utc=datetime.now(timezone.utc).isoformat(), **result, **metadata)
                    generate_recall_atomic_json(path, record)
                    done.append(record)
                    break
                except Exception as exc:
                    if attempt < 4:
                        time.sleep(min(2 ** attempt, 8))
                    else:
                        with (output / 'errors.jsonl').open('a') as h:
                            h.write(json.dumps(dict(base, status='error', error_type=type(exc).__name__)) + '\n')
            new += 1
            print(f'{len(done)}/{len(rows)} successful; {new} items attempted this run', flush=True)
        temp = output / 'judge_raw_outputs.tmp'
        with temp.open('w', encoding='utf-8') as h:
            for record in done:
                h.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
            h.flush()
            os.fsync(h.fileno())
        os.replace(temp, output / 'judge_raw_outputs.jsonl')

def run_recall_judges_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scored-texts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--providers', nargs='+', choices=list(run_recall_judges_MODELS), default=list(run_recall_judges_MODELS))
    p.add_argument('--keys', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--max-new', type=int)
    a = p.parse_args(argv)
    if len(set(a.providers)) != len(a.providers):
        p.error('Duplicate providers')
    if a.max_new is not None and a.max_new < 1:
        p.error('--max-new must be positive')
    rows = run_recall_judges_targets(a.scored_texts, a.providers)
    guard = {'providers': a.providers, 'models': {k: run_recall_judges_MODELS[k] for k in a.providers}, 'input_sha256': hashlib.sha256(a.scored_texts.read_bytes()).hexdigest(), 'code_sha256': {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    if not a.execute:
        print(json.dumps(dict(guard, requests=len(rows), execution='plan only'), indent=2))
        return
    if a.keys is None:
        p.error('--keys local JSON is required with --execute')
    keys = json.loads(a.keys.read_text())
    for provider in a.providers:
        if not isinstance(keys.get(provider), str) or not keys[provider].strip():
            raise ValueError('Missing local API key for ' + provider)
    callers = {provider: recall_judge_api_configure(provider, keys[provider]) for provider in a.providers}
    packages = {'openai': 'openai', 'anthropic': 'anthropic', 'gemini': 'google-genai'}
    guard['sdk_versions'] = {provider: importlib.metadata.version(packages[provider]) for provider in a.providers}
    run_recall_judges_execute(rows, a.output, guard, callers, a.max_new)

def assemble(argv=None):
    """Combine verified S13 panels without recalculating their analyses."""
    parser=argparse.ArgumentParser()
    for name in ['a','b','c','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError('Choose a fresh output directory')
    panels={name:pd.read_csv(getattr(args,name),dtype=str,keep_default_na=False) for name in ['a','b','c']}
    for name,expected in [('a',4),('b',2),('c',2)]:
        if len(panels[name])!=expected:raise ValueError('Unexpected panel size: '+name)
    args.output.mkdir(parents=True)
    for name,frame in panels.items():
        stem='S13_Table_'+name.upper()
        frame.to_csv(args.output/(stem+'.csv'),index=False)
        lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
        lines+=['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)]
        (args.output/(stem+'.md')).write_text('\n'.join(lines)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'input_sha256':{name:hashlib.sha256(getattr(args,name).read_bytes()).hexdigest() for name in panels},'recalculated':False},indent=2))

def main():
    import sys
    operations={'prepare':prepare_recall_inputs_main,'generate':generate_recall_main,'judge':run_recall_judges_main,'score-distributions':score_distribution_recall_main,'score-questions':score_question_recall_main,'validate-judges':validate_judge_logs_main,'consensus':build_recall_consensus_main,'summarize':summarize_utas_recall_main,'tables':tables,'assemble':assemble}
    if len(sys.argv)<2 or sys.argv[1] not in operations:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=list(operations));p.parse_args()
    else:operations[sys.argv[1]](sys.argv[2:])
if __name__=='__main__':main()
