"""Shared annual UTAS coding and local workbook loading; no question text."""
from __future__ import annotations
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from .constants import ISSUES
import numpy as np
import pandas as pd
MAPPING_COLUMNS = {
    '年度別変数': ['調査年'],
    '政党対応': ['調査年', '元コード／文字列', '公式ラベル', 'プロンプト表現'],
    '新旧対応': ['調査年', '元コード／文字列', 'プロンプト表現', '標準区分'],
}

OPTION_TEXT = {1: '賛成', 2: 'どちらかと言えば賛成', 3: 'どちらとも言えない', 4: 'どちらかと言えば反対', 5: '反対'}
LABEL_TO_NUM = {label: value for value, label in OPTION_TEXT.items()}

@dataclass(frozen=True)
class YearConfig:
    year: int
    chamber: str
    filename: str
    party_column: str
    incumbency_column: str
    issue_columns: tuple[str | None, ...]
    candidate_mask: Callable[[pd.DataFrame], pd.Series]
    elected_mask: Callable[[pd.DataFrame], pd.Series]
    label_mode: bool = False

def _all(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=frame.index)
YEAR_CONFIGS = {2003: YearConfig(2003, '衆議院', '2003data.csv', 'partycod', 'incumben', ('defense', 'smallgov', 'publicen', 'keynes', 'nkorea', 'safety'), _all, lambda d: d['won'].isin([1, 2, 3])), 2004: YearConfig(2004, '参議院', '2004data.csv', 'party', 'incumben', ('defence', 'smallgov', 'publicen', 'keynes', 'nkorea', None), lambda d: d['mode'].eq(1), lambda d: d['won'].eq(1)), 2005: YearConfig(2005, '衆議院', '2005data.csv', 'partycod', 'incumben', ('defence', 'smallgov', 'publecen', 'keynes', 'nkorea', 'safety'), _all, lambda d: d['won'].isin([1, 2, 3])), 2007: YearConfig(2007, '参議院', '2007data.csv', 'partycod', 'incumben', ('defence', 'smallgov', 'publicen', 'keynes', 'nkorea', 'safety'), lambda d: d['reelect'].eq('改選'), lambda d: d['won'].eq('当選'), True), 2009: YearConfig(2009, '衆議院', '2009data.csv', 'PARTY', 'INCUMB', ('Q9_2', 'Q9_9', 'Q9_11', 'Q9_12', 'Q9_6', 'Q9_18'), _all, lambda d: d['RESULT'].isin([1, 2, 3])), 2010: YearConfig(2010, '参議院', '2010data.csv', 'PARTY', 'INCUMB', ('Q7_2', 'Q7_9', 'Q7_13', 'Q7_14', 'Q7_6', 'Q7_22'), lambda d: d['RESULT'].isin([0, 1]), lambda d: d['RESULT'].eq(1)), 2012: YearConfig(2012, '衆議院', '2012data.csv', 'PARTY', 'INCUMB', ('Q5_2', 'Q5_7', 'Q5_8', 'Q5_9', 'Q5_5', 'Q5_15'), _all, lambda d: d['RESULT'].isin([1, 2, 3])), 2013: YearConfig(2013, '参議院', '2013data.csv', 'PARTY', 'INCUMB', ('Q4_1', 'Q4_7', 'Q4_8', 'Q4_9', 'Q4_4', 'Q4_14'), lambda d: ~d['INCUMB'].eq(66), lambda d: d['RESULT'].eq(1)), 2014: YearConfig(2014, '衆議院', '2014data.csv', 'PARTY', 'INCUMB', ('Q6_1', 'Q6_5', 'Q6_6', 'Q6_7', 'Q6_3', 'Q6_11'), _all, lambda d: d['RESULT'].isin([1, 2, 3])), 2016: YearConfig(2016, '参議院', '2016data.csv', 'PARTY', 'INCUMB', ('Q3_1', 'Q3_5', 'Q3_6', 'Q3_7', 'Q3_3', 'Q3_13'), lambda d: ~d['RESULT'].eq(66), lambda d: d['RESULT'].eq(1)), 2017: YearConfig(2017, '衆議院', '2017data.csv', 'PARTY', 'INCUMB', ('Q4_1', 'Q4_6', 'Q4_7', 'Q4_8', 'Q4_3', 'Q4_11'), _all, lambda d: d['RESULT'].isin([2, 3, 4])), 2019: YearConfig(2019, '参議院', '2019data.csv', 'PARTY', 'INCUMB', ('Q4_1', 'Q4_4', 'Q4_5', 'Q4_6', 'Q4_3', 'Q4_9'), lambda d: ~d['RESULT'].eq(66), lambda d: d['RESULT'].eq(1)), 2021: YearConfig(2021, '衆議院', '2021data.csv', 'PARTY', 'INCUMB', ('Q6_1', 'Q6_6', 'Q6_7', 'Q6_8', 'Q6_3', 'Q6_13'), _all, lambda d: d['RESULT'].isin([2, 3, 4])), 2022: YearConfig(2022, '参議院', '2022data.csv', 'PARTY', 'INCUMB', ('Q4_1', 'Q4_9', 'Q4_10', 'Q4_11', 'Q4_3', 'Q4_19'), lambda d: ~d['RESULT'].eq(66), lambda d: d['RESULT'].eq(1)), 2024: YearConfig(2024, '衆議院', '2024data.csv', 'PARTY', 'INCUMB', ('Q4_1', 'Q4_5', 'Q4_6', 'Q4_7', 'Q4_2', 'Q4_12'), _all, lambda d: d['RESULT'].isin([2, 3, 4])), 2025: YearConfig(2025, '参議院', '2025data.csv', 'PARTY', 'INCUMB', ('Q4_1', 'Q4_6', 'Q4_7', 'Q4_8', 'Q4_3', 'Q4_15'), _all, lambda d: d['RESULT'].eq(1))}

def normalize_code(value):
    """Normalize Excel/CSV code types without changing literal strings."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return int(number) if number.is_integer() else number
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none', 'na'}:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number

def read_csv_flexibly(path: Path) -> tuple[pd.DataFrame, str]:
    raw = path.read_bytes()
    last_error: Exception | None = None
    for encoding in ('utf-8-sig', 'cp932', 'shift_jis', 'utf-16'):
        try:
            frame = pd.read_csv(io.StringIO(raw.decode(encoding)), sep=',')
            frame.columns = [str(column).strip() for column in frame.columns]
            return (frame, encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f'Could not read {path}: {last_error}')

def load_mapping_tables(workbook_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if workbook_path.suffix.lower() == '.json':
        payload = json.loads(workbook_path.read_text(encoding='utf-8'))
        if payload.get('schema_version') != 1:
            raise ValueError('Unsupported UTAS mapping schema')
        tables = payload['tables']
        frames = []
        for sheet, columns in MAPPING_COLUMNS.items():
            frame = pd.DataFrame(tables[sheet])
            if set(frame.columns) != set(columns):
                raise ValueError(f'Invalid mapping columns: {sheet}')
            frames.append(frame[columns])
        years, parties, incumbencies = frames
    else:
        years = pd.read_excel(workbook_path, sheet_name='年度別変数')
        parties = pd.read_excel(workbook_path, sheet_name='政党対応')
        incumbencies = pd.read_excel(workbook_path, sheet_name='新旧対応')
    for frame in (years, parties, incumbencies):
        frame['調査年'] = pd.to_numeric(frame['調査年'], errors='raise').astype(int)
        frame['元コード／文字列'] = frame.get('元コード／文字列', pd.Series(index=frame.index)).map(normalize_code)
    return (years, parties, incumbencies)

def _mapping_for_year(mapping: pd.DataFrame, year: int, value_column: str) -> dict:
    rows = mapping.loc[mapping['調査年'].eq(year)]
    if rows['元コード／文字列'].duplicated().any():
        duplicated = rows.loc[rows['元コード／文字列'].duplicated(), '元コード／文字列'].tolist()
        raise ValueError(f'{year}: duplicate mapping codes {duplicated}')
    return dict(zip(rows['元コード／文字列'], rows[value_column]))

def _clean_issue(series: pd.Series, label_mode: bool) -> pd.Series:
    if label_mode:
        cleaned = series.map(LABEL_TO_NUM)
    else:
        cleaned = pd.to_numeric(series, errors='coerce')
    return cleaned.where(cleaned.isin([1, 2, 3, 4, 5])).astype(float)

def build_utas_records(raw_dir: Path, workbook_path: Path, years: Iterable[int]=tuple(YEAR_CONFIGS)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return candidate records and a year-level QC table.

    Each output row is one actual candidate-wave record.  It contains the six
    cleaned UTAS answers, mapped party/incumbency labels, candidate/elected
    flags, and a complete-response flag matching the original pooled analysis
    (five available issues in 2004; all six in every other wave).
    """
    _, parties, incumbencies = load_mapping_tables(workbook_path)
    records: list[pd.DataFrame] = []
    qc: list[dict] = []
    for year in years:
        cfg = YEAR_CONFIGS[int(year)]
        frame, encoding = read_csv_flexibly(raw_dir / cfg.filename)
        candidate_mask = cfg.candidate_mask(frame).fillna(False).astype(bool)
        elected_mask = cfg.elected_mask(frame).fillna(False).astype(bool)
        subset = frame.loc[candidate_mask].copy()
        party_map = _mapping_for_year(parties, year, 'プロンプト表現')
        party_official_map = _mapping_for_year(parties, year, '公式ラベル')
        incumb_prompt_map = _mapping_for_year(incumbencies, year, 'プロンプト表現')
        incumb_std_map = _mapping_for_year(incumbencies, year, '標準区分')
        party_codes = subset[cfg.party_column].map(normalize_code)
        incumb_codes = subset[cfg.incumbency_column].map(normalize_code)
        unknown_party = sorted(set(party_codes.dropna()) - set(party_map), key=str)
        unknown_incumb = sorted(set(incumb_codes.dropna()) - set(incumb_prompt_map), key=str)
        if unknown_party or unknown_incumb:
            raise ValueError(f'{year}: unmapped party={unknown_party}, incumbency={unknown_incumb}')
        official_party_labels = party_codes.map(party_official_map)
        missing_official_codes = sorted(set(party_codes.loc[party_codes.notna() & official_party_labels.isna()]), key=str)
        if missing_official_codes:
            raise ValueError(f'{year}: party codes with missing official labels={missing_official_codes}')
        out = pd.DataFrame(index=subset.index)
        out['year'] = year
        out['chamber'] = cfg.chamber
        out['raw_row'] = subset.index.astype(int)
        out['party_code'] = party_codes
        out['party'] = official_party_labels
        out['party_prompt'] = party_codes.map(party_map)
        out['incumbency_code'] = incumb_codes
        out['incumbency'] = incumb_codes.map(incumb_std_map)
        out['incumbency_prompt'] = incumb_codes.map(incumb_prompt_map)
        out['elected'] = elected_mask.loc[subset.index].astype(bool).to_numpy()
        for issue, column in zip(ISSUES, cfg.issue_columns):
            out[issue] = np.nan if column is None else _clean_issue(subset[column], cfg.label_mode)
        available = [issue for issue, column in zip(ISSUES, cfg.issue_columns) if column is not None]
        out['complete_response'] = out[available].notna().all(axis=1)
        out['any_response'] = out[available].notna().any(axis=1)
        out['persona_complete'] = out[['party_prompt', 'incumbency_prompt']].notna().all(axis=1)
        records.append(out.reset_index(drop=True))
        qc.append({'year': year, 'encoding': encoding, 'raw_n': len(frame), 'candidate_n': int(candidate_mask.sum()), 'elected_n': int((candidate_mask & elected_mask).sum()), 'complete_response_n': int(out['complete_response'].sum()), 'persona_complete_n': int(out['persona_complete'].sum())})
    combined = pd.concat(records, ignore_index=True)
    return (combined, pd.DataFrame(qc))

def is_residual_party(label: object) -> bool:
    text = '' if pd.isna(label) else str(label)
    residual_markers = ('無所属', '諸派', 'その他', '支持政党なし', '該当なし', '不明')
    return any((marker in text for marker in residual_markers))
