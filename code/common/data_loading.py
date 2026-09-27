"""Load local inputs without altering row order, coding, or sample definitions."""
from pathlib import Path
import numpy as np
import pandas as pd
from .constants import THEMES, ISSUES, BLOCK_COLUMNS, ISSUE_COLUMNS_JP, EXPECTED_YEARS, PUBLIC_FILES

def clean_utas_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["調査年"] = pd.to_numeric(frame["調査年"], errors="raise").astype(int)
    for column in ISSUE_COLUMNS_JP:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[~frame[column].isin([1, 2, 3, 4, 5]), column] = np.nan
    years = sorted(frame["調査年"].unique().tolist())
    if years != EXPECTED_YEARS:
        raise ValueError(f"Unexpected election years in {path.name}: {years}")
    if frame.loc[frame["調査年"].eq(2004), "治安"].notna().any():
        raise ValueError("Public Safety must be structurally missing for every 2004 record")
    rows_2004 = frame.loc[frame["調査年"].eq(2004)].dropna(subset=ISSUE_COLUMNS_JP[:-1])
    rows_other = frame.loc[~frame["調査年"].eq(2004)].dropna(subset=ISSUE_COLUMNS_JP)
    cleaned = (
        pd.concat([rows_2004, rows_other], ignore_index=True)
        .sort_values("調査年", kind="stable")
        .reset_index(drop=True)
    )
    cleaned_years = sorted(cleaned["調査年"].unique().tolist())
    if cleaned_years != EXPECTED_YEARS:
        raise ValueError(
            f"UTAS cleaning removed an entire election year in {path.name}: {cleaned_years}"
        )
    return cleaned

def load_statements(directory):
    frames, sources, canonical_blocks = {}, {}, None
    for issue in ISSUES:
        candidates = [directory / PUBLIC_FILES[issue], directory / THEMES[issue]["csv"]]
        present = [p for p in candidates if p.exists()]
        if len(present) != 1:
            raise ValueError(f"{issue}: supply exactly one supported CSV filename")
        source = present[0]
        f = pd.read_csv(source)
        needed = BLOCK_COLUMNS + ["Stance_Value", "Original_ID", "Generated_Text"]
        if set(needed) - set(f.columns) or len(f) != 4320:
            raise ValueError(f"{issue}: incorrect columns or statement count")
        stance = pd.to_numeric(f.Stance_Value, errors="raise")
        if not stance.isin([1,2,3,4,5]).all() or f[needed].isna().any().any():
            raise ValueError(f"{issue}: missing metadata or invalid stance")
        if f.Generated_Text.astype(str).str.strip().eq("").any():
            raise ValueError(f"{issue}: empty statement")
        generator = f.Original_ID.astype(str).str.replace(r"-ID-\d+$", "", regex=True)
        blocks = sorted(set(f[BLOCK_COLUMNS].astype(str).apply(tuple, axis=1)))
        counts = f.assign(generator=generator).groupby(BLOCK_COLUMNS+["Stance_Value", "generator"]).size()
        if len(blocks) != 288 or generator.nunique() != 3 or len(counts) != 4320 or not counts.eq(1).all():
            raise ValueError(f"{issue}: unexpected context/stance/generator balance")
        if canonical_blocks is not None and blocks != canonical_blocks:
            raise ValueError("Context blocks differ across issues")
        canonical_blocks = blocks
        # Preserve all columns and the original row order: activations depend on it.
        frames[issue], sources[issue] = f, source
    return frames, sources

def load_matrix(path):
    frame=pd.read_csv(path,index_col=0)
    aliases={'Security':'Public Safety','Social':'Social Welfare','Public':'Public Works','Fiscal':'Fiscal Stimulus','Nkorea':'North Korea'}
    frame=frame.rename(index=aliases,columns=aliases)
    if frame.index.duplicated().any() or frame.columns.duplicated().any():raise ValueError('Duplicate matrix labels')
    if set(frame.index)!=set(ISSUES) or set(frame.columns)!=set(ISSUES):raise ValueError('Expected exactly six policy issues')
    x=frame.loc[ISSUES,ISSUES].to_numpy(float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite matrix')
    return x
