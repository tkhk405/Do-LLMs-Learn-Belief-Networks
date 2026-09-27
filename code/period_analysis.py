"""Three-period UTAS comparisons, release-period comparisons, and S12/S13A tables."""
from __future__ import annotations
from pathlib import Path
from itertools import combinations,permutations
import argparse,hashlib,json,platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
from common.configuration import load_paths
from common.constants import ISSUES,ISSUE_COLUMNS_JP
from common.data_loading import load_matrix,clean_utas_frame
from common.statistics import exact_mantel
SAMPLES={'elected':'Elected Diet members','all_candidates':'All candidates'}
MEASURES=['Transfer performance','Cosine similarity']


PERIODS = {'2003-2007': [2003, 2004, 2005, 2007], '2009-2012': [2009, 2010, 2012], '2013-2025': [2013, 2014, 2016, 2017, 2019, 2021, 2022, 2024, 2025]}

JP = ['防衛力強化', '小さな政府', '公共事業', '財政出動', '北朝鮮', '治安']

EN = ['Defense', 'Social Welfare', 'Public Works', 'Fiscal Stimulus', 'North Korea', 'Public Safety']

FILES = {'elected': '谷口朝日_6項目統合_当選者（非改選抜き）.csv', 'all_candidates': '谷口朝日_6項目統合_全候補者（非改選除く）.csv'}

def three_periods(args):
    paths = load_paths(args.config)
    OUT = args.output
    if OUT.exists():
        raise FileExistsError('Choose a fresh output directory')
    refs = {'Transfer performance': load_matrix(args.transfer), 'Cosine similarity': load_matrix(args.cosine)}
    refs['Transfer performance'] = (refs['Transfer performance'] + refs['Transfer performance'].T) / 2
    inputs = {s: paths['utas_' + s + '_csv'] for s in ['elected', 'all_candidates']}
    OUT.mkdir(parents=True)
    records = []
    counts = []
    changes = []
    checks = []
    hashes = {}
    upper = np.triu_indices(6, 1)
    for sample, name in FILES.items():
        p = inputs[sample]
        hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        d = pd.read_csv(p, encoding='utf-8-sig')
        d['調査年'] = pd.to_numeric(d['調査年'], errors='raise').astype(int)
        d[JP] = d[JP].apply(pd.to_numeric, errors='coerce')
        d[JP] = d[JP].where(d[JP].isin([1, 2, 3, 4, 5]))
        d.loc[d['調査年'].eq(2004), '治安'] = np.nan
        d = pd.concat([d[d['調査年'].eq(2004)].dropna(subset=JP[:-1]), d[~d['調査年'].eq(2004)].dropna(subset=JP)])
        assert set(d['調査年']) == set(sum(PERIODS.values(), []))
        mats = {}
        for period, years in PERIODS.items():
            v = d.loc[d['調査年'].isin(years), JP].rename(columns=dict(zip(JP, EN)))
            m = v.corr(method='spearman')
            mats[period] = m.to_numpy()
            n = v.notna().astype(int).T.dot(v.notna().astype(int))
            m.to_csv(OUT / f'{sample}_{period}_correlations.csv')
            n.to_csv(OUT / f'{sample}_{period}_pairwise_n.csv')
            counts.append(dict(sample=sample, period=period, records=len(v), pairwise_n_min=int(n.to_numpy()[upper].min()), pairwise_n_max=int(n.to_numpy()[upper].max())))
        for a, b in combinations(PERIODS, 2):
            x, y = (mats[a], mats[b])
            rho = float(spearmanr(x[upper], y[upper]).statistic)
            null = [float(spearmanr(x[np.ix_(perm, perm)][upper], y[upper]).statistic) for perm in permutations(range(6))]
            extreme = int(np.sum(np.asarray(null) >= rho - 1e-12))
            delta = y[upper] - x[upper]
            records.append(dict(sample=sample, period_1=a, period_2=b, spearman_rho=rho, p_one_sided_exact=extreme / 720, extreme_permutations=extreme, n_permutations=720, mean_absolute_change=float(np.abs(delta).mean()), maximum_absolute_change=float(np.abs(delta).max())))
            for i, j in zip(*upper):
                changes.append(dict(sample=sample, period_1=a, period_2=b, issue_1=EN[i], issue_2=EN[j], correlation_1=x[i, j], correlation_2=y[i, j], change=y[i, j] - x[i, j]))
    result = pd.DataFrame(records)
    result.to_csv(OUT / 'between_period_results.csv', index=False)
    pd.DataFrame(counts).to_csv(OUT / 'sample_counts.csv', index=False)
    pd.DataFrame(changes).to_csv(OUT / 'issue_pair_changes.csv', index=False)
    (OUT / 'manifest.json').write_text(json.dumps(dict(inputs_sha256=hashes, periods=PERIODS, python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), ensure_ascii=False, indent=2))
    llm_rows = []
    for sample in FILES:
        for period in PERIODS:
            u = load_matrix(OUT / f'{sample}_{period}_correlations.csv')
            for measure, m in refs.items():
                r, _ = exact_mantel(m, u)
                llm_rows.append({'sample': sample, 'period': period, 'model': 'Gemma', 'measure': measure, 'spearman_rho': r['rho'], 'p_one_sided_exact': r['p_one_sided'], 'extreme_permutations': r['extreme_permutations'], 'n_permutations': r['permutations'], 'n_issue_pairs': r['issue_pairs']})
    pd.DataFrame(llm_rows).to_csv(OUT / 'gemma_period_comparisons.csv', index=False)
    print('Three-period comparisons complete')

def release_periods(a):
    paths = load_paths(a.config)
    if a.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    t = load_matrix(a.transfer)
    t = (t + t.T) / 2
    c = load_matrix(a.cosine)
    if not np.allclose(c, c.T):
        raise ValueError('Expected symmetric cosine matrix')
    raw = pd.read_csv(a.utas_2026, encoding='cp932')
    raw = raw[raw.RESPONSE.eq(1)]
    cols = ['Q4_1', 'Q4_6', 'Q4_7', 'Q4_8', 'Q4_3', 'Q4_13']
    rename = dict(zip(ISSUE_COLUMNS_JP, ISSUES))
    ix = np.triu_indices(6, 1)
    rows = []
    counts = []
    matrices = {}
    sizes = {}
    files = [a.utas_2026, a.transfer, a.cosine]
    for sample in ['elected', 'all_candidates']:
        source = paths['utas_' + sample + '_csv']
        files.append(source)
        clean = clean_utas_frame(source)
        pre = clean[clean['調査年'].between(2003, 2024)]
        for year, g in pre.groupby('調査年'):
            counts.append({'sample': sample, 'period': '2003-2024', 'year': int(year), 'records': len(g)})
        early = pre[ISSUE_COLUMNS_JP].rename(columns=rename)
        old = clean[clean['調査年'].eq(2025)][ISSUE_COLUMNS_JP].rename(columns=rename)
        new = raw[raw.RESULT.isin([2, 3, 4])] if sample == 'elected' else raw
        new = new[cols].rename(columns=dict(zip(cols, ISSUES))).apply(pd.to_numeric, errors='coerce')
        new = new.where(new.isin(range(1, 6))).dropna()
        for year, d in [(2025, old), (2026, new)]:
            counts.append({'sample': sample, 'period': '2025-2026', 'year': year, 'records': len(d)})
        for period, d in [('2003-2024', early), ('2025-2026', pd.concat([old, new], ignore_index=True))]:
            m = d.corr(method='spearman')
            valid = d.notna().astype(int)
            n = valid.T @ valid
            matrices[sample, period] = m
            sizes[sample, period] = n
            for measure, ref in [('Transfer performance', t), ('Cosine similarity', c)]:
                stats, _ = exact_mantel(ref, m.to_numpy())
                rows.append({'sample': sample, 'period': period, 'measure': measure, 'rho': stats['rho'], 'p': stats['p_one_sided'], 'extreme_permutations': stats['extreme_permutations'], 'permutations': 720, 'records': len(d), 'pairwise_n_min': int(n.to_numpy()[ix].min()), 'pairwise_n_max': int(n.to_numpy()[ix].max())})
    a.output.mkdir(parents=True)
    for (sample, period), m in matrices.items():
        m.to_csv(a.output / f'{sample}_{period}_correlation.csv')
        sizes[sample, period].to_csv(a.output / f'{sample}_{period}_pairwise_n.csv')
    pd.DataFrame(rows).to_csv(a.output / 'comparisons.csv', index=False)
    pd.DataFrame(counts).to_csv(a.output / 'sample_counts.csv', index=False)
    files += [Path(__file__), Path(__file__).parent / 'common/data_loading.py', Path(__file__).parent / 'common/statistics.py']
    (a.output / 'manifest.json').write_text(json.dumps({'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, 'periods': ['2003-2024', '2025-2026'], 'post_release_missing_policy': 'complete_six'}, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

def one(frame, **criteria):
    selected = frame
    for key, value in criteria.items():
        selected = selected[selected[key] == value]
    if len(selected) != 1:
        raise ValueError('Expected one row: ' + str(criteria))
    return selected.iloc[0]

def pair(rho, p):
    if not np.isfinite([rho, p]).all() or not -1 <= rho <= 1 or (not 0 <= p <= 1):
        raise ValueError('Invalid correlation or p-value')
    return f'{rho:.3f} ({p:.3f})'

def tables(a):
    if a.output.exists():raise FileExistsError('Choose a fresh output directory')
    inputs={};frames={}
    def read(key,path,expected):
        frame=pd.read_csv(path)
        if len(frame)!=expected:raise ValueError('Unexpected number of rows: '+str(path))
        inputs[key]=hashlib.sha256(path.read_bytes()).hexdigest();frames[key]=frame
        return frame
    between=read('between_periods',a.periods/'between_period_results.csv',6)
    period=read('period_gemma',a.periods/'gemma_period_comparisons.csv',12)
    release=read('release',a.release/'comparisons.csv',8)
    tables={};rows=[]
    for scope, label in SAMPLES.items():
        for i, first in enumerate(PERIODS):
            row = {'UTAS sample': label, 'Period': first}
            for j, second in enumerate(PERIODS):
                if j < i:
                    row[second] = ''
                elif j == i:
                    row[second] = '—'
                else:
                    v = one(between, sample=scope, period_1=first, period_2=second)
                    row[second] = pair(v.spearman_rho, v.p_one_sided_exact)
            rows.append(row)
    tables['S12_Table_A']=pd.DataFrame(rows);rows=[]
    for scope, label in SAMPLES.items():
        for measure in MEASURES:
            row = {'UTAS sample': label, 'Measure': measure}
            for time in PERIODS:
                v = one(period, sample=scope, period=time, measure=measure, model='Gemma')
                row[time] = pair(v.spearman_rho, v.p_one_sided_exact)
            rows.append(row)
    tables['S12_Table_B']=pd.DataFrame(rows);rows=[]
    for scope, label in SAMPLES.items():
        for measure in MEASURES:
            row = {'Gemma measure': measure, 'UTAS sample': label}
            for time, title in [('2025-2026', 'Post-release surveys (2025-2026)'), ('2003-2024', 'Pre-release surveys (2003-2024), reference')]:
                v = one(release, sample=scope, period=time, measure=measure)
                pair(v.rho, v.p)
                row[title + ' Spearman ρ'] = f'{v.rho:.3f}'
                row[title + ' p'] = f'{v.p:.3f}'
            rows.append(row)
    tables['S13_Table_A']=pd.DataFrame(rows);rows=[]
    a.output.mkdir(parents=True)
    for name,f in tables.items():
        f.to_csv(a.output/(name+'.csv'),index=False)
        lines=['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']
        lines+=['| '+' | '.join(map(str,row))+' |' for row in f.itertuples(index=False,name=None)]
        (a.output/(name+'.md')).write_text('\n'.join(lines)+'\n')
    for name,f in frames.items():f.to_csv(a.output/(name+'_unrounded.csv'),index=False)
    (a.output/'manifest.json').write_text(json.dumps({'input_sha256':inputs,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'panels':list(tables)},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True)
    for op in ['three-periods','release-periods','tables']:
        q=sub.add_parser(op);q.add_argument('--output',type=Path,required=True)
        if op=='tables':
            q.add_argument('--periods',type=Path,required=True);q.add_argument('--release',type=Path,required=True)
        else:
            q.add_argument('--config');q.add_argument('--transfer',type=Path,required=True);q.add_argument('--cosine',type=Path,required=True)
            if op=='release-periods':q.add_argument('--utas-2026',type=Path,required=True)
    a=p.parse_args()
    {'three-periods':three_periods,'release-periods':release_periods,'tables':tables}[a.operation](a)
if __name__=='__main__':main()
