"""Respondent-weighted party/year rank analysis and S11 Table export."""
from __future__ import annotations
from pathlib import Path
import argparse,json,hashlib,itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common.utas_records import build_utas_records,is_residual_party,YEAR_CONFIGS
from common.constants import ISSUES
from common.data_loading import load_matrix
KEYS=['year','chamber','party']
SAMPLES={'elected':'Elected Diet members','all_candidates':'All candidates'}
MEASURES=['Transfer performance','Cosine similarity']


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]

def matrix_rank_correlation(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, float, int]:
    """Spearman correlation between the finite upper-triangle entries."""
    av = upper_triangle(a.loc[ISSUES, ISSUES].to_numpy(float))
    bv = upper_triangle(b.loc[ISSUES, ISSUES].to_numpy(float))
    keep = np.isfinite(av) & np.isfinite(bv)
    if keep.sum() < 3:
        return (np.nan, np.nan, int(keep.sum()))
    result = spearmanr(av[keep], bv[keep])
    return (float(result.statistic), float(result.pvalue), int(keep.sum()))

def exact_label_permutation_test(observed: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float | int]:
    """Exact Mantel-style test over all 6! joint row/column permutations."""
    observed = observed.loc[ISSUES, ISSUES]
    reference = reference.loc[ISSUES, ISSUES]
    rho, asymptotic_p, n_elements = matrix_rank_correlation(observed, reference)
    if not np.isfinite(rho) or n_elements != 15:
        return {'rho': rho, 'spearman_asymptotic_p': asymptotic_p, 'exact_one_sided_p': np.nan, 'exact_two_sided_p': np.nan, 'n_matrix_elements': n_elements, 'n_label_permutations': 0}
    permuted_rhos: list[float] = []
    ref_array = reference.to_numpy(float)
    for permutation in itertools.permutations(range(len(ISSUES))):
        permuted = pd.DataFrame(ref_array[np.ix_(permutation, permutation)], index=ISSUES, columns=ISSUES)
        permuted_rho, _, _ = matrix_rank_correlation(observed, permuted)
        if np.isfinite(permuted_rho):
            permuted_rhos.append(permuted_rho)
    values = np.asarray(permuted_rhos)
    return {'rho': rho, 'spearman_asymptotic_p': asymptotic_p, 'exact_one_sided_p': float(np.sum(values >= rho) / len(values)), 'exact_two_sided_p': float(np.sum(np.abs(values) >= abs(rho)) / len(values)), 'n_matrix_elements': n_elements, 'n_label_permutations': int(len(values))}

def analyze(args):
    OUT = args.output
    if OUT.exists():
        raise FileExistsError('Choose a fresh output directory')
    records, qc = build_utas_records(args.raw_dir, args.mapping_workbook)
    paths = {'Gemma probing': args.transfer, 'Gemma cosine': args.cosine}
    llm = {}
    for k, v in paths.items():
        m = load_matrix(v)
        if not np.allclose(m, m.T, atol=1e-12, rtol=0):
            raise ValueError('Supply symmetric reference matrices')
        llm[k] = pd.DataFrame(m, index=ISSUES, columns=ISSUES)
    OUT.mkdir(parents=True)
    qc.to_csv(OUT / 'year_sample_qc.csv', index=False)
    comparisons = []
    cells = []
    checks = []
    for sample in ['elected', 'all_candidates']:
        mask = records['any_response'] & records['complete_response']
        if sample == 'elected':
            mask &= records['elected']
        f = records.loc[mask].copy()
        f = f.loc[f.party.notna() & ~f.party.map(is_residual_party)].copy()
        matrices = {k: pd.DataFrame(np.eye(6), index=ISSUES, columns=ISSUES) for k in ['unadjusted', 'between_mean_ranks', 'within_residual_ranks']}
        for i, left in enumerate(ISSUES):
            for j in range(i + 1, 6):
                right = ISSUES[j]
                pair = f[KEYS + [left, right]].dropna().copy()
                pair = pair.loc[pair.groupby(KEYS)[left].transform('size').ge(2)]
                ranks = pair[[left, right]].rank(method='average')
                w = pair[KEYS].join(ranks)
                means = w.groupby(KEYS)[[left, right]].transform('mean')
                residual = ranks - means
                for name, data in [('unadjusted', ranks), ('between_mean_ranks', means), ('within_residual_ranks', residual)]:
                    value = data.corr(method='pearson').iloc[0, 1]
                    matrices[name].iloc[i, j] = matrices[name].iloc[j, i] = value
                g = w.groupby(KEYS)[[left, right]].mean()
                n = w.groupby(KEYS).size().reindex(g.index).to_numpy()
                a = g.to_numpy()
                centered = a - np.average(a, axis=0, weights=n)
                cov = (centered * n[:, None]).T @ centered / n.sum()
                weighted = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
                assert np.isclose(weighted, matrices['between_mean_ranks'].iloc[i, j], atol=1e-12)
                total_cov = np.cov(ranks.to_numpy().T, bias=True)
                decomposition = np.cov(means.to_numpy().T, bias=True) + np.cov(residual.to_numpy().T, bias=True)
                assert np.allclose(total_cov, decomposition, rtol=1e-10, atol=1e-08)
                checks.append(dict(sample=sample, left=left, right=right, n=len(pair), groups=len(g), weighted_equivalence_error=abs(weighted - matrices['between_mean_ranks'].iloc[i, j]), covariance_decomposition_error=np.max(abs(total_cov - decomposition))))
                raw = pair.groupby(KEYS)[[left, right]].mean()
                for key, row in g.iterrows():
                    cells.append(dict(sample=sample, year=key[0], chamber=key[1], party=key[2], left=left, right=right, n=int(w.groupby(KEYS).size().loc[key]), mean_rank_left=row[left], mean_rank_right=row[right], mean_response_left=raw.loc[key, left], mean_response_right=raw.loc[key, right]))
        for kind, matrix in matrices.items():
            matrix.to_csv(OUT / f'{sample}_{kind}_matrix.csv')
            for model, reference in llm.items():
                result = exact_label_permutation_test(matrix, reference)
                comparisons.append(dict(sample=sample, component=kind, model=model, **result))
        print(sample, 'verified and calculated', flush=True)
    pd.DataFrame(comparisons).to_csv(OUT / 'llm_comparisons.csv', index=False)
    pd.DataFrame(cells).to_csv(OUT / 'party_wave_pair_means_and_weights.csv', index=False)
    pd.DataFrame(checks).to_csv(OUT / 'verification.csv', index=False)
    inputs = [args.mapping_workbook, *paths.values()] + [args.raw_dir / c.filename for c in YEAR_CONFIGS.values()]
    manifest = {'group_keys': KEYS, 'model': 'Gemma', 'weight': 'pair-specific respondent count per party/year/chamber cell', 'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))

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

def table(a):
    if a.output.exists():raise FileExistsError('Choose a fresh output directory')
    path=a.party/'llm_comparisons.csv'
    party=pd.read_csv(path)
    if len(party)!=12:raise ValueError('Expected 12 comparisons')
    rows=[]
    for scope, label in SAMPLES.items():
        for measure, model in zip(MEASURES, ['Gemma probing', 'Gemma cosine']):
            row = {'UTAS sample': label, 'Gemma matrix used for comparison': measure + ' matrix'}
            for component, title in [('unadjusted', 'Individual responses'), ('between_mean_ranks', 'Party means within each survey year'), ('within_residual_ranks', 'Individual deviations from these means')]:
                v = one(party, sample=scope, component=component, model=model)
                row[title] = pair(v.rho, v.exact_one_sided_p)
            rows.append(row)
    frame=pd.DataFrame(rows)
    a.output.mkdir(parents=True)
    frame.to_csv(a.output/'S11_Table.csv',index=False)
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines+=['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)]
    (a.output/'S11_Table.md').write_text('\n'.join(lines)+'\n')
    party.to_csv(a.output/'party_unrounded.csv',index=False)
    (a.output/'manifest.json').write_text(json.dumps({'input_sha256':{'party':hashlib.sha256(path.read_bytes()).hexdigest()},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'panels':['S11_Table']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True)
    for op in ['analyze','table']:
        q=sub.add_parser(op);q.add_argument('--output',type=Path,required=True)
        for name in (['raw-dir','mapping-workbook','transfer','cosine'] if op=='analyze' else ['party']):q.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    {'analyze':analyze,'table':table}[a.operation](a)
if __name__=='__main__':main()
