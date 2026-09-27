"""Fixed-head refitted-probe sensitivity: prepare, run, summarize, table."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,json,fcntl,importlib.metadata,hashlib,tempfile
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common.configuration import load_paths
from common.data_loading import load_statements,clean_utas_frame
from common.constants import THEMES,ISSUES,BLOCK_COLUMNS
from common.statistics import utas_matrix
N_BLOCKS=288
ALPHAS=[.01,.1,1.,10.,100.,1000.]
def learning_dependencies():
    global StandardScaler,StratifiedKFold,mord,Parallel,delayed,parallel_config
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    import mord
    from joblib import Parallel,delayed,parallel_config


def rows_from_weights(block_rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    block_sequence = np.repeat(np.arange(N_BLOCKS), weights.astype(int))
    return block_rows[block_sequence].reshape(-1)

def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(len(ISSUES), 1)]

def matrix_spearman(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    return float(spearmanr(upper_triangle(matrix_a), upper_triangle(matrix_b))[0])

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def analysis_array(x, model):
    return np.asarray(x) if model == 'Gemma' else np.asarray(x, dtype=np.float64)

def cosine_original(a, b):
    na, nb = (np.linalg.norm(a), np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return np.nan
    return np.dot(a, b) / (na * nb)

def cv_folds(y):
    learning_dependencies()
    f = np.empty(len(y), int)
    for k, (_, val) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
        f[val] = k
    return f

def fit_probe(x, y, rows, folds, model):
    learning_dependencies()
    x = analysis_array(x[rows], model)
    y = y[rows]
    f = folds[rows]
    scores = {a: [] for a in ALPHAS}
    for k in range(5):
        tr = f != k
        va = ~tr
        if len(np.unique(y[tr])) != 5 or len(np.unique(y[va])) < 2:
            raise ValueError('Invalid bootstrap CV fold')
        sc = StandardScaler()
        xt = sc.fit_transform(x[tr])
        xv = sc.transform(x[va])
        for a in ALPHAS:
            m = mord.LogisticAT(alpha=a).fit(xt, y[tr])
            rho = float(spearmanr(y[va], m.predict(xv)).statistic)
            if np.isfinite(rho):
                scores[a].append(rho)
    avg = {a: float(np.mean(v)) for a, v in scores.items() if v}
    if not avg:
        raise ValueError('All CV predictions constant')
    alpha = max(avg, key=avg.get)
    sc = StandardScaler()
    m = mord.LogisticAT(alpha=alpha).fit(sc.fit_transform(x), y)
    out = dict(coef=m.coef_, theta=m.theta_, mean=sc.mean_, scale=sc.scale_, alpha=np.array(alpha), cv_rho=np.array(avg[alpha]))
    if not all((np.isfinite(v).all() for v in out.values())):
        raise ValueError('Nonfinite fit')
    return out

def fit_checkpoint(path, x, y, rows, folds, model='Gemma'):
    path = Path(path)
    if path.exists():
        with np.load(path) as z:
            if set(z.files) != {'coef', 'theta', 'mean', 'scale', 'alpha', 'cv_rho'} or not all((np.isfinite(z[k]).all() for k in z.files)):
                raise ValueError(f'Invalid checkpoint {path}')
        return str(path)
    atomic_npz(path, **fit_probe(x, y, rows, folds, model))
    return str(path)

def matrices(arrays, labels, rows, heads, th, ch, folder, model):
    learning_dependencies()
    tp = np.empty((6, 6))
    cs = np.eye(6)
    dirs = {}
    for ti, target in enumerate(ISSUES):
        rr = rows[target]
        y = labels[target][rr]
        values = np.empty((6, len(th)))
        for hi, head in enumerate(th):
            x = analysis_array(arrays[target][rr, heads.index(head), :], model)
            x = StandardScaler().fit_transform(x)
            for si, source in enumerate(ISSUES):
                with np.load(folder / f'{si}_{hi}.npz') as z:
                    coef = z['coef']
                values[si, hi] = spearmanr(y, x @ coef).statistic
        tp[:, ti] = values.mean(axis=1)
        vv = []
        for head in ch:
            x = analysis_array(arrays[target][rr, heads.index(head), :], model)
            a = x[y <= 1]
            d = x[y >= 3]
            v = d.mean(0) - a.mean(0)
            sd = np.concatenate([a, d]).std(0)
            v = v / np.where(sd > 1e-10, sd, 1.0)
            vv.append(v)
        dirs[target] = np.asarray(vv)
    for i, a in enumerate(ISSUES):
        for j, b in enumerate(ISSUES):
            cs[i, j] = np.mean([cosine_original(x, y) for x, y in zip(dirs[a], dirs[b])])
    tp = (tp + tp.T) / 2
    if not np.isfinite(tp).all() or not np.isfinite(cs).all():
        raise ValueError('Nonfinite matrices')
    return (tp, cs)

def sample_rows(blocks, seed, rep):
    rng = np.random.default_rng(np.random.SeedSequence([seed, rep, 0]))
    w = rng.multinomial(288, np.full(288, 1 / 288))
    return {i: rows_from_weights(b, w) for i, b in blocks.items()}

def sample_utas(frame, seed, rep, scope):
    rng = np.random.default_rng(np.random.SeedSequence([seed, rep, 1, scope]))
    groups = frame.groupby('調査年', sort=True).indices.values()
    ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
    return utas_matrix(frame.iloc[ix])

def summarize_progress(out, reps):
    records = []
    for r in range(reps):
        path = out / 'replicates' / f'{r:04d}.npz'
        if not path.exists():
            continue
        with np.load(path) as z:
            for scope in ['elected', 'all_candidates']:
                for method in ['transfer', 'cosine']:
                    records.append({'replicate': r, 'scope': scope, 'method': method, 'rho': matrix_spearman(z[method], z[scope])})
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df.to_csv(out / 'replicate_correlations.csv', index=False)
    summary = df.groupby(['scope', 'method']).rho.agg(n='count', mean='mean', lower=lambda x: x.quantile(0.025), upper=lambda x: x.quantile(0.975)).reset_index()
    summary['complete'] = summary.n.eq(reps)
    summary.to_csv(out / 'sensitivity_summary.csv', index=False)
    return summary

def prepare(a):
    paths = load_paths(a.config)
    frames, sources = load_statements(paths['synthetic_statements_dir'])

    def headlist(name):
        f = pd.read_csv(a.heads / name)
        v = f[['layer', 'head']].to_numpy()
        if not np.isfinite(v).all() or not np.equal(v, v.astype(int)).all():
            raise ValueError('Noninteger heads')
        pairs = [tuple(map(int, x)) for x in v]
        if len(set(pairs)) != len(pairs) or not pairs:
            raise ValueError('Duplicate/empty heads')
        return pairs
    th = headlist('common_top20_heads.csv')
    ch = sorted(headlist('union_heads.csv'))
    heads = sorted(set(th) | set(ch))
    layers, nh, dim = (42, 16, 256) if a.model == 'gemma' else (32, 32, 128)
    if len(th) != 20 or any((l not in range(layers) or h not in range(nh) for l, h in heads)):
        raise ValueError('Invalid head selection')
    design = {'version': 'fixed-head-refit-v2-original-dtypes', 'model': a.model.title(), 'mode': 'full', 'reps': 1000, 'seed': 20260913, 'heads_transfer': th, 'heads_cosine': ch}
    signature = {'model': a.model, 'heads': heads, 'geometry': [layers, nh, dim], 'statements': {i: digest(p) for i, p in sources.items()}, 'selection': {n: digest(a.heads / n) for n in ['common_top20_heads.csv', 'union_heads.csv']}, 'script': digest(Path(__file__))}
    signature = json.loads(json.dumps(signature))
    mp = a.output / 'manifest.json'
    old = json.loads(mp.read_text()) if mp.exists() else None
    if old and old['signature'] != signature:
        raise ValueError('Changed inputs/settings: use a fresh directory')
    hashes = {}
    input_hashes = {}
    for issue, c in THEMES.items():
        files = {}
        root = paths[a.model + '_activations_dir']
        for l in sorted({l for l, h in heads}):
            name = f'{c['vec']}_layer_{l:02d}.npy'
            candidates = [root / name]
            if a.model == 'gemma' and c['subdir']:
                candidates.append(root / c['subdir'] / name)
            found = [p for p in candidates if p.exists()]
            if len(found) != 1:
                raise ValueError('Missing/ambiguous activation')
            files[l] = found[0]
        ih = {p.name: digest(p) for p in files.values()}
        input_hashes[issue] = ih
        dest = a.output / f'{c['prefix']}.npy'
        if old and dest.name in old['sha256']:
            if old['activation_sha256'][issue] != ih or digest(dest) != old['sha256'][dest.name]:
                raise ValueError('Changed activation/cache')
            hashes[dest.name] = old['sha256'][dest.name]
            continue
        tmp = dest.with_suffix('.tmp.npy')
        arr = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float16, shape=(4320, len(heads), dim))
        for l, p in files.items():
            raw = np.load(p, mmap_mode='r')
            if raw.shape != (4320, nh, dim) or raw.dtype != np.float16:
                raise ValueError('Unexpected activation shape/dtype')
            for j, (layer, h) in enumerate(heads):
                if layer == l:
                    arr[:, j] = raw[:, h]
        if not np.isfinite(arr).all():
            raise ValueError('Nonfinite activations')
        arr.flush()
        del arr
        tmp.replace(dest)
        hashes[dest.name] = digest(dest)
        print(f'{a.model}: {issue} prepared', flush=True)
    atomic_json(mp, {'signature': signature, 'sha256': hashes, 'activation_sha256': input_hashes})
    atomic_json(a.output / 'design.json', design)

def run(a):
    learning_dependencies()
    paths = load_paths(a.config)
    design = json.loads(a.design.read_text())
    if design['version'] != 'fixed-head-refit-v2-original-dtypes':
        raise ValueError('Wrong source design')
    model = design['model']
    th = [tuple(h) for h in design['heads_transfer']]
    ch = [tuple(h) for h in design['heads_cosine']]
    heads = sorted(set(th) | set(ch))
    source_manifest = json.loads((a.inputs / 'manifest.json').read_text())
    if [tuple(h) for h in source_manifest['signature']['heads']] != heads:
        raise ValueError('Input head order mismatch')
    frames, sources = load_statements(paths['synthetic_statements_dir'])
    arrays = {}
    labels = {}
    blocks = {}
    hashes = {}
    for i, c in THEMES.items():
        p = a.inputs / f'{c['prefix']}.npy'
        hashes[p.name] = digest(p)
        if hashes[p.name] != source_manifest['sha256'][p.name]:
            raise ValueError('Input checksum mismatch')
        arrays[i] = np.load(p, mmap_mode='r')
        dim = 256 if model == 'Gemma' else 128
        if arrays[i].shape != (4320, len(heads), dim) or arrays[i].dtype != np.float16:
            raise ValueError('Invalid prepared activations')
        labels[i] = frames[i].Stance_Value.to_numpy(int) - 1
        keys = frames[i][BLOCK_COLUMNS].astype(str).apply(tuple, axis=1)
        blocks[i] = np.stack([np.flatnonzero(keys.map(lambda x: x == k).to_numpy()) for k in sorted(set(keys))])
        hashes[i] = digest(sources[i])
        recorded = source_manifest['signature'].get('statements', {})
        if recorded and recorded.get(i) != hashes[i]:
            raise ValueError('Statement rows differ from prepared inputs')
    utas = {s: clean_utas_frame(paths['utas_' + s + '_csv']) for s in ['elected', 'all_candidates']}
    hashes.update({s: digest(paths['utas_' + s + '_csv']) for s in utas})
    versions = {k: importlib.metadata.version(k) for k in ['numpy', 'scipy', 'pandas', 'scikit-learn', 'mord', 'joblib']}
    pins = {'numpy': '1.26.4', 'scipy': '1.13.1', 'pandas': '3.0.1', 'scikit-learn': '1.8.0', 'mord': '0.7', 'joblib': '1.4.2'}
    if versions != pins:
        raise RuntimeError('Use requirements-sensitivity.txt')
    spec = {'version': design['version'], 'model': model, 'mode': design['mode'], 'reps': a.reps, 'seed': design['seed'], 'heads_transfer': th, 'heads_cosine': ch, 'inputs': hashes, 'versions': versions, 'source_design_sha256': digest(a.design), 'scripts': {p.name: digest(p) for p in [Path(__file__), Path(__file__).parent/'common/data_loading.py', Path(__file__).parent/'common/constants.py', Path(__file__).parent/'common/statistics.py']}}
    spec = json.loads(json.dumps(spec))
    mp = a.output / 'run_manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != spec:
        raise ValueError('Changed run configuration')
    if not mp.exists() and ((a.output / 'replicates').exists() or (a.output / 'probes').exists()):
        raise ValueError('Unidentified checkpoints')
    atomic_json(mp, spec)
    folds = {i: cv_folds(labels[i]) for i in ISSUES}
    new = 0
    for r in range(a.reps):
        done = a.output / 'replicates' / f'{r:04d}.npz'
        if done.exists():
            with np.load(done) as z:
                if set(z.files) != {'transfer', 'cosine', 'elected', 'all_candidates'} or not all((z[k].shape == (6, 6) and np.isfinite(z[k]).all() for k in z.files)):
                    raise ValueError('Invalid replicate')
            continue
        if (a.output / 'STOP').exists():
            break
        rows = sample_rows(blocks, design['seed'], r)
        folder = a.output / 'probes' / f'{r:04d}'
        with parallel_config(backend='loky', inner_max_num_threads=1):
            Parallel(n_jobs=a.jobs)((delayed(fit_checkpoint)(folder / f'{si}_{hi}.npz', arrays[i][:, heads.index(h), :], labels[i], rows[i], folds[i], model) for si, i in enumerate(ISSUES) for hi, h in enumerate(th)))
        tp, cs = matrices(arrays, labels, rows, heads, th, ch, folder, model)
        u = {s: sample_utas(f, design['seed'], r, k) for k, (s, f) in enumerate(utas.items())}
        atomic_npz(done, transfer=tp, cosine=cs, **u)
        new += 1
        atomic_json(a.output / 'progress.json', {'completed': len(list((a.output / 'replicates').glob('*.npz'))), 'total': a.reps})
        print(f'{model}: replicate {r + 1}/{a.reps} saved', flush=True)
        if a.max_new and new >= a.max_new:
            break
    return summarize_progress(a.output, a.reps)

def summarize(source, output):
    if output.exists():
        raise FileExistsError('Choose a fresh output directory')
    manifest = json.loads((source / 'run_manifest.json').read_text())
    if manifest['version'] != 'fixed-head-refit-v2-original-dtypes' or manifest['mode'] != 'full':
        raise ValueError('Expected the full refitted-probe v2 analysis')
    reps = manifest['reps']
    if reps != 1000:
        raise ValueError('Expected 1000 replicates for the reported analysis')
    expected = {f'{i:04d}.npz' for i in range(reps)}
    if {p.name for p in (source / 'replicates').glob('*.npz')} != expected:
        raise ValueError('Incomplete or unexpected replicate files')
    ix = np.triu_indices(6, 1)
    rows = []
    hashes = {}
    for r in range(reps):
        path = source / 'replicates' / f'{r:04d}.npz'
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:
            for k in ['transfer', 'cosine', 'elected', 'all_candidates']:
                if z[k].shape != (6, 6) or not np.isfinite(z[k]).all():
                    raise ValueError('Invalid replicate matrix')
            for scope in ['elected', 'all_candidates']:
                for method in ['transfer', 'cosine']:
                    rho = float(spearmanr(z[method][ix], z[scope][ix])[0])
                    if not np.isfinite(rho):
                        raise ValueError('Undefined correlation')
                    rows.append({'replicate': r, 'scope': scope, 'method': method, 'rho': rho})
    df = pd.DataFrame(rows)
    result = df.groupby(['scope', 'method']).rho.agg(n='count', mean='mean', lower=lambda x: x.quantile(0.025), upper=lambda x: x.quantile(0.975)).reset_index()
    result['complete'] = result.n.eq(reps)
    output.mkdir(parents=True)
    df.to_csv(output / 'replicate_correlations.csv', index=False)
    result.to_csv(output / 'sensitivity_summary.csv', index=False)
    record = {'source_run': manifest, 'replicate_sha256': hashes, 'versions': {k: importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scipy']}, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / 'aggregation_manifest.json').write_text(json.dumps(record, indent=2))
    return result

def table(a):
    if a.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    rows = []
    for model in ['gemma', 'llama']:
        s = pd.read_csv(getattr(a, model + '_summary'))
        m = pd.read_csv(getattr(a, model + '_main'))
        if len(s) != 4 or s.duplicated(['scope', 'method']).any() or (not s.n.eq(1000).all()) or (not s.complete.eq(True).all()):
            raise ValueError('Expected four completed 1000-replicate summaries')
        if not m.model.eq(model).all() or m.comparison.duplicated().any():
            raise ValueError('Wrong model or duplicate main comparisons')
        for scope in ['elected', 'all_candidates']:
            for method in ['transfer', 'cosine']:
                selected = s[(s.scope == scope) & (s.method == method)]
                original = m[m.comparison == f'{method}_vs_utas_{scope}']
                if len(selected) != 1 or len(original) != 1:
                    raise ValueError('Missing table cell')
                v = selected.iloc[0]
                rho = float(original.iloc[0].rho)
                if not np.isfinite([rho, v.lower, v.upper]).all() or not -1 <= v.lower <= v.upper <= 1:
                    raise ValueError('Invalid interval')
                rows.append(dict(scope=scope, model=model.title(), measure='Transfer performance' if method == 'transfer' else 'Cosine similarity', original_rho=rho, lower=v.lower, upper=v.upper, n=1000))
    frame = pd.DataFrame(rows)
    frame['scope'] = pd.Categorical(frame.scope, ['elected', 'all_candidates'], ordered=True)
    frame = frame.sort_values(['scope', 'model'], kind='stable')
    a.output.mkdir(parents=True)
    frame.to_csv(a.output / 'S10_Table_values.csv', index=False)
    text = ['| Population | Model | Measure | Original Spearman ρ | 95% resampling sensitivity interval |', '|---|---|---|---:|---|']
    for r in frame.itertuples():
        text.append(f'| {('Elected Diet members' if r.scope == 'elected' else 'All candidates')} | {r.model} | {r.measure} | {r.original_rho:.3f} | [{r.lower:.3f}, {r.upper:.3f}] |')
    (a.output / 'S10_Table.md').write_text('\n'.join(text) + '\n')
    (a.output / 'manifest.json').write_text(json.dumps({'input_sha256': {key: hashlib.sha256(getattr(a, key).read_bytes()).hexdigest() for key in ['gemma_summary', 'llama_summary', 'gemma_main', 'llama_main']}, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True)
    for op in ['prepare','run','summarize','table']:
        q=sub.add_parser(op);q.add_argument('--output',type=Path,required=True)
        if op in ['prepare','run']:q.add_argument('--config')
        if op=='prepare':
            q.add_argument('--model',choices=['gemma','llama'],required=True);q.add_argument('--heads',type=Path,required=True)
        if op=='run':
            q.add_argument('--inputs',type=Path,required=True);q.add_argument('--design',type=Path,required=True)
            q.add_argument('--reps',type=int,default=1000);q.add_argument('--jobs',type=int,default=1);q.add_argument('--max-new',type=int)
        if op=='summarize':q.add_argument('--source',type=Path,required=True)
        if op=='table':
            for arg in ['gemma-summary','llama-summary','gemma-main','llama-main']:q.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    if a.operation in ['prepare','run']:
        if a.operation=='run' and (a.reps<1 or a.jobs<1 or (a.max_new is not None and a.max_new<1)):p.error('Counts must be positive')
        a.output.mkdir(parents=True,exist_ok=True)
        with (a.output/(a.operation+'.lock')).open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result={'prepare':prepare,'run':run}[a.operation](a)
            if result is not None:print(result.to_string(index=False))
    elif a.operation=='summarize':print(summarize(a.source,a.output).to_string(index=False))
    else:table(a)
if __name__=='__main__':main()
