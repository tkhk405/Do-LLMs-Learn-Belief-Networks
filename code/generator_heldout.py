"""Gemma generator-held-out training, selection, evaluation, aggregation and figures."""
from __future__ import annotations
import os
for _key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_key]='1'
from pathlib import Path
from types import SimpleNamespace
from itertools import permutations
import argparse, json, hashlib, importlib.metadata
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from common.configuration import load_paths
from common.data_loading import load_statements, load_matrix
from common.constants import THEMES, ISSUES
from common.statistics import exact_mantel
from common.checkpoints import sha
RANDOM_STATE=42
ALPHAS=np.logspace(-2,3,6)

def learning_dependencies():
    global mord, StratifiedKFold, StandardScaler
    import mord
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

def plotting_dependencies():
    global mpl, plt, TwoSlopeNorm, Line2D, Image
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from PIL import Image


def identify_generator(original_id):
    value = str(original_id).lower()
    if value.startswith('gpt-'):
        return 'GPT'
    if value.startswith('claude-'):
        return 'Claude'
    if value.startswith('gemini-'):
        return 'Gemini'
    raise ValueError(f'Unknown Original_ID prefix: {original_id}')

def calc_spearman(y_true, y_pred):
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    rho, _ = spearmanr(y_true, y_pred)
    return rho if not np.isnan(rho) else np.nan

def process_one_head_heldout(head, layer_data_train, labels_train_ordinal, alphas):
    learning_dependencies()
    X = layer_data_train[:, head, :]
    y = labels_train_ordinal
    alpha_scores = {float(a): [] for a in alphas}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in cv.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        for alpha in alphas:
            model = mord.LogisticAT(alpha=float(alpha))
            try:
                model.fit(X_train, y[train_idx])
                pred = model.predict(X_val)
                rho = calc_spearman(y[val_idx] + 1, pred + 1)
                if not np.isnan(rho):
                    alpha_scores[float(alpha)].append(rho)
            except Exception:
                continue
    mean_scores = {a: np.mean(v) if v else np.nan for a, v in alpha_scores.items()}
    valid = {a: v for a, v in mean_scores.items() if not np.isnan(v)}
    best_alpha = max(valid, key=valid.get) if valid else float(alphas[0])
    best_rho = valid.get(best_alpha, 0.0)
    final_scaler = StandardScaler()
    X_full = final_scaler.fit_transform(X)
    final_model = mord.LogisticAT(alpha=best_alpha)
    final_model.fit(X_full, y)
    return {'head': head, 'rho': float(best_rho), 'alpha': float(best_alpha), 'coef': final_model.coef_.reshape(-1).astype(np.float32), 'theta': final_model.theta_.reshape(-1).astype(np.float32), 'scaler_mean': final_scaler.mean_.astype(np.float32), 'scaler_scale': final_scaler.scale_.astype(np.float32)}

def train(a):
    learning_dependencies()
    paths = load_paths(a.config)
    if any((x not in range(42) for x in a.layers)) or any((x not in range(16) for x in a.heads)):
        raise ValueError('Invalid indices')
    frames, sources = load_statements(paths['synthetic_statements_dir'])
    versions = {k: importlib.metadata.version(k) for k in ['numpy', 'scipy', 'scikit-learn', 'mord']}
    for issue in a.issues or list(THEMES):
        cfg = THEMES[issue]
        f = frames[issue]
        gen = f.Original_ID.map(identify_generator)
        for g in ['GPT', 'Claude', 'Gemini']:
            if not all((int(((gen == g) & (f.Stance_Value == s)).sum()) == 288 for s in range(1, 6))):
                raise ValueError('Unexpected generator/stance balance')
        mask = (gen != a.heldout).to_numpy()
        y = f.loc[mask, 'Stance_Value'].to_numpy(int) - 1
        for layer in a.layers:
            base = paths['gemma_activations_dir']
            candidates = [base / f'{cfg['vec']}_layer_{layer:02d}.npy']
            if cfg['subdir']:
                candidates.append(base / cfg['subdir'] / candidates[0].name)
            found = [x for x in candidates if x.exists()]
            if len(found) != 1:
                raise ValueError('Expected one activation file')
            v = found[0]
            data = np.load(v, mmap_mode='r')
            if data.shape != (4320, 16, 256) or not np.isfinite(data).all():
                raise ValueError('Invalid activations')
            train = np.asarray(data[mask])
            signature = {'heldout': a.heldout, 'vector': sha(v), 'statements': sha(sources[issue]), 'runner': sha(Path(__file__)), 'kernel': sha(Path(__file__)), 'versions': versions, 'training_rows': int(mask.sum()), 'dtype': str(data.dtype)}
            for head in a.heads:
                dest = a.output / f'heldout_{a.heldout}' / 'checkpoints' / f'{cfg['prefix']}_layer_{layer:02d}_head_{head:02d}.npz'
                dest.parent.mkdir(parents=True, exist_ok=True)
                sig = json.dumps({**signature, 'head': head}, sort_keys=True)
                if dest.exists():
                    with np.load(dest) as z:
                        if str(z['signature'].item()) != sig:
                            raise ValueError('Checkpoint mismatch')
                    continue
                r = process_one_head_heldout(head, train, y, ALPHAS)
                r = {k: np.asarray(v, dtype=np.float32) for k, v in r.items() if k != 'head'}
                if not all((np.isfinite(v).all() for v in r.values())):
                    raise ValueError('Invalid fit')
                tmp = dest.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, **r, signature=sig)
                tmp.replace(dest)
                print(f'{a.heldout}: {issue}, layer {layer}, head {head} saved', flush=True)

def cosine(a, b):
    na, nb = (np.linalg.norm(a), np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return np.nan
    return float(np.dot(a, b) / (na * nb))

def evaluate(a):
    learning_dependencies()
    paths = load_paths(a.config)
    frames, sources = load_statements(paths['synthetic_statements_dir'])
    common = pd.read_csv(a.heads / 'transfer_common_top_heads.csv')
    union = pd.read_csv(a.heads / 'cosine_union_heads.csv')
    transfer_heads = list(zip(common.layer.astype(int), common['head'].astype(int)))
    cosine_heads = sorted(zip(union.layer.astype(int), union['head'].astype(int)))
    if len(transfer_heads) != 20 or len(set(transfer_heads)) != 20:
        raise ValueError('Exactly 20 distinct transfer heads required')
    if not cosine_heads or len(set(cosine_heads)) != len(cosine_heads) or any((l not in range(42) or h not in range(16) for l, h in transfer_heads + cosine_heads)):
        raise ValueError('Invalid selected heads')
    coef = {}
    signature = {'heldout': a.heldout, 'script': sha(Path(__file__)), 'common': sha(a.heads / 'transfer_common_top_heads.csv'), 'union': sha(a.heads / 'cosine_union_heads.csv'), 'inputs': {}}
    import importlib.metadata
    signature['versions'] = {k: importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scipy', 'scikit-learn']}
    for issue, c in THEMES.items():
        cp = a.scores / f'{c['prefix']}_coef.npy'
        coef[issue] = np.load(cp, mmap_mode='r')
        signature['inputs'][cp.name] = sha(cp)
        signature['inputs'][issue] = sha(sources[issue])
    a.output.mkdir(parents=True, exist_ok=True)
    checkpoint = a.output / 'run.json'
    if checkpoint.exists():
        if json.loads(checkpoint.read_text()) != signature:
            raise ValueError('Changed inputs/settings: use a fresh output directory')
    elif any(a.output.iterdir()):
        raise ValueError('Existing output has no run manifest')
    else:
        checkpoint.write_text(json.dumps(signature, indent=2))
    layers = sorted({l for l, h in transfer_heads + cosine_heads})
    directions = {}
    transfer = []
    for layer in layers:
        th = [h for l, h in transfer_heads if l == layer]
        ch = [h for l, h in cosine_heads if l == layer]
        files = {}
        for issue, c in THEMES.items():
            root = paths['gemma_activations_dir']
            candidates = [root / f'{c['vec']}_layer_{layer:02d}.npy']
            if c['subdir']:
                candidates.append(root / c['subdir'] / candidates[0].name)
            existing = [x for x in candidates if x.exists()]
            if len(existing) != 1:
                raise ValueError('Missing/ambiguous activation')
            files[issue] = existing[0]
        hashes = json.dumps({i: sha(f) for i, f in files.items()}, sort_keys=True)
        cache = a.output / f'layer_{layer:02d}.npz'
        if cache.exists():
            with np.load(cache) as z:
                if str(z['inputs'].item()) != hashes:
                    raise ValueError('Activation hash changed')
                tr = z['transfer']
                di = z['directions']
        else:
            tr = np.full((len(th), 6, 6), np.nan)
            di = []
            for ti, issue in enumerate(ISSUES):
                raw = np.load(files[issue], mmap_mode='r')
                labels = frames[issue].Stance_Value.to_numpy(int)
                if len(raw) != len(labels):
                    raise ValueError('Activation row count mismatch')
                ds = []
                for h in sorted(set(th + ch)):
                    x = np.asarray(raw[:, h, :])
                    held = frames[issue].Original_ID.map(identify_generator).to_numpy() == a.heldout
                    if not np.isfinite(x).all():
                        raise ValueError('Nonfinite activation')
                    if h in th:
                        scaled = StandardScaler().fit_transform(x[held])
                        for si, source in enumerate(ISSUES):
                            tr[th.index(h), si, ti] = spearmanr(labels[held], scaled @ coef[source][layer, h]).statistic
                    if h in ch:
                        split_directions = []
                        for mask in [~held, held]:
                            lo = np.asarray(x[mask & np.isin(labels, [1, 2])], dtype=np.float32)
                            hi = np.asarray(x[mask & np.isin(labels, [4, 5])], dtype=np.float32)
                            delta = hi.mean(axis=0) - lo.mean(axis=0)
                            sigma = np.concatenate([lo, hi], axis=0).std(axis=0, ddof=0)
                            split_directions.append(delta / np.where(sigma > 1e-10, sigma, 1.0))
                        ds.append((h, np.stack(split_directions)))
                di.append(np.stack([dict(ds)[h] for h in ch]) if ch else np.empty((0, 2, raw.shape[2])))
            di = np.stack(di)
            tmp = cache.with_suffix('.tmp.npz')
            np.savez(tmp, inputs=hashes, transfer=tr, directions=di)
            tmp.replace(cache)
        if not np.isfinite(tr).all() or not np.isfinite(di).all():
            raise ValueError('Invalid cross-issue output')
        for k, h in enumerate(th):
            for si, s in enumerate(ISSUES):
                for ti, t in enumerate(ISSUES):
                    transfer.append({'source': s, 'target': t, 'layer': layer, 'head': h, 'rho': tr[k, si, ti]})
        for k, h in enumerate(ch):
            directions[layer, h] = di[:, k, :]
        print(f'{a.heldout}: layer {layer} complete ({layers.index(layer) + 1}/{len(layers)})', flush=True)
    long = pd.DataFrame(transfer)
    long['heldout_generator'] = a.heldout
    long = long.rename(columns={'rho': 'rho_target_standardized'})
    tm = long.groupby(['source', 'target'], sort=False).rho_target_standardized.mean().unstack().loc[ISSUES, ISSUES]
    cm = np.empty((6, 6))
    cr = []
    for i in range(6):
        for j in range(6):
            values = []
            for l, h in cosine_heads:
                v = cosine(directions[l, h][i, 0], directions[l, h][j, 1])
                values.append(v)
                cr.append({'heldout_generator': a.heldout, 'source_train_issue': ISSUES[i], 'target_heldout_issue': ISSUES[j], 'layer': l, 'head': h, 'cosine': v})
            cm[i, j] = np.nanmean(values)
    if not np.isfinite(cm).all():
        raise ValueError('Undefined cosine')
    (a.output / 'transfer').mkdir(exist_ok=True)
    (a.output / 'cosine').mkdir(exist_ok=True)
    tm.to_csv(a.output / 'transfer/transfer_matrix_target_standardized.csv')
    long.to_csv(a.output / 'transfer/transfer_per_head.csv', index=False)
    pd.DataFrame(cm, index=ISSUES, columns=ISSUES).to_csv(a.output / 'cosine/cosine_cross_split_asymmetric.csv')
    pd.DataFrame((cm + cm.T) / 2, index=ISSUES, columns=ISSUES).to_csv(a.output / 'cosine/cosine_cross_split_symmetric.csv')
    pd.DataFrame(cr).to_csv(a.output / 'cosine/cross_split_cosine_per_head.csv', index=False)

def aggregate(path, fold, measure):
    f = pd.read_csv(path).replace({'Security': 'Public Safety'})
    a, b, v = ('source', 'target', 'rho_target_standardized') if measure == 'transfer' else ('source_train_issue', 'target_heldout_issue', 'cosine')
    if not f.heldout_generator.eq(fold).all():
        raise ValueError('Incorrect held-out fold')
    if f.duplicated([a, b, 'layer', 'head']).any():
        raise ValueError('Duplicate per-head values')
    if not np.isfinite(f[v]).all():
        raise ValueError('Nonfinite per-head values')
    groups = f.groupby([a, b], sort=False)
    expected = {(x, y) for x in ISSUES for y in ISSUES}
    if set(groups.groups) != expected:
        raise ValueError('Missing or unexpected issue pair')
    headsets = [set(zip(g.layer, g['head'])) for _, g in groups]
    if not headsets[0] or any((h != headsets[0] for h in headsets)):
        raise ValueError('Different selected heads across issue pairs')
    if measure == 'transfer' and len(headsets[0]) != 20:
        raise ValueError('Expected 20 transfer heads')
    matrix = groups[v].mean().unstack().loc[ISSUES, ISSUES].to_numpy()
    return (matrix, len(headsets[0]))

def summarize(a):
    if a.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    rows = []
    matrices = {}
    inputs = []
    nulls = {}
    for measure in ['transfer', 'cosine']:
        refs = {}
        for name, file in [('original', f'pooled_original_{measure}_matrix.csv'), ('elected', 'utas_elected_spearman_matrix.csv'), ('all_candidates', 'utas_all_candidates_spearman_matrix.csv')]:
            path = a.results / file
            inputs.append(path)
            x = load_matrix(path)
            refs[name] = (x + x.T) / 2
        for fold in ['GPT', 'Claude', 'Gemini']:
            path = a.results / f'heldout_{fold}' / measure / ('transfer_per_head.csv' if measure == 'transfer' else 'cross_split_cosine_per_head.csv')
            inputs.append(path)
            m, n = aggregate(path, fold, measure)
            matrices[f'{measure}_{fold}_directional'] = m
            sym = (m + m.T) / 2
            matrices[f'{measure}_{fold}_symmetric'] = sym
            for name, ref in refs.items():
                stats, null = exact_mantel(sym, ref)
                rows.append({'measure': measure, 'heldout_generator': fold, 'reference': name, 'selected_heads': n, **stats})
                nulls[f'{measure}_{fold}_{name}'] = null
    a.output.mkdir(parents=True)
    for name, m in matrices.items():
        pd.DataFrame(m, index=ISSUES, columns=ISSUES).to_csv(a.output / f'{name}.csv')
    pd.DataFrame(rows).to_csv(a.output / 'comparisons.csv', index=False)
    np.savez(a.output / 'permutation_distributions.npz', **nulls)
    manifest = {'input_sha256': {str(p.relative_to(a.results)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, 'stage': 'saved per-head aggregation; no probe refitting or head reselection'}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

def prepare_figures(a):
    if a.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    comparisons = pd.read_csv(a.aggregated / 'comparisons.csv')
    if len(comparisons) != 18 or comparisons.duplicated(['measure', 'heldout_generator', 'reference']).any():
        raise ValueError('Expected 18 unique comparisons')
    inputs = [a.aggregated / 'comparisons.csv', a.original_transfer, a.original_cosine, a.elected, a.all_candidates]
    refs = {'transfer': load_matrix(a.original_transfer), 'cosine': load_matrix(a.original_cosine), 'elected': load_matrix(a.elected), 'all_candidates': load_matrix(a.all_candidates)}
    output_matrices = {'pooled_original_transfer_matrix.csv': refs['transfer'], 'pooled_original_cosine_matrix.csv': refs['cosine'], 'utas_elected_spearman_matrix.csv': refs['elected'], 'utas_all_candidates_spearman_matrix.csv': refs['all_candidates']}
    summaries = {}
    labels = {'transfer': {'original': 'target-standardized transfer vs pooled original transfer', 'elected': 'target-standardized transfer vs UTAS elected', 'all_candidates': 'target-standardized transfer vs UTAS all candidates'}, 'cosine': {'original': 'cross-split vs pooled original cosine', 'elected': 'cross-split cosine vs UTAS elected', 'all_candidates': 'cross-split cosine vs UTAS all candidates'}}
    for measure in ['transfer', 'cosine']:
        rows = []
        fold_matrices = []
        for fold in ['GPT', 'Claude', 'Gemini']:
            path = a.aggregated / f'{measure}_{fold}_directional.csv'
            inputs.append(path)
            m = load_matrix(path)
            fold_matrices.append(m)
            filename = 'transfer_matrix_target_standardized.csv' if measure == 'transfer' else 'cosine_cross_split_asymmetric.csv'
            output_matrices[f'heldout_{fold}/{measure}/{filename}'] = m
            for reference in ['original', 'elected', 'all_candidates']:
                r = refs[measure] if reference == 'original' else refs[reference]
                stats, _ = exact_mantel((m + m.T) / 2, (r + r.T) / 2)
                saved = comparisons[(comparisons.measure == measure) & (comparisons.heldout_generator == fold) & (comparisons.reference == reference)]
                if len(saved) != 1 or not np.allclose([stats['rho'], stats['p_one_sided']], saved[['rho', 'p_one_sided']].iloc[0].to_numpy(float), atol=1e-12, rtol=0):
                    raise ValueError('Aggregation statistics disagree with matrices')
                rows.append({'heldout_generator': fold, 'comparison': labels[measure][reference], 'mantel_spearman': stats['rho'], 'p_one_sided': stats['p_one_sided']})
        mean = np.mean(fold_matrices, axis=0)
        for reference in ['original', 'elected', 'all_candidates']:
            r = refs[measure] if reference == 'original' else refs[reference]
            stats, _ = exact_mantel((mean + mean.T) / 2, (r + r.T) / 2)
            rows.append({'heldout_generator': 'THREE_FOLD_MEAN', 'comparison': labels[measure][reference], 'mantel_spearman': stats['rho'], 'p_one_sided': stats['p_one_sided']})
        summaries[measure] = pd.DataFrame(rows)
    a.output.mkdir(parents=True)
    names = ['Security' if s == 'Public Safety' else s for s in ISSUES]
    for file, m in output_matrices.items():
        path = a.output / file
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(m, index=names, columns=names).to_csv(path)
    for measure, frame in summaries.items():
        frame.to_csv(a.output / f'{measure}_comparison_summary.csv', index=False)
    (a.output / 'manifest.json').write_text(json.dumps({'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, 'verified_fold_comparisons': 18}, indent=2))

def select(scores, output, checkpoints=False):
    if output.exists():
        raise FileExistsError('Choose a fresh output directory')
    arrays = {}
    hashes = {}
    full = {}
    fold_signatures = set()
    for c in THEMES.values():
        prefix = c['prefix']
        if checkpoints:
            items = []
            signatures = []
            for l in range(42):
                row = []
                for h in range(16):
                    p = scores / f'{prefix}_layer_{l:02d}_head_{h:02d}.npz'
                    with np.load(p) as z:
                        row.append({k: z[k].copy() for k in ['rho', 'alpha', 'coef', 'theta', 'scaler_mean', 'scaler_scale']})
                        signatures.append(json.loads(str(z['signature'].item())))
                    hashes[p.name] = sha(p)
                items.append(row)
            fold_signatures.update(((s['heldout'], s['kernel'], s['runner'], json.dumps(s['versions'], sort_keys=True)) for s in signatures))
            if len(fold_signatures) != 1:
                raise ValueError('Mixed checkpoints')
            full[prefix] = {k: np.array([[v[k] for v in row] for row in items], dtype=np.float32) for k in items[0][0]}
            arrays[prefix] = full[prefix]['rho']
        else:
            p = scores / f'{prefix}_rho.npy'
            arrays[prefix] = np.load(p)
            hashes[p.name] = sha(p)
        if arrays[prefix].shape != (42, 16) or not np.isfinite(arrays[prefix]).all():
            raise ValueError('Incomplete scores')
    stack = np.stack(list(arrays.values()))
    counts = np.sum(~np.isnan(stack), axis=0)
    avg = np.divide(np.nansum(stack, axis=0), counts, out=np.full((42, 16), np.nan), where=counts > 0)

    def top(v):
        return sorted([(l, h, float(v[l, h])) for l in range(42) for h in range(16)], key=lambda x: x[2], reverse=True)[:20]
    output.mkdir(parents=True)
    pd.DataFrame(top(avg), columns=['layer', 'head', 'mean_train_rho']).to_csv(output / 'transfer_common_top_heads.csv', index=False)
    union = set()
    for prefix, v in arrays.items():
        chosen = top(v)
        union.update(((l, h) for l, h, _ in chosen))
        pd.DataFrame(chosen, columns=['layer', 'head', 'train_rho']).to_csv(output / f'cosine_top_heads_{prefix}.csv', index=False)
    pd.DataFrame(sorted(union), columns=['layer', 'head']).to_csv(output / 'cosine_union_heads.csv', index=False)
    if checkpoints:
        (output / 'probing_full').mkdir()
        for prefix, d in full.items():
            for k, v in d.items():
                np.save(output / 'probing_full' / f'{prefix}_{k}.npy', v)
    (output / 'manifest.json').write_text(json.dumps({'input_hashes': hashes, 'union_size': len(union)}, indent=2))

def exact_p(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = (a + a.T) / 2
    b = (b + b.T) / 2
    u = np.triu_indices(len(a), 1)
    x = rankdata(a[u])
    x -= x.mean()
    x /= np.linalg.norm(x)

    def rho(m):
        y = rankdata(m[u])
        y -= y.mean()
        y /= np.linalg.norm(y)
        return float(x @ y)
    observed = rho(b)
    vals = [rho(b[np.ix_(p, p)]) for p in permutations(range(len(a)))]
    return sum((v >= observed - 1e-12 for v in vals)) / len(vals)

def draw_summary_table(ax, folds, original_values, original_p, categories, colors):
    pos = ax.get_position()
    table = ax.figure.add_axes([pos.x0 + pos.width * 0.62, pos.y0, pos.width * 0.38, pos.height])
    ax.set_position([pos.x0, pos.y0, pos.width * 0.57, pos.height])
    table.set_xlim(0, 1)
    table.set_ylim(-0.7, 14.1)
    table.axis('off')
    table.text(0.04, 13.8, 'Analysis', weight='bold', fontsize=9, va='bottom')
    table.text(0.93, 13.8, '$p$', fontsize=10, ha='right', va='bottom')
    for k, label in enumerate(categories):
        top = 12.8 - k * 5
        group = folds[folds.comparison_label == label].set_index('heldout_generator')
        for j, model in enumerate(['Original analysis', 'GPT', 'Claude', 'Gemini']):
            y = top - j
            orig = j == 0
            rho = original_values[label] if orig else float(group.loc[model, 'mantel_spearman'])
            pv = original_p[label] if orig else float(group.loc[model, 'p_one_sided'])
            color = 'black' if orig else colors[model]
            marker = 'X' if orig else 'o'
            edge = 'black' if orig or model == 'GPT' else 'white'
            ax.scatter(rho, y, s=32, marker=marker, c=color, edgecolors=edge, linewidths=0.6, zorder=3)
            table.scatter(0.055, y, s=28, marker=marker, c=color, edgecolors=edge, linewidths=0.6)
            table.text(0.14, y, 'Original' if orig else model + ' held out', fontsize=8.5, va='center')
            table.text(0.96, y, '—' if pv is None else f'{pv:.3f}', fontsize=8.5, ha='right', va='center')
            ax.axhline(y, color='#eeeeee', lw=0.45, zorder=0)
            ax.annotate(f'{rho:.3f}', (rho, y), xytext=(-7, 0), textcoords='offset points', va='center', ha='right', fontsize=8)
        for a in [ax, table]:
            a.axhline(top + 0.6, color='#999999', lw=0.6)
    ax.set_ylim(-0.7, 14.1)
    ax.set_yticks([11.3, 6.3, 1.3], categories)
    ax.set_xlim(0, 1.06)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel('Mantel Spearman ρ')
    ax.grid(axis='x', color='#dddddd', lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)

transfer_RESULTS = None

transfer_OUTPUT = None

transfer_ISSUES = ['Defense', 'Social Welfare', 'Public Works', 'Fiscal Stimulus', 'North Korea', 'Security']

transfer_DISPLAY_LABELS = ['Defense', 'Social\nwelfare', 'Public\nworks', 'Fiscal\nstimulus', 'North\nKorea', 'Public\nsafety']

transfer_GENERATORS = ['GPT', 'Claude', 'Gemini']

transfer_GENERATOR_COLORS = {'GPT': 'white', 'Claude': '#da7756', 'Gemini': '#078EFA'}

transfer_COMPARISON_LABELS = {'target-standardized transfer vs pooled original transfer': 'Original analysis', 'target-standardized transfer vs UTAS elected': 'UTAS elected', 'target-standardized transfer vs UTAS all candidates': 'UTAS candidates'}

def transfer_configure_style() -> None:
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': True, 'axes.linewidth': 0.8})

def transfer_load_matrices() -> dict[str, pd.DataFrame]:
    original_path = transfer_RESULTS / 'pooled_original_transfer_matrix.csv'
    original = pd.read_csv(original_path, index_col=0).loc[transfer_ISSUES, transfer_ISSUES]
    if original.shape != (6, 6) or original.isna().any().any():
        raise ValueError(f'Invalid original transfer matrix: {original_path}')
    matrices: dict[str, pd.DataFrame] = {'Original analysis': original}
    for generator in transfer_GENERATORS:
        path = transfer_RESULTS / f'heldout_{generator}' / 'transfer' / 'transfer_matrix_target_standardized.csv'
        matrix = pd.read_csv(path, index_col=0).loc[transfer_ISSUES, transfer_ISSUES]
        if matrix.shape != (6, 6) or matrix.isna().any().any():
            raise ValueError(f'Invalid transfer matrix: {path}')
        matrices[generator] = matrix
    return matrices

def transfer_matrix_spearman(matrix_a: pd.DataFrame, matrix_b: pd.DataFrame) -> float:
    """Spearman correlation across the 15 symmetrized off-diagonal pairs."""
    values_a = matrix_a.loc[transfer_ISSUES, transfer_ISSUES].to_numpy(dtype=float)
    values_b = matrix_b.loc[transfer_ISSUES, transfer_ISSUES].to_numpy(dtype=float)
    sym_a = (values_a + values_a.T) / 2
    sym_b = (values_b + values_b.T) / 2
    upper = np.triu_indices(len(transfer_ISSUES), k=1)
    return float(pd.Series(sym_a[upper]).corr(pd.Series(sym_b[upper]), method='spearman'))

def transfer_load_original_reference_values(original: pd.DataFrame) -> dict[str, float]:
    elected = pd.read_csv(transfer_RESULTS / 'utas_elected_spearman_matrix.csv', index_col=0).loc[transfer_ISSUES, transfer_ISSUES]
    candidates = pd.read_csv(transfer_RESULTS / 'utas_all_candidates_spearman_matrix.csv', index_col=0).loc[transfer_ISSUES, transfer_ISSUES]
    values = {'Original analysis': 1.0, 'UTAS elected': transfer_matrix_spearman(original, elected), 'UTAS candidates': transfer_matrix_spearman(original, candidates)}
    if not np.isclose(values['UTAS elected'], 0.8678571428571429):
        raise ValueError('Original UTAS-elected correlation does not match the manuscript')
    if not np.isclose(values['UTAS candidates'], 0.6107142857142855):
        raise ValueError('Original UTAS-candidates correlation does not match the manuscript')
    return values

def transfer_load_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(transfer_RESULTS / 'transfer_comparison_summary.csv')
    selected = summary[summary['comparison'].isin(transfer_COMPARISON_LABELS)].copy()
    selected['comparison_label'] = selected['comparison'].map(transfer_COMPARISON_LABELS)
    folds = selected[selected['heldout_generator'].isin(transfer_GENERATORS)].copy()
    mean_matrix = selected[selected['heldout_generator'] == 'THREE_FOLD_MEAN'].copy()
    expected = len(transfer_GENERATORS) * len(transfer_COMPARISON_LABELS)
    if len(folds) != expected or len(mean_matrix) != len(transfer_COMPARISON_LABELS):
        raise ValueError('Missing generator-held-out summary rows')
    if folds['mantel_spearman'].isna().any() or mean_matrix['mantel_spearman'].isna().any():
        raise ValueError('Missing Mantel correlations')
    return (folds, mean_matrix)

def transfer_add_panel_label(ax: mpl.axes.Axes, label: str, x: float=-0.18, y: float=1.34) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight='bold', ha='left', va='top')

def transfer_format_heatmap_value(value: float) -> str:
    """Format correlations to two decimal places, including the leading zero."""
    return f'{value:.2f}'.replace('-', '−')

def transfer_plot_heatmap(ax: mpl.axes.Axes, matrix: pd.DataFrame, title: str, norm: TwoSlopeNorm, cmap: mpl.colors.Colormap, show_ylabels: bool) -> mpl.image.AxesImage:
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
    ax.set_xticks(range(6), transfer_DISPLAY_LABELS, rotation=90)
    ax.set_yticks(range(6), transfer_DISPLAY_LABELS if show_ylabels else [''] * 6)
    ax.set_title(title, pad=7, fontweight='normal')
    ax.tick_params(axis='both', length=0, pad=2)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.75)
    ax.tick_params(which='minor', bottom=False, left=False)
    for row in range(6):
        for col in range(6):
            value = values[row, col]
            red, green, blue, _ = cmap(norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = 'white' if luminance < 0.48 else 'black'
            ax.text(col, row, transfer_format_heatmap_value(value), ha='center', va='center', fontsize=9, color=color, fontweight='bold' if row == col else 'normal')
    return image

def transfer_plot_summary(ax: mpl.axes.Axes, folds: pd.DataFrame, mean_matrix: pd.DataFrame, original_values: dict[str, float]) -> None:
    categories = list(transfer_COMPARISON_LABELS.values())
    y_positions = {label: 2 - idx for idx, label in enumerate(categories)}
    offsets = {'GPT': 0.23, 'Claude': 0.0, 'Gemini': -0.23}
    original = transfer_load_matrices()['Original analysis']
    original_p = {'Original analysis': None}
    for label, filename in [('UTAS elected', 'utas_elected_spearman_matrix.csv'), ('UTAS candidates', 'utas_all_candidates_spearman_matrix.csv')]:
        reference = pd.read_csv(transfer_RESULTS / filename, index_col=0).loc[transfer_ISSUES, transfer_ISSUES]
        original_p[label] = exact_p(original, reference)
    draw_summary_table(ax, folds, original_values, original_p, categories, transfer_GENERATOR_COLORS)

def transfer_create_figure() -> tuple[Path, Path, Path]:
    transfer_configure_style()
    matrices = transfer_load_matrices()
    folds, mean_matrix = transfer_load_summary()
    original_values = transfer_load_original_reference_values(matrices['Original analysis'])
    transfer_OUTPUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.5, 10.5), facecolor='white')
    grid = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.055], height_ratios=[1, 1, 1.55], left=0.155, right=0.92, bottom=0.065, top=0.965, wspace=0.23, hspace=0.48)
    cmap = mpl.colormaps['RdBu_r']
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    image = None
    panel_specs = [('Original analysis', 'Original analysis'), ('GPT', 'GPT held out'), ('Claude', 'Claude held out'), ('Gemini', 'Gemini held out')]
    panel_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for panel_index, ((matrix_key, title), (row, col)) in enumerate(zip(panel_specs, panel_positions)):
        ax = fig.add_subplot(grid[row, col])
        image = transfer_plot_heatmap(ax, matrices[matrix_key], title, norm, cmap, show_ylabels=col == 0)
        transfer_add_panel_label(ax, f'({chr(65 + panel_index)})', x=-0.24 if col == 0 else -0.13, y=1.12)
    color_ax = fig.add_subplot(grid[:2, 2])
    colorbar = fig.colorbar(image, cax=color_ax, ticks=[-1, -0.5, 0, 0.5, 1])
    colorbar.set_label('Transfer performance (ρ)', rotation=90, labelpad=7)
    colorbar.ax.tick_params(labelsize=8, width=0.8, length=3)
    summary_ax = fig.add_subplot(grid[2, :2])
    transfer_plot_summary(summary_ax, folds, mean_matrix, original_values)
    transfer_add_panel_label(summary_ax, '(E)', x=-0.31, y=1.07)
    pdf_path = transfer_OUTPUT / 'S8_Fig.pdf'
    png_path = transfer_OUTPUT / 'S8_Fig.png'
    tif_path = transfer_OUTPUT / 'S8_Fig.tif'
    fig.savefig(pdf_path, facecolor='white')
    fig.savefig(png_path, dpi=300, facecolor='white', transparent=False)
    fig.savefig(tif_path, dpi=600, facecolor='white', transparent=False, pil_kwargs={'compression': 'tiff_lzw'})
    with Image.open(tif_path) as image:
        image.convert('RGB').save(tif_path, format='TIFF', compression='tiff_lzw', dpi=(600, 600))
    plt.close(fig)
    return (pdf_path, png_path, tif_path)

transfer_renderer = SimpleNamespace(load_matrices=transfer_load_matrices, load_summary=transfer_load_summary, matrix_spearman=transfer_matrix_spearman, ISSUES=transfer_ISSUES, GENERATORS=transfer_GENERATORS, create_figure=transfer_create_figure)

cosine_RESULTS = None

cosine_OUTPUT = None

cosine_ISSUES = ['Defense', 'Social Welfare', 'Public Works', 'Fiscal Stimulus', 'North Korea', 'Security']

cosine_DISPLAY_LABELS = ['Defense', 'Social\nwelfare', 'Public\nworks', 'Fiscal\nstimulus', 'North\nKorea', 'Public\nsafety']

cosine_GENERATORS = ['GPT', 'Claude', 'Gemini']

cosine_GENERATOR_COLORS = {'GPT': 'white', 'Claude': '#da7756', 'Gemini': '#078EFA'}

cosine_COMPARISON_LABELS = {'cross-split vs pooled original cosine': 'Original analysis', 'cross-split cosine vs UTAS elected': 'UTAS elected', 'cross-split cosine vs UTAS all candidates': 'UTAS candidates'}

def cosine_configure_style() -> None:
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': True, 'axes.linewidth': 0.8})

def cosine_load_matrices() -> dict[str, pd.DataFrame]:
    original_path = cosine_RESULTS / 'pooled_original_cosine_matrix.csv'
    original = pd.read_csv(original_path, index_col=0).loc[cosine_ISSUES, cosine_ISSUES]
    if original.shape != (6, 6) or original.isna().any().any():
        raise ValueError(f'Invalid original cosine matrix: {original_path}')
    matrices: dict[str, pd.DataFrame] = {'Original analysis': original}
    for generator in cosine_GENERATORS:
        path = cosine_RESULTS / f'heldout_{generator}' / 'cosine' / 'cosine_cross_split_asymmetric.csv'
        matrix = pd.read_csv(path, index_col=0).loc[cosine_ISSUES, cosine_ISSUES]
        if matrix.shape != (6, 6) or matrix.isna().any().any():
            raise ValueError(f'Invalid cross-split cosine matrix: {path}')
        matrices[generator] = matrix
    return matrices

def cosine_matrix_spearman(matrix_a: pd.DataFrame, matrix_b: pd.DataFrame) -> float:
    """Spearman correlation across the 15 symmetrized off-diagonal pairs."""
    values_a = matrix_a.loc[cosine_ISSUES, cosine_ISSUES].to_numpy(dtype=float)
    values_b = matrix_b.loc[cosine_ISSUES, cosine_ISSUES].to_numpy(dtype=float)
    sym_a = (values_a + values_a.T) / 2
    sym_b = (values_b + values_b.T) / 2
    upper = np.triu_indices(len(cosine_ISSUES), k=1)
    return float(pd.Series(sym_a[upper]).corr(pd.Series(sym_b[upper]), method='spearman'))

def cosine_load_original_reference_values(original: pd.DataFrame) -> dict[str, float]:
    elected = pd.read_csv(cosine_RESULTS / 'utas_elected_spearman_matrix.csv', index_col=0).loc[cosine_ISSUES, cosine_ISSUES]
    candidates = pd.read_csv(cosine_RESULTS / 'utas_all_candidates_spearman_matrix.csv', index_col=0).loc[cosine_ISSUES, cosine_ISSUES]
    values = {'Original analysis': 1.0, 'UTAS elected': cosine_matrix_spearman(original, elected), 'UTAS candidates': cosine_matrix_spearman(original, candidates)}
    if not np.isclose(values['UTAS elected'], 0.85):
        raise ValueError('Original UTAS-elected correlation does not match the analysis')
    if not np.isclose(values['UTAS candidates'], 0.625):
        raise ValueError('Original UTAS-candidates correlation does not match the analysis')
    return values

def cosine_load_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(cosine_RESULTS / 'cosine_comparison_summary.csv')
    selected = summary[summary['comparison'].isin(cosine_COMPARISON_LABELS)].copy()
    selected['comparison_label'] = selected['comparison'].map(cosine_COMPARISON_LABELS)
    folds = selected[selected['heldout_generator'].isin(cosine_GENERATORS)].copy()
    mean_matrix = selected[selected['heldout_generator'] == 'THREE_FOLD_MEAN'].copy()
    expected = len(cosine_GENERATORS) * len(cosine_COMPARISON_LABELS)
    if len(folds) != expected or len(mean_matrix) != len(cosine_COMPARISON_LABELS):
        raise ValueError('Missing generator-held-out cosine summary rows')
    if folds['mantel_spearman'].isna().any() or mean_matrix['mantel_spearman'].isna().any():
        raise ValueError('Missing Mantel correlations')
    return (folds, mean_matrix)

def cosine_add_panel_label(ax: mpl.axes.Axes, label: str, x: float=-0.18, y: float=1.34) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight='bold', ha='left', va='top')

def cosine_plot_heatmap(ax: mpl.axes.Axes, matrix: pd.DataFrame, title: str, norm: TwoSlopeNorm, cmap: mpl.colors.Colormap, show_ylabels: bool) -> mpl.image.AxesImage:
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
    ax.set_xticks(range(6), cosine_DISPLAY_LABELS, rotation=90)
    ax.set_yticks(range(6), cosine_DISPLAY_LABELS if show_ylabels else [''] * 6)
    ax.set_title(title, pad=7, fontweight='normal')
    ax.tick_params(axis='both', length=0, pad=2)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.75)
    ax.tick_params(which='minor', bottom=False, left=False)
    for row in range(6):
        for col in range(6):
            value = values[row, col]
            red, green, blue, _ = cmap(norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = 'white' if luminance < 0.48 else 'black'
            ax.text(col, row, f'{value:.2f}'.replace('-', '−'), ha='center', va='center', fontsize=9, color=color, fontweight='bold' if row == col else 'normal')
    return image

def cosine_plot_summary(ax: mpl.axes.Axes, folds: pd.DataFrame, mean_matrix: pd.DataFrame, original_values: dict[str, float]) -> None:
    categories = list(cosine_COMPARISON_LABELS.values())
    y_positions = {label: 2 - idx for idx, label in enumerate(categories)}
    offsets = {'GPT': 0.23, 'Claude': 0.0, 'Gemini': -0.23}
    original = cosine_load_matrices()['Original analysis']
    original_p = {'Original analysis': None}
    for label, filename in [('UTAS elected', 'utas_elected_spearman_matrix.csv'), ('UTAS candidates', 'utas_all_candidates_spearman_matrix.csv')]:
        reference = pd.read_csv(cosine_RESULTS / filename, index_col=0).loc[cosine_ISSUES, cosine_ISSUES]
        original_p[label] = exact_p(original, reference)
    draw_summary_table(ax, folds, original_values, original_p, categories, cosine_GENERATOR_COLORS)

def cosine_create_figure() -> tuple[Path, Path, Path, Path]:
    cosine_configure_style()
    matrices = cosine_load_matrices()
    folds, mean_matrix = cosine_load_summary()
    original_values = cosine_load_original_reference_values(matrices['Original analysis'])
    cosine_OUTPUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.5, 10.5), facecolor='white')
    grid = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.055], height_ratios=[1, 1, 1.55], left=0.155, right=0.92, bottom=0.065, top=0.965, wspace=0.23, hspace=0.48)
    cmap = mpl.colormaps['RdBu_r']
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    image = None
    panel_specs = [('Original analysis', 'Original analysis'), ('GPT', 'GPT held out'), ('Claude', 'Claude held out'), ('Gemini', 'Gemini held out')]
    panel_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for panel_index, ((matrix_key, title), (row, col)) in enumerate(zip(panel_specs, panel_positions)):
        ax = fig.add_subplot(grid[row, col])
        image = cosine_plot_heatmap(ax, matrices[matrix_key], title, norm, cmap, show_ylabels=col == 0)
        cosine_add_panel_label(ax, f'({chr(65 + panel_index)})', x=-0.24 if col == 0 else -0.13, y=1.12)
    color_ax = fig.add_subplot(grid[:2, 2])
    colorbar = fig.colorbar(image, cax=color_ax, ticks=[-1, -0.5, 0, 0.5, 1])
    colorbar.set_label('Cosine similarity', rotation=90, labelpad=7)
    colorbar.ax.tick_params(labelsize=8, width=0.8, length=3)
    summary_ax = fig.add_subplot(grid[2, :2])
    cosine_plot_summary(summary_ax, folds, mean_matrix, original_values)
    cosine_add_panel_label(summary_ax, '(E)', x=-0.31, y=1.07)
    pdf_path = cosine_OUTPUT / 'S9_Fig.pdf'
    png_path = cosine_OUTPUT / 'S9_Fig.png'
    tif_path = cosine_OUTPUT / 'S9_Fig.tif'
    fig.savefig(pdf_path, facecolor='white')
    fig.savefig(png_path, dpi=300, facecolor='white', transparent=False)
    fig.savefig(tif_path, dpi=600, facecolor='white', transparent=False, pil_kwargs={'compression': 'tiff_lzw'})
    with Image.open(tif_path) as image_tif:
        image_tif.convert('RGB').save(tif_path, format='TIFF', compression='tiff_lzw', dpi=(600, 600))
    plt.close(fig)
    return (pdf_path, png_path, tif_path)

cosine_renderer = SimpleNamespace(load_matrices=cosine_load_matrices, load_summary=cosine_load_summary, matrix_spearman=cosine_matrix_spearman, ISSUES=cosine_ISSUES, GENERATORS=cosine_GENERATORS, create_figure=cosine_create_figure)

def figures(a):
    global transfer_RESULTS, transfer_OUTPUT, cosine_RESULTS, cosine_OUTPUT
    plotting_dependencies()
    transfer_RESULTS = cosine_RESULTS = a.results
    transfer_OUTPUT = cosine_OUTPUT = a.output
    if a.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    checks = []
    sources = {}
    for measure, module in [('transfer', transfer_renderer), ('cosine', cosine_renderer)]:
        module.RESULTS = a.results
        module.OUTPUT = a.output
        matrices = module.load_matrices()
        folds, _ = module.load_summary()
        references = {'Original analysis': matrices['Original analysis']}
        for label, file in [('UTAS elected', 'utas_elected_spearman_matrix.csv'), ('UTAS candidates', 'utas_all_candidates_spearman_matrix.csv')]:
            references[label] = pd.read_csv(a.results / file, index_col=0).loc[module.ISSUES, module.ISSUES]
        for _, row in folds.iterrows():
            x = matrices[row.heldout_generator]
            y = references[row.comparison_label]
            rho = module.matrix_spearman(x, y)
            pv = exact_p(x, y)
            if not np.isclose(rho, row.mantel_spearman, rtol=0, atol=1e-12) or not np.isclose(pv, row.p_one_sided, rtol=0, atol=1e-12):
                raise ValueError('Matrix/summary mismatch')
            checks.append({'measure': measure, 'generator': row.heldout_generator, 'reference': row.comparison_label, 'rho': rho, 'p': pv})
        files = [a.results / f'pooled_original_{measure}_matrix.csv', a.results / f'{measure}_comparison_summary.csv', a.results / 'utas_elected_spearman_matrix.csv', a.results / 'utas_all_candidates_spearman_matrix.csv']
        for generator in module.GENERATORS:
            files.append(a.results / f'heldout_{generator}' / measure / ('transfer_matrix_target_standardized.csv' if measure == 'transfer' else 'cosine_cross_split_asymmetric.csv'))
        for path in files:
            sources[str(path.relative_to(a.results))] = hashlib.sha256(path.read_bytes()).hexdigest()
    a.output.mkdir(parents=True)
    transfer_create_figure()
    cosine_create_figure()
    (a.output / 'verification.json').write_text(json.dumps({'input_sha256': sources, 'recomputed_fold_comparisons': checks}, indent=2))
    print('Rendered S8 Fig and S9 Fig')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='operation',required=True)
    for command in ['train','select','evaluate','summarize','prepare-figures','figures']:
        q=sub.add_parser(command)
        q.add_argument('--output',type=Path,required=True)
        if command in ['train','evaluate']:
            q.add_argument('--heldout',choices=['GPT','Claude','Gemini'],required=True)
            q.add_argument('--config')
        if command=='train':
            q.add_argument('--issues',nargs='+',choices=list(THEMES))
            q.add_argument('--layers',nargs='+',type=int,default=list(range(42)))
            q.add_argument('--heads',nargs='+',type=int,default=list(range(16)))
        if command in ['select','evaluate']:q.add_argument('--scores',type=Path,required=True)
        if command=='select':q.add_argument('--checkpoints',action='store_true')
        if command=='evaluate':q.add_argument('--heads',type=Path,required=True)
        if command in ['summarize','figures']:q.add_argument('--results',type=Path,required=True)
        if command=='prepare-figures':
            for key in ['aggregated','original-transfer','original-cosine','elected','all-candidates']:
                q.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.operation=='select':select(a.scores,a.output,a.checkpoints)
    else:{'train':train,'evaluate':evaluate,'summarize':summarize,'prepare-figures':prepare_figures,'figures':figures}[a.operation](a)

if __name__=='__main__':main()
