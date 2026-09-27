"""Ordinal probing: train, collect, select heads, and export associated figures/tables.

Saved-result commands do not train probes. Train is explicit and resumes only
matching per-head checkpoints. Gemma and Llama retain separate numerical kernels.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
from itertools import combinations
from types import SimpleNamespace
import argparse, hashlib, json, importlib.metadata
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common.configuration import ROOT, load_paths
from common.constants import THEMES
from common.data_loading import load_statements
ALPHAS=np.logspace(-2,3,6)
ISSUES=[('Defense','Defense'),('Social welfare','Social'),('Public works','Public'),('Fiscal stimulus','Fiscal'),('North Korea','Nkorea'),('Public safety','Security')]

def _learning_dependencies():
    global mord, StratifiedKFold, StandardScaler
    import mord
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler



# 1. Model-specific fitting. Numerical bodies preserved.

def _spearman_gemma(y_true, y_pred):
    """全データをそのまま使用してスピアマン相関を計算。"""
    if len(y_true) < 3:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    c, _ = spearmanr(y_true, y_pred)
    return c if not np.isnan(c) else np.nan

def fit_head_gemma(h_idx, layer_data, labels_ordinal, ALPHAS):
    """1つのヘッドに対するCV + 最終学習を実行（LogisticAT使用）"""
    _learning_dependencies()
    X = layer_data[:, h_idx, :]
    y = labels_ordinal

    # 各AlphaのOOF予測を一時保存
    oof_preds_temp = {a: np.zeros_like(y, dtype=float) for a in ALPHAS}
    alpha_cv_scores = {a: [] for a in ALPHAS}

    # ★StratifiedKFoldに変更：各foldでラベル分布を維持
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        for alpha in ALPHAS:
            # ★LogisticATを使用（All-Threshold: 絶対誤差を考慮した損失関数）
            model = mord.LogisticAT(alpha=alpha)

            try:
                model.fit(X_train_scaled, y_train)
                y_pred_val = model.predict(X_val_scaled)
            except Exception:
                continue

            oof_preds_temp[alpha][val_idx] = y_pred_val

            # 元スケールに戻して評価
            y_val_original = y_val + 1
            y_pred_val_original = y_pred_val + 1

            score = _spearman_gemma(y_val_original, y_pred_val_original)
            if not np.isnan(score):
                alpha_cv_scores[alpha].append(score)

    # 平均スコア算出
    avg_scores = {}
    for a, scores in alpha_cv_scores.items():
        avg_scores[a] = np.mean(scores) if len(scores) > 0 else np.nan

    # ベストスコア決定
    if all(np.isnan(v) for v in avg_scores.values()):
        best_alpha = ALPHAS[0]
        best_score = 0.0
    else:
        valid_scores = {k: v for k, v in avg_scores.items() if not np.isnan(v)}
        if valid_scores:
            best_alpha = max(valid_scores, key=valid_scores.get)
            best_score = valid_scores[best_alpha]
        else:
            best_alpha = ALPHAS[0]
            best_score = 0.0

    # 全データ再学習
    scaler_final = StandardScaler()
    X_final = scaler_final.fit_transform(X)
    final_model = mord.LogisticAT(alpha=best_alpha)

    coef = np.zeros(X.shape[1])
    theta = np.zeros(4)  # 5カテゴリなので4つのしきい値

    try:
        final_model.fit(X_final, y)
        coef = final_model.coef_.flatten()  # LogisticATも1次元の係数
        # ★LogisticATはtheta_を持つ（学習されたしきい値）
        if hasattr(final_model, 'theta_') and final_model.theta_ is not None:
            theta = final_model.theta_
    except Exception:
        pass

    # OOF予測（元スケール）
    oof_preds = oof_preds_temp[best_alpha] + 1

    return h_idx, best_score, best_alpha, coef, theta, oof_preds

def _spearman_llama(y_true, y_pred):
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    value = spearmanr(y_true, y_pred).statistic
    return value if np.isfinite(value) else np.nan

def fit_head_llama(h_idx, layer_data, labels_ordinal):
    """1つのヘッドに対するCV + 最終学習を実行（LogisticAT使用）"""
    _learning_dependencies()
    X = np.asarray(layer_data[:, h_idx, :], dtype=np.float64)
    y = labels_ordinal

    # 各AlphaのOOF予測を一時保存
    oof_preds_temp = {a: np.zeros_like(y, dtype=float) for a in ALPHAS}
    alpha_cv_scores = {a: [] for a in ALPHAS}

    # ★StratifiedKFoldに変更：各foldでラベル分布を維持
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        for alpha in ALPHAS:
            # ★LogisticATを使用（All-Threshold: 絶対誤差を考慮した損失関数）
            model = mord.LogisticAT(alpha=alpha)

            try:
                model.fit(X_train_scaled, y_train)
                y_pred_val = model.predict(X_val_scaled)
            except Exception as exc:
                raise RuntimeError(f"Probe CV failed: head={h_idx}, alpha={alpha}") from exc

            oof_preds_temp[alpha][val_idx] = y_pred_val

            # 元スケールに戻して評価
            y_val_original = y_val + 1
            y_pred_val_original = y_pred_val + 1

            score = _spearman_llama(y_val_original, y_pred_val_original)
            if not np.isnan(score):
                alpha_cv_scores[alpha].append(score)

    # 平均スコア算出
    avg_scores = {}
    for a, scores in alpha_cv_scores.items():
        avg_scores[a] = np.mean(scores) if len(scores) > 0 else np.nan

    # ベストスコア決定
    if all(np.isnan(v) for v in avg_scores.values()):
        raise RuntimeError(f"No valid CV score: head={h_idx}")
    else:
        valid_scores = {k: v for k, v in avg_scores.items() if not np.isnan(v)}
        if valid_scores:
            best_alpha = max(valid_scores, key=valid_scores.get)
            best_score = valid_scores[best_alpha]
        else:
            best_alpha = ALPHAS[0]
            best_score = 0.0

    # 全データ再学習
    scaler_final = StandardScaler()
    X_final = scaler_final.fit_transform(X)
    final_model = mord.LogisticAT(alpha=best_alpha)

    coef = np.zeros(X.shape[1])
    theta = np.zeros(4)  # 5カテゴリなので4つのしきい値

    try:
        final_model.fit(X_final, y)
        coef = final_model.coef_.flatten()  # LogisticATも1次元の係数
        # ★LogisticATはtheta_を持つ（学習されたしきい値）
        if hasattr(final_model, 'theta_') and final_model.theta_ is not None:
            theta = final_model.theta_
    except Exception as exc:
        raise RuntimeError(f"Final probe fit failed: head={h_idx}, alpha={best_alpha}") from exc

    # OOF予測（元スケール）
    oof_preds = oof_preds_temp[best_alpha] + 1

    if not (np.isfinite(coef).all() and np.isfinite(theta).all() and np.isfinite(oof_preds).all()):
        raise RuntimeError(f"Non-finite probe output: head={h_idx}")
    return h_idx, best_score, best_alpha, coef, theta, oof_preds

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

# 2. Checkpointed training and full-run aggregation.

def train(a):
 paths=load_paths(a.config);shape=json.loads((ROOT/'config/extraction.json').read_text())[a.model]
 layers=a.layers if a.layers is not None else list(range(shape['layers']));heads=a.heads if a.heads is not None else list(range(shape['heads']))
 if not layers or not heads or min(layers)<0 or max(layers)>=shape['layers'] or min(heads)<0 or max(heads)>=shape['heads']:raise ValueError('Invalid layer/head indices')
 frames,sources=load_statements(paths['synthetic_statements_dir']);kernel=SimpleNamespace(ALPHAS=ALPHAS,process_one_head=fit_head_gemma if a.model=='gemma' else fit_head_llama)
 out=paths['output_dir']/'probing'/a.model;out.mkdir(parents=True,exist_ok=True)
 versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','mord']}
 for issue in a.issues or list(THEMES):
  cfg=THEMES[issue];labels=frames[issue].Stance_Value.to_numpy(int)-1
  for layer in layers:
   base=paths[a.model+'_activations_dir'];candidates=[base/f'{cfg["vec"]}_layer_{layer:02d}.npy']
   if a.model=='gemma' and cfg['subdir']:candidates.append(base/cfg['subdir']/candidates[0].name)
   found=[p for p in candidates if p.exists()]
   if len(found)!=1:raise ValueError('Need exactly one activation file for '+issue)
   vector=found[0];data=np.load(vector,mmap_mode='r')
   if data.shape!=(4320,shape['heads'],shape['head_dim']) or not np.isfinite(data).all():raise ValueError('Invalid activations')
   signature={'vector':sha(vector),'statements':sha(sources[issue]),'kernel':sha(Path(__file__)),'runner':sha(Path(__file__)),'versions':versions,'input_dtype':str(data.dtype)}
   for head in heads:
    file=out/f'{cfg["prefix"]}_layer_{layer:02d}_head_{head:02d}.npz';sig={**signature,'head':head};encoded=json.dumps(sig,sort_keys=True)
    if file.exists():
     with np.load(file) as z:
      if str(z['signature'].item())!=encoded:raise ValueError('Checkpoint input/settings mismatch')
      if not all(np.isfinite(z[k]).all() for k in ['rho','alpha','coef','theta','preds']):raise ValueError('Corrupt checkpoint')
     continue
    args=(head,data,labels,kernel.ALPHAS) if a.model=='gemma' else (head,data,labels)
    h,rho,alpha,coef,theta,preds=kernel.process_one_head(*args)
    if not all(np.isfinite(x).all() for x in [rho,alpha,coef,theta,preds]) or not np.any(coef):raise ValueError('Invalid/failed fit')
    tmp=file.with_suffix('.tmp.npz');np.savez(tmp,rho=rho,alpha=alpha,coef=coef,theta=theta,preds=preds,signature=encoded);tmp.replace(file)
    print(f'{a.model}: {issue}, layer {layer}, head {head} saved',flush=True)

def collect(source,output,model):
 if output.exists():raise FileExistsError('Choose a fresh output directory')
 spec=json.loads((ROOT/'config/extraction.json').read_text())[model];L,H,D=spec['layers'],spec['heads'],spec['head_dim']
 # Preflight completeness before writing any summaries.
 for c in THEMES.values():
  issue_signature=None
  for l in range(L):
   for h in range(H):
    if not (source/f'{c["prefix"]}_layer_{l:02d}_head_{h:02d}.npz').exists():raise ValueError('Incomplete probing run; cannot form full head rankings')
    with np.load(source/f'{c["prefix"]}_layer_{l:02d}_head_{h:02d}.npz') as z:
     sig=json.loads(str(z['signature'].item()))
     if sig.pop('head')!=h:raise ValueError('Checkpoint head mismatch')
     sig.pop('vector')
     if issue_signature is None:issue_signature=sig
     elif sig!=issue_signature:raise ValueError('Mixed probing configurations')
 output.mkdir(parents=True)
 for c in THEMES.values():
  prefix=c['prefix'];arrays={'rho':np.empty((L,H)),'alpha':np.empty((L,H)),'coef':np.empty((L,H,D)),'theta':np.empty((L,H,4))}
  for l in range(L):
   predictions=[]
   for h in range(H):
    with np.load(source/f'{prefix}_layer_{l:02d}_head_{h:02d}.npz') as z:
     for key,a in arrays.items():
      if not np.isfinite(z[key]).all():raise ValueError('Invalid checkpoint')
      a[l,h]=z[key]
     predictions.append(z['preds'])
   np.save(output/f'{prefix}_layer_{l:02d}_preds.npy',np.stack(predictions,axis=1))
  for key,suffix in [('rho','heatmap_rho'),('alpha','bestAlpha'),('coef','coef'),('theta','theta')]:np.save(output/f'{prefix}_{suffix}_full.npy',arrays[key])

# 3. Head selection and numerical tables.

def select(directory,output,model):
 if output.exists():raise FileExistsError('Choose a fresh output directory')
 shape=(42,16) if model=='gemma' else (32,32)
 arrays={};hashes={}
 for issue,c in THEMES.items():
  p=directory/f'{c["prefix"]}_heatmap_rho_full.npy';v=np.load(p)
  if v.shape!=shape or not np.isfinite(v).all():raise ValueError('Incomplete rho arrays')
  arrays[c['prefix']]=v;hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 rows=[]
 for layer in range(shape[0]):
  for head in range(shape[1]):
   row={'layer':layer,'head':head};row.update({'rho_'+k:float(v[layer,head]) for k,v in arrays.items()});row['mean_rho']=float(np.mean([v[layer,head] for v in arrays.values()]));rows.append(row)
 scores=pd.DataFrame(rows).sort_values(['mean_rho','layer','head'],ascending=[False,True,True]);common=scores.head(20)
 selected={k:scores.sort_values(['rho_'+k,'layer','head'],ascending=[False,True,True]).head(20) for k in arrays}
 union=sorted(set().union(*(set(zip(d.layer,d['head'])) for d in selected.values())))
 output.mkdir(parents=True);common.to_csv(output/'common_top20_heads.csv',index=False)
 for k,d in selected.items():d.to_csv(output/f'per_theme_top20_{k}.csv',index=False)
 pd.DataFrame(union,columns=['layer','head']).to_csv(output/'union_heads.csv',index=False)
 (output/'manifest.json').write_text(json.dumps({'input_sha256':hashes,'tie_break':'layer then head ascending','union_size':len(union)},indent=2))
 return common,selected,union

def export_tables(a):
 if a.output.exists():raise FileExistsError('Choose a fresh output directory')
 a.output.mkdir(parents=True)
 tables={};hashes={}
 for model,summary_no,union_no,common_no in [('gemma',3,6,8),('llama',4,7,9)]:
  directory=getattr(a,model+'_scores');common,selected,union=select(directory,a.output/(model+'_selection'),model)
  arrays={}
  for title,prefix in ISSUES:
   path=directory/f'{prefix}_heatmap_rho_full.npy';arrays[prefix]=np.load(path);hashes[model+'/'+path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
  rows=[]
  for title,prefix in ISSUES:
   v=arrays[prefix];best=np.unravel_index(np.argmax(v),v.shape)
   rows.append({'Issue':title,'Max ρ':f'{v.max():.3f}','Best layer':int(best[0]),'Mean ρ (top-20 heads)':f"{common['rho_'+prefix].mean():.3f}",'Mean ρ (all heads)':f'{v.mean():.3f}'})
  tables[f'S{summary_no}_Table']=pd.DataFrame(rows)
  sets={prefix:set(zip(frame.layer,frame['head'])) for prefix,frame in selected.items()}
  rows=[]
  for layer,head in union:
   row={'Layer':layer,'Head':head};count=0
   for title,prefix in ISSUES:
    included=(layer,head) in sets[prefix];count+=included;row[title]=f'{arrays[prefix][layer,head]:.3f}' if included else '---'
   row['Number of issues']=count;rows.append(row)
  tables[f'S{union_no}_Table']=pd.DataFrame(rows).sort_values(['Number of issues','Layer','Head'],ascending=[False,True,True])
  rows=[]
  for rank,(_,v) in enumerate(common.iterrows(),1):
   row={'Rank':rank,'Layer':int(v.layer),'Head':int(v['head']),'Mean':f'{v.mean_rho:.3f}'}
   row.update({title:f"{v['rho_'+prefix]:.3f}" for title,prefix in ISSUES});rows.append(row)
  tables[f'S{common_no}_Table']=pd.DataFrame(rows)
  if model=='llama':
   pairs=[]
   for (left,s),(right,t) in combinations(sets.items(),2):pairs.append({'issue_1':left,'issue_2':right,'shared_heads':len(s&t),'jaccard':len(s&t)/len(s|t)})
   f=pd.DataFrame(pairs);f.to_csv(a.output/'llama_overlap_unrounded.csv',index=False)
   intersection=len(set.intersection(*sets.values()))
   tables['S5_Table']=pd.DataFrame([
    ('Shared heads between each pair of issue-specific top-20 sets',f'{f.shared_heads.min()}/20--{f.shared_heads.max()}/20'),
    ('Jaccard index between issue-specific top-20 sets',f'{f.jaccard.min():.3f}--{f.jaccard.max():.3f}'),
    ('Heads in the union of the six top-20 sets (maximum: 120)',f'{len(union)}/120 ({len(union)/120*100:.1f}%)'),
    ('Heads common to all six top-20 sets',f'{intersection}/20 ({intersection/20*100:.1f}%)')],columns=['Measure','Llama'])
 for name,f in tables.items():
  f.to_csv(a.output/(name+'.csv'),index=False)
  lines=['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']
  lines+=['| '+' | '.join(map(str,row))+' |' for row in f.itertuples(index=False,name=None)]
  (a.output/(name+'.md')).write_text('\n'.join(lines)+'\n')
 (a.output/'manifest.json').write_text(json.dumps({'input_sha256':hashes,'tables':sorted(tables),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))

# 4. Rendering associated saved probing results.

manifest=[]

def load(p):
 a=np.load(p);manifest.append({'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});return a

def save(fig,stem):
 dpi=600 if stem in ('Fig2','Fig3') else 300
 fig.savefig(OUT/f'{stem}.pdf');fig.savefig(OUT/f'{stem}.png',dpi=dpi)
 with Image.open(OUT/f'{stem}.png') as im:im.convert('RGB').save(OUT/f'{stem}.tif',compression='tiff_lzw',dpi=(dpi,dpi))
 plt.close(fig)

def rankplot(directory,stem):
 fig,axs=plt.subplots(6,1,figsize=(5.2,6.5));fig.subplots_adjust(left=.12,right=.83,top=.97,bottom=.08,hspace=.35)
 for i,(ax,(title,prefix)) in enumerate(zip(axs,ISSUES)):
  a=load(directory/f'{prefix}_heatmap_rho_full.npy');layers,heads=a.shape
  z=np.sort(a,axis=1)[:,::-1].T
  assert np.all(np.diff(z,axis=0)<=0)
  im=ax.imshow(z,origin='upper',aspect='auto',interpolation='nearest',cmap='YlGnBu',vmin=0,vmax=1)
  ax.text(.01,.88,f'({chr(65+i)}) {title}',transform=ax.transAxes,fontsize=9,va='top',bbox={'boxstyle':'round,pad=0.2','facecolor':'white','alpha':.8,'edgecolor':'none'})
  ax.set_yticks([0,heads//2-1,heads-1],labels=[1,heads//2,heads]);ax.set_xticks(range(0,layers,5));ax.tick_params(labelsize=8)
  if i<5:ax.set_xticklabels([])
  else:ax.set_xlabel('Layer',fontsize=11)
 fig.text(.025,.525,'Head rank',va='center',rotation='vertical',fontsize=11)
 cb=fig.colorbar(im,cax=fig.add_axes([.86,.08,.015,.89]));cb.set_label('Spearman ρ',fontsize=10);cb.ax.tick_params(labelsize=8)
 save(fig,stem)

def stanceplot(model,stem):
 fig,axs=plt.subplots(2,3,figsize=(7.5,5));fig.subplots_adjust(left=.065,right=.96,bottom=.10,top=.94,wspace=.49,hspace=.38)
 boxes=[]
 for ax,(title,prefix) in zip(axs.flat,ISSUES):
  z=proportions(model,prefix)
  assert np.allclose(z.sum(axis=0),1)
  pd.DataFrame(z,index=range(1,6),columns=range(1,6)).to_csv(OUT/f'{stem}_{prefix}.csv')
  im=ax.imshow(z,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto',interpolation='nearest')
  for i in range(5):
   for j in range(5):ax.text(j,i,f'{z[i,j]:.3f}',ha='center',va='center',fontsize=6.6,color='white' if z[i,j]>.6 else 'black')
  ax.set(title=title,xlabel='Ground truth',ylabel='Predicted category',xticks=range(5),yticks=range(5),xticklabels=range(1,6),yticklabels=range(1,6))
  ax.title.set_fontsize(9);ax.xaxis.label.set_fontsize(8);ax.yaxis.label.set_fontsize(8);ax.tick_params(labelsize=7)
  cax=make_axes_locatable(ax).append_axes('right',size='4%',pad=.045);cb=fig.colorbar(im,cax=cax);cb.ax.tick_params(labelsize=7);boxes.append((ax,cax))
 fig.canvas.draw()
 for ax,cax in boxes:
  assert np.allclose([ax.get_position().y0,ax.get_position().y1],[cax.get_position().y0,cax.get_position().y1])
 save(fig,stem)

def proportions(model,prefix):
 config=CONFIG[model.lower()]
 rho=load(Path(config['rho_dir'])/f'{prefix}_heatmap_rho_full.npy')
 y=load(Path(config['labels_dir'])/f'{prefix}_labels.npy').astype(int)
 if model=='Gemma':indices=np.argsort(rho.ravel())[::-1][:20]
 else:
  top=pd.read_csv(Path(config['top20_dir'])/f'per_theme_top20_{prefix}.csv')
  indices=[int(r.layer)*rho.shape[1]+int(r['head']) for _,r in top.iterrows()]
 counts=np.zeros((5,5),int)
 assert len(indices)==20 and len(set(indices))==20
 assert np.isin(y,[1,2,3,4,5]).all()
 for index in indices:
  layer,head=divmod(int(index),rho.shape[1])
  directory=Path(config['predictions_dir'])
  if config.get('prediction_format','npy')=='layer_npz':
   path=directory/f'{prefix}_layer_{layer:02d}.npz'
   with np.load(path) as z:pred=z['preds'][:,head]
   manifest.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
  else:pred=load(directory/f'{prefix}_layer_{layer:02d}_preds.npy')[:,head]
  assert pred.shape==y.shape and np.isin(pred,[1,2,3,4,5]).all()
  np.add.at(counts,(pred.astype(int)-1,y-1),1)
 assert (counts.sum(axis=0)>0).all()
 return counts/counts.sum(axis=0,keepdims=True)

def meanplot(model,stem):
 fig,ax=plt.subplots(figsize=(5.2,3.5))
 if model=='Gemma':fig.subplots_adjust(left=.15,right=.97,bottom=.16,top=.96)
 else:fig.subplots_adjust(left=.12,right=.86,top=.95,bottom=.18)
 for i,(label,prefix) in enumerate(ISSUES):
  a=load(Path(CONFIG[model.lower()]['rho_dir'])/f'{prefix}_heatmap_rho_full.npy');layers=a.shape[0]
  if model=='Gemma':ax.plot(np.arange(layers),np.mean(a,axis=1),marker='o',markersize=2,lw=1,label=label)
  else:ax.plot(range(layers),a.mean(axis=1),'-',label=label,marker=['o','s','^','D','v','P'][i],markersize=4,alpha=.8,linewidth=1.5,color=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b'][i])
 if model=='Gemma':
  ax.set(xlabel='Layer',ylabel='Mean Spearman ρ',ylim=(.35,.95));ax.legend(fontsize=8);ax.grid(alpha=.2)
 else:
  ax.set_xlabel('Layer',fontsize=12);ax.set_ylabel('Mean Spearman ρ',fontsize=12);ax.legend(loc='lower right',fontsize=9);ax.grid(True,alpha=.3);ax.set_xlim(-.5,layers-.5);ax.set_ylim(.35,.9);ax.set_xticks(range(0,layers,5));ax.tick_params(labelsize=10)
 save(fig,stem)

def render_figures(args):
 global OUT, CONFIG, plt, Image, make_axes_locatable, manifest
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from mpl_toolkits.axes_grid1 import make_axes_locatable
 from PIL import Image
 manifest=[]
 OUT=args.output
 if OUT.exists():raise FileExistsError('Choose a fresh output directory')
 CONFIG=json.loads(args.inputs.read_text())
 for model in CONFIG:
  for key,value in CONFIG[model].items():
   if key.endswith('_dir'):
    path=Path(value).expanduser();CONFIG[model][key]=str(path if path.is_absolute() else args.inputs.resolve().parent/path)
 OUT.mkdir(parents=True)
 plt.rcdefaults();plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
 rankplot(Path(CONFIG['llama']['rho_dir']),'S3_Fig')
 stanceplot('Gemma','S4_Fig');stanceplot('Llama','S5_Fig');meanplot('Llama','S2_Fig')
 plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold','axes.unicode_minus':True})
 meanplot('Gemma','Fig2')
 # Gemma submission ranking has its own typography and title boxes.
 arrays=[load(Path(CONFIG['gemma']['rho_dir'])/f'{prefix}_heatmap_rho_full.npy') for _,prefix in ISSUES]
 fig,axs=plt.subplots(6,1,figsize=(5.2,6.5));fig.subplots_adjust(left=.12,right=.83,top=.97,bottom=.08,hspace=.35)
 for i,(ax,a,(label,_)) in enumerate(zip(axs,arrays,ISSUES)):
  z=np.sort(a,axis=1)[:,::-1].T;assert np.all(np.diff(z,axis=0)<=0)
  im=ax.imshow(z,origin='upper',aspect='auto',interpolation='nearest',cmap='YlGnBu',vmin=0,vmax=1);ax.text(.01,.88,f'({chr(65+i)}) {label}',transform=ax.transAxes,fontsize=9,va='top',bbox=dict(fc='white',alpha=.8,ec='none'));ax.set_yticks([0,7,15],[1,8,16]);ax.set_xticks(range(0,42,5))
  if i<5:ax.set_xticklabels([])
  else:ax.set_xlabel('Layer')
 fig.text(.025,.525,'Head rank',va='center',rotation='vertical',fontsize=10);cb=fig.colorbar(im,cax=fig.add_axes([.86,.08,.015,.89]));cb.set_label('Spearman ρ');save(fig,'Fig3')
 (OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2))

# 5. Public operations: no implicit training or API calls.
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='operation',required=True)
    p=sub.add_parser('train',help='Fit probes from locally supplied activations; resume matching checkpoints')
    p.add_argument('--config');p.add_argument('--model',required=True,choices=['gemma','llama'])
    p.add_argument('--issues',nargs='+',choices=list(THEMES));p.add_argument('--layers',nargs='+',type=int);p.add_argument('--heads',nargs='+',type=int)
    p=sub.add_parser('collect',help='Collect a complete model run into arrays')
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',choices=['gemma','llama'],required=True)
    p=sub.add_parser('select',help='Select common and issue-specific top-20 heads')
    p.add_argument('--scores',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',choices=['gemma','llama'],required=True)
    p=sub.add_parser('tables',help='Export S3–S9 Tables for both models')
    p.add_argument('--gemma-scores',type=Path,required=True);p.add_argument('--llama-scores',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('figures',help='Export Fig2/3 and S2–S5 Fig from saved results')
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.operation=='train':train(a)
    elif a.operation=='collect':collect(a.source,a.output,a.model)
    elif a.operation=='select':select(a.scores,a.output,a.model)
    elif a.operation=='tables':export_tables(a)
    else:render_figures(a)

if __name__=='__main__':main()
