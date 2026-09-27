"""Compute directional transfer using common top-20 heads; does not refit probes."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,json,importlib.metadata
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from common.constants import ISSUES,THEMES
from common.configuration import load_paths
from common.data_loading import load_statements
from common.checkpoints import sha,activation_files,start_run,atomic_cache

def calculate_layer(model,files,frames,coefficients,layer,heads):
    from sklearn.preprocessing import StandardScaler
    values=np.full((len(heads),6,6),np.nan)
    for ti,target in enumerate(ISSUES):
        raw=np.load(files[target],mmap_mode='r');labels=frames[target].Stance_Value.to_numpy(int)
        if len(raw)!=len(labels):raise ValueError('Activation row count mismatch')
        for h in sorted(heads):
            x=np.asarray(raw[:,h,:],dtype=np.float64) if model=='llama' else np.asarray(raw[:,h,:])
            if not np.isfinite(x).all():raise ValueError('Nonfinite activation')
            scaled=StandardScaler().fit_transform(x)
            for si,source in enumerate(ISSUES):values[heads.index(h),si,ti]=spearmanr(labels,scaled@coefficients[source][layer,h]).statistic
    if not np.isfinite(values).all():raise ValueError('Invalid transfer output')
    return values

def aggregate(rows,output):
    long=pd.DataFrame(rows);matrix=long.groupby(['source','target'],sort=False).rho.mean().unstack().loc[ISSUES,ISSUES]
    matrix.to_csv(output/'transfer_directional.csv');((matrix+matrix.T)/2).to_csv(output/'transfer_symmetric.csv');long.to_csv(output/'transfer_per_head.csv',index=False)
    return matrix

def run(a):
    paths=load_paths(a.config);frames,sources=load_statements(paths['synthetic_statements_dir'])
    selected=pd.read_csv(a.heads/'common_top20_heads.csv');heads=list(zip(selected.layer.astype(int),selected['head'].astype(int)))
    if len(heads)!=20 or len(set(heads))!=20:raise ValueError('Exactly 20 distinct transfer heads required')
    coefficients={};signature={'model':a.model,'script':sha(Path(__file__)),'common':sha(a.heads/'common_top20_heads.csv'),'inputs':{},'versions':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn']}}
    signature['shared']={n:sha(Path(__file__).parent/'common'/n) for n in ['constants.py','configuration.py','data_loading.py','checkpoints.py']}
    for issue,c in THEMES.items():
        p=a.scores/f'{c["prefix"]}_coef_full.npy';coefficients[issue]=np.load(p,mmap_mode='r');signature['inputs'][p.name]=sha(p);signature['inputs'][issue]=sha(sources[issue])
    start_run(a.output,signature);rows=[];layers=sorted({l for l,h in heads})
    for layer in layers:
        selected_heads=[h for l,h in heads if l==layer];files=activation_files(paths,a.model,layer);hashes=json.dumps({k:sha(v) for k,v in files.items()},sort_keys=True);cache=a.output/f'layer_{layer:02d}.npz'
        if cache.exists():
            with np.load(cache) as z:
                if str(z['inputs'].item())!=hashes:raise ValueError('Activation hash changed')
                values=z['transfer']
        else:
            values=calculate_layer(a.model,files,frames,coefficients,layer,selected_heads);atomic_cache(cache,inputs=hashes,transfer=values)
        if values.shape!=(len(selected_heads),6,6) or not np.isfinite(values).all():raise ValueError('Invalid transfer cache')
        for k,h in enumerate(selected_heads):
            for si,s in enumerate(ISSUES):
                for ti,t in enumerate(ISSUES):rows.append({'source':s,'target':t,'layer':layer,'head':h,'rho':values[k,si,ti]})
        print(f'{a.model}: layer {layer} complete ({layers.index(layer)+1}/{len(layers)})',flush=True)
    aggregate(rows,a.output)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['gemma','llama'],required=True);p.add_argument('--config');p.add_argument('--scores',type=Path,required=True);p.add_argument('--heads',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
if __name__=='__main__':main()
