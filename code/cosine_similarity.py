"""Compute sigma-standardized activation-contrast cosine on the union of top heads.
Directions are class-mean activation differences, not fitted probe coefficients.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,json,importlib.metadata
import numpy as np,pandas as pd
from common.constants import ISSUES
from common.configuration import load_paths
from common.data_loading import load_statements
from common.checkpoints import sha,activation_files,start_run,atomic_cache

def cosine(a,b):
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    if na<1e-10 or nb<1e-10:return np.nan
    return np.dot(a,b)/(na*nb)

def calculate_layer(model,files,frames,heads):
    directions=[]
    for issue in ISSUES:
        raw=np.load(files[issue],mmap_mode='r');labels=frames[issue].Stance_Value.to_numpy(int)
        if len(raw)!=len(labels):raise ValueError('Activation row count mismatch')
        ds=[]
        for h in sorted(heads):
            x=np.asarray(raw[:,h,:],dtype=np.float64) if model=='llama' else np.asarray(raw[:,h,:])
            if not np.isfinite(x).all():raise ValueError('Nonfinite activation')
            hi=x[np.isin(labels,[4,5])];lo=x[np.isin(labels,[1,2])]
            delta=hi.mean(axis=0)-lo.mean(axis=0);sigma=np.concatenate([hi,lo],axis=0).std(axis=0,ddof=0)
            ds.append((h,delta/np.where(sigma>1e-10,sigma,1.0)))
        directions.append(np.stack([dict(ds)[h] for h in heads]))
    result=np.stack(directions)
    if not np.isfinite(result).all():raise ValueError('Invalid direction output')
    return result

def aggregate(directions,heads,output):
    matrix=np.eye(6);rows=[]
    for i in range(6):
        for j in range(i+1,6):
            values=[]
            for l,h in heads:
                v=cosine(directions[l,h][i],directions[l,h][j]);values.append(v);rows.append({'issue_1':ISSUES[i],'issue_2':ISSUES[j],'layer':l,'head':h,'cosine':v})
            matrix[i,j]=matrix[j,i]=np.mean([v for v in values if not np.isnan(v)])
    pd.DataFrame(matrix,index=ISSUES,columns=ISSUES).to_csv(output/'cosine_sigma_own.csv');pd.DataFrame(rows).to_csv(output/'cosine_per_head.csv',index=False)
    return matrix

def run(a):
    paths=load_paths(a.config);frames,sources=load_statements(paths['synthetic_statements_dir']);selected=pd.read_csv(a.heads/'union_heads.csv');heads=sorted(zip(selected.layer.astype(int),selected['head'].astype(int)))
    if not heads or len(set(heads))!=len(heads):raise ValueError('Expected unique union heads')
    signature={'model':a.model,'script':sha(Path(__file__)),'union':sha(a.heads/'union_heads.csv'),'inputs':{k:sha(v) for k,v in sources.items()},'versions':{k:importlib.metadata.version(k) for k in ['numpy','pandas']}}
    signature['shared']={n:sha(Path(__file__).parent/'common'/n) for n in ['constants.py','configuration.py','data_loading.py','checkpoints.py']}
    start_run(a.output,signature);directions={};layers=sorted({l for l,h in heads})
    for layer in layers:
        selected_heads=[h for l,h in heads if l==layer];files=activation_files(paths,a.model,layer);hashes=json.dumps({k:sha(v) for k,v in files.items()},sort_keys=True);cache=a.output/f'layer_{layer:02d}.npz'
        if cache.exists():
            with np.load(cache) as z:
                if str(z['inputs'].item())!=hashes:raise ValueError('Activation hash changed')
                values=z['directions']
        else:values=calculate_layer(a.model,files,frames,selected_heads);atomic_cache(cache,inputs=hashes,directions=values)
        if values.shape[:2]!=(6,len(selected_heads)) or values.ndim!=3 or not np.isfinite(values).all():raise ValueError('Invalid direction cache')
        for k,h in enumerate(selected_heads):directions[layer,h]=values[:,k,:]
        print(f'{a.model}: layer {layer} complete ({layers.index(layer)+1}/{len(layers)})',flush=True)
    aggregate(directions,heads,a.output)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['gemma','llama'],required=True);p.add_argument('--config');p.add_argument('--heads',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
if __name__=='__main__':main()
