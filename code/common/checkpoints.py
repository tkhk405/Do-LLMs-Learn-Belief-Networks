"""Input hashing and guarded, atomic layer checkpoints."""
from pathlib import Path
import json,hashlib
import numpy as np
from .constants import THEMES

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def activation_files(paths,model,layer):
    result={}
    for issue,c in THEMES.items():
        base=paths[model+'_activations_dir'];p=base/f'{c["vec"]}_layer_{layer:02d}.npy';candidates=[p]
        if model=='gemma' and c['subdir']:candidates.append(base/c['subdir']/p.name)
        found=[p for p in candidates if p.exists()]
        if len(found)!=1:raise ValueError('Missing/ambiguous activation for '+issue)
        result[issue]=found[0]
    return result

def start_run(output,signature):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);p=output/'run.json'
    if p.exists():
        if json.loads(p.read_text())!=signature:raise ValueError('Changed inputs/settings: use a fresh output directory')
    elif any(output.iterdir()):raise ValueError('Existing output has no run manifest')
    else:p.write_text(json.dumps(signature,indent=2))

def atomic_cache(path,**values):
    tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**values);tmp.replace(path)
