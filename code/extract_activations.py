"""Extract o_proj input activations; smoke mode by default, no API generation."""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
from pathlib import Path
import argparse,json,hashlib,importlib.metadata
import numpy as np
from common.configuration import ROOT,load_paths
from common.data_loading import load_statements
from common.constants import ISSUES,THEMES

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def save_json(path,value):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def main():
    global model,tokenizer,NUM_HEADS,HEAD_DIM,CONCAT_DIM,MAX_LENGTH,SAVE_DTYPE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['gemma','llama'],required=True)
    p.add_argument('--config');p.add_argument('--mode',choices=['smoke','full'],default='smoke')
    p.add_argument('--revision',help='Hugging Face commit; required for Gemma because its original commit was not recorded')
    p.add_argument('--plan-only',action='store_true',help='Validate inputs and print extraction settings without loading a model')
    args=p.parse_args();paths=load_paths(args.config)
    settings=json.loads((ROOT/'config/extraction.json').read_text());shape=settings[args.model]
    model_id=json.loads((ROOT/'config/models.json').read_text())[args.model]
    revision=args.revision or shape['model_revision']
    if not revision:raise ValueError('Specify an explicit --revision for Gemma; do not silently use a moving model version')
    frames,sources=load_statements(paths['synthetic_statements_dir'])
    layers=list(range(2 if args.mode=='smoke' else shape['layers']));n=16 if args.mode=='smoke' else 4320
    specification={'model':model_id,'revision':revision,'mode':args.mode,'layers':layers,'rows_per_issue':n,'heads':shape['heads'],'head_dim':shape['head_dim'],'batch_size':8,'max_length':4096,'padding_side':'left','input_sha256':{k:digest(v) for k,v in sources.items()},'script_sha256':digest(Path(__file__)),'kernel_sha256':digest(Path(__file__))}
    if args.plan_only:
        print(json.dumps(specification,indent=2));return
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; no CPU fallback')
    if args.model=='llama' and 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('The analyzed Llama protocol requires an A100 GPU')
    out=paths['output_dir']/'activations'/args.model/args.mode
    manifest=out/'extraction_manifest.json'
    if out.exists() and not manifest.exists():raise ValueError('Output exists without manifest; choose a new output_dir')
    previous=json.loads(manifest.read_text()) if manifest.exists() else None
    versions={k:importlib.metadata.version(k) for k in ['torch','transformers','tokenizers','accelerate','bitsandbytes','numpy']}
    specification['versions']=versions
    specification['gpu']=torch.cuda.get_device_name(0)
    if previous and previous['specification']!=specification:raise ValueError('Run settings changed; choose a new output_dir')
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision);tokenizer.padding_side='left'
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,quantization_config=quant,device_map='auto',trust_remote_code=True)
    if model.training:raise ValueError('Model must be in evaluation mode')
    actual=(model.config.num_hidden_layers,model.config.num_attention_heads,getattr(model.config,'head_dim',model.config.hidden_size//model.config.num_attention_heads))
    if actual!=(shape['layers'],shape['heads'],shape['head_dim']):raise ValueError('Unexpected model architecture')
    effective={'quantization':quant.to_dict(),'attention':model.config._attn_implementation,'use_cache':model.config.use_cache,'tokenizer_sha256':hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()}
    if previous and previous['effective']!=effective:raise ValueError('Effective model settings changed')
    state=previous or {'specification':specification,'effective':effective,'outputs':{}}
    out.mkdir(parents=True,exist_ok=True);save_json(manifest,state)
    model=model;tokenizer=tokenizer;NUM_HEADS=shape['heads'];HEAD_DIM=shape['head_dim'];CONCAT_DIM=shape['heads']*shape['head_dim'];MAX_LENGTH=4096;SAVE_DTYPE=np.float16
    for issue in ISSUES:
        texts=frames[issue].Generated_Text.iloc[:n].tolist()
        for layer in layers:
            name=f'{THEMES[issue]["vec"]}_layer_{layer:02d}.npy';path=out/name
            if name in state['outputs']:
                if not path.exists() or digest(path)!=state['outputs'][name]:raise ValueError('Saved activation hash mismatch')
                continue
            result=extract_layer(texts,layer,8)
            if not np.isfinite(result).all():raise ValueError('Nonfinite activation')
            tmp=path.with_suffix('.tmp.npy');np.save(tmp,result);tmp.replace(path)
            state['outputs'][name]=digest(path);save_json(manifest,state)
            print(f'{len(state["outputs"])}/{6*len(layers)} layers complete',flush=True)
    print('Extraction complete')

def last_nonpadding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    import torch
    if attention_mask.ndim != 2 or not torch.all(attention_mask.sum(dim=1) > 0):
        raise ValueError('Every sample must contain at least one non-padding token.')
    seq_len = attention_mask.shape[1]
    return seq_len - 1 - attention_mask.flip(dims=[1]).long().argmax(dim=1)

def extract_layer(texts: list[str], layer_idx: int, batch_size: int) -> np.ndarray:
    import torch
    from tqdm.auto import tqdm
    state = {'attention_mask': None, 'chunks': [], 'calls': 0}

    def o_proj_hook(module, inputs, output):
        hidden_states = inputs[0]
        mask = state['attention_mask']
        if mask is None:
            raise RuntimeError('attention_mask was not set before the forward pass')
        if hidden_states.shape[-1] != CONCAT_DIM:
            raise ValueError(f'Layer {layer_idx}: expected last dimension {CONCAT_DIM}, got {hidden_states.shape[-1]}')
        mask = mask.to(hidden_states.device)
        last_idx = last_nonpadding_indices(mask)
        batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        if not torch.all(mask[batch_idx, last_idx] == 1):
            raise RuntimeError('A padding position was selected')
        if not torch.all(last_idx == hidden_states.shape[1] - 1):
            raise RuntimeError('Left padding required for Gemma-matched readout')
        last_token = hidden_states[:, -1, :]
        per_head = last_token.reshape(hidden_states.shape[0], NUM_HEADS, HEAD_DIM)
        state['chunks'].append(per_head.detach().cpu().numpy())
        state['calls'] += 1
    layer_module = model.model.layers[layer_idx].self_attn.o_proj
    handle = layer_module.register_forward_hook(o_proj_hook)
    try:
        for start in tqdm(range(0, len(texts), batch_size), desc=f'layer {layer_idx:02d} batches', leave=False):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH)
            encoded = encoded.to(model.device)
            state['attention_mask'] = encoded['attention_mask']
            with torch.no_grad():
                model(**encoded)
            state['attention_mask'] = None
            del encoded, batch
            torch.cuda.empty_cache()
    finally:
        handle.remove()
    expected_calls = (len(texts) + batch_size - 1) // batch_size
    if state['calls'] != expected_calls:
        raise RuntimeError(f'Layer {layer_idx}: hook calls={state['calls']}, expected={expected_calls}')
    result = np.concatenate(state['chunks'], axis=0).astype(SAVE_DTYPE, copy=False)
    expected_shape = (len(texts), NUM_HEADS, HEAD_DIM)
    if result.shape != expected_shape:
        raise ValueError(f'Layer {layer_idx}: expected {expected_shape}, got {result.shape}')
    return result

if __name__=='__main__':main()
