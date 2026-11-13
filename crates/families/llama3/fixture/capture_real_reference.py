#!/usr/bin/env python3
"""Offline eager reference using mlx==0.32.4 and mlx-lm==0.31.2.

Loads exact frozen quantized weights with upstream mlx-lm, retaining codes and
casting floating parameters to f32 to match native execution. No compilation,
model downloads, conversion, runtime Python dependency or sampling. Captures
all vocabulary logits for final prefill and two fixed decode positions.
"""
import argparse, hashlib, importlib.metadata, json
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_model
from mlx_lm.models.cache import make_prompt_cache
from tokenizers import Tokenizer

p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--profile',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
profile=next(x for x in json.loads(a.corpus.read_text())['profiles'] if x['id']==a.profile)
for f in profile['artifact']['files']:
    h=hashlib.sha256()
    with (a.artifact/f['path']).open('rb') as file:
        for chunk in iter(lambda:file.read(8*1024*1024),b''): h.update(chunk)
    assert h.hexdigest()==f['sha256'],f['path']
mx.disable_compile()
model,config=load_model(a.artifact,lazy=False)
model.apply(lambda x: x.astype(mx.float32) if mx.issubdtype(x.dtype,mx.floating) else x)
mx.eval(model.parameters())
tokenizer=Tokenizer.from_file(str(a.artifact/'tokenizer.json'))
texts=['<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is 2 + 2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
       '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nAnswer concisely.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nExplain why the sky is blue.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n']
result={'profile':a.profile,'artifact_sha256':profile['artifact']['sha256'],'versions':{n:importlib.metadata.version(n) for n in ['mlx','mlx-lm','numpy','tokenizers']},'compile_enabled':False,'dtype':'float32 floating parameters; original packed codes','cases':[]}
for text in texts:
    tokens=tokenizer.encode(text,add_special_tokens=False).ids
    cache=make_prompt_cache(model)
    logits=model(mx.array([tokens]),cache=cache)[:,-1,:];mx.eval(logits)
    rows=[np.array(logits[0]).tolist()]
    for token in [19,13]:
        logits=model(mx.array([[token]]),cache=cache)[:,-1,:];mx.eval(logits);rows.append(np.array(logits[0]).tolist())
    result['cases'].append({'tokens':tokens,'decode_tokens':[19,13],'logits':rows})
result['peak_memory_bytes']=mx.get_peak_memory()
a.output.write_text(json.dumps(result,allow_nan=False))
print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
