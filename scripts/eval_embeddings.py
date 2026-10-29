"""Portable sequential BM25/dense/RRF evaluation on fixed TechQA chunks."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import resource
import time
from pathlib import Path
from scripts.audit_techqa import ROOT, RAW, sha256, stats
from scripts.baseline import BM25, retrieval_scores
from scripts.chunking import answer_coverage
from scripts.retrieval import rrf, pack_context, summarize


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(temp,path)


def format_input(spec, text, query=False):
    return spec['query_prefix' if query else 'document_prefix']+text


def structured_chunk_texts(chunks):
    texts=[]
    for chunk in chunks:
        text=chunk.get('search_text')
        if not isinstance(text,str) or not text.strip():
            raise ValueError('Structured chunks require non-empty search_text')
        texts.append(text)
    return texts


def configure_text_input(model, model_name):
    # ST 6.1 automatically converts text to messages when message is advertised.
    # Gemma's pinned processor has no chat template; this arm uses only text.
    if model_name == 'gemma2':
        module=model[0]
        module.modality_config={'text':dict(module.modality_config['text'])}


def rank_top(scores, chunk_keys, limit=100):
    import numpy as np
    values=np.asarray(scores)
    if values.shape!=(len(chunk_keys),) or not np.isfinite(values).all():
        raise ValueError('Invalid search scores')
    # lexsort gives a deterministic chunk-ID tie break, including ties at the cutoff.
    return np.lexsort((np.asarray(chunk_keys),-values))[:limit].tolist()


def score_run(questions, candidates, chunks, docs, context_tokenizer, structured=False):
    rows=[]
    for q in questions:
        ranked=candidates[q['QUESTION_ID']]['ranked']
        documents=list(dict.fromkeys(chunks[i]['document_id'] for i in ranked))
        began=time.perf_counter()
        context=pack_context(ranked,chunks,docs,context_tokenizer,2048)
        context_ms=(time.perf_counter()-began)*1000
        known=q['ANSWERABLE']=='Y'
        if known and not structured:
            intervals=[p for p in context['passages'] if p['document_id']==q['DOCUMENT']]
            coverage=answer_coverage(int(q['START_OFFSET']),int(q['END_OFFSET']),intervals)['union_fraction']
        else:
            coverage=None
        context.update(answer_fraction=coverage,complete=(coverage==1 if known and not structured else None))
        if structured:
            context['coverage_status']='unavailable: new HTML coordinates; original answer mapping not scored'
        rows.append({'question_id':q['QUESTION_ID'],'scores':retrieval_scores(q['DOCUMENT'] if known else None,documents),
                     'ranked_document_ids':documents,'ranked_chunk_ids':[chunks[i]['chunk_id'] for i in ranked],
                     'search_ms':candidates[q['QUESTION_ID']]['search_ms'],
                     'query_encode_ms':candidates[q['QUESTION_ID']].get('query_encode_ms',0),
                     'context_pack_ms':context_ms,'context':context})
    if structured:
        # Keep all answerable questions in retrieval denominators without treating
        # unmappable original-text offsets as zero coverage.
        summary_rows=[dict(r,context=dict(r['context'],answer_fraction=0,complete=False)) for r in rows]
        result=summarize(summary_rows)
        result['context']['mean_answer_fraction']=None
        result['context']['full_answer_fraction']=None
        result['context']['coverage_status']='unavailable: new HTML coordinates; original answer mapping not scored'
    else:
        result=summarize(rows)
    result['search_ms']=stats([r['search_ms'] for r in rows])
    result['query_encode_ms']=stats([r['query_encode_ms'] for r in rows])
    result['context_pack_ms']=stats([r['context_pack_ms'] for r in rows])
    return rows,result


def download_model(spec):
    from huggingface_hub import snapshot_download
    started=time.perf_counter()
    directory=Path(snapshot_download(spec['model_id'],revision=spec['revision'],
                                    allow_patterns=[f['rfilename'] for f in spec['files']],token=False))
    verified={}
    for asset in spec['files']:
        path=directory/asset['rfilename']
        if path.stat().st_size!=asset['size']:
            raise ValueError('Model asset size mismatch: '+asset['rfilename'])
        digest=sha256(path)
        if 'lfs' in asset:
            if digest!=asset['lfs']['sha256']:
                raise ValueError('Model checkpoint/tokenizer hash mismatch')
        else:
            raw=path.read_bytes()
            git_hash=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            if git_hash!=asset['blobId']:
                raise ValueError('Model configuration hash mismatch')
        verified[asset['rfilename']]=digest
    return directory,verified,time.perf_counter()-started


def encode_cache(model, texts, lengths, directory, key, dimension, batch_size, token_budget, order=None):
    import numpy as np
    import torch
    # A length-sorted order keeps batch members at a similar length, so padding is
    # shared instead of wasted. Padding is masked, so rows are unchanged within
    # floating-point tolerance; only batch composition differs.
    order=list(range(len(texts))) if order is None else list(order)
    if len(order)!=len(texts) or sorted(order)!=list(range(len(texts))):
        raise ValueError('Batching order must be a permutation of the inputs')
    if order!=list(range(len(texts))):
        lengths=[lengths[i] for i in order]
    directory.mkdir(parents=True,exist_ok=True)
    state_path=directory/'state.json'
    vector_path=directory/'vectors.npy'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state['key']!=key or state['total']!=len(texts) or state['dimension']!=dimension:
            raise ValueError('Incompatible embedding checkpoint')
        vectors=np.load(vector_path,mmap_mode='r+')
        if vectors.shape!=(len(texts),dimension) or vectors.dtype!=np.float32:
            raise ValueError('Invalid cached vector array')
        done=state['done']
        if not 0<=done<=len(texts):
            raise ValueError('Invalid checkpoint row count')
        for start in range(0,done,4096):
            previous=vectors[order[start:min(start+4096,done)]]
            if not np.isfinite(previous).all() or not np.allclose(np.linalg.norm(previous,axis=1),1,atol=.002):
                raise ValueError('Corrupt completed vectors')
    else:
        vectors=np.lib.format.open_memmap(vector_path,mode='w+',dtype=np.float32,shape=(len(texts),dimension))
        state={'key':key,'total':len(texts),'dimension':dimension,'done':0,'encode_seconds':0,
               'length_sorted':order!=list(range(len(texts))),
               'peak_cuda_allocated':0,'peak_cuda_reserved':0}
        save_json(state_path,state)
    start=state['done']
    while start<len(texts):
        end=start
        longest=0
        while end<len(texts) and end-start<batch_size:
            proposed=max(longest,lengths[end])
            if end>start and (end-start+1)*proposed>token_budget:
                break
            longest=proposed;end+=1
        try:
            torch.cuda.synchronize()
            began=time.perf_counter()
            with torch.inference_mode():
                encoded=model.encode([texts[i] for i in order[start:end]],batch_size=end-start,prompt='',normalize_embeddings=True,
                                     convert_to_numpy=True,show_progress_bar=False)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            if batch_size==1:
                raise
            batch_size=max(1,batch_size//2)
            torch.cuda.empty_cache()
            print('CUDA OOM: reducing batch cap to',batch_size,flush=True)
            continue
        if encoded.shape!=(end-start,dimension) or not np.isfinite(encoded).all():
            raise ValueError('Invalid embedding output')
        if not np.allclose(np.linalg.norm(encoded,axis=1),1,atol=.002):
            raise ValueError('Embeddings are not normalized')
        vectors[order[start:end]]=encoded.astype(np.float32)
        vectors.flush()
        state['encode_seconds']+=time.perf_counter()-began
        state.update(done=end,effective_batch_size=batch_size,
                     peak_cuda_allocated=max(state['peak_cuda_allocated'],torch.cuda.max_memory_allocated()),
                     peak_cuda_reserved=max(state['peak_cuda_reserved'],torch.cuda.max_memory_reserved()))
        save_json(state_path,state)
        if start==0 or end==len(texts) or end//512>start//512:
            rate=end/state['encode_seconds'] if state['encode_seconds'] else 0
            eta=(len(texts)-end)/rate/60 if rate else float('inf')
            print(f'Embedded {end}/{len(texts)} rows | {rate:.1f} rows/s | ETA {eta:.1f} min (encoding only)',flush=True)
        start=end
    return vectors,state


def main(args):
    import numpy as np
    from tokenizers import Tokenizer
    if not args.tag.replace('_','').isalnum():
        raise ValueError('Use a simple alphanumeric/underscore run tag')
    variant_root=args.chunk_root if args.chunk_root.is_absolute() else ROOT/args.chunk_root
    source_config=json.loads((variant_root/'config.json').read_text())
    chunk_path=variant_root/args.variant/'chunks.jsonl'
    chunk_metrics=json.loads((variant_root/args.variant/'metrics.json').read_text())
    if sha256(chunk_path)!=chunk_metrics['chunks_sha256']:
        raise ValueError('Chunk file changed')
    structured=source_config.get('representation')=='structured_html_v1'
    if args.variant=='structured_512' and not structured:
        raise ValueError('structured_512 requires representation=structured_html_v1')
    if structured and args.variant!='structured_512':
        raise ValueError('structured_html_v1 requires --variant structured_512')
    if structured and source_config.get('document_count')!=28482 and not args.smoke:
        raise ValueError('Full structured evaluation requires all 28,482 source documents; use --smoke for partial artifacts')
    base=RAW/'TechQA/training_and_dev'
    for name in ['training_Q_A.json','training_dev_technotes.json']:
        if sha256(base/name)!=source_config['source_hashes'][name]:
            raise ValueError('Original input changed: '+name)
    split_path=ROOT/'data/splits/techqa_v1.json'
    if sha256(split_path)!=source_config['split_sha256']:
        raise ValueError('Question split changed')
    split=json.loads(split_path.read_text())
    questions=[q for q in json.loads((base/'training_Q_A.json').read_text()) if q['QUESTION_ID'] in split['tuning_ids']]
    documents_path=variant_root/'documents.json' if structured else base/'training_dev_technotes.json'
    docs=json.loads(documents_path.read_text())
    if structured and len(docs)!=source_config.get('document_count'):
        raise ValueError('Structured document count does not match its source config')
    chunks=[json.loads(line) for line in chunk_path.open()]
    if len(chunks)!=chunk_metrics['chunk_count'] or len(questions)!=120:
        raise ValueError('Unexpected corpus/query membership')
    if args.smoke:
        chunks=chunks[:128];questions=questions[:3]
    keys=[c['chunk_id'] for c in chunks]
    if structured:
        bodies=structured_chunk_texts(chunks)
    else:
        bodies=[docs[c['document_id']]['text'][c['start']:c['end']] for c in chunks]
    if len(set(keys))!=len(keys):
        raise ValueError('Duplicate chunk IDs')
    context_dir=ROOT/'data/tokenizers/bge_wordpiece'
    context_source=json.loads((context_dir/'source.json').read_text())
    if sha256(context_dir/'tokenizer.json')!=context_source['tokenizer_sha256']:
        raise ValueError('Context tokenizer changed')
    context_tokenizer=Tokenizer.from_file(str(context_dir/'tokenizer.json'))
    context_tokenizer.no_truncation();context_tokenizer.no_padding()
    config={'variant':args.variant,'model':args.model,'smoke':args.smoke,'chunk_sha256':sha256(chunk_path),
            'split_sha256':sha256(split_path),'source_hashes':source_config['source_hashes'],
            'track':'B: all 28,482 supplied documents','candidate_depth':100,'fusion':'chunk-level equal-weight RRF, one-based ranks, k=60; then deduplicate document IDs',
            'context':'2048 reference tokens including source headers; at most 5 documents/2 chunks per document; merge overlap; skip chunks exceeding remaining budget',
            'context_tokenizer':context_source,'representation':source_config.get('representation','original prepared text'),
            'input':('structured chunk search_text; model-native task wrappers' if structured else 'same body-only text; model-native task wrappers; no source titles/product prefixes'),
            'query_ids':[q['QUESTION_ID'] for q in questions],
            'code_sha256':{p.name:sha256(p) for p in [Path(__file__),ROOT/'scripts/retrieval.py',ROOT/'scripts/baseline.py',ROOT/'scripts/chunking.py']},
            'software':{p:importlib.metadata.version(p) for p in ['numpy','tokenizers']},'platform':platform.platform(),'python':platform.python_version()}
    if structured:
        config['documents_sha256']=sha256(documents_path)
        config['input_policy']={'representation':'structured_html_v1','chunks_sha256':config['chunk_sha256'],
                                'documents_sha256':config['documents_sha256']}
    output=ROOT/'data/retrieval_runs'/f'{args.model}_{args.variant}_{args.tag}{"_smoke" if args.smoke else ""}'
    lexical_path=ROOT/'data/retrieval_runs'/f'bm25_{args.variant}_{args.tag}'
    if args.model=='bm25':
        started=time.perf_counter()
        index=BM25({str(i):body for i,body in enumerate(bodies)})
        build_seconds=time.perf_counter()-started
        candidates={}
        for q in questions:
            started=time.perf_counter()
            scores=index.scores(q['QUESTION_TITLE']+'\n'+q['QUESTION_TEXT'])
            ranked=sorted((int(i) for i in scores),key=lambda i:(-scores[str(i)],keys[i]))[:100]
            candidates[q['QUESTION_ID']]={'ranked':ranked,'search_ms':(time.perf_counter()-started)*1000}
        rows,metrics=score_run(questions,candidates,chunks,docs,context_tokenizer,structured)
        metrics['index_seconds']=build_seconds
        config['run_key']=fingerprint(config)
        if output.exists():
            raise FileExistsError('Preserve existing BM25 run')
        output.mkdir(parents=True)
        save_json(output/'config.json',config);save_json(output/'metrics.json',metrics)
        save_json(output/'predictions.json',rows);save_json(output/'candidates.json',candidates)
        print(json.dumps(metrics,indent=2),flush=True)
        return
    if args.smoke is False:
        lexical_config=json.loads((lexical_path/'config.json').read_text())
        if lexical_config['chunk_sha256']!=config['chunk_sha256'] or lexical_config['query_ids']!=config['query_ids']:
            raise ValueError('BM25 baseline is not compatible with this run')
        if lexical_config['context']!=config['context'] or lexical_config['context_tokenizer']!=config['context_tokenizer']:
            raise ValueError('BM25 context policy differs from this comparison')
        if lexical_config.get('representation','original prepared text')!=config['representation'] or lexical_config.get('input_policy','plain_text_v1')!=config.get('input_policy','plain_text_v1'):
            raise ValueError('BM25 representation/input policy differs from this comparison')
        lexical=json.loads((lexical_path/'candidates.json').read_text())
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU evaluation requires CUDA; refusing silent CPU fallback')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_grad_enabled(False)
    config.update(device=torch.cuda.get_device_name(0),cuda=torch.version.cuda,dtype=args.dtype,tf32=False,
                  attn_implementation=args.attn,batch_size=args.batch_size,padded_token_budget=args.token_budget,
                  length_sorted_batches=args.sort_batches)
    config['software'].update({p:importlib.metadata.version(p) for p in ['torch','transformers','sentence-transformers','huggingface-hub']})
    spec=json.loads((ROOT/'models-embeddings.json').read_text())['models'][args.model]
    config['model_spec']=spec
    directory,assets,download_seconds=download_model(spec)
    config['verified_assets']=assets
    from sentence_transformers import SentenceTransformer
    began=time.perf_counter()
    model_dtype={'float32':torch.float32,'float16':torch.float16}[args.dtype]
    model=SentenceTransformer(str(directory),device='cuda',trust_remote_code=False,local_files_only=True,config_kwargs=spec['config_kwargs'],
                              model_kwargs={'dtype':model_dtype,'attn_implementation':args.attn})
    if args.dtype=='float32':
        model.float()
    model.eval();model.default_prompt_name=None;model.max_seq_length=spec['max_length']
    configure_text_input(model,args.model)
    config.setdefault('input_policy','plain_text_v1')
    config['model_load_seconds']=time.perf_counter()-began
    config['download_verify_seconds']=download_seconds
    config['actual_parameters']=sum(p.numel() for p in model.parameters())
    if model.get_sentence_embedding_dimension()!=spec['dimension']:
        raise ValueError('Unexpected embedding dimension')
    from transformers import AutoTokenizer
    native_tokenizer=AutoTokenizer.from_pretrained(str(directory),trust_remote_code=False)
    texts=[format_input(spec,b) for b in bodies]+[format_input(spec,q['QUESTION_TITLE']+'\n'+q['QUESTION_TEXT'],True) for q in questions]
    lengths=[]
    for start in range(0,len(texts),256):
        lengths.extend(native_tokenizer(texts[start:start+256],truncation=False,add_special_tokens=True,return_length=True)['length'])
    if max(lengths)>spec['max_length']:
        raise ValueError('Formatted inputs exceed native model context; re-chunk, do not truncate')
    for i in [0,len(bodies)-1,len(texts)-1]:
        actual=int(model.tokenize([texts[i]])['attention_mask'].sum().item())
        if actual!=lengths[i]:
            raise ValueError('SentenceTransformer/native tokenizer input lengths differ')
    config['max_document_tokens']=max(lengths[:len(bodies)])
    config['max_query_tokens']=max(lengths[len(bodies):])
    encoder_code=hashlib.sha256((inspect.getsource(encode_cache)+inspect.getsource(format_input)+inspect.getsource(configure_text_input)).encode()).hexdigest()
    cache_key=fingerprint({'texts':fingerprint(texts),'assets':assets,'environment':config['software'],
                           'device':config['device'],'dtype':args.dtype,'tf32':False,'encoder':encoder_code,
                           'attn_implementation':args.attn,'batch_size':args.batch_size,
                           'padded_token_budget':args.token_budget,'length_sorted_batches':args.sort_batches})
    config['cache_key']=cache_key
    if os.environ.get('TECHQA_RESUME_CACHE_KEY',cache_key)!=cache_key:
        raise ValueError('Resume cache identity changed; refusing to restart encoding silently')
    print(f'Input preflight passed: {len(bodies):,} chunks, {len(questions)} queries; cache {cache_key}',flush=True)
    # Timing observations are excluded from identity so an interrupted run can resume.
    config['run_key']=fingerprint({k:v for k,v in config.items() if not k.endswith('_seconds')})
    if output.exists():
        previous=json.loads((output/'config.json').read_text())
        if previous['run_key']!=config['run_key']:
            raise ValueError('Existing result uses different configuration; choose a new tag')
        if (output/'metrics.json').exists():
            print('Complete compatible result exists; skipping',output,flush=True)
            return
    else:
        output.mkdir(parents=True)
    save_json(output/'config.json',config)
    order=sorted(range(len(texts)),key=lambda i:(lengths[i],i)) if args.sort_batches else None
    vectors,state=encode_cache(model,texts,lengths,ROOT/'data/embedding_cache'/cache_key,cache_key,spec['dimension'],
                               args.batch_size,args.token_budget,order)
    if args.smoke:
        # Catch precision/batch differences before an expensive full corpus run.
        for i in range(len(bodies),len(texts)):
            with torch.inference_mode():
                single=model.encode([texts[i]],batch_size=1,prompt='',normalize_embeddings=True,
                                    convert_to_numpy=True,show_progress_bar=False)[0]
            if not np.allclose(single,vectors[i],atol=.001):
                raise ValueError('Smoke: single-query and cached-batch embeddings differ')
        save_json(output/'metrics.json',{'smoke':'passed','rows':len(texts),'dimension':spec['dimension'],
                                       'max_input_tokens':max(lengths),'encoding':state,'quality_scores':'not computed for smoke subset'})
        print('GPU/model/prompt/normalization smoke test passed',flush=True)
        return
    dense={};fused={}
    print('Embeddings ready. Evaluating dense search and hybrid RRF...',flush=True)
    index_by_key={key:i for i,key in enumerate(keys)}
    for number,q in enumerate(questions):
        torch.cuda.synchronize()
        began=time.perf_counter()
        with torch.inference_mode():
            online_query=model.encode([texts[len(bodies)+number]],batch_size=1,prompt='',
                                      normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)[0]
        torch.cuda.synchronize()
        query_ms=(time.perf_counter()-began)*1000
        if not np.allclose(online_query,vectors[len(bodies)+number],atol=.001):
            raise ValueError('Single-query and cached-batch embeddings unexpectedly differ')
        began=time.perf_counter()
        similarities=np.asarray(vectors[:len(bodies)] @ online_query)
        ranked=rank_top(similarities,keys)
        dense[q['QUESTION_ID']]={'ranked':ranked,'search_ms':(time.perf_counter()-began)*1000,'query_encode_ms':query_ms}
        began=time.perf_counter()
        merged,_=rrf([keys[i] for i in lexical[q['QUESTION_ID']]['ranked']],[keys[i] for i in ranked])
        fused[q['QUESTION_ID']]={'ranked':[index_by_key[key] for key in merged],
                                 'search_ms':(time.perf_counter()-began)*1000,'query_encode_ms':query_ms}
        if (number+1)%10==0:
            print(f'Retrieval evaluated {number+1}/{len(questions)} queries',flush=True)
    print('Packing and scoring dense contexts...',flush=True)
    rows,dense_metrics=score_run(questions,dense,chunks,docs,context_tokenizer,structured)
    print('Packing and scoring hybrid contexts...',flush=True)
    hybrid_rows,hybrid_metrics=score_run(questions,fused,chunks,docs,context_tokenizer,structured)
    metrics={'dense':dense_metrics,'hybrid':hybrid_metrics,'encoding':state,
             'latency_scope':'dense CPU matrix search/rank only; hybrid fusion-only timing; no answer model; not complete request latency',
             'vector_bytes':vectors.nbytes,'peak_rss_os_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    save_json(output/'predictions_dense.json',rows);save_json(output/'predictions_hybrid.json',hybrid_rows)
    save_json(output/'metrics.json',metrics)
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,choices=['bm25','gemma2','bge_m3','qwen3'])
    parser.add_argument('--variant',default='recursive_512',choices=['recursive_256','recursive_512','section_256','section_512','structured_512'])
    parser.add_argument('--chunk-root',type=Path,default=Path('data/runs/chunking_v1'))
    parser.add_argument('--tag',default='colab_v1')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--dtype',default='float32',choices=['float32','float16'])
    parser.add_argument('--attn',default='eager',choices=['eager','sdpa'])
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--token-budget',type=int,default=8192)
    parser.add_argument('--sort-batches',action='store_true')
    args=parser.parse_args()
    if args.model=='gemma2' and args.dtype=='float16':
        parser.error('EmbeddingGemma 2 does not support FP16 activations; use float32 on T4')
    if args.batch_size<1 or args.token_budget<1:
        parser.error('Batch size and token budget must be positive')
    (ROOT/'data').mkdir(exist_ok=True)
    lock=ROOT/'data/embedding_worker.lock'
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    try:
        main(args)
    finally:
        lock.unlink()
