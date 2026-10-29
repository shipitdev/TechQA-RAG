"""Shared rank fusion, evidence packing and explicit retrieval/context summaries."""
import collections


def rrf(first, second, depth=100, k=60):
    scores=collections.defaultdict(float)
    for branch in [first,second]:
        for rank,chunk_id in enumerate(list(dict.fromkeys(branch))[:depth],1):
            scores[chunk_id]+=1/(k+rank)
    ranked=sorted(scores,key=lambda i:(-scores[i],i))
    return ranked,dict(scores)


def pack_context(ranked, chunks, docs, tokenizer, budget=2048):
    passages=[]
    selected=collections.Counter()
    skipped_budget=0
    def render(rows):
        return '\n\n'.join(f'[{p["document_id"]}:{p["start"]}-{p["end"]}]\n'+
                            docs[p['document_id']]['text'][p['start']:p['end']] for p in rows)
    for chunk_id in ranked:
        c=chunks[chunk_id];pid=c['document_id']
        if selected[pid]>=2 or (pid not in selected and len(selected)>=5):
            continue
        if any(p['document_id']==pid and p['start']<=c['start'] and p['end']>=c['end'] for p in passages):
            continue
        proposal=[dict(p) for p in passages]
        new={'document_id':pid,'start':c['start'],'end':c['end']}
        for p in proposal:
            if p['document_id']==pid and p['start']<=new['end'] and new['start']<=p['end']:
                p['start']=min(p['start'],new['start']);p['end']=max(p['end'],new['end'])
                break
        else:
            proposal.append(new)
        text=render(proposal)
        if len(tokenizer.encode(text,add_special_tokens=False).ids)>budget:
            skipped_budget+=1
            continue
        passages=proposal
        selected[pid]+=1
    text=render(passages)
    return {'passages':passages,'text':text,'tokens':len(tokenizer.encode(text,add_special_tokens=False).ids),
            'skipped_budget':skipped_budget}


def summarize(rows):
    positive=[r for r in rows if any(v is not None for v in r['scores'].values())]
    return {'queries':len(rows),'answerable_scored':len(positive),'original_unanswerable_excluded':len(rows)-len(positive),
            'retrieval':{k:sum(r['scores'][k] for r in positive)/len(positive) for k in positive[0]['scores']} if positive else {},
            'context':{'mean_answer_fraction':sum(r['context']['answer_fraction'] for r in positive)/len(positive) if positive else None,
                       'full_answer_fraction':sum(r['context']['complete'] for r in positive)/len(positive) if positive else None,
                       'mean_reference_tokens':sum(r['context']['tokens'] for r in rows)/len(rows) if rows else None}}
