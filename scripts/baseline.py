"""First TechQA baseline: grouped tuning questions and whole-body BM25 search."""
import collections
import hashlib
import heapq
import json
import math
import platform
import random
import re
import resource
import time
from pathlib import Path
from scripts.audit_techqa import ROOT, RAW, sha256, normalized_question, stats


def group_questions(rows, docs):
    parent = {q['QUESTION_ID']: q['QUESTION_ID'] for q in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    owners = {}
    for q in rows:
        keys = [('question', normalized_question(q))]
        if q['ANSWERABLE'] == 'Y':
            keys += [('document', q['DOCUMENT']),
                     ('body', hashlib.sha256(docs[q['DOCUMENT']]['text'].encode()).hexdigest())]
        for key in keys:
            if key in owners:
                parent[find(q['QUESTION_ID'])] = find(owners[key])
            else:
                owners[key] = q['QUESTION_ID']
    groups = collections.defaultdict(list)
    for qid in sorted(parent):
        groups[find(qid)].append(qid)
    return sorted(groups.values())


def retrieval_scores(gold, ranked):
    names = ['recall_at_5','recall_at_10','mrr_at_10','ndcg_at_10']
    if gold is None:
        return dict.fromkeys(names, None)
    ranked = list(dict.fromkeys(ranked))
    rank = ranked.index(gold)+1 if gold in ranked else math.inf
    return dict(zip(names, [int(rank<=5),int(rank<=10),1/rank if rank<=10 else 0,
                           1/math.log2(rank+1) if rank<=10 else 0]))


def tokenize(text):
    # shortcut: simple word tokens split paths/punctuation; test technical-token alternatives later.
    return re.findall(r'\w+', text.lower())


class BM25:
    def __init__(self, documents):
        self.postings = collections.defaultdict(list)
        self.lengths = {}
        for docid, text in documents.items():
            counts = collections.Counter(tokenize(text))
            self.lengths[docid] = sum(counts.values())
            for word, count in counts.items():
                self.postings[word].append((docid,count))
        self.average_length = sum(self.lengths.values())/max(1,len(self.lengths))

    def scores(self, query, allowed=None):
        scores = collections.defaultdict(float)
        n = len(self.lengths)
        for word in sorted(set(tokenize(query))):
            postings = self.postings.get(word, [])
            idf = math.log(1+(n-len(postings)+.5)/(len(postings)+.5))
            for docid, tf in postings:
                if allowed is not None and docid not in allowed:
                    continue
                denominator = tf+1.5*(.25+.75*self.lengths[docid]/self.average_length)
                scores[docid] += idf*tf*2.5/denominator
        return dict(scores)

    def search(self, query, allowed=None):
        scores = self.scores(query, allowed)
        return heapq.nsmallest(10, scores.items(),key=lambda x:(-x[1],x[0]))


def main():
    output = ROOT/'data/runs/bm25_whole_v1'
    split_path = ROOT/'data/splits/techqa_v1.json'
    if output.exists():
        raise FileExistsError('Completed/partial run exists; preserve it and use a reviewed new run version')
    base = RAW/'TechQA/training_and_dev'
    source = json.loads((RAW/'source.json').read_text())
    files = ['training_Q_A.json','dev_Q_A.json','training_dev_technotes.json']
    hashes = {}
    for name in files:
        path = base/name
        hashes[name] = sha256(path)
        assert hashes[name] == source['extracted_files']['TechQA/training_and_dev/'+name]['sha256']
    rows = json.loads((base/files[0]).read_text())
    dev = json.loads((base/files[1]).read_text())
    docs = json.loads((base/files[2]).read_text())
    groups = group_questions(rows,docs)
    shuffled = list(groups)
    random.Random(42).shuffle(shuffled)
    tuning = []
    for group in shuffled:
        if len(tuning)<120:
            tuning.extend(group)
    tuning = sorted(tuning)
    train_ids = sorted({q['QUESTION_ID'] for q in rows}-set(tuning))
    train_texts = {normalized_question(q) for q in rows}
    train_gold = {q['DOCUMENT'] for q in rows if q['ANSWERABLE']=='Y'}
    train_gold_bodies = {sha256_text(docs[d]['text']) for d in train_gold}
    cleaner_dev = [q['QUESTION_ID'] for q in dev if normalized_question(q) not in train_texts
                   and (q['ANSWERABLE']!='Y' or (q['DOCUMENT'] not in train_gold
                   and sha256_text(docs[q['DOCUMENT']]['text']) not in train_gold_bodies))]
    split = {'seed':42,'policy':'shuffle whole groups until at least 120 tuning rows; no outcome-based balancing',
             'group_rules':['normalized question','same answer document','identical answer body'],
             'source_hashes':hashes,'groups':groups,'practice_ids':train_ids,'tuning_ids':tuning,
             'final_dev_ids':[q['QUESTION_ID'] for q in dev], 'supplementary_dev_ids':cleaner_dev}
    assert set(train_ids).isdisjoint(tuning) and len(train_ids)+len(tuning)==600
    assert all(set(g)<=set(tuning) or set(g)<=set(train_ids) for g in groups)
    split_path.parent.mkdir(parents=True,exist_ok=True)
    if split_path.exists():
        assert json.loads(split_path.read_text())==split, 'Existing split differs; do not overwrite'
    else:
        split_path.write_text(json.dumps(split,indent=2)+'\n')
    config = {'track':'B: all supplied benchmark documents','document_count':len(docs),
              'representation':'whole original body only; metadata retained in source, not searchable',
              'chunking':'no splitting; whole-document lexical reference only',
              'bm25':{'k1':1.5,'b':.75,'idf':'log(1+(N-df+0.5)/(df+0.5))','query_tf':'unique terms'},
              'tokenizer':'Unicode word runs, lowercase, no stopwords or stemming',
              'tie_break':'lexicographic document ID','source_hashes':hashes,
              'split_sha256':sha256(split_path),'script_sha256':sha256(Path(__file__)),
              'audit_helper_sha256':sha256(ROOT/'scripts/audit_techqa.py'),
              'python':platform.python_version(),'platform':platform.platform(),
              'no_answer_evaluation':'not performed; original negatives not verified for track B'}
    start=time.perf_counter()
    index=BM25({k:d['text'] for k,d in docs.items()})
    index_seconds=time.perf_counter()-start
    predictions=[]
    for q in rows:
        if q['QUESTION_ID'] not in tuning:
            continue
        start=time.perf_counter()
        hits=index.search(q['QUESTION_TITLE']+'\n'+q['QUESTION_TEXT'])
        elapsed=time.perf_counter()-start
        gold=q['DOCUMENT'] if q['ANSWERABLE']=='Y' else None
        predictions.append({'question_id':q['QUESTION_ID'],'original_answerable':q['ANSWERABLE'],
                            'results':[{'document_id':d,'score':s} for d,s in hits],
                            'search_ms':elapsed*1000,'scores':retrieval_scores(gold,[d for d,s in hits])})
    scored=[p for p in predictions if p['original_answerable']=='Y']
    metrics={'queries':len(predictions),'answerable_scored':len(scored),
             'original_unanswerable_excluded':len(predictions)-len(scored),
             'retrieval':{k:sum(p['scores'][k] for p in scored)/len(scored) for k in scored[0]['scores']},
             'index_seconds':index_seconds,'search_ms':stats([p['search_ms'] for p in predictions]),
             'peak_rss_os_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'latency_scope':'search only, no embeddings, reranking, context packing or generation'}
    output.mkdir(parents=True)
    for name,value in [('config',config),('metrics',metrics),('predictions',predictions)]:
        (output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


if __name__=='__main__':
    main()
