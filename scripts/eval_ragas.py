"""Collect verified TechQA answers, then evaluate saved cases with RAGAS offline."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from scripts.audit_techqa import ROOT, RAW, sha256

FIELDS = {
    'context_precision': ('user_input', 'reference', 'retrieved_contexts'),
    'context_recall': ('user_input', 'reference', 'retrieved_contexts'),
    'faithfulness': ('user_input', 'response', 'retrieved_contexts'),
    'answer_relevancy': ('user_input', 'response'),
    'factual_correctness': ('response', 'reference'),
    'answer_completeness': ('response', 'reference'),
}


class BudgetExceeded(RuntimeError):
    pass


class RequestBudget:
    def __init__(self, limit):
        if limit < 0:
            raise ValueError('Request budget must be nonnegative')
        self.limit, self.sent, self.exhausted = limit, 0, False

    def consume(self):
        if self.sent >= self.limit:
            self.exhausted = True
            raise BudgetExceeded('Evaluation HTTP request budget exhausted')
        self.sent += 1


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def retry_delay(headers):
    # Groq's integer Retry-After can round below the fractional refill time.
    try:
        seconds = float(headers.get('retry-after', 58))
        return max(5, seconds + 2) if math.isfinite(seconds) else 60
    except (TypeError, ValueError):
        return 60


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def select_questions(rows, tuning_ids, limit):
    lookup = {q['QUESTION_ID']: q for q in rows}
    if limit < 1 or len(lookup) != len(rows) or len(set(tuning_ids)) != len(tuning_ids):
        raise ValueError('Invalid limit or duplicate question IDs')
    if any(not qid.startswith('TRAIN_') or qid not in lookup for qid in tuning_ids):
        raise ValueError('Only the saved training tuning split is allowed')
    return [lookup[qid] for qid in sorted(tuning_ids) if lookup[qid]['ANSWERABLE'] == 'Y'][:limit]


def make_case(q, original_body, retrieved, answer):
    start, end = int(q['START_OFFSET']), int(q['END_OFFSET'])
    if q['ANSWERABLE'] != 'Y' or not 0 <= start < end <= len(original_body) or original_body[start:end] != q['ANSWER']:
        raise ValueError('Reference does not match the annotated original-body span')
    sources = retrieved.get('sources', [])
    return {'question_id': q['QUESTION_ID'], 'user_input': q['QUESTION_TITLE'] + '\n' + q['QUESTION_TEXT'],
            # The writer receives JSON sources including titles/IDs, not just body strings.
            'retrieved_contexts': [json.dumps(s, ensure_ascii=False) for s in sources],
            'writer_evidence_json': json.dumps(sources, ensure_ascii=False), 'sources': sources,
            'selected_passages': retrieved.get('selected_passages', []),
            'reference': original_body[start:end],
            'reference_provenance': {'view': 'original_body_span_v1', 'document_id': q['DOCUMENT'],
                                     'start': start, 'end': end, 'body_sha256': hashlib.sha256(original_body.encode()).hexdigest()},
            'status': answer['status'], 'response': answer.get('answer', ''),
            'citations': answer.get('citations', []), 'retrieval_metrics': retrieved.get('metrics', {})}


def metric_inputs(name, case):
    return {key: case[key] for key in FIELDS[name]}


async def cached_judgment(generate, settings, prompt, response_model, directory, journal, http_calls, *, draw=0):
    key = fingerprint([settings, prompt, response_model.model_json_schema(), draw])
    path = directory / (key + '.json')
    started = time.perf_counter()
    offset = len(http_calls)
    if path.exists():
        stored = json.loads(path.read_text())
        result = response_model.model_validate(stored['output'])
        journal.append({'cache_hit': True, 'key': key, 'decision': stored['output'],
                        'seconds': time.perf_counter() - started, 'http_calls': []})
        return result
    try:
        result = await generate(prompt, response_model)
        save(path, {'output': result.model_dump(mode='json')})
        return result
    finally:
        journal.append({'cache_hit': False, 'key': key,
            'decision': result.model_dump(mode='json') if 'result' in locals() else None,
            'seconds': time.perf_counter() - started, 'http_calls': http_calls[offset:]})


async def evaluate_case(case, scorers, *, timeout=300, journal=None, budget=None):
    results = {}
    for name, scorer in scorers.items():
        started = time.perf_counter()
        offset = len(journal) if journal is not None else 0
        if (('retrieved_contexts' in FIELDS[name] and not case['retrieved_contexts']) or
                ('response' in FIELDS[name] and case['status'] != 'answered')):
            results[name] = {'state': 'skipped', 'score': None, 'reason': 'no_context_or_no_released_answer'}
            continue
        try:
            async with asyncio.timeout(timeout):
                result = await scorer.ascore(**metric_inputs(name, case))
            value = float(result.value)
            results[name] = {'state': 'success' if math.isfinite(value) else 'undefined',
                             'score': value if math.isfinite(value) else None}
        except Exception as exc:
            # Provider messages can contain inputs/headers; retain a safe category only.
            results[name] = {'state': 'budget_exhausted' if budget and budget.exhausted else 'error',
                             'score': None, 'error_type': type(exc).__name__}
        results[name]['seconds'] = round(time.perf_counter() - started, 3)
        if journal is not None:
            results[name]['judge_calls'] = journal[offset:]
        print(f"  {name}: {results[name]['state']} {results[name]['score']}", flush=True)
    return {'question_id': case['question_id'], 'status': case['status'], 'metrics': results}


def summarize(rows):
    summary = {'attempted': len(rows), 'outcomes': dict(Counter(r['status'] for r in rows)), 'metrics': {}}
    for name in dict.fromkeys(n for r in rows for n in r['metrics']):
        values = [r['metrics'][name] for r in rows if name in r['metrics']]
        counts = Counter(v['state'] for v in values)
        scores = [v['score'] for v in values if v['state'] == 'success']
        summary['metrics'][name] = {state: counts[state] for state in ('success', 'error', 'undefined', 'skipped', 'budget_exhausted')}
        summary['metrics'][name]['mean'] = sum(scores) / len(scores) if scores else None
    return summary


def collect(args):
    from rag_app.retriever import Retriever
    from rag_app.generation import AnswerGenerator, GroqREST, GenerationError
    split_path = ROOT / 'data/splits/techqa_v1.json'
    split = json.loads(split_path.read_text())
    base = RAW / 'TechQA/training_and_dev'
    for name in ('training_Q_A.json', 'training_dev_technotes.json'):
        if sha256(base / name) != split['source_hashes'][name]:
            raise ValueError('Pinned source changed: ' + name)
    selected = select_questions(json.loads((base / 'training_Q_A.json').read_text()), split['tuning_ids'], args.limit)
    if not selected:
        raise ValueError('No answerable training cases')
    original_docs = json.loads((base / 'training_dev_technotes.json').read_text())
    retriever = Retriever()
    provider = GroqREST()
    generator = AnswerGenerator(provider.json_call)
    identity = {'split_sha256': sha256(split_path), 'source_hashes': split['source_hashes'],
                'index': retriever.index_identity(), 'model': provider.model,
                'code': {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'rag_app').glob('*.py'))},
                'scope': 'standalone canonical answer; no memory, answer cache, UI or reference labels in inference'}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'identity': identity, 'question_ids': [q['QUESTION_ID'] for q in selected],
                'selection': 'first sorted answerable IDs in saved tuning split; no outcome selection',
                'excluded_original_negatives': sum(q['ANSWERABLE'] != 'Y' for q in
                    json.loads((base / 'training_Q_A.json').read_text()) if q['QUESTION_ID'] in split['tuning_ids'])}
    old_manifest = args.output / 'manifest.json'
    if old_manifest.exists() and json.loads(old_manifest.read_text()) != manifest:
        raise ValueError('Existing collection differs; use a new output directory')
    save(old_manifest, manifest)
    try:
        for q in selected:
            path = args.output / (q['QUESTION_ID'] + '.json')
            if path.exists():
                print(q['QUESTION_ID'] + ': saved case reused', flush=True)
                continue
            question = q['QUESTION_TITLE'] + '\n' + q['QUESTION_TEXT']
            print(q['QUESTION_ID'] + ': retrieval → writer → verifier', flush=True)
            started = time.perf_counter()
            retrieved = retriever.retrieve(question)
            provider.usage.clear()
            provider.deadline = time.monotonic() + 180
            try:
                answer = generator.generate(question, retrieved['sources'])
            except GenerationError as exc:
                answer = {'status': 'generation_error', 'answer': '', 'error_type': exc.category}
            case = make_case(q, original_docs[q['DOCUMENT']]['text'], retrieved, answer)
            case['generation'] = {'seconds': time.perf_counter() - started,
                                  'provider_calls': list(provider.usage), 'error_type': answer.get('error_type')}
            save(path, case)
            print(f"  saved {case['status']}; {len(case['sources'])} passages", flush=True)
    finally:
        retriever.close()


async def run(args):
    os.environ['RAGAS_DO_NOT_TRACK'] = 'true'
    import httpx
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy, FactualCorrectness
    from rag_app.retriever import Retriever

    manifest = json.loads((args.cases / 'manifest.json').read_text())
    ids = manifest['question_ids'][:args.limit]
    if len(set(ids)) != len(ids) or any(not qid.startswith('TRAIN_') or not qid.replace('_', '').isalnum() for qid in ids):
        raise ValueError('Invalid training case IDs')
    cases = [json.loads((args.cases / (qid + '.json')).read_text()) for qid in ids]
    if not cases or any(c['question_id'] != qid for c, qid in zip(cases, ids)):
        raise ValueError('Case membership mismatch')
    key = os.getenv('GROQ_API_KEY') or os.getenv('GROK_API_KEY')
    if not key and args.max_requests:
        raise ValueError('Groq key missing')
    retriever = Retriever() if 'answer_relevancy' in args.metrics else None
    budget = RequestBudget(args.max_requests)
    settings = {'provider': 'groq', 'judge': args.judge, 'temperature': 0, 'reasoning_effort': 'low',
                'transport': {'sdk_max_retries': 3, 'retry_after_buffer_seconds': 2, 'minimum_retry_seconds': 5},
                'max_tokens': 4096, 'strictness': 3, 'timeout_per_metric': args.timeout,
                'packages': {p: importlib.metadata.version(p) for p in
                    (('ragas', 'openai', 'instructor', 'langchain-community') +
                     (('sentence-transformers', 'torch', 'transformers', 'torchvision') if retriever else ()))},
                'implementation_sha256': sha256(Path(__file__)), 'embedding': retriever._gemma_policy() if retriever else None,
                'embedding_policy': 'same formatted Gemma query role for original/generated questions; normalized cosine',
                'ragas_source_sha256': fingerprint([(str(p.relative_to(Path(__import__('ragas').__file__).parent)), sha256(p))
                    for p in sorted(Path(__import__('ragas').__file__).parent.rglob('*.py'))])}
    journal, http_calls = [], []
    args.cache.mkdir(parents=True, exist_ok=True)

    async def request_hook(request):
        budget.consume()  # Includes SDK retries; blocks before any request is sent.
        request.extensions['eval_started'] = time.perf_counter()
        request.extensions['eval_call_index'] = len(http_calls)
        http_calls.append({'status': None, 'usage': {}, 'seconds': None})

    async def response_hook(response):
        await response.aread()
        try:
            usage = response.json().get('usage', {})
        except (ValueError, AttributeError):
            usage = {}
        http_calls[response.request.extensions['eval_call_index']].update(
            {'status': response.status_code, 'usage': usage,
             'seconds': time.perf_counter() - response.request.extensions['eval_started']})
        if response.status_code == 429:
            delay = retry_delay(response.headers)
            http_calls[response.request.extensions['eval_call_index']]['retry_after_seconds'] = delay
            response.headers['retry-after'] = str(delay)
            print(f'  Groq quota: waiting about {delay:g}s before retry', flush=True)

    client = AsyncOpenAI(api_key=key or 'cache-only-no-key', base_url='https://api.groq.com/openai/v1', timeout=60, max_retries=3,
        http_client=httpx.AsyncClient(event_hooks={'request': [request_hook], 'response': [response_hook]}))
    backend = llm_factory(args.judge, client=client, temperature=0, max_tokens=4096, reasoning_effort='low')

    class CachedJudge(InstructorBaseRagasLLM):
        def __init__(self, draws_per_prompt=1):
            self.draws = Counter()
            self.draws_per_prompt = draws_per_prompt

        def generate(self, prompt, response_model):
            return asyncio.run(self.agenerate(prompt, response_model))

        async def agenerate(self, prompt, response_model):
            identity = fingerprint([prompt, response_model.model_json_schema()])
            draw = self.draws[identity] % self.draws_per_prompt
            self.draws[identity] += 1
            # Reuse identical grading work; only relevancy needs three separate draws.
            return await cached_judgment(backend.agenerate, settings, prompt, response_model,
                                         args.cache, journal, http_calls, draw=draw)

    embedding_counts = {'query_cache_hits': 0, 'query_cache_misses': 0}

    class LocalGemma(BaseRagasEmbedding):
        def embed_text(self, text, **kwargs):
            return retriever._query_vector(text, embedding_counts)

        async def aembed_text(self, text, **kwargs):
            return self.embed_text(text)

    judge = CachedJudge()
    factories = {'context_precision': lambda: ContextPrecision(llm=judge),
                 'context_recall': lambda: ContextRecall(llm=judge),
                 'faithfulness': lambda: Faithfulness(llm=judge),
                 'answer_relevancy': lambda: AnswerRelevancy(llm=CachedJudge(3), embeddings=LocalGemma(), strictness=3),
                 'factual_correctness': lambda: FactualCorrectness(llm=judge, mode='f1'),
                 'answer_completeness': lambda: FactualCorrectness(llm=judge, mode='recall')}
    scorers = {name: factories[name]() for name in dict.fromkeys(args.metrics)}
    rows = []
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'settings': settings,
              'manifest': manifest, 'selected_question_ids': ids, 'selected_metrics': list(scorers),
              'cases_sha256': {c['question_id']: sha256(args.cases / (c['question_id'] + '.json')) for c in cases},
              'limitations': ['training tuning pilot, not final development', 'same model family writer/judge; manual review needed',
                              'answer metrics conditional on released answers; no global refusal accuracy', 'cost not calculated; actual token usage recorded']}
    started = time.perf_counter()
    try:
        for case in cases:
            print(case['question_id'] + ': RAGAS scoring', flush=True)
            rows.append(await evaluate_case(case, scorers, timeout=args.timeout, journal=journal, budget=budget))
            report.update({'results': rows, 'summary': summarize(rows), 'evaluation_seconds': time.perf_counter() - started,
                           'judge_cache_hits': sum(x['cache_hit'] for x in journal),
                           'judge_cache_misses': sum(not x['cache_hit'] for x in journal),
                           'http_calls': http_calls, 'embedding_cache': embedding_counts,
                           'request_budget': {'limit': budget.limit, 'sent': budget.sent, 'exhausted': budget.exhausted}})
            save(args.output, report)
    finally:
        await client.close()
        if retriever:
            retriever.close()
    print(json.dumps(report['summary'], indent=2), flush=True)


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    c = subs.add_parser('collect')
    c.add_argument('--limit', type=int, default=3)
    c.add_argument('--output', type=Path, default=ROOT / 'data/ragas/pilot_v1')
    r = subs.add_parser('run')
    r.add_argument('--cases', type=Path, default=ROOT / 'data/ragas/pilot_v1')
    r.add_argument('--output', type=Path)
    r.add_argument('--metrics', nargs='+', choices=tuple(FIELDS), required=True)
    r.add_argument('--limit', type=int, default=3)
    r.add_argument('--max-requests', type=int, default=12, help='Maximum HTTP attempts including retries; 0 is cache-only')
    r.add_argument('--cache', type=Path, default=ROOT / 'data/ragas/judge_cache')
    r.add_argument('--judge', default='openai/gpt-oss-120b')
    r.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    if args.command == 'collect':
        collect(args)
    else:
        if args.timeout <= 0 or args.limit < 1 or args.max_requests < 0:
            parser.error('timeout/limit must be positive; max-requests must be nonnegative')
        if args.output is None:
            args.output = ROOT / 'data/ragas/results' / ('run_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
        asyncio.run(run(args))


if __name__ == '__main__':
    main()
