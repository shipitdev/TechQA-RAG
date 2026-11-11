"""Collect supplied-evidence component cases and real API workflow cases for RAGAS."""
import argparse
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from fastapi.testclient import TestClient

from rag_app.api import create_app
from rag_app.generation import AnswerGenerator, ConversationResolver, GenerationError, configured_provider
from rag_app.retriever import Retriever
from rag_app.tracing import trace_span
from scripts.audit_techqa import RAW, ROOT
from scripts.eval_ragas import save
from scripts.eval_suite import load_cases


def stream_result(response):
    for block in response.text.split('\n\n'):
        if block.startswith('event: final\n'):
            return json.loads(block.split('data: ', 1)[1])
        if block.startswith('event: error\n'):
            error = json.loads(block.split('data: ', 1)[1])
            return {'status': 'generation_error', 'answer': '', 'citations': [],
                    'sources': [], 'metrics': error.get('metrics', {})}
    raise ValueError('API response has no terminal event')


def collect_components(pipeline, output, question_ids):
    manifest = json.loads((pipeline / 'manifest.json').read_text())
    cases = {c['question_id']: c for c in load_cases(pipeline)}
    if len(set(question_ids)) != len(question_ids) or not set(question_ids) <= set(manifest['question_ids']):
        raise ValueError('Component IDs must belong to the pipeline cohort')
    documents = json.loads((RAW / 'TechQA/training_and_dev/training_dev_technotes.json').read_text())
    provider = configured_provider()
    for qid in question_ids:
        case = dict(cases[qid])
        doc = case['reference_provenance']['document_id']
        evidence = [{'id': 'S1', 'document_id': doc, 'text': documents[doc]['text']}]
        provider.usage.clear()
        provider.attempts.clear()
        generator = AnswerGenerator(provider.json_call)
        started = time.perf_counter()
        with trace_span('Component / Generator / ' + qid, inputs={'question': case['user_input']},
                        metadata={'evidence_mode': 'annotated_original_article'}) as run:
            try:
                answer = generator.generate(case['user_input'], evidence)
            except GenerationError as exc:
                answer = {'status': 'generation_error', 'answer': '', 'citations': [], 'error_type': exc.category}
            case.update(status=answer['status'], response=answer['answer'], citations=answer['citations'],
                        sources=evidence, retrieved_contexts=[json.dumps(evidence[0], ensure_ascii=False)],
                        evaluation_layer='component', langsmith_run_id=str(run.id) if run else None,
                        generation={'seconds': time.perf_counter() - started, 'trace': generator.trace,
                                    'provider_calls': list(provider.usage), 'http_attempts': list(provider.attempts),
                                    'error_type': answer.get('error_type')})
            if run:
                run.end(outputs={'status': case['status'], 'response': case['response']})
        save(output / (qid + '.json'), case)
    save(output / 'manifest.json', {**manifest, 'question_ids': question_ids,
         'evaluation_layer': 'component', 'scope': 'generator with supplied original annotated article'})


def collect_workflow(pipeline, output, root_id, followup):
    manifest = json.loads((pipeline / 'manifest.json').read_text())
    if root_id not in manifest['question_ids']:
        raise ValueError('Workflow root must belong to the pipeline cohort')
    base = {c['question_id']: c for c in load_cases(pipeline)}[root_id]
    provider = configured_provider()
    generator = AnswerGenerator(provider.json_call)
    search = Retriever()
    old_ui = os.environ.get('ENABLE_GENUI')
    os.environ['ENABLE_GENUI'] = '0'
    try:
        with TestClient(create_app(search, generator, ConversationResolver(provider.json_call), provider)) as client:
            turns = [(root_id, base['user_input']), (root_id + '_FOLLOWUP', followup)]
            root_status = None
            for qid, question in turns:
                started = time.perf_counter()
                with trace_span('Workflow / ' + qid, inputs={'question': question}) as run:
                    result = stream_result(client.post('/ask', json={'question': question}))
                    if root_status is None:
                        root_status = result['status']
                    sources = result.get('sources', [])
                    case = {**base, 'question_id': qid, 'user_input': question,
                            'status': result['status'], 'response': result['answer'], 'citations': result['citations'],
                            'sources': sources, 'retrieved_contexts': [json.dumps(s, ensure_ascii=False) for s in sources],
                            'evaluation_layer': 'workflow', 'langsmith_run_id': str(run.id) if run else None,
                            'generation': {'seconds': time.perf_counter() - started, 'trace': list(generator.trace),
                                'provider_calls': result['metrics'].get('provider_calls', []),
                                'http_attempts': result['metrics'].get('http_attempts', []),
                                'error_type': result['metrics'].get('error_category')}}
                    if qid.endswith('_FOLLOWUP'):
                        case['user_input'] += '\nConversation question: ' + base['user_input']
                    if run:
                        run.end(outputs={'status': case['status'], 'response': case['response']})
                    save(output / (qid + '.json'), case)
            repeated = stream_result(client.post('/ask', json={'question': base['user_input']}))
            client.post('/session/end')
            save(output / 'contracts.json', {'turns': 2, 'same_session': True,
                 'cache_check_applicable': root_status == 'answered',
                 'repeat_answer_cache_hit': repeated['metrics'].get('answer_cache_hit', False),
                 'session_cleared': 'techqa_session' not in client.cookies})
    finally:
        search.close()
        if old_ui is None:
            os.environ.pop('ENABLE_GENUI', None)
        else:
            os.environ['ENABLE_GENUI'] = old_ui
    save(output / 'manifest.json', {**manifest, 'question_ids': [q for q, _ in turns],
         'evaluation_layer': 'workflow', 'scope': 'API follow-up turns and session/cache contracts; canonical text'})


if __name__ == '__main__':
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--component-ids', nargs='+', required=True)
    parser.add_argument('--workflow-root', required=True)
    parser.add_argument('--followup', required=True)
    args = parser.parse_args()
    collect_components(args.pipeline, args.output / 'component', args.component_ids)
    collect_workflow(args.pipeline, args.output / 'workflow', args.workflow_root, args.followup)
