"""Import saved evaluation measurements into LangSmith without new model calls."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4, uuid5

from dotenv import load_dotenv
from langsmith.run_trees import RunTree
from langsmith.utils import LangSmithNotFoundError

from rag_app.tracing import get_client
from scripts.eval_suite import build_report, load_cases


def publish(directory, client, ragas=None, *, dataset_name=None, experiment_prefix='techqa-saved-pipeline'):
    cases = load_cases(directory)
    report = build_report(cases, ragas)
    identity_file = directory / 'manifest.json' if directory.is_dir() else directory
    identity = report['cohort_id']
    feedback_cohort = all(c['question_id'].startswith('FEEDBACK_') for c in cases)
    name = dataset_name or (('techqa-feedback-' if feedback_cohort else 'techqa-training-') + identity[:12])
    try:
        dataset = client.read_dataset(dataset_name=name)
    except LangSmithNotFoundError:
        dataset = client.create_dataset(name, description='TechQA evaluation. References are used for scoring.')
    existing = {e.inputs['question_id']: e for e in client.list_examples(dataset_id=dataset.id)}
    for case in cases:
        example = existing.get(case['question_id'])
        if example and (example.inputs.get('question') != case['user_input'] or
                        (example.outputs or {}).get('reference') != case['reference']):
            raise ValueError('Existing dataset question/reference differs; use a versioned dataset')
    missing = [{'inputs': {'question_id': c['question_id'], 'question': c['user_input']},
                'outputs': {'reference': c['reference']}} for c in cases if c['question_id'] not in existing]
    if missing:
        client.create_examples(dataset_id=dataset.id, examples=missing)
        existing = {e.inputs['question_id']: e for e in client.list_examples(dataset_id=dataset.id)}
    if not all(c['question_id'] in existing for c in cases):
        raise ValueError('Dataset example creation is incomplete')
    experiment_name = experiment_prefix + '-' + str(uuid4())[:8]
    metadata = {'cohort_id': identity, 'mode': 'recorded_result_import',
                'evaluation_layer': cases[0].get('evaluation_layer', 'pipeline'),
                'collection_artifact_sha256': hashlib.sha256(identity_file.read_bytes()).hexdigest(),
                'split': 'reviewed_user_feedback' if feedback_cohort else 'training_tuning'}
    project = client.create_project(experiment_name, reference_dataset_id=dataset.id,
        num_examples=len(cases), num_repetitions=1, metadata=metadata,
        description='Saved measurements imported with original duration, usage and failure status.')
    rows = {r['question_id']: r for r in report['cases']}
    for case in cases:
        qid = case['question_id']
        row = rows[qid]
        gen = case.get('generation', {})
        end = datetime.now(timezone.utc)
        seconds = gen.get('seconds')
        measured = isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds >= 0
        calls = gen.get('provider_calls', [])
        outputs = {'question_id': qid, 'response': case['response'], 'status': case['status'],
                   'sources': case['sources'], 'original_trace_id': case.get('langsmith_run_id'),
                   'evaluation_layer': case.get('evaluation_layer', 'pipeline'),
                   'model_calls': [{k: call.get(k) for k in ('provider', 'model')} for call in calls]}
        run = RunTree(name=case.get('evaluation_layer', 'pipeline').title() + ' / ' + qid, run_type='chain', client=client,
            project_name=experiment_name, reference_example_id=existing[qid].id,
            inputs={'question_id': qid, 'question': case['user_input']}, outputs=outputs,
            start_time=end - timedelta(seconds=seconds) if measured else end, end_time=end,
            error=(gen.get('error_type') or 'generation_error') if case['status'] == 'generation_error' else None,
            extra={'metadata': {**metadata, 'timestamp_basis': 'import_time',
                                'duration_source': 'saved_generation_seconds' if measured else 'unavailable'}})
        run.post()
        if calls:
            usage = {'input_tokens': sum(c.get('prompt_tokens') or 0 for c in calls),
                     'output_tokens': sum(c.get('output_tokens', c.get('completion_tokens')) or 0 for c in calls),
                     'total_tokens': sum(c.get('total_tokens') or 0 for c in calls)}
            child = run.create_child('Pipeline / Recorded model usage', run_type='llm',
                inputs={'messages': []}, outputs={'usage_metadata': usage}, start_time=end, end_time=end,
                extra={'metadata': {'ls_provider': calls[0].get('provider'),
                                    'ls_model_name': calls[0].get('model'), 'aggregate_usage': True}})
            child.post()
        feedback = [
            {'key': 'response_status', 'value': case['status']},
            {'key': 'selected_gold_document_hit', 'score': row['selected_gold_document_hit']},
            {'key': 'pipeline_error', 'score': int(case['status'] == 'generation_error')},
            {'key': 'citation_contract', 'score': int(row['citation_contract']['pass'])
             if row['citation_contract']['state'] == 'checked' else None,
             'comment': 'Citation format and source IDs.'},
            {'key': 'recorded_latency_seconds', 'score': seconds if measured else None},
            {'key': 'recorded_tokens', 'score': sum(c.get('total_tokens') or 0 for c in calls)},
            {'key': 'reference_claim_coverage', 'score': row['reference_claim_coverage']['score'],
             'comment': 'Custom reference-claim coverage from cached reverse NLI judgments, version 1.'},
            {'key': 'cost_status', 'value': 'free_route' if calls and all(str(c.get('model', '')).endswith(':free') for c in calls) else 'not_recorded'},
        ]
        for metric, value in row['ragas'].items():
            scored = value['state'] == 'success' and not value['needs_review']
            reason = 'Judge review required' if value['needs_review'] else value.get('reason') or value['state']
            feedback.extend([
                {'key': 'ragas_' + metric, 'score': value['score'] if scored else None, 'comment': reason},
                {'key': 'ragas_state_' + metric, 'value': 'needs_review' if value['needs_review'] else value['state'], 'comment': reason},
                {'key': 'ragas_scored_' + metric, 'score': int(scored)},
                {'key': 'ragas_eligible_' + metric, 'score': int(metric.startswith('context_') or case['status'] == 'answered')},
            ])
        for item in feedback:
            client.create_feedback(run.id, **item, session_id=project.id, trace_id=run.trace_id,
                feedback_id=uuid5(run.id, item['key']), source_info={'imported_measurement': True})
    client.flush()
    client.update_project(project.id, end_time=datetime.now(timezone.utc))
    url = dataset.url + '/compare?selectedSessions=' + str(project.id) if dataset.url else str(project.id)
    print('Published experiment', experiment_name, '\n' + url, flush=True)
    return SimpleNamespace(experiment_id=str(project.id), experiment_name=experiment_name, url=url)


if __name__ == '__main__':
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--ragas', type=Path)
    parser.add_argument('--dataset-name')
    parser.add_argument('--experiment-prefix', default='techqa-saved-pipeline')
    args = parser.parse_args()
    publish(args.cases, get_client(), json.loads(args.ragas.read_text()) if args.ragas else None,
            dataset_name=args.dataset_name, experiment_prefix=args.experiment_prefix)
