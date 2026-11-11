import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from scripts.publish_langsmith import publish


def client_for(cases):
    client = Mock()
    client.read_dataset.return_value = SimpleNamespace(id=uuid4(), url='https://example.com/dataset')
    client.list_examples.return_value = [SimpleNamespace(id=uuid4(), inputs={'question_id': c['question_id'], 'question': c['user_input']}, outputs={'reference': c['reference']}) for c in cases]
    client.create_project.return_value = SimpleNamespace(id=uuid4())
    return client


def test_saved_experiment_imports_measurements_before_run_creation(tmp_path):
    payload = json.loads(Path('evaluations/fixtures/offline_demo.json').read_text())
    case = payload['cases'][0]
    case['status'] = 'generation_error'
    case['generation'] = {'seconds': 12, 'error_type': 'http_429', 'provider_calls': [
        {'provider': 'tokenharbor', 'model': 'deepseek-v4.1-flash:free',
         'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14, 'private': 'omit'}]}
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps(payload))
    client = client_for(payload['cases'])
    with patch('scripts.publish_langsmith.RunTree') as tree:
        tree.return_value.id = uuid4()
        publish(path, client)
    root = tree.call_args_list[0].kwargs
    assert (root['end_time'] - root['start_time']).total_seconds() == 12
    assert root['error'] == 'http_429'
    assert root['outputs']['model_calls'] == [{'provider': 'tokenharbor', 'model': 'deepseek-v4.1-flash:free'}]
    assert client.evaluate.call_count == 0
    assert client.update_run.call_count == 0
    assert any(c.kwargs.get('key') == 'pipeline_error' and c.kwargs.get('score') == 1
               for c in client.create_feedback.call_args_list)


def test_dataset_identity_survives_changed_answers(tmp_path):
    payload = json.loads(Path('evaluations/fixtures/offline_demo.json').read_text())
    client = client_for(payload['cases'])
    with patch('scripts.publish_langsmith.RunTree') as tree:
        tree.return_value.id = uuid4()
        publish(Path('evaluations/fixtures/offline_demo.json'), client)
        name = client.read_dataset.call_args.kwargs['dataset_name']
        payload['cases'][0]['response'] = 'Changed. [S1]'
        path = tmp_path / 'changed.json'
        path.write_text(json.dumps(payload))
        publish(path, client)
    assert client.read_dataset.call_args.kwargs['dataset_name'] == name


def test_unscored_metric_has_explicit_state_and_coverage(tmp_path):
    payload = json.loads(Path('evaluations/fixtures/offline_demo.json').read_text())
    cases = payload['cases']
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps(payload))
    ragas = {'results': [{'question_id': c['question_id'], 'status': c['status'],
              'metrics': {'answer_relevancy': {'state': 'skipped', 'score': None, 'reason': 'quota'}}} for c in cases],
             'settings': {}, 'http_calls': []}
    client = client_for(cases)
    with patch('scripts.publish_langsmith.RunTree') as tree:
        tree.return_value.id = uuid4()
        publish(path, client, ragas)
    fields = [c.kwargs for c in client.create_feedback.call_args_list]
    assert any(x.get('key') == 'ragas_state_answer_relevancy' and x.get('value') == 'skipped' for x in fields)
    assert any(x.get('key') == 'ragas_scored_answer_relevancy' and x.get('score') == 0 for x in fields)


def test_custom_dataset_rejects_changed_reference():
    import pytest
    payload = json.loads(Path('evaluations/fixtures/offline_demo.json').read_text())
    client = client_for(payload['cases'])
    client.list_examples.return_value[0].outputs['reference'] = 'Different reference'
    with pytest.raises(ValueError, match='question/reference differs'):
        publish(Path('evaluations/fixtures/offline_demo.json'), client, dataset_name='existing')
