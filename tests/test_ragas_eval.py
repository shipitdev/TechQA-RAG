import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.eval_ragas import select_questions, make_case, metric_inputs, evaluate_case, summarize, cached_judgment, retry_delay, RequestBudget, BudgetExceeded


def question(qid='TRAIN_Q1', answerable='Y'):
    return dict(QUESTION_ID=qid, ANSWERABLE=answerable, QUESTION_TITLE='Which setting?',
                QUESTION_TEXT='Explain.', DOCUMENT='gold', START_OFFSET=4, END_OFFSET=10, ANSWER='fix=42')


def case():
    return make_case(question(), 'Use fix=42 now',
                     {'sources': [{'id': 'S1', 'document_id': 'other', 'text': 'Retrieved content', 'title': 'Title'}],
                      'selected_passages': [{'start': 0, 'end': 17}], 'metrics': {}},
                     {'status': 'answered', 'answer': 'Use the setting. [S1]', 'citations': ['S1']})


def test_selection_is_training_tuning_only_and_does_not_cherry_pick():
    rows = [question('TRAIN_Q2'), question('TRAIN_Q1'), question('TRAIN_Q3', 'N')]
    assert [q['QUESTION_ID'] for q in select_questions(rows, ['TRAIN_Q1', 'TRAIN_Q3'], 5)] == ['TRAIN_Q1']
    with pytest.raises(ValueError): select_questions(rows, ['DEV_Q1'], 1)
    with pytest.raises(ValueError): select_questions(rows, ['TRAIN_Q1'], 0)


def test_reference_is_checked_but_never_inserted_in_retrieved_context():
    c = case()
    assert c['reference'] == 'fix=42'
    assert 'fix=42' not in ''.join(c['retrieved_contexts'])
    assert c['sources'][0]['document_id'] == 'other'
    with pytest.raises(ValueError): make_case(question(), 'wrong', {}, {})


def test_each_metric_receives_only_its_required_fields():
    c = case()
    assert set(metric_inputs('context_precision', c)) == {'user_input', 'reference', 'retrieved_contexts'}
    assert 'reference' not in metric_inputs('faithfulness', c)
    assert set(metric_inputs('answer_relevancy', c)) == {'user_input', 'response'}


def test_retry_after_has_buffer_for_fractional_token_refill():
    assert retry_delay({'retry-after': '2'}) == 5
    assert retry_delay({'retry-after': '.7'}) == 5
    assert retry_delay({}) == 60
    assert retry_delay({'retry-after': 'invalid'}) == 60
    assert retry_delay({'retry-after': 'nan'}) == 60


def test_request_budget_counts_retries_and_cache_only_blocks_before_sending():
    budget = RequestBudget(2)
    budget.consume(); budget.consume()
    with pytest.raises(BudgetExceeded): budget.consume()
    assert budget.sent == 2 and budget.exhausted
    with pytest.raises(BudgetExceeded): RequestBudget(0).consume()
    with pytest.raises(ValueError): RequestBudget(-1)


def test_withheld_answers_keep_retrieval_scores_but_skip_answer_scores():
    c = case(); c['status'] = 'insufficient_evidence'
    async def scorer(**kwargs): return SimpleNamespace(value=0.5)
    result = asyncio.run(evaluate_case(c, {n: SimpleNamespace(ascore=scorer) for n in
        ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']}, timeout=1))
    assert result['metrics']['context_recall']['score'] == 0.5
    assert result['metrics']['faithfulness']['state'] == 'skipped'
    assert result['metrics']['answer_relevancy']['state'] == 'skipped'


def test_errors_and_nan_are_not_zero_scores_and_summary_has_denominators():
    async def bad(**kwargs): raise TimeoutError()
    async def nan(**kwargs): return SimpleNamespace(value=float('nan'))
    result = asyncio.run(evaluate_case(case(), {'context_precision': SimpleNamespace(ascore=bad),
                                               'faithfulness': SimpleNamespace(ascore=nan)}, timeout=1))
    assert result['metrics']['context_precision']['state'] == 'error'
    assert result['metrics']['faithfulness']['state'] == 'undefined'
    summary = summarize([result])
    assert summary['metrics']['context_precision']['mean'] is None
    assert summary['metrics']['context_precision']['error'] == 1
    assert summary['metrics']['faithfulness']['success'] == 0


def test_judge_cache_preserves_three_independent_relevancy_draws_and_invalidates_inputs():
    class Output:
        def model_dump(self, **kwargs): return {'verdict': 1}
        @staticmethod
        def model_json_schema(): return {'type': 'object'}
        @staticmethod
        def model_validate(value):
            assert value == {'verdict': 1}
            return Output()
    calls = []
    async def generate(prompt, schema):
        calls.append(prompt)
        return Output()
    async def check(directory):
        journal = []
        for _ in range(2):
            for draw in range(3):
                await cached_judgment(generate, {'model': 'one'}, 'same prompt', Output,
                                      Path(directory), journal, [], draw=draw)
        assert len(calls) == 3
        assert sum(j['cache_hit'] for j in journal) == 3
        await cached_judgment(generate, {'model': 'two'}, 'same prompt', Output,
                              Path(directory), journal, [], draw=0)
        await cached_judgment(generate, {'model': 'one'}, 'changed prompt', Output,
                              Path(directory), journal, [], draw=0)
        assert len(calls) == 5
    with tempfile.TemporaryDirectory() as directory:
        asyncio.run(check(directory))


def test_real_ragas_aggregations_with_controlled_marking_not_network_calls(monkeypatch):
    monkeypatch.setenv('RAGAS_DO_NOT_TRACK', 'true')
    pytest.importorskip('ragas')
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy, FactualCorrectness

    class Judge(InstructorBaseRagasLLM):
        def generate(self, prompt, response_model):
            return asyncio.run(self.agenerate(prompt, response_model))
        async def agenerate(self, prompt, response_model):
            values = {
                'ContextPrecisionOutput': {'reason': 'Relevant', 'verdict': 1},
                'ContextRecallOutput': {'classifications': [
                    {'statement': 'One', 'reason': 'Supported', 'attributed': 1},
                    {'statement': 'Two', 'reason': 'Missing', 'attributed': 0}]},
                'StatementGeneratorOutput': {'statements': ['One', 'Two']},
                'ClaimDecompositionOutput': {'claims': ['One', 'Two']},
                'NLIStatementOutput': {'statements': [
                    {'statement': 'One', 'reason': 'Supported', 'verdict': 1},
                    {'statement': 'Two', 'reason': 'Missing', 'verdict': 0}]},
                'AnswerRelevanceOutput': {'question': 'Which setting?', 'noncommittal': 0},
            }
            return response_model.model_validate(values[response_model.__name__])

    class Embeddings(BaseRagasEmbedding):
        def embed_text(self, text, **kwargs): return [1., 0.]
        async def aembed_text(self, text, **kwargs): return self.embed_text(text)

    judge = Judge()
    results = asyncio.run(evaluate_case(case(), {
        'context_precision': ContextPrecision(llm=judge), 'context_recall': ContextRecall(llm=judge),
        'faithfulness': Faithfulness(llm=judge),
        'answer_relevancy': AnswerRelevancy(llm=judge, embeddings=Embeddings()),
        'factual_correctness': FactualCorrectness(llm=judge, mode='f1'),
        'answer_completeness': FactualCorrectness(llm=judge, mode='recall')}))['metrics']
    assert results['context_precision']['score'] == pytest.approx(1)
    assert results['context_recall']['score'] == pytest.approx(.5)
    assert results['faithfulness']['score'] == pytest.approx(.5)
    assert results['answer_relevancy']['score'] == pytest.approx(1)
    assert results['factual_correctness']['score'] == pytest.approx(.5)
    assert results['answer_completeness']['score'] == pytest.approx(.5)


@pytest.mark.parametrize('category, expected_calls', [('http_429', 1), ('quota_deadline', 2)])
def test_collection_distinguishes_account_limits_from_request_deadlines(tmp_path, monkeypatch, category, expected_calls):
    import json
    import scripts.eval_ragas as evaluation
    import rag_app.retriever as retrieval
    import rag_app.generation as generation
    root = tmp_path
    base = root / 'raw/TechQA/training_and_dev'
    base.mkdir(parents=True)
    rows = [question('TRAIN_Q1'), question('TRAIN_Q2')]
    (base / 'training_Q_A.json').write_text(json.dumps(rows))
    (base / 'training_dev_technotes.json').write_text(json.dumps({'gold': {'text': 'Use fix=42 now'}}))
    split = root / 'data/splits/techqa_v1.json'
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps({'tuning_ids': ['TRAIN_Q1', 'TRAIN_Q2'], 'source_hashes': {
        name: evaluation.sha256(base / name) for name in ['training_Q_A.json', 'training_dev_technotes.json']}}))
    monkeypatch.setattr(evaluation, 'ROOT', root)
    monkeypatch.setattr(evaluation, 'RAW', root / 'raw')
    search = SimpleNamespace(index_identity=lambda: 'test', close=lambda: None,
                             retrieve=lambda q: {'sources': [], 'metrics': {}})
    monkeypatch.setattr(retrieval, 'Retriever', lambda: search)
    provider = SimpleNamespace(model='test', json_call=lambda *a: None, usage=[], attempts=[])
    monkeypatch.setattr(generation, 'GroqREST', lambda **kwargs: provider)
    attempts = []

    class QuotaWriter:
        trace = []
        def generate(self, question, sources):
            attempts.append(question)
            raise generation.GenerationError('quota', category=category)

    monkeypatch.setattr(generation, 'AnswerGenerator', lambda call: QuotaWriter())
    monkeypatch.delenv('LANGSMITH_API_KEY', raising=False)
    output = root / 'output'
    args = SimpleNamespace(limit=2, output=output, max_requests=5, sample_seed=None)
    evaluation.collect(args)
    assert len(attempts) == expected_calls
    assert (output / 'TRAIN_Q1.json').exists()
    assert (output / 'TRAIN_Q2.json').exists() == (expected_calls == 2)
    assert len(json.loads((output / 'completed.json').read_text())['cases']) == expected_calls
    evaluation.collect(args)
    assert len(attempts) == 2
    archived = list((output / 'attempts').rglob('*.json'))
    assert len(archived) == (1 if category == 'http_429' else 0)


def test_unreviewed_feedback_is_rejected_before_judging(tmp_path):
    pytest.importorskip('ragas')
    import json
    from scripts.eval_ragas import run
    qid = 'FEEDBACK_' + 'a' * 32
    (tmp_path / 'manifest.json').write_text(json.dumps({'kind': 'reviewed_user_feedback', 'question_ids': [qid]}))
    (tmp_path / (qid + '.json')).write_text(json.dumps({'question_id': qid, 'review_status': 'pending', 'reference': 'Unreviewed claim'}))
    with pytest.raises(ValueError, match='approved references'):
        asyncio.run(run(SimpleNamespace(cases=tmp_path, limit=1)))


def test_shared_dollar_budget_reserves_then_settles_and_reuses_cache_discount(tmp_path):
    import json
    from scripts.eval_budget import EvaluationBudget, BudgetLimitError
    ledger = EvaluationBudget(tmp_path / "budget.json", limit=0.02)
    body = {"messages": [{"role": "user", "content": "test"}], "max_tokens": 100}
    ticket = ledger.begin(body)
    ledger.finish(ticket, {"prompt_tokens": 100, "completion_tokens": 10,
                          "prompt_tokens_details": {"cached_tokens": 90}}, 200)
    data = json.loads((tmp_path / "budget.json").read_text())
    assert data["requests"][ticket]["cached_input_tokens"] == 90
    assert data["requests"][ticket]["value_usd"] < 0.00002
    with pytest.raises(BudgetLimitError):
        ledger.begin({"messages": [], "max_tokens": 100000})


def test_workflow_stream_parser_keeps_generation_errors():
    from scripts.eval_layers import stream_result
    response = SimpleNamespace(text='event: error\ndata: {"metrics":{"error_category":"http_429"}}\n\n')
    result = stream_result(response)
    assert result["status"] == "generation_error"
    assert result["metrics"]["error_category"] == "http_429"


def test_public_workflow_collector_uses_api_memory_and_cache(tmp_path, monkeypatch):
    import json
    import rag_app.api as api
    import scripts.eval_layers as layers
    payload = json.loads(Path('evaluations/fixtures/offline_demo.json').read_text())['cases'][0]
    pipeline = tmp_path / 'pipeline'
    pipeline.mkdir()
    (pipeline / 'manifest.json').write_text(json.dumps({'question_ids': [payload['question_id']]}))
    (pipeline / (payload['question_id'] + '.json')).write_text(json.dumps(payload))
    monkeypatch.setattr(api, 'ROOT', tmp_path)

    class Search:
        def index_identity(self): return 'controlled'
        def retrieve(self, question): return {'sources': payload['sources'], 'metrics': {}}
        def close(self): pass

    def reply(prompt, schema):
        if 'dependent' in schema['properties']:
            data = json.loads(prompt[1]['content'])
            dependent = data['original_question'] == 'What follows that edit?'
            return {'dependent': dependent, 'question': 'Beacon restart?' if dependent else data['original_question'],
                    'needs_clarification': False}
        if 'verdict' in schema['properties']:
            return {'verdict': 'complete', 'supported': True, 'complete': True,
                    'missing_steps': [], 'version_conflicts': [], 'findings': [], 'unsupported_claims': []}
        return {'status': 'answered', 'answer': 'Restart the connector. [S1]', 'citations': ['S1']}

    provider = SimpleNamespace(json_call=reply, usage=[], attempts=[], model='controlled', provider='controlled', max_tokens=100)
    monkeypatch.setattr(layers, 'configured_provider', lambda: provider)
    monkeypatch.setattr(layers, 'Retriever', Search)
    output = tmp_path / 'workflow'
    layers.collect_workflow(pipeline, output, payload['question_id'], 'What follows that edit?')
    checks = json.loads((output / 'contracts.json').read_text())
    assert checks['repeat_answer_cache_hit'] and checks['session_cleared']
    assert json.loads((output / (payload['question_id'] + '_FOLLOWUP.json')).read_text())['status'] == 'answered'


def test_component_collector_supplies_gold_article_instead_of_search_context(tmp_path, monkeypatch):
    import json
    import scripts.eval_layers as layers
    c = json.loads(Path('evaluations/fixtures/offline_demo.json').read_text())['cases'][0]
    pipeline = tmp_path / 'pipeline'
    pipeline.mkdir()
    (pipeline / 'manifest.json').write_text(json.dumps({'question_ids': [c['question_id']]}))
    (pipeline / (c['question_id'] + '.json')).write_text(json.dumps(c))
    raw = tmp_path / 'raw'
    folder = raw / 'TechQA/training_and_dev'
    folder.mkdir(parents=True)
    (folder / 'training_dev_technotes.json').write_text(json.dumps({'beacon': {'text': 'GOLD_ARTICLE'}}))
    seen = []
    class Generator:
        trace = []
        def __init__(self, callback): pass
        def generate(self, question, sources):
            seen.append(sources)
            return {'status':'answered','answer':'Approved [S1]','citations':['S1']}
    monkeypatch.setattr(layers, 'RAW', raw)
    monkeypatch.setattr(layers, 'AnswerGenerator', Generator)
    monkeypatch.setattr(layers, 'configured_provider', lambda: SimpleNamespace(usage=[], attempts=[], json_call=None))
    output = tmp_path / 'component'
    layers.collect_components(pipeline, output, [c['question_id']])
    assert seen[0][0]['text'] == 'GOLD_ARTICLE'
    assert json.loads((output / (c['question_id'] + '.json')).read_text())['evaluation_layer'] == 'component'
