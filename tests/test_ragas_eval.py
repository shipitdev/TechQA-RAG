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
