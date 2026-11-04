import copy
import json

import pytest

from scripts.eval_suite import build_report, compare_reports, judge_version_flags, load_cases


def cases():
    return [dict(question_id='DEMO_1', user_input='Which setting?', reference='Use TIP 2.1.',
                 reference_provenance={'document_id': 'gold', 'view': 'test', 'start': 0, 'end': 12},
                 status='answered', response='Use TIP 2.1. [S1]', citations=['S1'],
                 sources=[{'id': 'S1', 'document_id': 'gold', 'text': 'Use TIP 2.1.'}],
                 generation={'seconds': 2, 'provider_calls': [{'total_tokens': 10}]})]


def test_free_report_uses_selected_evidence_not_search_recall_and_redacts_text():
    report = build_report(cases())
    assert report['metrics']['selected_gold_document_hit_rate'] == 1
    assert report['metrics']['answer_coverage'] == 1
    assert report['metrics']['citation_contract_pass_rate'] == 1
    assert report['operations']['generation_reported_tokens'] == 10
    assert 'Which setting?' not in json.dumps(report)
    assert 'Use TIP 2.1.' not in json.dumps(report)
    assert 'recall_at_10' not in report['metrics']


def test_unknown_citations_and_withheld_cases_stay_visible():
    rows = cases()
    rows[0]['response'] = 'Use TIP 2.1. [S9]'
    rows += [dict(rows[0], question_id='DEMO_2', status='insufficient_evidence',
                  response='Cannot answer.', citations=[], sources=[])]
    report = build_report(rows)
    assert report['metrics']['answer_coverage'] == .5
    assert report['metrics']['citation_contract_pass_rate'] == 0
    assert report['outcomes']['insufficient_evidence'] == 1
    assert report['cases'][1]['citation_contract']['state'] == 'skipped'


def test_judge_version_guard_flags_changed_claim_but_not_reordering():
    raw = {'metrics': {'faithfulness': {'judge_calls': [
        {'decision': {'statements': ['TIP 1 is unsupported.']}}]}}}
    flags = judge_version_flags(cases()[0], raw)
    assert flags and flags[0]['metric'] == 'faithfulness'
    raw['metrics']['faithfulness']['judge_calls'][0]['decision']['statements'] = ['For TIP 2.1, use the setting.']
    assert judge_version_flags(cases()[0], raw) == []


def test_comparison_rejects_reference_changes_and_reports_per_case_regressions():
    original = cases(); candidate = copy.deepcopy(original)
    candidate[0]['sources'][0]['document_id'] = 'different'
    result = compare_reports(build_report(original), build_report(candidate))
    assert result['regressions'][0]['metric'] == 'selected_gold_document_hit'
    candidate[0]['reference'] = 'A changed answer key'
    with pytest.raises(ValueError, match='cohort'):
        compare_reports(build_report(original), build_report(candidate))


def test_duplicate_cases_and_ragas_membership_mismatches_are_rejected():
    with pytest.raises(ValueError): build_report(cases() * 2)
    with pytest.raises(ValueError):
        build_report(cases(), {'results': [{'question_id': 'OTHER', 'metrics': {}}]})


def test_changed_embedding_identity_prevents_semantic_comparison_and_flagged_score_is_visible():
    raw = {'settings': {'judge': 'same', 'embedding': {'revision': 'first'}},
           'results': [{'question_id': 'DEMO_1', 'status': 'answered', 'metrics': {
               'faithfulness': {'state': 'success', 'score': .5, 'judge_calls': [
                   {'decision': {'statements': ['TIP 1 is unsupported.']}}]}}}]}
    baseline = build_report(cases(), raw)
    assert baseline['ragas_summary']['metrics']['faithfulness']['mean'] == .5
    assert baseline['ragas_summary']['metrics']['faithfulness']['flagged_for_review'] == 1
    raw['settings']['embedding']['revision'] = 'second'
    candidate = build_report(cases(), raw)
    assert not compare_reports(baseline, candidate)['semantic_scores_comparable']


def test_case_loader_rejects_manifest_path_traversal(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'question_ids': ['../outside']}))
    with pytest.raises(ValueError): load_cases(tmp_path)
