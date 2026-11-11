"""Free evaluation reports and paired regression comparisons over saved RAG cases."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

from rag_app.generation import AnswerGenerator
from scripts.eval_ragas import fingerprint, save, summarize

CONTRACT = 'saved_case_checks_v1'
VERSION = re.compile(r'\b([A-Za-z][A-Za-z0-9_-]{1,24})\s+v?(\d+(?:\.\d+){0,4})\b')


def load_cases(path):
    path = Path(path)
    if path.is_file():
        return json.loads(path.read_text())['cases']
    manifest = json.loads((path / 'manifest.json').read_text())
    ids = manifest['question_ids']
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', i) for i in ids):
        raise ValueError('Invalid case manifest membership')
    rows = [json.loads((path / (qid + '.json')).read_text()) for qid in ids]
    if [r['question_id'] for r in rows] != ids:
        raise ValueError('Case IDs differ from manifest')
    return rows


def judge_version_flags(case, result):
    flags = {}
    for metric, score in result.get('metrics', {}).items():
        original = case['reference'] if metric == 'context_recall' else case['response']
        if metric in ('factual_correctness', 'answer_completeness'):
            original += '\n' + case['reference']
        allowed = defaultdict(set)
        for label, version in VERSION.findall(original):
            allowed[label.lower()].add(version)
        for call in score.get('judge_calls', []):
            decision = call.get('decision') or {}
            claims = decision.get('statements', decision.get('classifications', decision.get('claims', [])))
            for claim in claims:
                text = claim if isinstance(claim, str) else claim.get('statement', '')
                for label, version in VERSION.findall(text):
                    if label.lower() in allowed and version not in allowed[label.lower()]:
                        key = (metric, label.lower(), version)
                        flags[key] = {'metric': metric, 'label': label, 'introduced_version': version,
                                      'allowed_versions': sorted(allowed[label.lower()]), 'needs_review': True}
    return list(flags.values())


def reference_claim_coverage(metric):
    """Aggregate cached reference-to-response NLI from the pinned factual metric."""
    groups = []
    claims = None
    for call in metric.get('judge_calls', []):
        decision = call.get('decision') or {}
        if 'claims' in decision:
            claims = decision['claims']
        if 'statements' in decision and claims is not None:
            statements = decision['statements']
            if (statements and len(statements) == len(claims)
                    and all(isinstance(s, dict) and s.get('verdict') in (0, 1) for s in statements)
                    and [' '.join(s.get('statement', '').split()) for s in statements]
                        == [' '.join(c.split()) for c in claims]):
                groups.append(statements)
            claims = None
    if metric.get('state') != 'success' or len(groups) != 2:
        return {'state': 'unavailable', 'score': None}
    reference_verdicts = groups[-1]
    return {'state': 'derived_from_cached_judge',
            'score': sum(s['verdict'] for s in reference_verdicts) / len(reference_verdicts),
            'reference_claims': len(reference_verdicts), 'contract': 'reference_claim_coverage_v1'}


def build_report(cases, ragas=None):
    if not cases or len({c['question_id'] for c in cases}) != len(cases):
        raise ValueError('Expected nonempty unique cases')
    for c in cases:
        if c['status'] not in {'answered', 'insufficient_evidence', 'needs_clarification', 'generation_error'}:
            raise ValueError('Unknown outcome')
        if not all(isinstance(c[k], str) for k in ('question_id', 'user_input', 'reference', 'response')):
            raise ValueError('Malformed case text')
        ids = [s['id'] for s in c['sources']]
        if len(set(ids)) != len(ids) or not all(isinstance(i, str) for i in ids):
            raise ValueError('Duplicate or malformed source IDs')
    judged = {}
    if ragas is not None:
        judged = {r['question_id']: r for r in ragas['results']}
        if len(judged) != len(ragas['results']) or set(judged) != {c['question_id'] for c in cases}:
            raise ValueError('RAGAS and case membership differ')
    rows, citations, hits, seconds = [], [], [], []
    tokens = 0
    for c in cases:
        gold_document = c['reference_provenance'].get('document_id')
        hit = int(any(s.get('document_id') == gold_document for s in c['sources'])) if gold_document else None
        if hit is not None:
            hits.append(hit)
        citation = {'state': 'skipped', 'pass': None}
        if c['status'] == 'answered':
            error = AnswerGenerator._citation_error({'answer': c['response'], 'citations': c['citations']},
                                                     {s['id'] for s in c['sources']})
            citation = {'state': 'checked', 'pass': error is None, 'reason': error}
            citations.append(int(error is None))
        flags = judge_version_flags(c, judged.get(c['question_id'], {}))
        metrics = {}
        for name, m in judged.get(c['question_id'], {}).get('metrics', {}).items():
            value = m.get('score')
            if m['state'] == 'success' and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError('Successful judge score must be finite')
            metrics[name] = {'state': m['state'], 'score': value,
                             'needs_review': any(f['metric'] == name for f in flags),
                             'reason': m.get('reason') or m.get('error_type')}
        rows.append({'question_id': c['question_id'], 'outcome': c['status'],
                     'selected_gold_document_hit': hit, 'selected_passage_count': len(c['sources']),
                     'citation_contract': citation, 'ragas': metrics, 'judge_integrity_flags': flags,
                     'reference_claim_coverage': reference_claim_coverage(judged.get(c['question_id'], {}).get('metrics', {}).get('answer_completeness', {}))})
        gen = c.get('generation', {})
        elapsed = gen.get('seconds')
        if isinstance(elapsed, (int, float)) and math.isfinite(elapsed):
            seconds.append(elapsed)
        tokens += sum(call.get('total_tokens') or 0 for call in gen.get('provider_calls', []))
    identity = sorted((c['question_id'], c['user_input'], c['reference'], c['reference_provenance'])
                      for c in cases)
    report = {'schema': CONTRACT, 'cohort_id': fingerprint(identity), 'case_count': len(cases),
              'outcomes': dict(Counter(c['status'] for c in cases)),
              'metrics': {'answer_coverage': sum(c['status'] == 'answered' for c in cases) / len(cases),
                          'selected_gold_document_hit_rate': sum(hits) / len(hits) if hits else None,
                          'citation_contract_pass_rate': sum(citations) / len(citations) if citations else None},
              'denominators': {'selected_gold_document_hit_rate': len(hits), 'citation_contract_pass_rate': len(citations)},
              'operations': {'generation_seconds': seconds, 'generation_reported_tokens': tokens,
                             'generation_usage_cases': sum('provider_calls' in c.get('generation', {}) for c in cases),
                             'cost_usd': None}, 'cases': rows,
              'limits': ['Selected-document hit is not search Recall@K or semantic evidence recall.',
                         'Citation validity does not establish claim support.',
                         'Version flags are heuristics for review; raw scores are never adjusted.',
                         'No style, toxicity/PII, live feedback or drift scores are computed.']}
    if ragas is not None:
        report['ragas_summary'] = summarize(ragas['results'])
        for name, summary in report['ragas_summary']['metrics'].items():
            summary['flagged_for_review'] = sum(row['ragas'].get(name, {}).get('needs_review', False) for row in rows)
        settings = ragas['settings']
        report['judge'] = {k: settings.get(k) for k in ('provider', 'judge', 'temperature', 'strictness',
            'max_tokens', 'embedding', 'embedding_policy', 'ragas_source_sha256', 'implementation_sha256')}
        report['judge']['packages'] = settings.get('packages', {})
        report['judge_comparison_id'] = fingerprint(report['judge'])
        report['operations']['evaluation'] = {
            'seconds': ragas.get('evaluation_seconds'), 'judge_cache_hits': ragas.get('judge_cache_hits'),
            'judge_cache_misses': ragas.get('judge_cache_misses'), 'http_attempts': len(ragas.get('http_calls', [])),
            'reported_tokens': sum((call.get('usage') or {}).get('total_tokens') or 0 for call in ragas.get('http_calls', [])),
            'scope': 'this recorded scoring pass only; earlier cache-populating passes cost extra'}
    return report


def compare_reports(baseline, candidate):
    if baseline['schema'] != candidate['schema'] or baseline['cohort_id'] != candidate['cohort_id']:
        raise ValueError('Comparison requires identical cohort, references and metric contract')
    old = {r['question_id']: r for r in baseline['cases']}
    if set(old) != {r['question_id'] for r in candidate['cases']}:
        raise ValueError('Comparison requires identical case membership')
    semantic = bool(baseline.get('judge_comparison_id') and
                    baseline.get('judge_comparison_id') == candidate.get('judge_comparison_id'))
    changes, outcomes, regressions = [], [], []
    for row in candidate['cases']:
        previous = old[row['question_id']]
        if previous['outcome'] != row['outcome']:
            outcomes.append({'question_id': row['question_id'], 'before': previous['outcome'], 'after': row['outcome']})
        pairs = {'selected_gold_document_hit': (previous['selected_gold_document_hit'], row['selected_gold_document_hit'])}
        if previous['citation_contract']['state'] == row['citation_contract']['state'] == 'checked':
            pairs['citation_contract_pass'] = (int(previous['citation_contract']['pass']), int(row['citation_contract']['pass']))
        if semantic:
            for name in previous['ragas'].keys() & row['ragas'].keys():
                a, b = previous['ragas'][name], row['ragas'][name]
                if a['state'] == b['state'] == 'success' and not a['needs_review'] and not b['needs_review']:
                    pairs[name] = (a['score'], b['score'])
                elif a['state'] != b['state']:
                    changes.append({'question_id': row['question_id'], 'metric': name,
                                    'before_state': a['state'], 'after_state': b['state']})
        for name, (a, b) in pairs.items():
            item = {'question_id': row['question_id'], 'metric': name, 'before': a, 'after': b, 'delta': b-a}
            changes.append(item)
            if b < a - 1e-9:
                regressions.append(item)
    return {'cohort_id': baseline['cohort_id'], 'changes': changes, 'outcome_changes': outcomes,
            'regressions': regressions, 'semantic_scores_comparable': semantic,
            'interpretation': 'Paired observed differences, not statistical significance or automatic release approval.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    report = subs.add_parser('report')
    report.add_argument('--cases', type=Path, required=True)
    report.add_argument('--ragas', type=Path)
    report.add_argument('--output', type=Path, required=True)
    compare = subs.add_parser('compare')
    compare.add_argument('--baseline', type=Path, required=True)
    compare.add_argument('--candidate', type=Path, required=True)
    compare.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'report':
        raw = json.loads(args.ragas.read_text()) if args.ragas else None
        output = build_report(load_cases(args.cases), raw)
    else:
        output = compare_reports(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    save(args.output, output)
    print(json.dumps({k: v for k, v in output.items() if k in ('case_count', 'metrics', 'outcomes', 'regressions')}, indent=2))
    print('Saved', args.output, '(no model calls)')


if __name__ == '__main__':
    main()
