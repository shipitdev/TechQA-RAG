"""Audit pinned TechQA inputs without loading models or reading dev examples."""
import argparse
import collections
import hashlib
import json
import re
import statistics
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/techqa'
ARCHIVE_SHA = '6b094ef9a69718f727ce8d7e15c4d961e51032cefaa952e0d6af9d176d7ba118'
REVISION = '60437bc79ab217679682217598a3693cab78365b'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stats(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    return {'count': len(values), 'min': values[0], 'median': statistics.median(values),
            'mean': statistics.mean(values), 'p95': values[max(0, (95 * len(values) + 99) // 100 - 1)],
            'max': values[-1]}


def span_valid(question, documents):
    if question['ANSWERABLE'] != 'Y':
        return None
    doc = documents.get(question['DOCUMENT'])
    if doc is None:
        return False
    start, end = int(question['START_OFFSET']), int(question['END_OFFSET'])
    return 0 <= start < end <= len(doc['text']) and doc['text'][start:end] == question['ANSWER']


def normalized_question(q):
    return ' '.join((q['QUESTION_TITLE'] + '\n' + q['QUESTION_TEXT']).lower().split())


def self_check():
    doc = {'a': {'text': 'x α answer z'}}
    q = {'ANSWERABLE': 'Y', 'DOCUMENT': 'a', 'START_OFFSET': 4, 'END_OFFSET': 10, 'ANSWER': 'answer'}
    assert span_valid(q, doc)
    assert not span_valid(dict(q, END_OFFSET=11), doc)
    assert not span_valid(dict(q, DOCUMENT='missing'), doc)
    assert span_valid({'ANSWERABLE': 'N'}, doc) is None
    assert stats([]) == {'count': 0}
    assert stats(range(1, 101))['p95'] == 95
    assert normalized_question({'QUESTION_TITLE': ' A ', 'QUESTION_TEXT': 'B\nC'}) == 'a b c'


def extract_benchmark():
    archive = RAW / 'TechQA.tar.gz'
    actual_sha = sha256(archive)
    if actual_sha != ARCHIVE_SHA:
        raise ValueError('Archive hash mismatch or incomplete download')
    inventory = []
    extracted = {}
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():
                raise ValueError(f'Unsafe archive member: {member.name}')
            inventory.append({'name': member.name, 'bytes': member.size, 'regular_file': member.isfile()})
            wanted = (member.name in ('TechQA/README.txt', 'TechQA/CDLA-Permissive-v1.0.pdf')
                      or member.name.startswith('TechQA/training_and_dev/')
                      or member.name.startswith('TechQA/validation/'))
            if not wanted or not member.isfile():
                continue
            if member.size > 512 * 1024 * 1024:
                raise ValueError(f'Unexpected benchmark file size: {member.name}')
            content = tar.extractfile(member).read()
            digest = hashlib.sha256(content).hexdigest()
            target = RAW.joinpath(*path.parts)
            if target.exists():
                if sha256(target) != digest:
                    raise ValueError(f'Refusing to overwrite changed input: {target}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as f:
                    f.write(content)
            extracted[member.name] = {'bytes': len(content), 'sha256': digest}
    manifest = {'revision': REVISION, 'url': f'https://huggingface.co/datasets/PrimeQA/TechQA/resolve/{REVISION}/TechQA.tar.gz',
                'archive_bytes': archive.stat().st_size, 'archive_sha256': actual_sha,
                'members': inventory, 'extracted_files': extracted,
                'license': 'Bundled README: CDLA-Permissive-1.0; hosting metadata: apache-2.0; retain bundled terms'}
    (RAW / 'source.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def audit_split(questions, documents, include_examples=False):
    ids = [q['QUESTION_ID'] for q in questions]
    groups = collections.defaultdict(list)
    for q in questions:
        groups[normalized_question(q)].append(q['QUESTION_ID'])
    positive = [q for q in questions if q['ANSWERABLE'] == 'Y']
    candidate_ids = {d for q in questions for d in q['DOC_IDS']}
    invalid_spans = [q['QUESTION_ID'] for q in positive if not span_valid(q, documents)]
    out_of_bounds = [q['QUESTION_ID'] for q in positive if q['DOCUMENT'] not in documents
                    or not (0 <= int(q['START_OFFSET']) < int(q['END_OFFSET']) <= len(documents[q['DOCUMENT']]['text']))]
    out = {
        'questions': len(questions), 'unique_question_ids': len(set(ids)),
        'answerable': len(positive), 'unanswerable': sum(q['ANSWERABLE'] == 'N' for q in questions),
        'schemas': dict(collections.Counter(','.join(sorted(q)) for q in questions)),
        'candidate_counts': dict(collections.Counter(len(q['DOC_IDS']) for q in questions)),
        'duplicate_candidate_id_questions': [q['QUESTION_ID'] for q in questions if len(q['DOC_IDS']) != len(set(q['DOC_IDS']))],
        'unique_candidate_documents': len(candidate_ids),
        'missing_candidate_documents': sorted(candidate_ids - documents.keys()),
        'gold_not_in_candidates': [q['QUESTION_ID'] for q in positive if q['DOCUMENT'] not in q['DOC_IDS']],
        'answer_text_mismatch_ids': invalid_spans,
        'out_of_bounds_span_ids': out_of_bounds,
        'empty_answer_string_ids': [q['QUESTION_ID'] for q in positive if not q['ANSWER']],
        'mismatches_equal_after_whitespace_normalization': sum(
            ' '.join(documents[q['DOCUMENT']]['text'][int(q['START_OFFSET']):int(q['END_OFFSET'])].split())
            == ' '.join(q['ANSWER'].split()) for q in positive
            if q['QUESTION_ID'] in invalid_spans and q['DOCUMENT'] in documents),
        'unanswerable_nonempty_label_fields': [q['QUESTION_ID'] for q in questions if q['ANSWERABLE'] == 'N'
            and any(q[k] != '-' for k in ['DOCUMENT', 'START_OFFSET', 'END_OFFSET', 'ANSWER'])],
        'question_whitespace_words': stats([len((q['QUESTION_TITLE'] + ' ' + q['QUESTION_TEXT']).split()) for q in questions]),
        'answer_whitespace_words': stats([len(q['ANSWER'].split()) for q in positive]),
        'answer_chars': stats([len(q['ANSWER']) for q in positive]),
        'normalized_duplicate_question_groups': [g for g in groups.values() if len(g) > 1],
        'duplicate_query_groups_with_mixed_answerability': sum(
            len({q['ANSWERABLE'] for q in questions if q['QUESTION_ID'] in g}) > 1
            for g in groups.values() if len(g) > 1),
        'gold_unique_documents': len({q['DOCUMENT'] for q in positive}),
        'body_equals_title': sum(q['QUESTION_TEXT'].strip() == q['QUESTION_TITLE'].strip() for q in questions),
    }
    if include_examples:
        out['duplicate_query_details'] = []
        q_by_id = {q['QUESTION_ID']: q for q in questions}
        for ids_in_group in out['normalized_duplicate_question_groups']:
            rows = [q_by_id[qid] for qid in ids_in_group]
            out['duplicate_query_details'].append({
                'ids': ids_in_group, 'answerable_labels': [q['ANSWERABLE'] for q in rows],
                'same_candidate_sets': all(set(q['DOC_IDS']) == set(rows[0]['DOC_IDS']) for q in rows),
                'positive_gold_present_in_negative_pool': [q['DOCUMENT'] in n['DOC_IDS']
                    for q in rows if q['ANSWERABLE'] == 'Y' for n in rows if n['ANSWERABLE'] == 'N'],
            })
        selections = [('first_answerable', positive[0]),
                      ('longest_answer', max(positive, key=lambda q: len(q['ANSWER'].split()))),
                      ('first_unanswerable', next(q for q in questions if q['ANSWERABLE'] == 'N'))]
        code = next((q for q in positive if re.search(r'\b[A-Z]{2,}\d{3,}[A-Z]?\b', normalized_question(q).upper())), None)
        if code:
            selections.append(('error_code', code))
        out['training_examples'] = []
        for reason, q in selections:
            row = {'reason': reason, 'question_id': q['QUESTION_ID'], 'title': q['QUESTION_TITLE'],
                   'body': q['QUESTION_TEXT'], 'answerable': q['ANSWERABLE'], 'document': q['DOCUMENT'],
                   'start': q['START_OFFSET'], 'end': q['END_OFFSET'], 'answer': q['ANSWER']}
            if q['ANSWERABLE'] == 'Y':
                d = documents[q['DOCUMENT']]
                row.update(document_title=d['title'], document_metadata=d.get('metadata'),
                           context_excerpt=d['text'][max(0,int(q['START_OFFSET'])-160):int(q['END_OFFSET'])+160])
            out['training_examples'].append(row)
    return out


def run_audit():
    base = RAW / 'TechQA/training_and_dev'
    train = json.loads((base / 'training_Q_A.json').read_text())
    dev = json.loads((base / 'dev_Q_A.json').read_text())
    docs = json.loads((base / 'training_dev_technotes.json').read_text())
    if not isinstance(docs, dict):
        raise ValueError('Unexpected document collection schema')
    source = json.loads((RAW / 'source.json').read_text())
    for name, entry in source['extracted_files'].items():
        if sha256(RAW.joinpath(*PurePosixPath(name).parts)) != entry['sha256']:
            raise ValueError(f'Extracted input changed: {name}')
    result = {'source_archive_sha256': source['archive_sha256'], 'audit_script_sha256': sha256(Path(__file__)),
              'scope': 'All benchmark documents; train semantic examples; dev structural checks only; no full companion semantic audit',
              'train': audit_split(train, docs, True), 'dev_structure': audit_split(dev, docs)}
    text_groups = collections.defaultdict(list)
    meta_keys = collections.Counter()
    for key, d in docs.items():
        if not isinstance(d.get('text'), str):
            raise ValueError(f'Invalid text: {key}')
        text_groups[hashlib.sha256(d['text'].encode()).hexdigest()].append(key)
        meta_keys.update(d.get('metadata', {}).keys())
    metadata = {}
    for key in meta_keys:
        values = [d.get('metadata', {}).get(key) for d in docs.values()]
        metadata[key] = {'present': meta_keys[key], 'nonempty': sum(v not in (None, '', [], {}) for v in values),
                         'types': dict(collections.Counter(type(v).__name__ for v in values))}
    duplicates = [g for g in text_groups.values() if len(g) > 1]
    word_counts = [len(d['text'].split()) for d in docs.values()]
    result['documents'] = {
        'count': len(docs), 'schemas': dict(collections.Counter(','.join(sorted(d)) for d in docs.values())),
        'id_key_mismatches': [key for key,d in docs.items() if d.get('id') != key],
        'empty_text': [key for key,d in docs.items() if not d['text'].strip()],
        'empty_title_count': sum(not d.get('title', '').strip() for d in docs.values()),
        'body_whitespace_words': stats(word_counts), 'body_characters': stats([len(d['text']) for d in docs.values()]),
        'body_over_512_words': sum(n > 512 for n in word_counts), 'body_over_1024_words': sum(n > 1024 for n in word_counts),
        'with_blank_line_boundaries': sum(bool(re.search(r'\n\s*\n', d['text'])) for d in docs.values()),
        'with_any_newline': sum('\n' in d['text'] for d in docs.values()),
        'with_html_content': sum(bool(d.get('content', '').strip()) for d in docs.values()),
        'text_with_tag_like_strings': sum(bool(re.search(r'</?[a-zA-Z][^>]{0,120}>', d['text'])) for d in docs.values()),
        'metadata': metadata, 'exact_body_duplicate_groups': duplicates,
        'duplicate_groups_count': len(duplicates), 'duplicate_documents_excess': sum(len(g)-1 for g in duplicates),
    }
    train_docs = {d for q in train for d in q['DOC_IDS']}
    dev_docs = {d for q in dev for d in q['DOC_IDS']}
    train_gold = {q['DOCUMENT'] for q in train if q['ANSWERABLE'] == 'Y'}
    dev_gold = {q['DOCUMENT'] for q in dev if q['ANSWERABLE'] == 'Y'}
    train_texts = {normalized_question(q) for q in train}
    result['overlap'] = {
        'train_dev_question_ids': sorted({q['QUESTION_ID'] for q in train} & {q['QUESTION_ID'] for q in dev}),
        'train_dev_normalized_question_matches': sum(normalized_question(q) in train_texts for q in dev),
        'candidate_doc_intersection': len(train_docs & dev_docs), 'candidate_doc_union': len(train_docs | dev_docs),
        'gold_doc_intersection': len(train_gold & dev_gold),
        'same_body_groups_with_train_and_dev_gold': sum(bool(set(g)&train_gold) and bool(set(g)&dev_gold) for g in duplicates),
        'dev_question_ids_matching_train_text': [q['QUESTION_ID'] for q in dev if normalized_question(q) in train_texts],
        'train_question_ids_matching_dev_text': [q['QUESTION_ID'] for q in train if normalized_question(q) in {normalized_question(d) for d in dev}],
    }
    sections = json.loads((base / 'training_dev_technotes.sections.json').read_text())
    invalid_sections = []
    section_counts, section_words = [], []
    discontinuous_docs = []
    for key, d in sections.items():
        section_counts.append(len(d['sections']))
        previous_end = 0
        continuous = True
        for s in d['sections']:
            start, end = s['start'], s['end']
            section_words.append(len(s['text'].split()))
            if not (0 <= start <= end <= len(d['text']) and d['text'][start:end] == s['text']):
                invalid_sections.append({'document': key, 'section_id': s['id'], 'start': start, 'end': end})
            if start != previous_end:
                continuous = False
            previous_end = end
        if not continuous or previous_end != len(d['text']):
            discontinuous_docs.append(key)
    result['sections'] = {
        'documents': len(sections), 'missing_document_ids': sorted(docs.keys() - sections.keys()),
        'body_exact_match_documents': sum(k in docs and d['text'] == docs[k]['text'] for k,d in sections.items()),
        'invalid_sections': invalid_sections, 'discontinuous_or_incomplete_documents': discontinuous_docs,
        'sections_per_document': stats(section_counts), 'section_whitespace_words': stats(section_words),
        'empty_or_whitespace_only_sections': sum(not s['text'].strip() for d in sections.values() for s in d['sections']),
        'section_heading_counts': dict(collections.Counter(s['id'] for d in sections.values() for s in d['sections'])),
        'train_answers_fully_inside_one_section': sum(any(s['start'] <= int(q['START_OFFSET'])
            and s['end'] >= int(q['END_OFFSET']) for s in sections[q['DOCUMENT']]['sections'])
            for q in train if q['ANSWERABLE'] == 'Y'),
    }
    validation_path = RAW / 'TechQA/validation/validation_reference.json'
    if validation_path.exists():
        val = json.loads(validation_path.read_text())
        if isinstance(val, dict):
            val = list(val.values())
        result['validation'] = {'questions': len(val),
                                'ids_in_dev': sum(q['QUESTION_ID'] in {r['QUESTION_ID'] for r in dev} for q in val),
                                'question_texts_matching_dev': sum(normalized_question(q) in {normalized_question(d) for d in dev} for q in val)}
    output = ROOT / 'data/audit_techqa.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('train','documents')}, indent=2))
    print('Saved', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract', action='store_true', help='Verify complete archive and extract benchmark files only')
    parser.add_argument('--self-check', action='store_true', help='Run artificial checks without data')
    args = parser.parse_args()
    self_check()
    if not args.self_check:
        if args.extract:
            extract_benchmark()
        run_audit()
