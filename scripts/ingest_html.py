"""Convert TechQA HTML to a resumable, structured, searchable corpus with Docling."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import time
from pathlib import Path

from scripts.audit_techqa import ROOT, RAW, sha256, stats
from scripts.chunking import build_chunks

SOURCE = RAW / 'TechQA/training_and_dev/training_dev_technotes.json'
TOKENIZER_DIR = ROOT / 'data/tokenizers/bge_wordpiece'
OUTPUT = ROOT / 'data/runs/html_v1'
CHUNKS_OUTPUT = OUTPUT / 'structured_512'
CACHE = ROOT / 'data/ingestion/html_v1'
SPLIT = ROOT / 'data/splits/techqa_v1.json'


def _clean_markdown(markdown):
    markdown = markdown.replace('\r\n', '\n').replace('\r', '\n')
    markdown = re.sub(r'\n{3,}', '\n\n', markdown).strip()
    return markdown + ('\n' if markdown else '')


def _repeat_table_headers(markdown, tokenizer, budget=512):
    """Keep compact Markdown tables intact; split oversized tables into header-repeating row groups."""
    lines = markdown.splitlines()
    out, i = [], 0
    while i < len(lines):
        if i + 1 >= len(lines) or not lines[i].lstrip().startswith('|') or not re.match(r'^\s*\|?\s*:?-+', lines[i + 1]):
            out.append(lines[i]); i += 1; continue
        table = [lines[i], lines[i + 1]]
        i += 2
        while i < len(lines) and lines[i].lstrip().startswith('|'):
            table.append(lines[i]); i += 1
        if len(tokenizer.encode('\n'.join(table), add_special_tokens=False).ids) <= budget:
            out.extend(table)
            continue
        header = table[:2]
        rows, group, groups = table[2:], [], []
        for row in rows:
            candidate = header + group + [row]
            if group and len(tokenizer.encode('\n'.join(candidate), add_special_tokens=False).ids) > budget:
                groups.append(group)
                group = [row]
            else:
                group.append(row)
        if group:
            groups.append(group)
        for group_number, group in enumerate(groups):
            if group_number:
                out.extend(['', '[Table continued; repeated header.]'])
            out.extend(header + group)
        if i < len(lines):
            out.append('')
    return _clean_markdown('\n'.join(out))


def _convert(html, converter):
    # Docling stays optional for imports and tests; conversion uses only its HTML backend.
    from docling.datamodel.base_models import InputFormat
    result = converter.convert_string(html, format=InputFormat.HTML)
    return _clean_markdown(result.document.export_to_markdown(
        compact_tables=True, escape_underscores=False, escape_html=False))


def _source_interval(body, original):
    if not body:
        return None, None
    start = original.find(body)
    if start >= 0 and original.find(body, start + 1) < 0:
        return start, start + len(body)
    return None, None


def _section_at(markdown, offset):
    headings = list(re.finditer(r'(?m)^#{1,6}\s+(.+?)\s*#*$', markdown[:offset]))
    return headings[-1].group(1) if headings else ''


def _blocks(markdown):
    lines = markdown.splitlines(keepends=True)
    starts, cursor = [], 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)
    result, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1; continue
        start = i
        if lines[i].lstrip().startswith('```'):
            i += 1
            while i < len(lines):
                closing = lines[i].lstrip().startswith('```')
                i += 1
                if closing:
                    break
            kind = 'code'
        elif i + 1 < len(lines) and lines[i].lstrip().startswith('|') and re.match(r'^\s*\|?\s*:?-+', lines[i + 1]):
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith('|'):
                i += 1
            kind = 'table'
        else:
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith('```'):
                if i + 1 < len(lines) and lines[i].lstrip().startswith('|') and re.match(r'^\s*\|?\s*:?-+', lines[i + 1]):
                    break
                i += 1
            kind = 'prose'
        end_line = i - 1
        result.append({'start': starts[start], 'end': starts[end_line] + len(lines[end_line]), 'kind': kind})
    return result


def _structured_chunks(markdown, tokenizer, budget=512):
    """Chunk prose recursively; keep each complete table group and code block atomic where possible."""
    blocks, chunks, prose = _blocks(markdown), [], None

    def flush_prose():
        nonlocal prose
        if prose:
            left, right = prose
            for chunk in build_chunks(markdown[left:right], [], tokenizer, 'recursive', budget):
                chunks.append(dict(chunk, start=chunk['start'] + left, end=chunk['end'] + left, structure='prose'))
            prose = None

    i = 0
    while i < len(blocks):
        block = blocks[i]
        if block['kind'] == 'prose':
            prose = (block['start'], block['end']) if prose is None else (prose[0], block['end'])
            i += 1
            continue
        flush_prose()
        body = markdown[block['start']:block['end']]
        size = len(tokenizer.encode(body, add_special_tokens=False).ids)
        note = None
        if block['kind'] == 'table' and i + 1 < len(blocks) and blocks[i + 1]['kind'] == 'prose':
            candidate = blocks[i + 1]
            note_text = markdown[candidate['start']:candidate['end']]
            if re.match(r'(?is)^\s*(?:note|footnote|requires|important|warning|exception)\b', note_text):
                note = candidate
        chunk_end = block['end']
        note_included = False
        if note and size + len(tokenizer.encode(markdown[block['end']:note['end']], add_special_tokens=False).ids) <= budget:
            chunk_end, note_included = note['end'], True
        if size <= budget:
            chunk_body_tokens = len(tokenizer.encode(markdown[block['start']:chunk_end], add_special_tokens=False).ids)
            chunks.append({'start': block['start'], 'end': chunk_end, 'body_tokens': chunk_body_tokens,
                           'section_ids': [], 'structure': block['kind'] + ('_with_note' if note_included else ''),
                           'note_included': note_included,
                           'following_note_start': note['start'] if note and not note_included else None,
                           'following_note_end': note['end'] if note and not note_included else None})
            if note_included:
                i += 1
        else:
            # Oversized indivisible structures are split only as a flagged fallback.
            for chunk in build_chunks(body, [], tokenizer, 'recursive', budget):
                chunks.append(dict(chunk, start=chunk['start'] + block['start'],
                                   end=chunk['end'] + block['start'],
                                   structure=block['kind'] + '_split', note_included=False,
                                   following_note_start=note['start'] if note else None,
                                   following_note_end=note['end'] if note else None))
        i += 1
    flush_prose()
    return sorted(chunks, key=lambda c: c['start'])


def _doc_record(docid, source, markdown):
    return {'document_id': docid, 'text': markdown, 'title': source.get('title', ''),
            'metadata': source.get('metadata', {})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, help='convert only the first N sorted documents')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    import tokenizers
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    if tokenizers.__version__ != '0.23.2':
        raise ValueError('Use requirements-chunking.txt for the recorded tokenizer implementation')
    tokenizer_record = json.loads((TOKENIZER_DIR / 'source.json').read_text())
    if sha256(TOKENIZER_DIR / 'tokenizer.json') != tokenizer_record['tokenizer_sha256']:
        raise ValueError('Tokenizer fingerprint changed')
    tokenizer = tokenizers.Tokenizer.from_file(str(TOKENIZER_DIR / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    converter = DocumentConverter(allowed_formats=[InputFormat.HTML])
    docs = json.loads(SOURCE.read_text())
    selected = sorted(docs)[:args.limit]
    source_hash = sha256(SOURCE)
    qa_hash = sha256(RAW / 'TechQA/training_and_dev/training_Q_A.json')
    source_manifest = json.loads((RAW / 'source.json').read_text())
    expected_source = source_manifest['extracted_files']['TechQA/training_and_dev/training_dev_technotes.json']['sha256']
    expected_qa = source_manifest['extracted_files']['TechQA/training_and_dev/training_Q_A.json']['sha256']
    if source_hash != expected_source or qa_hash != expected_qa:
        raise ValueError('TechQA source fingerprint differs from the recorded extraction')
    split_hash = sha256(SPLIT)
    script_hash = sha256(Path(__file__))
    docling_version = importlib.metadata.version('docling')
    cache_fingerprint = {'source_hash': source_hash, 'tokenizer_sha256': tokenizer_record['tokenizer_sha256'],
                         'script_sha256': script_hash, 'docling_version': docling_version, 'budget': 512}
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records, all_chunks, counts, sizes = {}, [], [], []
    started = time.perf_counter()
    for n, docid in enumerate(selected, 1):
        raw = docs[docid]
        cache_path = CACHE / (hashlib.sha256(docid.encode()).hexdigest() + '.json')
        html_hash = hashlib.sha256(raw.get('content', '').encode()).hexdigest()
        cached = json.loads(cache_path.read_text()) if cache_path.exists() else None
        if cached and all(cached.get(k) == v for k, v in cache_fingerprint.items()) and cached.get('html_hash') == html_hash:
            markdown = cached['text']
        else:
            markdown = _convert(raw.get('content', ''), converter)
            markdown = _repeat_table_headers(markdown, tokenizer)
            temp = cache_path.with_suffix('.tmp')
            temp.write_text(json.dumps({**cache_fingerprint, 'html_hash': html_hash, 'text': markdown}, ensure_ascii=False))
            os.replace(temp, cache_path)
        empty_fallback = not markdown.strip()
        if empty_fallback:
            markdown = _clean_markdown(raw.get('text', ''))
        record = _doc_record(docid, raw, markdown)
        record['html_empty_fallback'] = empty_fallback
        records[docid] = record
        chunks = _structured_chunks(markdown, tokenizer, 512)
        counts.append(len(chunks))
        for chunk in chunks:
            body = markdown[chunk['start']:chunk['end']]
            src_start, src_end = _source_interval(body, raw.get('text', ''))
            section = _section_at(markdown, chunk['start'])
            all_chunks.append(dict(chunk, chunk_id=f'html_v1:{docid}:{chunk["start"]}',
                                   document_id=docid, source_start=src_start, source_end=src_end,
                                   source_coordinate='prepared_view', structure=chunk['structure'],
                                   empty_html_fallback=empty_fallback,
                                   search_text='\n'.join(filter(None, [record['title'],
                                       record['metadata'].get('productName', ''), section, body])),
                                   section=section))
            sizes.append(chunk['body_tokens'])
        if n % 500 == 0:
            print(f'{n}/{len(selected)} documents', flush=True)
    # Replace final views atomically so an interrupted run leaves its per-document cache usable.
    (OUTPUT / 'documents.json.tmp').write_text(json.dumps(records, ensure_ascii=False))
    os.replace(OUTPUT / 'documents.json.tmp', OUTPUT / 'documents.json')
    CHUNKS_OUTPUT.mkdir(parents=True, exist_ok=True)
    with (CHUNKS_OUTPUT / 'chunks.jsonl.tmp').open('w') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    os.replace(CHUNKS_OUTPUT / 'chunks.jsonl.tmp', CHUNKS_OUTPUT / 'chunks.jsonl')
    config = {'version': 'html_v1', 'source_hashes': {'training_dev_technotes.json': source_hash,
                                                       'training_Q_A.json': qa_hash},
              'split_sha256': split_hash,
              'tokenizer_sha256': tokenizer_record['tokenizer_sha256'],
              'tokenizer': tokenizer_record, 'chunk_method': 'structured blocks with recursive prose', 'body_token_budget': 512,
              'overlap_fraction': .15, 'source_coordinate': 'prepared_view; source_start/end only exact raw-text matches',
              'representation': 'structured_html_v1',
              'code_sha256': {name: sha256(ROOT / 'scripts' / name) for name in ['chunking.py', 'ingest_html.py']},
              'software': {'docling': docling_version, 'docling-core': importlib.metadata.version('docling-core'),
                           'tokenizers': tokenizers.__version__},
              'qa_labels_loaded': False, 'document_count': len(records),
              'python': platform.python_version(), 'limit': args.limit}
    (OUTPUT / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    metrics = {'documents': len(records), 'chunk_count': len(all_chunks),
               'chunks_per_document': stats(counts), 'body_tokens_per_chunk': stats(sizes),
               'elapsed_seconds': time.perf_counter() - started,
               'chunks_sha256': sha256(CHUNKS_OUTPUT / 'chunks.jsonl')}
    (CHUNKS_OUTPUT / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
