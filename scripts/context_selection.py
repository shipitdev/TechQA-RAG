"""Metadata-aware ranking and evidence packing for retrieved chunks."""
import math
import numbers
import re

from scripts.retrieval import rrf


_METADATA_LABELS = re.compile(
    r"\b(?:product\s*(?:alias(?:es)?|synonyms?)|aliases?|synonyms?|"
    r"document\s+information|segment|product|component|platform|version|edition)\b",
    re.I,
)


def _chunk(chunks, chunk_id):
    return chunks[chunk_id]


def _body(chunk, docs):
    doc = docs[chunk['document_id']]
    return doc['text'][chunk['start']:chunk['end']]


def is_supporting_metadata(chunk, docs):
    """True only for recognizable label/value metadata or alias tables."""
    body = _body(chunk, docs).strip()
    heading = chunk.get('section', '')
    if not body:
        return False
    embedded_heading = re.match(r'^#{1,6}\s+(.+?)\s*\n(.*)$', body, re.S)
    heading_text = embedded_heading.group(1).strip() if embedded_heading else heading
    heading_body = embedded_heading.group(2).strip() if embedded_heading else body
    # Alias headings in these sources often contain only a short list of names.
    if re.fullmatch(r'\s*(?:product\s*)?(?:alias(?:es)?(?:\s*/\s*synonym(?:s)?)?|synonym(?:s)?)\s*', heading_text, re.I):
        return len(heading_body.split()) <= 40 and not re.search(
            r'\b(cve|vulnerabilit\w*|install|upgrade|configure|command|error|fix|workaround)\b', heading_body, re.I
        ) and not re.search(r'[.!?]', heading_body)
    if re.fullmatch(r'\s*document\s+information\s*', heading_text, re.I):
        if len(heading_body.split()) <= 40 and bool(
                re.search(r'(?m)^\s*(?:[-*]\s*)?\w[\w /-]*:\s*\S+', heading_body)):
            return True
    if not _METADATA_LABELS.search(heading + '\n' + body):
        return False
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    table = sum(line.startswith('|') for line in lines)
    # Keep technical tables and mixed prose available as primary evidence.
    if table >= 2:
        header = ' '.join(lines[:3]).lower()
        aliases = bool(re.search(r'\b(alias|synonym)\b', lines[0], re.I))
        header_cells = [cell.strip().strip(' :').lower() for cell in lines[0].strip('|').split('|')]
        if re.search(r'\b(cve|vulnerabilit\w*|fix|workaround|command|setting|procedure|resolution|symptom)\b', header):
            return False
        identity_columns = {'segment', 'product', 'component', 'platform', 'version', 'edition'}
        labels = set(header_cells) == identity_columns
        document_info = ('document information' in heading_text.lower() or
                         'document information' in body.lower()) and set(header_cells) == {'field', 'value'}
        if document_info:
            fields = [line.strip().strip('|').split('|')[0].strip().lower() for line in lines[2:]]
            document_info = bool(fields) and all(field in {
                'title', 'author', 'date', 'published', 'publication date', 'revision', 'document id',
                'product', 'component', 'platform', 'version', 'edition', 'segment'
            } for field in fields)
        return aliases or labels or document_info
    # A heading by itself cannot classify a long prose passage as metadata.
    return False


def rerank_candidates(ranked_ids, chunks, docs, ce_scores, depth=20):
    if depth < 1:
        raise ValueError('depth must be positive')
    dense = list(dict.fromkeys(ranked_ids))[:100]
    primary = [i for i in dense if not is_supporting_metadata(_chunk(chunks, i), docs)]
    metadata = [i for i in dense if is_supporting_metadata(_chunk(chunks, i), docs)]
    cross = primary[:depth]
    scores = []
    for i in cross:
        try:
            score = ce_scores[i] if isinstance(ce_scores, dict) else ce_scores[len(scores)]
        except (KeyError, IndexError, TypeError):
            raise ValueError(f'Missing cross-encoder score for candidate {i}') from None
        if not isinstance(score, numbers.Real) or not math.isfinite(score):
            raise ValueError(f'Invalid cross-encoder score for candidate {i}')
        scores.append((i, float(score)))
    ce_order = [i for i, _ in sorted(scores, key=lambda pair: -pair[1])]
    ce_with_fallback = ce_order + primary[len(cross):]
    merged, _ = rrf(primary, ce_with_fallback, depth=100, k=60)
    return merged + metadata, {'dense_candidates': len(dense), 'primary_candidates': len(primary),
                               'metadata_candidates': len(metadata), 'reranked': len(cross)}


def pack_evidence(ranked_ids, chunks, docs, tokenizer, budget=4096, per_doc=5, max_docs=8,
                  skip_metadata=True, include_notes=True):
    if budget < 1 or per_doc < 1 or max_docs < 1:
        raise ValueError('budget, per_doc and max_docs must be positive')
    passages, dropped = [], {'duplicate': 0, 'metadata': 0, 'quota': 0, 'budget': 0}

    def render(rows):
        return '\n\n'.join(f"[{p['document_id']}:{p['start']}-{p['end']}]\n" +
                            docs[p['document_id']]['text'][p['start']:p['end']] for p in rows)

    primary_counts = {}
    for chunk_id in ranked_ids:
        c = _chunk(chunks, chunk_id)
        pid, start, end = c['document_id'], c['start'], c['end']
        metadata = skip_metadata and is_supporting_metadata(c, docs)
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(docs[pid]['text']):
            raise ValueError(f'Invalid source offsets for candidate {chunk_id}')
        if metadata and skip_metadata:
            dropped['metadata'] += 1
            continue
        same = [p for p in passages if p['document_id'] == pid]
        overlap = [p for p in same if p['start'] <= end and start <= p['end']]
        if any(p['start'] <= start and p['end'] >= end for p in same):
            dropped['duplicate'] += 1
            continue
        if not metadata and primary_counts.get(pid, 0) >= per_doc:
            dropped['quota'] += 1
            continue
        if pid not in {p['document_id'] for p in passages} and len({p['document_id'] for p in passages}) >= max_docs:
            dropped['quota'] += 1
            continue
        # Merge every overlap, including overlaps joined transitively by this chunk.
        merged = {'document_id': pid, 'start': min([start] + [p['start'] for p in overlap]),
                  'end': max([end] + [p['end'] for p in overlap])}
        if not overlap and c.get('source_start') is not None and c.get('source_end') is not None:
            merged.update(source_start=c['source_start'], source_end=c['source_end'])
        proposal = list(passages)
        if overlap:
            insert_at = min(passages.index(p) for p in overlap)
            proposal = [p for p in passages if p not in overlap]
            proposal.insert(insert_at, merged)
        else:
            proposal.append(merged)
        note_start, note_end = c.get('following_note_start'), c.get('following_note_end')
        if (note_start is None) != (note_end is None):
            raise ValueError(f'Incomplete note offsets for candidate {chunk_id}')
        if include_notes and note_start is not None and not any(
                p['document_id'] == pid and p['start'] <= note_start and p['end'] >= note_end for p in proposal):
            if not isinstance(note_start, int) or not isinstance(note_end, int) or not 0 <= note_start < note_end <= len(docs[pid]['text']):
                raise ValueError(f'Invalid note offsets for candidate {chunk_id}')
            note = {'document_id': pid, 'start': note_start, 'end': note_end}
            note_overlap = [p for p in proposal if p['document_id'] == pid and
                            p['start'] <= note_end and note_start <= p['end']]
            note = {'document_id': pid,
                    'start': min([note_start] + [p['start'] for p in note_overlap]),
                    'end': max([note_end] + [p['end'] for p in note_overlap])}
            note_proposal = [p for p in proposal if p not in note_overlap]
            note_proposal.insert(min((proposal.index(p) for p in note_overlap), default=len(proposal)), note)
            if len(tokenizer.encode(render(note_proposal), add_special_tokens=False).ids) <= budget:
                proposal = note_proposal
        if len(tokenizer.encode(render(proposal), add_special_tokens=False).ids) > budget:
            dropped['budget'] += 1
            continue
        passages = proposal
        if not metadata:
            primary_counts[pid] = primary_counts.get(pid, 0) + 1

    text = render(passages)
    tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
    return {'passages': passages, 'text': text, 'tokens': tokens, 'diagnostics': dropped}
