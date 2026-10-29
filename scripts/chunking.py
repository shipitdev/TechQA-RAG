"""Source-preserving recursive/section chunking; no embedding inference."""
from bisect import bisect_right
import math
import re


def build_chunks(text, sections, tokenizer, method, max_tokens, overlap_fraction=.15):
    if method not in ('recursive','section') or max_tokens < 1 or not 0 <= overlap_fraction < 1:
        raise ValueError('Invalid chunk method, token limit or overlap')
    if method == 'section':
        previous = 0
        for section in sections:
            if section['start'] != previous or not previous <= section['end'] <= len(text):
                raise ValueError('Sections must partition the original body')
            previous = section['end']
        if sections and previous != len(text):
            raise ValueError('Sections do not cover the body')
    encoding = tokenizer.encode(text, add_special_tokens=False)
    offsets = encoding.offsets
    if not offsets:
        return []
    starts = [s for s,e in offsets]
    ends = [e for s,e in offsets]
    if any(s < 0 or e < s or e > len(text) for s,e in offsets) or starts != sorted(starts):
        raise ValueError('Unexpected tokenizer character offsets')
    def token_boundaries(chars):
        return sorted({bisect_right(ends,c) for c in chars})
    preferences = [token_boundaries(m.end() for m in re.finditer(r'\n[ \t\r]*\n+',text)),
                   token_boundaries(m.end() for m in re.finditer(r'[.!?](?=\s)',text)),
                   token_boundaries(m.end() for m in re.finditer(r'\n',text))]
    if method == 'section':
        preferences.insert(0,token_boundaries(s['end'] for s in sections))
    chunks = []
    start = 0
    overlap = math.floor(max_tokens*overlap_fraction)
    while start < len(offsets):
        end = min(start+max_tokens,len(offsets))
        if end < len(offsets):
            minimum = start+max(overlap+1, math.ceil(max_tokens/2))
            for boundaries in preferences:
                i = bisect_right(boundaries,end)-1
                if i >= 0 and boundaries[i] >= minimum:
                    end = boundaries[i]
                    break
        char_start = 0 if start == 0 else starts[start]
        while end > start:
            char_end = len(text) if end == len(offsets) else starts[end]
            token_count = len(tokenizer.encode(text[char_start:char_end],add_special_tokens=False).ids)
            if token_count <= max_tokens:
                break
            end -= 1
        if end == start:
            raise ValueError('Cannot fit a single source token under the requested limit')
        chunks.append({'start':char_start,'end':char_end,'body_tokens':token_count,
                       'section_ids':[s['id'] for s in sections
                                      if s['start'] < char_end and s['end'] > char_start]})
        if end == len(offsets):
            break
        start = max(start+1,end-overlap)
    return chunks


def answer_coverage(start, end, chunks):
    if not 0 <= start < end:
        raise ValueError('Expected a nonempty answer interval')
    intersections = sorted((max(start,c['start']),min(end,c['end'])) for c in chunks
                           if c['start'] < end and c['end'] > start)
    covered = 0
    previous_end = start
    for left,right in intersections:
        covered += max(0,right-max(left,previous_end))
        previous_end = max(previous_end,right)
    return {'full_in_one_chunk':any(c['start']<=start and c['end']>=end for c in chunks),
            'best_single_fraction':max((b-a for a,b in intersections),default=0)/(end-start),
            'union_fraction':covered/(end-start)}
