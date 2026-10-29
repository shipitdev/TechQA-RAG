import re
import unittest
from types import SimpleNamespace
from scripts.chunking import build_chunks, answer_coverage


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        offsets = [m.span() for m in re.finditer(r'\S+', text)]
        return SimpleNamespace(offsets=offsets, ids=list(range(len(offsets))))


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer()

    def test_unicode_offsets_roundtrip_and_token_limit(self):
        text = '  α café <domain>\nrun command now.\n\nKeep backup before restarting.  '
        for method in ['recursive','section']:
            sections = [{'start':0,'end':len(text),'id':'body'}]
            chunks = build_chunks(text, sections, self.tokenizer, method, 6)
            self.assertTrue(all(c['body_tokens'] <= 6 for c in chunks))
            self.assertEqual(chunks[0]['start'],0)
            self.assertEqual(chunks[-1]['end'],len(text))
            covered = set()
            for c in chunks:
                self.assertGreater(c['end'],c['start'])
                covered.update(range(c['start'],c['end']))
            self.assertEqual(covered,set(range(len(text))))
            self.assertEqual(build_chunks(text, sections, self.tokenizer, method, 6),chunks)

    def test_sections_keep_useful_boundary_and_merge_small_neighbors(self):
        text = 'a b c d e f g h i j k l'
        ends = [4,12,len(text)]
        sections = [{'start':s,'end':e,'id':str(i)} for i,(s,e) in enumerate(zip([0]+ends[:-1],ends))]
        chunks = build_chunks(text, sections, self.tokenizer,'section',8,0)
        self.assertEqual(chunks[0]['end'],12)
        self.assertEqual(chunks[0]['section_ids'],['0','1'])

    def test_oversized_section_makes_progress_and_overlap(self):
        text = ' '.join('word'+str(i) for i in range(50))
        sections = [{'start':0,'end':len(text),'id':'long'}]
        chunks = build_chunks(text,sections,self.tokenizer,'section',10,.2)
        self.assertLess(len(chunks),10)
        self.assertTrue(all(a['start']<b['start']<=a['end'] for a,b in zip(chunks,chunks[1:])))
        self.assertTrue(any(b['start']<a['end'] for a,b in zip(chunks,chunks[1:])))

    def test_short_and_empty_text(self):
        self.assertEqual(build_chunks('  ',[],self.tokenizer,'recursive',10),[])
        chunks=build_chunks('one two',[],self.tokenizer,'recursive',10)
        self.assertEqual([(c['start'],c['end']) for c in chunks],[(0,7)])

    def test_coverage_distinguishes_union_from_one_complete_chunk(self):
        chunks=[{'start':0,'end':6},{'start':4,'end':10}]
        score=answer_coverage(2,9,chunks)
        self.assertFalse(score['full_in_one_chunk'])
        self.assertEqual(score['best_single_fraction'],5/7)
        self.assertEqual(score['union_fraction'],1)
        self.assertEqual(answer_coverage(2,9,[])['union_fraction'],0)

    def test_exact_answer_endpoints_count_as_complete(self):
        score = answer_coverage(2,9,[{'start':2,'end':9}])
        self.assertTrue(score['full_in_one_chunk'])
        self.assertEqual(score['best_single_fraction'],1)
        self.assertEqual(score['union_fraction'],1)

    def test_answer_longer_than_cap_is_not_called_fully_preserved(self):
        text = ' '.join(['step']*30)
        chunks = build_chunks(text,[],self.tokenizer,'recursive',5)
        score = answer_coverage(0,len(text),chunks)
        self.assertFalse(score['full_in_one_chunk'])
        self.assertLess(score['best_single_fraction'],1)
        self.assertEqual(score['union_fraction'],1)

    def test_bad_settings_or_sections_fail(self):
        for size,overlap in [(0,.15),(10,1),(10,-.1)]:
            with self.assertRaises(ValueError):
                build_chunks('word',[],self.tokenizer,'recursive',size,overlap)
        with self.assertRaises(ValueError):
            build_chunks('word',[{'start':1,'end':4,'id':'bad'}],self.tokenizer,'section',10)
        with self.assertRaises(ValueError):
            answer_coverage(5,5,[])
