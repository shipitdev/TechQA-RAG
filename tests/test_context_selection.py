import unittest

from scripts.context_selection import is_supporting_metadata, pack_evidence, rerank_candidates


class Encoding:
    def __init__(self, text):
        self.ids = text.split()


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return Encoding(text)


def fixture(body, section=''):
    doc = {'text': body}
    return {'document_id': 'd', 'start': 0, 'end': len(body), 'section': section}, {'d': doc}


class ContextSelectionTests(unittest.TestCase):
    def test_metadata_labels_do_not_hide_technical_tables_or_long_body(self):
        alias, docs = fixture('| Product Alias | Canonical Product |\n| --- | --- |\n| Foo | Bar |', 'Product Aliases')
        self.assertTrue(is_supporting_metadata(alias, docs))
        support, docs = fixture('| Component | Vulnerability | Fix |\n| --- | --- | --- |\n| X | CVE-1 | Upgrade |', 'Product Information')
        self.assertFalse(is_supporting_metadata(support, docs))
        support_matrix, docs = fixture('| Product | Version |\n| --- | --- |\n| Widget | 2.1 |', 'Supported Products')
        self.assertFalse(is_supporting_metadata(support_matrix, docs))
        identity, docs = fixture('| Segment | Product | Component | Platform | Version | Edition |\n'
                                 '| --- | --- | --- | --- | --- | --- |\n| A | B | C | D | 1 | E |')
        self.assertTrue(is_supporting_metadata(identity, docs))
        body, docs = fixture('Product release notes explain the security impact and required patch in detail. ' * 30,
                             'Document Information')
        self.assertFalse(is_supporting_metadata(body, docs))

    def test_document_information_requires_metadata_fields(self):
        fields, docs = fixture('| Field | Value |\n| --- | --- |\n| Title | Setup Guide |\n| Version | 2 |',
                               'Document Information')
        self.assertTrue(is_supporting_metadata(fields, docs))
        technical, docs = fixture('| Field | Value |\n| --- | --- |\n| Procedure | Run command |',
                                  'Document Information')
        self.assertFalse(is_supporting_metadata(technical, docs))

    def test_short_plain_alias_section_is_metadata(self):
        chunk, docs = fixture('WMB IIB', 'Product Alias')
        self.assertTrue(is_supporting_metadata(chunk, docs))
        embedded, docs = fixture('# Product Alias\nWMB IIB')
        self.assertTrue(is_supporting_metadata(embedded, docs))
        slash_heading, docs = fixture('# Product Alias/Synonym\n\nWMB IIB\n')
        self.assertTrue(is_supporting_metadata(slash_heading, docs))
        mixed, docs = fixture('WMB IIB requires upgrading the service to fix CVE-1.', 'Product Alias')
        self.assertFalse(is_supporting_metadata(mixed, docs))
        inherited, docs = fixture('# Technical Details\nUse the product to apply the critical security fix.',
                                  'Product Alias')
        self.assertFalse(is_supporting_metadata(inherited, docs))

    def test_rerank_fails_on_missing_and_nonfinite_scores(self):
        c, docs = fixture('Actual technical guidance and configuration steps.')
        with self.assertRaisesRegex(ValueError, 'Missing cross-encoder'):
            rerank_candidates([0], [c], docs, [])
        with self.assertRaisesRegex(ValueError, 'Invalid cross-encoder'):
            rerank_candidates([0], [c], docs, [float('nan')])

    def test_rerank_places_metadata_after_scored_primary(self):
        primary, docs = fixture('Use this technical procedure.')
        meta = {'document_id': 'm', 'start': 0, 'end': 47, 'section': 'Product Aliases'}
        docs['m'] = {'text': '| Product Alias | Name |\n| --- | --- |\n| Foo | Bar |'}
        meta['end'] = len(docs['m']['text'])
        ids, stats = rerank_candidates([0, 1], [primary, meta], docs, [1.0])
        self.assertEqual(ids, [0, 1])
        self.assertEqual(stats['metadata_candidates'], 1)

    def test_pack_merges_transitive_overlaps_and_keeps_note_without_neighbors(self):
        text = 'abcdefghij NOTE: qualification. unrelated neighbor.'
        docs = {'d': {'text': text}}
        chunks = [
            {'document_id': 'd', 'start': 0, 'end': 5},
            {'document_id': 'd', 'start': 4, 'end': 9},
            {'document_id': 'd', 'start': 9, 'end': 31,
             'following_note_start': 11, 'following_note_end': 31},
            {'document_id': 'd', 'start': 32, 'end': len(text)},
        ]
        result = pack_evidence([0, 1, 2], chunks, docs, Tokenizer(), budget=50)
        self.assertEqual(len(result['passages']), 1)
        self.assertEqual((result['passages'][0]['start'], result['passages'][0]['end']), (0, 31))
        self.assertNotIn('unrelated neighbor', result['text'])

    def test_pack_skips_metadata_and_keeps_relevance_order(self):
        docs = {'z': {'text': 'A relevant primary passage.'}, 'a': {'text': 'WMB IIB'}}
        chunks = [
            {'document_id': 'z', 'start': 0, 'end': len(docs['z']['text'])},
            {'document_id': 'a', 'start': 0, 'end': len(docs['a']['text']), 'section': 'Product Alias'},
        ]
        result = pack_evidence([0, 1], chunks, docs, Tokenizer(), budget=100)
        self.assertEqual([p['document_id'] for p in result['passages']], ['z'])
        self.assertEqual(result['diagnostics']['metadata'], 1)

    def test_offsets_validated_and_note_switch_is_explicit(self):
        docs = {'d': {'text': 'body NOTE neighbor'}}
        chunks = [{'document_id': 'd', 'start': 0, 'end': 4,
                   'following_note_start': 5, 'following_note_end': 9},
                  {'document_id': 'd', 'start': 10, 'end': 18}]
        result = pack_evidence([0], chunks, docs, Tokenizer(), include_notes=False)
        self.assertEqual(result['passages'], [{'document_id': 'd', 'start': 0, 'end': 4}])
        with self.assertRaisesRegex(ValueError, 'Invalid source offsets'):
            pack_evidence([0], [{'document_id': 'd', 'start': 0, 'end': 99}], docs, Tokenizer())

    def test_overlap_counts_toward_quota_without_fabricating_raw_offsets(self):
        docs = {'d': {'text': 'abcdefghij klmnop'}}
        chunks = [
            {'document_id': 'd', 'start': 0, 'end': 6, 'source_start': 100, 'source_end': 106},
            {'document_id': 'd', 'start': 5, 'end': 10, 'source_start': 105, 'source_end': 110},
            {'document_id': 'd', 'start': 11, 'end': 17},
        ]
        result = pack_evidence([0, 1, 2], chunks, docs, Tokenizer(), per_doc=2)
        self.assertEqual(len(result['passages']), 1)
        self.assertNotIn('source_start', result['passages'][0])
        self.assertEqual(result['diagnostics']['quota'], 1)

    def test_budget_and_primary_quota(self):
        docs = {'d': {'text': 'one two three four five six seven eight nine ten'},
                'e': {'text': 'eleven twelve thirteen fourteen fifteen'}}
        chunks = [
            {'document_id': 'd', 'start': 0, 'end': 3},
            {'document_id': 'd', 'start': 4, 'end': 7},
            {'document_id': 'd', 'start': 8, 'end': 13},
            {'document_id': 'e', 'start': 0, 'end': 5},
        ]
        result = pack_evidence([0, 1, 2, 3], chunks, docs, Tokenizer(), budget=8, per_doc=2, max_docs=1)
        self.assertEqual(len(result['passages']), 2)
        self.assertEqual(result['diagnostics']['quota'], 2)
        self.assertLessEqual(result['tokens'], 8)


if __name__ == '__main__':
    unittest.main()
