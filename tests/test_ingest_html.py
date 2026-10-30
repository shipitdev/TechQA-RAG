import unittest

from scripts.ingest_html import _clean_markdown, _doc_record, _repeat_table_headers, _source_interval, _structured_chunks


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        ids = text.split()
        offsets = [(match.start(), match.end()) for match in __import__('re').finditer(r'\S+', text)]
        return type('Encoding', (), {'ids': ids, 'offsets': offsets})()


class HtmlIngestionTests(unittest.TestCase):
    def test_markdown_cleanup_preserves_structural_markdown(self):
        self.assertEqual(_clean_markdown('# Steps\n\n```sh\necho ok\n```\n\n\n|a|b|'),
                         '# Steps\n\n```sh\necho ok\n```\n\n|a|b|\n')

    def test_large_table_repeats_header_and_keeps_every_row(self):
        markdown = '| Key | Value |\n| --- | --- |\n| alpha | one |\n| beta | two |\n| gamma | three |'
        result = _repeat_table_headers(markdown, WordTokenizer(), budget=10)
        self.assertEqual(result.count('| Key | Value |'), 3)
        self.assertIn('[Table continued; repeated header.]', result)
        for row in ('alpha', 'beta', 'gamma'):
            self.assertIn(row, result)

    def test_source_coordinates_are_only_for_exact_matches(self):
        self.assertEqual(_source_interval('exact body', 'prefix exact body suffix'), (7, 17))
        self.assertEqual(_source_interval('normalized  body', 'normalized body'), (None, None))

    def test_prepared_document_schema_keeps_source_metadata(self):
        source = {'title': 'Title', 'metadata': {'productName': 'Product'}}
        self.assertEqual(_doc_record('doc-1', source, '# Body'),
                         {'document_id': 'doc-1', 'text': '# Body', 'title': 'Title',
                          'metadata': {'productName': 'Product'}})

    def test_structured_chunks_keep_compact_table_and_code_atomic(self):
        text = '# Setup\n\n| Key | Value |\n| --- | --- |\n| a | b |\n\n```sh\necho ok\n```\n'
        chunks = _structured_chunks(text, WordTokenizer(), budget=100)
        table = next(c for c in chunks if c['structure'] == 'table')
        code = next(c for c in chunks if c['structure'] == 'code')
        self.assertIn('| a | b |', text[table['start']:table['end']])
        self.assertIn('echo ok', text[code['start']:code['end']])
        self.assertFalse(any(c['structure'].endswith('_split') for c in chunks))

    def test_docling_compact_separator_is_still_a_table(self):
        text = '| Product | Support |\n| - | - |\n| A | Yes |\n'
        chunks = _structured_chunks(text, WordTokenizer(), budget=100)
        self.assertEqual(len(chunks),1)
        self.assertEqual(chunks[0]['structure'],'table')

    def test_large_table_row_groups_each_repeat_the_header(self):
        source = '| Key | Value |\n| --- | --- |\n| alpha | one |\n| beta | two |'
        grouped = _repeat_table_headers(source, WordTokenizer(), budget=15)
        chunks = _structured_chunks(grouped, WordTokenizer(), budget=15)
        tables = [grouped[c['start']:c['end']] for c in chunks if c['structure'] == 'table']
        self.assertEqual(len(tables), 2)
        self.assertTrue(all('| Key | Value |' in table and '| --- | --- |' in table for table in tables))
        self.assertIn('alpha', ''.join(tables))
        self.assertIn('beta', ''.join(tables))

    def test_qualifying_table_note_is_attached_or_referenced(self):
        text = '| Key | Value |\n| --- | --- |\n| a | b |\n\nNote: retain this qualifier.\n'
        chunks = _structured_chunks(text, WordTokenizer(), budget=30)
        table = next(c for c in chunks if c['structure'].startswith('table'))
        self.assertTrue(table['note_included'])
        self.assertIn('Note:', text[table['start']:table['end']])


if __name__ == '__main__':
    unittest.main()
