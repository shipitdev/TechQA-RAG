import unittest

from scripts.eval_embeddings import score_run, structured_chunk_texts


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return type('Encoding', (), {'ids': text.split()})()


class StructuredEmbeddingTests(unittest.TestCase):
    def test_context_packs_prepared_view_without_scoring_original_offsets(self):
        questions = [
            {'QUESTION_ID': 'yes', 'ANSWERABLE': 'Y', 'DOCUMENT': 'doc',
             'START_OFFSET': '10000', 'END_OFFSET': '10010'},
            {'QUESTION_ID': 'no', 'ANSWERABLE': 'N', 'DOCUMENT': '',
             'START_OFFSET': '', 'END_OFFSET': ''},
        ]
        chunks = [{'chunk_id': 'html:doc:0', 'document_id': 'doc',
                   'start': 0, 'end': 16, 'search_text': 'title product answer text'}]
        candidates = {qid: {'ranked': [0], 'search_ms': 1}
                      for qid in ('yes', 'no')}

        rows, metrics = score_run(questions, candidates, chunks,
                                  {'doc': {'text': 'view answer text'}},
                                  FakeTokenizer(), structured=True)

        self.assertEqual(rows[0]['context']['text'].splitlines()[-1], 'view answer text')
        self.assertIsNone(rows[0]['context']['answer_fraction'])
        self.assertIsNone(rows[0]['context']['complete'])
        self.assertIn('unavailable', rows[0]['context']['coverage_status'])
        self.assertEqual(metrics['answerable_scored'], 1)
        self.assertEqual(metrics['original_unanswerable_excluded'], 1)
        self.assertIsNone(metrics['context']['mean_answer_fraction'])
        self.assertIsNone(metrics['context']['full_answer_fraction'])
        self.assertEqual(metrics['retrieval']['recall_at_10'], 1)

    def test_structured_search_text_is_required(self):
        self.assertEqual(structured_chunk_texts([{'search_text': ' title + body '}]),
                         [' title + body '])
        for chunk in ({}, {'search_text': ''}, {'search_text': None}):
            with self.subTest(chunk=chunk), self.assertRaisesRegex(ValueError, 'search_text'):
                structured_chunk_texts([chunk])


if __name__ == '__main__':
    unittest.main()
