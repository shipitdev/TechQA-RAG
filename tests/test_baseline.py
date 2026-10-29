import math
import unittest
from scripts.baseline import group_questions, retrieval_scores, BM25


def question(qid, text, doc='-', answerable='N'):
    return {'QUESTION_ID': qid, 'QUESTION_TITLE': text, 'QUESTION_TEXT': '',
            'DOCUMENT': doc, 'ANSWERABLE': answerable}


class BaselineTests(unittest.TestCase):
    def test_transitive_groups_include_negative_duplicate(self):
        rows = [question('a','same','d','Y'), question('b','other','d','Y'), question('c',' SAME ')]
        self.assertEqual(group_questions(rows, {'d': {'text':'body'}}), [['a','b','c']])

    def test_identical_answer_bodies_group_different_ids(self):
        rows = [question('a','one','x','Y'), question('b','two','y','Y')]
        self.assertEqual(group_questions(rows, {'x':{'text':'same'},'y':{'text':'same'}}), [['a','b']])

    def test_ranking_deduplicates_and_scores_known_rank(self):
        result = retrieval_scores('gold', ['wrong','wrong','gold'])
        self.assertEqual(result['recall_at_5'], 1)
        self.assertEqual(result['mrr_at_10'], .5)
        self.assertAlmostEqual(result['ndcg_at_10'],1/math.log2(3))
        self.assertEqual(retrieval_scores('gold', [])['recall_at_5'],0)
        self.assertIsNone(retrieval_scores(None, ['x'])['recall_at_5'])

    def test_rank_cutoffs(self):
        for rank in [5,6,10,11]:
            scores = retrieval_scores('gold', ['x'+str(i) for i in range(rank-1)]+['gold'])
            self.assertEqual(scores['recall_at_5'], int(rank <= 5))
            self.assertEqual(scores['recall_at_10'], int(rank <= 10))
            self.assertAlmostEqual(scores['mrr_at_10'], 1/rank if rank <= 10 else 0)

    def test_bm25_matches_hand_computed_score_and_no_match(self):
        index = BM25({'a':'code code', 'b':'other other'})
        hits = index.search('code')
        expected=math.log(2)*(2*2.5)/(2+1.5)
        self.assertEqual(hits[0][0],'a')
        self.assertAlmostEqual(index.scores('code')['a'],expected)
        self.assertAlmostEqual(hits[0][1],expected)
        self.assertEqual(index.search('missing'),[])
        self.assertEqual(index.search('code', allowed={'b'}),[])
        self.assertEqual(index.search('code code'),hits)
