import unittest
from scripts.retrieval import rrf, pack_context, summarize
from test_chunking import WordTokenizer as DummyTokenizer


class RetrievalTests(unittest.TestCase):
    def test_rrf_deduplicates_each_branch_uses_one_based_ranks(self):
        ids,scores=rrf(['a','a','b'],['b','c'])
        self.assertEqual(ids[0],'b')
        self.assertAlmostEqual(scores['b'],1/62+1/61)
        self.assertAlmostEqual(scores['a'],1/61)
        self.assertEqual(rrf([],[])[0],[])

    def test_packing_merges_overlap_preserves_source_and_budget(self):
        docs={'d':{'text':'one two three four five six seven'},'x':{'text':'other'}}
        chunks=[{'document_id':'d','start':0,'end':13}, {'document_id':'d','start':8,'end':23},
                {'document_id':'x','start':0,'end':5}]
        context=pack_context([0,1,2],chunks,docs,DummyTokenizer(),30)
        self.assertEqual(context['passages'][0],{'document_id':'d','start':0,'end':23})
        self.assertLessEqual(context['tokens'],30)
        self.assertIn('one two three four five',context['text'])
        tiny=pack_context([0],chunks,docs,DummyTokenizer(),1)
        self.assertEqual(tiny['passages'],[])

    def test_aggregate_never_scores_global_unanswerable_as_correct(self):
        rows=[{'scores':{'recall_at_5':1},'context':{'answer_fraction':1,'complete':True,'tokens':10}},
              {'scores':{'recall_at_5':None},'context':{'answer_fraction':None,'complete':None,'tokens':5}}]
        result=summarize(rows)
        self.assertEqual(result['answerable_scored'],1)
        self.assertEqual(result['original_unanswerable_excluded'],1)
        self.assertEqual(result['retrieval']['recall_at_5'],1)
