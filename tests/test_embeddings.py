import contextlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scripts.eval_embeddings import rank_top, format_input, encode_cache, configure_text_input


class FakeOOM(Exception):
    pass


class FakeEncoder:
    def __init__(self, fail_second=False):
        self.inputs=[]
        self.calls=0
        self.fail_second=fail_second

    def encode(self,texts,**kwargs):
        self.calls+=1
        if self.fail_second and self.calls==2:
            raise RuntimeError('Simulated interruption')
        self.inputs.extend(texts)
        return np.asarray([[len(t)%2,1-len(t)%2] for t in texts],dtype=np.float32)


class EmbeddingTests(unittest.TestCase):
    def test_gemma_text_path_disables_automatic_chat_conversion(self):
        module=SimpleNamespace(modality_config={'text':{'method':'forward'},'message':{'format':'structured'}})
        configure_text_input([module], 'gemma2')
        self.assertEqual(module.modality_config, {'text':{'method':'forward'}})
        other=SimpleNamespace(modality_config={'text':{},'message':{}})
        configure_text_input([other], 'qwen3')
        self.assertIn('message', other.modality_config)

    def test_rank_ties_and_nonfinite_values(self):
        self.assertEqual(rank_top([.5,.5,.1],['b','a','c'],2),[1,0])
        with self.assertRaises(ValueError):
            rank_top([np.nan],['a'])

    def test_resume_only_encodes_unfinished_rows_and_rejects_corruption(self):
        cuda=SimpleNamespace(synchronize=lambda:None,empty_cache=lambda:None,
                             max_memory_allocated=lambda:0,max_memory_reserved=lambda:0,OutOfMemoryError=FakeOOM)
        torch=SimpleNamespace(cuda=cuda,inference_mode=contextlib.nullcontext)
        with tempfile.TemporaryDirectory() as folder, patch.dict('sys.modules',{'torch':torch}):
            path=Path(folder)
            with self.assertRaisesRegex(RuntimeError,'interruption'):
                encode_cache(FakeEncoder(True),['a','bb','ccc','dddd'],[1]*4,path,'key',2,2,10)
            self.assertEqual(json.loads((path/'state.json').read_text())['done'],2)
            encoder=FakeEncoder()
            vectors,state=encode_cache(encoder,['a','bb','ccc','dddd'],[1]*4,path,'key',2,2,10)
            self.assertEqual(encoder.inputs,['ccc','dddd'])
            self.assertEqual(state['done'],4)
            np.testing.assert_equal(vectors,[[1,0],[0,1],[1,0],[0,1]])
            vectors[0,0]=np.nan;vectors.flush()
            with self.assertRaisesRegex(ValueError,'Corrupt'):
                encode_cache(FakeEncoder(),['a','bb','ccc','dddd'],[1]*4,path,'key',2,2,10)

    def test_length_sorted_order_writes_rows_to_original_positions(self):
        cuda=SimpleNamespace(synchronize=lambda:None,empty_cache=lambda:None,
                             max_memory_allocated=lambda:0,max_memory_reserved=lambda:0,OutOfMemoryError=FakeOOM)
        torch=SimpleNamespace(cuda=cuda,inference_mode=contextlib.nullcontext)
        with tempfile.TemporaryDirectory() as folder, patch.dict('sys.modules',{'torch':torch}):
            path=Path(folder)
            encoder=FakeEncoder()
            vectors,state=encode_cache(encoder,['a','bb','ccc','dddd'],[1,2,3,4],path,'key',2,4,100,[2,0,3,1])
            self.assertEqual(encoder.inputs,['ccc','a','dddd','bb'])
            self.assertTrue(state['length_sorted'])
            np.testing.assert_equal(vectors,[[1,0],[0,1],[1,0],[0,1]])
            resumed=FakeEncoder()
            encode_cache(resumed,['a','bb','ccc','dddd'],[1,2,3,4],path,'key',2,4,100,[2,0,3,1])
            self.assertEqual(resumed.inputs,[])

    def test_native_prompts_are_task_appropriate(self):
        models=json.loads((Path(__file__).resolve().parents[1]/'models-embeddings.json').read_text())['models']
        self.assertEqual(format_input(models['bge_m3'],'question',True),'question')
        self.assertEqual(format_input(models['gemma2'],'question',True),'task: question answering | query: question')
        self.assertEqual(format_input(models['gemma2'],'body'),'title: none | text: body')
        self.assertIn('\nQuery:question',format_input(models['qwen3'],'question',True))
