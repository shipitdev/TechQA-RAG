from unittest.mock import patch
from rag_app.presentation import generate_presentation
from rag_app.generation import GenerationError

class Reply:
    def __init__(self,value):self.value=value
    def json(self):return self.value
class Judge:
    def __init__(self,pass_):self.pass_=pass_
    def json_call(self,prompt,schema):return {'openui':'root = AnswerText("Restart. [S1]")'}
    def _verify(self,*args):return {'verdict':'complete','supported':self.pass_,'complete':self.pass_,'missing_steps':[],'version_conflicts':[]}

def result():return {'answer':'Restart. [S1]','sources':[{'id':'S1','text':'Restart.'}],'citations':['S1']}

def test_only_verified_generated_ui_is_released():
    for approved in [True,False]:
        with patch('rag_app.presentation.httpx.get',return_value=Reply({'system_prompt':'safe components'})),patch('rag_app.presentation.httpx.post',return_value=Reply({'valid':True,'visible_text':'Restart. [S1]','citation_source_ids':['S1']})):
            output=generate_presentation(result(),'How?',Judge(approved))
            assert bool(output)==approved

def test_invalid_layout_falls_back_without_releasing():
    with patch('rag_app.presentation.httpx.get',return_value=Reply({'system_prompt':'safe components'})),patch('rag_app.presentation.httpx.post',return_value=Reply({'valid':False})):
        assert generate_presentation(result(),'?',Judge(True)) is None

def test_provider_failure_is_recorded_without_exposing_details():
    judge=Judge(True)
    def fail(*args):raise GenerationError('private provider detail',category='http_429')
    judge.json_call=fail
    output=result()
    with patch('rag_app.presentation.httpx.get',return_value=Reply({'system_prompt':'safe components'})):
        assert generate_presentation(output,'?',judge) is None
    assert output['metrics']['presentation_error']=='http_429'
    assert 'private' not in str(output)


def test_presentation_verifies_only_cited_sources_and_preserves_approved_answer():
    class Capture(Judge):
        def _verify(self,question,sources,draft):
            assert [s['id'] for s in sources]==['S1']
            assert draft['approved_answer']=='Restart. [S1]'
            return super()._verify(question,sources,draft)
    output=result();output['sources'].append({'id':'S2','text':'Unrelated passage.'})
    with patch('rag_app.presentation.httpx.get',return_value=Reply({'system_prompt':'safe components'})),patch('rag_app.presentation.httpx.post',return_value=Reply({'valid':True,'visible_text':'Restart. [S1]','citation_source_ids':['S1']})):
        assert generate_presentation(output,'How?',Capture(True)) is not None
