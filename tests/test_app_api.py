import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import rag_app.api as api


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    return tmp_path


class Search:
    def __init__(self):
        self.questions = []

    def retrieve(self, question):
        self.questions.append(question)
        return {"sources": [{"id": "S1", "text": "Restart the service.", "title": "Guide",
                             "document_id": "d", "chunk_id": "c", "url": ""}],
                "context_tokens": 9, "metrics": {"search_ms": 1}}


class Writer:
    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer or {"status": "answered", "answer": "Restart the service. [S1]", "citations": ["S1"]}

    def generate(self, question, sources):
        self.calls.append(question)
        return dict(self.answer)


class IndependentResolver:
    def resolve(self, question, context):
        return {"dependent": False, "question": "must be ignored", "needs_clarification": False}


def test_answer_session_and_regenerate(isolated_root):
    search, writer = Search(), Writer()
    client = TestClient(api.create_app(search, writer))
    r = client.post("/ask", json={"question": "How do I restart?"})
    assert "event: final" in r.text
    assert "techqa_session" in client.cookies
    r = client.post("/ask", json={"question": "No, generate it again"})
    assert '"regenerated": true' in r.text
    assert search.questions == ["How do I restart?", "How do I restart?"]
    assert len(writer.calls) == 2
    assert client.post("/session/end").status_code == 200
    assert "techqa_session" not in client.cookies


def test_input_and_origin(isolated_root):
    client = TestClient(api.create_app(Search(), Writer()))
    assert client.post("/ask", json={"question": " "}).status_code == 422
    assert client.post("/ask", json={"question": "hello"}, headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/session/end", headers={"Origin": "https://evil.example"}).status_code == 403


def test_standalone_exact_cache_ignores_old_history_and_regeneration_bypasses(isolated_root):
    search, writer = Search(), Writer()
    client = TestClient(api.create_app(search, writer, IndependentResolver()))
    first = client.post("/ask", json={"question": "How do I restart?"}).text
    second = client.post("/ask", json={"question": "How do I restart?"}).text
    assert '"answer_cache_hit": true' in second
    assert len(search.questions) == len(writer.calls) == 1
    assert writer.calls == ["How do I restart?"]

    regenerated = client.post("/ask", json={"question": "How do I restart?", "regenerate": True}).text
    assert '"regenerated": true' in regenerated
    assert len(search.questions) == len(writer.calls) == 2


def test_answer_cache_isolation_between_sessions(isolated_root):
    search, writer = Search(), Writer()
    app = api.create_app(search, writer, IndependentResolver())
    first_client, second_client = TestClient(app), TestClient(app)
    first_client.post("/ask", json={"question": "Same exact question?"})
    second = second_client.post("/ask", json={"question": "Same exact question?"}).text
    assert '"answer_cache_hit": false' in second
    assert len(search.questions) == len(writer.calls) == 2


def test_never_returns_internal_draft_fields_or_generation_errors(isolated_root):
    answer = {"status": "answered", "answer": "Verified answer. [S1]", "citations": ["S1"],
              "draft": "SECRET UNVERIFIED DRAFT"}
    client = TestClient(api.create_app(Search(), Writer(answer)))
    result = client.post("/ask", json={"question": "Question?"}).text
    assert "Verified answer" in result
    assert "SECRET UNVERIFIED DRAFT" not in result

    class BrokenWriter:
        def generate(self, question, sources):
            raise api.GenerationError("SECRET DRAFT IN PROVIDER EXCEPTION")

    broken = TestClient(api.create_app(Search(), BrokenWriter()))
    result = broken.post("/ask", json={"question": "Question?"}).text
    assert "SECRET DRAFT" not in result
    assert "Answer generation failed" in result


def test_error_metrics_are_content_free_and_retained(isolated_root):
    class BrokenWriter:
        def generate(self, question, sources):
            raise api.GenerationError("failed")

    client = TestClient(api.create_app(Search(), BrokenWriter()))
    client.post("/ask", json={"question": "private question text"})
    metrics = list((isolated_root / "data/app/metrics").glob("*.jsonl"))
    assert len(metrics) == 1
    record = json.loads(metrics[0].read_text().strip())
    assert record["status"] == "error"
    assert record["metrics"]["failed_provider_calls"] == 1
    assert "private question text" not in metrics[0].read_text()


def test_depleted_credits_show_billing_error_without_answer(isolated_root):
    class DepletedWriter:
        def generate(self, question, sources):
            raise api.GenerationError('provider billing details',category='http_402')
    client=TestClient(api.create_app(Search(),DepletedWriter()))
    response=client.post('/ask',json={'question':'How do I restart?'})
    assert 'Groq billing or credits need attention' in response.text
    assert 'event: final' not in response.text
    assert 'provider billing details' not in response.text


def test_active_index_change_reloads_default_retriever(isolated_root,monkeypatch):
    from rag_app import retriever as module
    directory=isolated_root/'data/indexes';directory.mkdir(parents=True)
    active=directory/'active.json';active.write_text('{"build_id":"one"}')
    instances=[]
    class ManagedSearch(Search):
        def __init__(self):
            super().__init__()
            self.index_manifest=json.loads(active.read_text())
            self.closed=False
            instances.append(self)
        def index_identity(self):return {'build':self.index_manifest['build_id']}
        def close(self):self.closed=True
    monkeypatch.setattr(module,'Retriever',ManagedSearch)
    writer=Writer();client=TestClient(api.create_app(generator=writer))
    first=client.post('/ask',json={'question':'Same question?'})
    active.write_text('{"build_id":"two"}')
    second=client.post('/ask',json={'question':'Same question?'})
    assert '"build": "one"' in first.text and '"build": "two"' in second.text
    assert len(instances)==2 and instances[0].closed
    assert len(writer.calls)==2
    assert '"answer_cache_hit": false' in second.text


def test_evidence_arrives_before_the_approved_answer(isolated_root):
    client = TestClient(api.create_app(Search(), Writer()))
    response = client.post('/ask', json={'question': 'How do I restart?'})
    assert response.text.index('event: evidence') < response.text.index('event: final')
    evidence = response.text.split('event: evidence\ndata: ', 1)[1].split('\n\n', 1)[0]
    assert set(json.loads(evidence)) == {'sources'}
    assert 'answer' not in json.loads(evidence)


def test_real_generator_streams_activity_without_private_verifier_text(isolated_root):
    from rag_app.generation import AnswerGenerator
    decisions = iter([
        {'status': 'answered', 'answer': 'Restart the service. [S1]', 'citations': ['S1']},
        {'verdict': 'complete', 'supported': True, 'complete': True, 'missing_steps': [],
         'version_conflicts': [], 'findings': ['private_verifier_canary'], 'unsupported_claims': []},
    ])
    writer = AnswerGenerator(lambda prompt, schema: next(decisions))
    client = TestClient(api.create_app(Search(), writer))
    response = client.post('/ask', json={'question': 'How do I restart?'})
    assert '"stage": "write"' in response.text
    assert '"stage": "verify"' in response.text
    assert 'private_verifier_canary' not in response.text
    assert response.text.index('"stage": "verify"') < response.text.index('event: final')
    assert writer.on_progress is None


def test_quota_error_explains_when_to_retry(isolated_root):
    from rag_app.generation import GenerationError

    class QuotaWriter:
        def generate(self, question, sources):
            raise GenerationError('quota', category='http_429')

    client = TestClient(api.create_app(Search(), QuotaWriter()))
    response = client.post('/ask', json={'question': 'How do I restart?'})
    assert 'Retry after your provider quota resets' in response.text
    assert 'event: final' not in response.text
