"""Full local handoff checks with real storage/search/packing and controlled models."""

import json
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import threading

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from rag_app.api import create_app
from rag_app.generation import AnswerGenerator
from rag_app.retriever import Retriever
from rag_app.storage import SourceStore
from rag_app.indexes import IndexManager
import rag_app.retriever as retriever_module
import rag_app.api as api


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=text.split())


def final(response):
    for block in response.text.split("\n\n"):
        if block.startswith("event: final\n"):
            return json.loads(block.split("data: ", 1)[1])
    raise AssertionError(response.text)


def test_v1_question_search_verify_cache_followup_regenerate_clear(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    monkeypatch.setattr(
        retriever_module, "_versions", lambda names: {n: "controlled" for n in names}
    )
    text = "Product Alpha version 2 requires restarting the service after changing its port."
    docs = {"d": {"document_id": "d", "title": "Alpha guide", "text": text}}
    chunk = {
        "chunk_id": "c",
        "document_id": "d",
        "start": 0,
        "end": len(text),
        "search_text": text,
        "structure": "prose",
    }
    dp, cp = tmp_path / "docs.json", tmp_path / "chunks.jsonl"
    dp.write_text(json.dumps(docs))
    cp.write_text(json.dumps(chunk) + "\n")
    with SourceStore(tmp_path / "source.sqlite") as source:
        source.load_prepared(dp, cp)
    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(
        "test",
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    )
    qdrant.upsert(
        "test",
        [
            models.PointStruct(
                id=1,
                vector=[1.0] + [0.0] * 767,
                payload={
                    "chunk_id": "c",
                    "doc_id": "d",
                    "corpus_version": "html_v1",
                    "index_build_id": "test",
                },
            )
        ],
    )
    manager = IndexManager(tmp_path / "indexes", qdrant)
    manifest = manager.adopt(
        tmp_path / "source.sqlite", "test", "test", {"model": "controlled"}
    )
    source = SourceStore(manager.source_path(manifest), readonly=True)
    search = Retriever(
        source_store=source,
        qdrant_client=qdrant,
        index_manifest=manifest,
        index_root=manager.root,
        embed_query=lambda _: [1.0] + [0.0] * 767,
        score_pairs=lambda pairs: [1.0] * len(pairs),
        tokenizer=WordTokenizer(),
        cache_path=tmp_path / "cache.sqlite",
    )
    calls = []

    def provider(prompt, schema):
        calls.append(prompt)
        if "verdict" in schema["properties"]:
            return {
                "verdict": "complete",
                "supported": True,
                "complete": True,
                "missing_steps": [],
                "version_conflicts": [],
                "findings": [],
                "unsupported_claims": [],
            }
        if "unrelated product" in prompt:
            return {"status": "insufficient_evidence", "answer": "", "citations": []}
        return {"status": "answered", "answer": text + " [S1]", "citations": ["S1"]}

    class Resolver:
        def resolve(self, question, context):
            dependent = question == "What about that version?"
            return {
                "dependent": dependent,
                "question": (
                    "Product Alpha version 2 port change?" if dependent else question
                ),
                "needs_clarification": False,
            }

    try:
        app = create_app(search, AnswerGenerator(provider), Resolver())
        client = TestClient(app)
        first = final(client.post("/ask", json={"question": "Alpha port change?"}))
        assert first["status"] == "answered" and first["sources"][0]["text"] == text
        assert first["metrics"]["index"]["build"] == "test"
        assert first["metrics"]["generation_stages"][1]["supported"] is True
        assert len(calls) == 2
        assert final(client.post("/ask", json={"question": "Alpha port change?"}))[
            "metrics"
        ]["answer_cache_hit"]
        assert len(calls) == 2
        assert (
            final(client.post("/ask", json={"question": "What about that version?"}))[
                "status"
            ]
            == "answered"
        )
        assert len(calls) == 4
        assert final(client.post("/ask", json={"question": "generate it again"}))[
            "metrics"
        ]["regenerated"]
        assert len(calls) == 6
        withheld = final(client.post("/ask", json={"question": "unrelated product?"}))
        assert (
            withheld["status"] == "insufficient_evidence"
            and withheld["citations"] == []
        )
        second = TestClient(app)
        assert not final(second.post("/ask", json={"question": "Alpha port change?"}))[
            "metrics"
        ]["answer_cache_hit"]
        client.post("/session/end")
        assert "techqa_session" not in client.cookies
    finally:
        search.close()
        source.close()
        qdrant.close()


def test_admission_is_bounded_and_drafts_stay_private(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    entered, release = threading.Event(), threading.Event()

    class Search:
        def retrieve(self, question):
            return {
                "sources": [{"id": "S1", "text": "Approved evidence"}],
                "metrics": {},
            }

    class SlowWriter:
        def generate(self, question, sources):
            entered.set()
            assert release.wait(5)
            return {
                "status": "answered",
                "answer": "Approved evidence [S1]",
                "citations": ["S1"],
                "draft": "private draft",
            }

    app = create_app(Search(), SlowWriter())

    def ask():
        return TestClient(app).post("/ask", json={"question": "test?"})

    with ThreadPoolExecutor(max_workers=6) as executor:
        first = executor.submit(ask)
        assert entered.wait(2)
        pending = [executor.submit(ask) for _ in range(5)]
        # The sixth request must be rejected while five workers occupy admission slots.
        from concurrent.futures import wait, FIRST_COMPLETED

        done, _ = wait(pending, timeout=3, return_when=FIRST_COMPLETED)
        try:
            assert done and next(iter(done)).result().status_code == 429
        finally:
            release.set()
        responses = [first.result()] + [f.result() for f in pending]
    assert sorted(r.status_code for r in responses) == [200] * 5 + [429]
    assert all("private draft" not in r.text for r in responses)
