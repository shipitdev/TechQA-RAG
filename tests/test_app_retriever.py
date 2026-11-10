from types import SimpleNamespace

import pytest

import rag_app.retriever as retriever_module
from rag_app.retriever import Retriever


@pytest.fixture(autouse=True)
def isolated_runtime_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / 'run.json'
    manifest.write_text('{"cache_key":"test-index"}')
    monkeypatch.setattr(retriever_module, 'RUN', manifest)
    monkeypatch.setattr(retriever_module, '_versions', lambda names: {name: 'test' for name in names})


class Source:
    documents = {
        "doc-a": {"document_id": "doc-a", "title": "A", "metadata": {"canonicalUrl": "https://a"},
                  "text": "Alpha support."},
        "doc-b": {"document_id": "doc-b", "title": "B", "metadata": {"canonicalUrl": "https://b"},
                  "text": "Beta support."},
    }
    chunks = {
        "html_v1:doc-a:0": {"chunk_id": "html_v1:doc-a:0", "document_id": "doc-a", "start": 0,
                            "end": 14, "search_text": "Alpha support.", "section": ""},
        "html_v1:doc-b:0": {"chunk_id": "html_v1:doc-b:0", "document_id": "doc-b", "start": 0,
                            "end": 13, "search_text": "Beta support.", "section": ""},
    }

    def get_chunk(self, chunk_id):
        return self.chunks[chunk_id]

    def get_document(self, docid):
        return self.documents[docid]


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=text.split())


class FakeQdrant:
    def __init__(self):
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=[
            SimpleNamespace(payload={"chunk_id": cid, "doc_id": docid, "corpus_version": "html_v1"}, score=score)
            for cid, docid, score in [
                ("html_v1:doc-a:0", "doc-a", 0.9), ("html_v1:doc-b:0", "doc-b", 0.8)]
        ])


def test_retrieve_uses_exact_search_audited_packing_and_cache(tmp_path, monkeypatch):
    qdrant, encoded, scored = FakeQdrant(), [], []
    packed_calls = []
    pack = retriever_module.pack_evidence

    def record_pack(*args, **kwargs):
        packed_calls.append(kwargs.copy())
        return pack(*args, **kwargs)

    monkeypatch.setattr(retriever_module, "pack_evidence", record_pack)

    def embed(text):
        encoded.append(text)
        return [1.0] + [0.0] * 767

    def score(pairs):
        scored.extend(pairs)
        return [2.0, 1.0]

    retriever = Retriever(source_store=Source(), qdrant_client=qdrant, embed_query=embed,
                          score_pairs=score, tokenizer=FakeTokenizer(), cache_path=tmp_path / "cache.sqlite")
    try:
        first = retriever.retrieve("question text")
        second = retriever.retrieve("question text")
    finally:
        retriever.close()

    assert len(encoded) == 1
    assert encoded[0].startswith("task: question answering | query: ")
    assert len(scored) == 2
    assert first["sources"][0] == {"id": "S1", "document_id": "doc-a", "chunk_id": "html_v1:doc-a:0",
                                  "title": "A", "url": "https://a", "text": "Alpha support."}
    assert first["selected_passages"][0]["source_id"] == "S1"
    assert first["context_tokens"] > 0
    assert packed_calls[0]["budget"] == 4096
    assert packed_calls[0]["per_doc"] == 5
    assert packed_calls[0]["max_docs"] == 8
    assert qdrant.calls[0]["limit"] == 100
    params = qdrant.calls[0]["search_params"]
    assert params.get("exact") if isinstance(params, dict) else params.exact
    assert first["metrics"]["query_cache_misses"] == 1
    assert second["metrics"]["query_cache_hits"] == 1
    assert second["metrics"]["pair_cache_hits"] == 2


def test_retrieve_rejects_empty_questions(tmp_path):
    retriever = Retriever(source_store=Source(), qdrant_client=FakeQdrant(), embed_query=lambda _: [0] * 768,
                          score_pairs=lambda pairs: [0] * len(pairs), tokenizer=FakeTokenizer(),
                          cache_path=tmp_path / "cache.sqlite")
    try:
        with pytest.raises(ValueError, match="non-empty"):
            retriever.retrieve("  ")
    finally:
        retriever.close()


def test_default_source_requires_imported_database(tmp_path, monkeypatch):
    monkeypatch.setattr(retriever_module, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="run scripts/import_qdrant.py first"):
        Retriever(cache_path=tmp_path / "cache.sqlite")
    assert not (tmp_path / "data/source.sqlite3").exists()


def test_index_identity_rejects_a_different_active_build(tmp_path):
    qdrant=FakeQdrant()
    r=Retriever(source_store=Source(),qdrant_client=qdrant,embed_query=lambda _: [1.]+[0.]*767,
                score_pairs=lambda pairs: [0.]*len(pairs),tokenizer=FakeTokenizer(),cache_path=tmp_path/'cache.sqlite')
    try:
        qdrant.scroll=lambda *a,**k: ([SimpleNamespace(payload={'index_build_id':r.run_config['cache_key']})],None)
        assert r.index_identity()['build']==r.run_config['cache_key']
        qdrant.scroll=lambda *a,**k: ([SimpleNamespace(payload={'index_build_id':'another-build'})],None)
        with pytest.raises(ValueError,match='index version'):r.index_identity()
    finally:r.close()


def test_retrieval_cache_evicts_old_entries(tmp_path):
    retriever = Retriever(source_store=Source(), qdrant_client=FakeQdrant(),
                          embed_query=lambda _: [1.] + [0.] * 767, score_pairs=lambda _: [],
                          tokenizer=FakeTokenizer(), cache_path=tmp_path/'bounded.sqlite')
    try:
        retriever.cache.executemany('INSERT INTO query_vectors VALUES (?,?)',
                                   ((str(i),'[]') for i in range(2050)))
        retriever._prune_cache()
        assert retriever.cache.execute('SELECT count(*) FROM query_vectors').fetchone()[0] == 2048
        assert retriever.cache.execute('SELECT 1 FROM query_vectors WHERE cache_key=?',('0',)).fetchone() is None
        assert retriever.cache.execute('SELECT 1 FROM query_vectors WHERE cache_key=?',('2049',)).fetchone()
    finally:
        retriever.close()


def test_missing_managed_activation_does_not_serve_legacy_data(tmp_path,monkeypatch):
    monkeypatch.setattr(retriever_module,'ROOT',tmp_path)
    version=tmp_path/'data/indexes/version';version.mkdir(parents=True)
    (version/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='activation is missing'):
        Retriever()
