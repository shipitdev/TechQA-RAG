import json

import pytest
from qdrant_client import QdrantClient, models

from rag_app.storage import SourceStore
from rag_app.indexes import IndexManager


def seed(tmp_path):
    client = QdrantClient(":memory:")
    client.create_collection(
        "original",
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    )
    docs = {
        k: {"document_id": k, "text": "Initial guidance for " + k, "title": k}
        for k in ["a", "b"]
    }
    chunks = [
        {
            "chunk_id": k + ":0",
            "document_id": k,
            "start": 0,
            "end": len(docs[k]["text"]),
            "search_text": docs[k]["text"],
            "structure": "prose",
        }
        for k in docs
    ]
    dp, cp = tmp_path / "docs.json", tmp_path / "chunks.jsonl"
    dp.write_text(json.dumps(docs))
    cp.write_text("".join(json.dumps(c) + "\n" for c in chunks))
    source_path = tmp_path / "source.sqlite"
    with SourceStore(source_path) as source:
        source.load_prepared(dp, cp)
    client.upsert(
        "original",
        [
            models.PointStruct(
                id=i + 1,
                vector=[1.0] + [0.0] * 767,
                payload={
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["document_id"],
                    "corpus_version": "html_v1",
                    "index_build_id": "baseline",
                },
            )
            for i, c in enumerate(chunks)
        ],
    )
    manager = IndexManager(tmp_path / "indexes", client)
    original = manager.adopt(source_path, "original", "baseline", {"model": "pinned"})
    return manager, original


def test_update_add_delete_reuses_unchanged_vectors_and_rollback(tmp_path):
    manager, original = seed(tmp_path)
    calls = []

    def encode(text):
        calls.append(text)
        return [0.0, 1.0] + [0.0] * 766

    def prepare(docid, doc):
        return {**doc, "document_id": docid}, [
            {
                "chunk_id": docid + ":0",
                "document_id": docid,
                "start": 0,
                "end": len(doc["text"]),
                "search_text": doc["text"],
                "structure": "prose",
            }
        ]

    staged = manager.stage(
        {"c": {"text": "New guidance"}}, ["b"], prepare, encode, {"model": "pinned"}
    )
    assert manager.active() == original
    assert staged["counts"] == {"documents": 2, "chunks": 2, "embedded": 1, "reused": 1}
    assert calls == ["New guidance"]
    manager.activate(staged["build_id"])
    assert manager.active()["build_id"] == staged["build_id"]
    with SourceStore(manager.source_path(staged), readonly=True) as source:
        assert source.get_document("c")["text"] == "New guidance"
        with pytest.raises(KeyError):
            source.get_document("b")
    manager.activate(original["build_id"])
    assert manager.active() == original
    assert manager.client.count("original").count == 2


def test_bad_update_cannot_change_active_or_original(tmp_path):
    manager, original = seed(tmp_path)
    with pytest.raises(ValueError, match="encoder"):
        manager.stage({}, ["b"], None, None, {"model": "different"})
    with pytest.raises(ValueError, match="missing"):
        manager.stage({}, ["absent"], None, None, {"model": "pinned"})
    assert manager.active() == original
    assert manager.client.count("original").count == 2


def test_validation_catches_equal_count_wrong_membership_and_source_corruption(
    tmp_path,
):
    manager, original = seed(tmp_path)
    manager.client.set_payload("original", {"chunk_id": "not-in-source"}, points=[1])
    with pytest.raises(ValueError, match="membership"):
        manager.validate(original)
    manager.source_path(original).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        manager.activate(original["build_id"])


def test_noop_is_free_and_stale_activation_is_rejected(tmp_path):
    manager, original = seed(tmp_path)
    doc = {"document_id": "a", "text": "Initial guidance for a", "title": "a"}
    staged = manager.stage(
        {"a": doc},
        [],
        lambda *args: pytest.fail("unchanged document reprocessed"),
        lambda *args: pytest.fail("unchanged document embedded"),
        {"model": "pinned"},
    )
    assert staged == original


def test_repeated_raw_update_is_free_and_parallel_staging_cannot_overwrite(tmp_path):
    manager, original = seed(tmp_path)

    def prepare(docid, doc):
        return {"document_id": docid, "text": doc["text"]}, [
            {
                "chunk_id": docid + ":new",
                "document_id": docid,
                "start": 0,
                "end": len(doc["text"]),
                "search_text": doc["text"],
            }
        ]

    def encode(_):
        return [1.0] + [0.0] * 767

    first = manager.stage(
        {"a": {"text": "Updated guidance"}}, [], prepare, encode, {"model": "pinned"}
    )
    second = manager.stage(
        {"b": {"text": "Other update"}}, [], prepare, encode, {"model": "pinned"}
    )
    manager.activate(first["build_id"])
    with pytest.raises(ValueError, match="Stale activation"):
        manager.activate(second["build_id"])
    assert manager.active() == first
    repeated = manager.stage(
        {"a": {"text": "Updated guidance"}},
        [],
        lambda *args: pytest.fail("reprocessed"),
        lambda *args: pytest.fail("reembedded"),
        {"model": "pinned"},
    )
    assert repeated == first
    manager.activate(original["build_id"])


def test_embedding_failure_leaves_original_untouched(tmp_path):
    manager, original = seed(tmp_path)

    def prepare(docid, doc):
        return {"document_id": docid, **doc}, [
            {
                "chunk_id": "new",
                "document_id": docid,
                "start": 0,
                "end": 3,
                "search_text": "new",
            }
        ]

    with pytest.raises(ValueError, match="Invalid embedding"):
        manager.stage(
            {"a": {"text": "new"}},
            [],
            prepare,
            lambda _: [0.0] * 768,
            {"model": "pinned"},
        )
    assert manager.active() == original
    assert [c.name for c in manager.client.get_collections().collections] == [
        "original"
    ]


def test_restore_rejects_corrupted_backup_before_upload(tmp_path):
    from scripts.manage_index import restore

    manager, original = seed(tmp_path)
    directory = tmp_path / "backup"
    directory.mkdir()
    (directory / "source.sqlite3").write_text("broken")
    (directory / "backup.json").write_text(
        json.dumps({"files": {"source.sqlite3": "wrong", "qdrant.snapshot": "wrong"}})
    )
    with pytest.raises(ValueError, match="hash"):
        restore(manager, directory)
    assert manager.active() == original


def test_adopt_rejects_missing_source_and_unsafe_identity(tmp_path):
    manager = IndexManager(tmp_path / "indexes", QdrantClient(":memory:"))
    with pytest.raises(FileNotFoundError):
        manager.adopt(tmp_path / "missing", "test", "baseline", {})
    assert not (tmp_path / "missing").exists()
    source = tmp_path / "source"
    source.write_text("placeholder")
    with pytest.raises(ValueError, match="identity"):
        manager.adopt(source, "test", "../outside", {})
    assert not (tmp_path / "outside").exists()


def test_explicit_activation_recovers_a_lost_pointer(tmp_path):
    manager, original = seed(tmp_path)
    (manager.root / "active.json").unlink()
    manager.activate(original["build_id"])
    assert manager.active() == original
