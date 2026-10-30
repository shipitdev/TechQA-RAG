import json
import sqlite3

import pytest

from rag_app.storage import SourceStore


def test_source_store_round_trip_and_missing_ids(tmp_path):
    documents = {"doc-1": {"document_id": "doc-1", "title": "Title", "text": "source"}}
    chunks = [
        {"chunk_id": "v1:doc-1:0", "document_id": "doc-1", "search_text": "first"},
        {"chunk_id": "v1:doc-1:1", "document_id": "doc-1", "search_text": "second"},
    ]
    docs_path, chunks_path = tmp_path / "docs.json", tmp_path / "chunks.jsonl"
    docs_path.write_text(json.dumps(documents))
    chunks_path.write_text("".join(json.dumps(row) + "\n" for row in chunks))

    with SourceStore(tmp_path / "source.sqlite") as source:
        source.load_prepared(docs_path, chunks_path)
        assert source.get_document("doc-1") == documents["doc-1"]
        assert source.get_chunk("v1:doc-1:1") == chunks[1]
        with pytest.raises(KeyError):
            source.get_chunk("query-vector")
        with pytest.raises(KeyError):
            source.get_document("missing")


def test_load_prepared_rolls_back_on_missing_document(tmp_path):
    db_path = tmp_path / "source.sqlite"
    with SourceStore(db_path) as source:
        source.db.execute("INSERT INTO documents VALUES (?, ?)", ("original", '{"document_id":"original"}'))
        source.db.commit()
        docs_path, chunks_path = tmp_path / "docs.json", tmp_path / "chunks.jsonl"
        docs_path.write_text(json.dumps({"new": {"document_id": "new"}}))
        chunks_path.write_text(json.dumps({"chunk_id": "orphan", "document_id": "absent"}) + "\n")
        with pytest.raises(ValueError, match="missing document"):
            source.load_prepared(docs_path, chunks_path)
        assert source.get_document("original") == {"document_id": "original"}
