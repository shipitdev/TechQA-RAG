"""Local SQLite source of truth for prepared TechQA documents and chunks."""
import json
import sqlite3
from pathlib import Path


class SourceStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS documents (document_id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, data TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS chunks_document_id ON chunks(document_id);
        """)

    def load_prepared(self, documents_path, chunks_path):
        """Replace source rows atomically from prepared JSON/JSONL artifacts."""
        docs = json.loads(Path(documents_path).read_text())
        with self.db:
            self.db.execute("DELETE FROM chunks")
            self.db.execute("DELETE FROM documents")
            self.db.executemany("INSERT INTO documents VALUES (?, ?)",
                                ((key, json.dumps(value)) for key, value in docs.items()))
            with Path(chunks_path).open() as stream:
                for line in stream:
                    chunk = json.loads(line)
                    if chunk["document_id"] not in docs:
                        raise ValueError(f"Chunk references missing document: {chunk['document_id']}")
                    self.db.execute("INSERT INTO chunks VALUES (?, ?, ?)",
                                    (chunk["chunk_id"], chunk["document_id"], json.dumps(chunk)))

    def _get(self, table, key_column, key):
        row = self.db.execute(f"SELECT data FROM {table} WHERE {key_column} = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row["data"])

    def get_chunk(self, chunk_id):
        return self._get("chunks", "chunk_id", chunk_id)

    def get_document(self, docid):
        return self._get("documents", "document_id", docid)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
