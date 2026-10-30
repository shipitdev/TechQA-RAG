#!/usr/bin/env python3
"""Import the prepared structured corpus and its cached Gemma vectors to Qdrant."""
import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_app.storage import SourceStore

CORPUS_VERSION = "html_v1"
CHUNK_PATH = ROOT / "data/runs/html_v1/structured_512/chunks.jsonl"
DOCUMENT_PATH = ROOT / "data/runs/html_v1/documents.json"
RUN_PATH = ROOT / "data/retrieval_runs/gemma2_structured_512_html_v1"
CONFIG_PATH = RUN_PATH / "config.json"
COLLECTION = "techqa_html_v1_gemma2"


def embedding_cache_path(config):
    key = config['cache_key']
    if not isinstance(key, str) or len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('Expected a SHA-256 embedding cache identity')
    return ROOT / 'data/embedding_cache' / key


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs():
    config = json.loads(CONFIG_PATH.read_text())
    cache_path = embedding_cache_path(config)
    chunks_hash, documents_hash = sha256(CHUNK_PATH), sha256(DOCUMENT_PATH)
    if chunks_hash != config["chunk_sha256"] or chunks_hash != config["input_policy"]["chunks_sha256"]:
        raise ValueError("Prepared chunk rows do not match the embedding run manifest")
    if documents_hash != config["documents_sha256"]:
        raise ValueError("Prepared documents do not match the embedding run manifest")
    state = json.loads((cache_path / "state.json").read_text())
    with CHUNK_PATH.open() as stream:
        chunks = [json.loads(line) for line in stream]
    vectors = np.load(cache_path / "vectors.npy", mmap_mode="r")
    if len(chunks) != 99_697 or len({row["chunk_id"] for row in chunks}) != len(chunks):
        raise ValueError("Prepared chunk identity/count check failed")
    if state["key"] != config["cache_key"] or state["total"] != 99_817 or state["done"] != state["total"]:
        raise ValueError("Embedding cache does not match the completed run")
    if vectors.shape != (state["total"], 768) or state["dimension"] != 768:
        raise ValueError(f"Expected a complete 768-dimensional cache, got {vectors.shape}")
    for start in range(0, len(chunks), 256):
        batch = vectors[start:start + 256]
        if not np.isfinite(batch).all() or not np.allclose(np.linalg.norm(batch, axis=1), 1, atol=1e-3):
            raise ValueError(f"Invalid or non-unit embedding vector in chunk batch starting at row {start}")
    # eval_embeddings.py constructs vectors as chunks in JSONL order, then 120 tuning queries.
    if len(config["query_ids"]) != 120:
        raise ValueError("Expected exactly 120 trailing tuning-query vectors")
    return chunks, vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="validate and summarize without connecting to Qdrant")
    parser.add_argument("--source-db", type=Path, default=ROOT / "data/source.sqlite3")
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    args = parser.parse_args()

    chunks, vectors = validate_inputs()
    print(f"validated {len(chunks)} chunk vectors; excluded {vectors.shape[0] - len(chunks)} query vectors")
    if args.dry_run:
        return

    from qdrant_client import QdrantClient, models

    with SourceStore(args.source_db) as source:
        source.load_prepared(DOCUMENT_PATH, CHUNK_PATH)
        client = QdrantClient(url=args.url)
        if client.collection_exists(COLLECTION):
            info = client.get_collection(COLLECTION).config.params.vectors
            if not isinstance(info, models.VectorParams) or info.size != 768 or info.distance != models.Distance.COSINE:
                raise ValueError(f"Collection {COLLECTION!r} has an incompatible vector schema")
        else:
            client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))
        index_build_id = json.loads(CONFIG_PATH.read_text())["cache_key"]
        for start in range(0, len(chunks), 256):
            batch = chunks[start:start + 256]
            client.upsert(COLLECTION, points=[models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{CORPUS_VERSION}:{chunk['chunk_id']}")),
                vector=vectors[start + offset].tolist(),
                payload={"chunk_id": chunk["chunk_id"], "doc_id": chunk["document_id"], "corpus_version": CORPUS_VERSION,
                         "index_build_id": index_build_id},
            ) for offset, chunk in enumerate(batch)], wait=True)
        print(f"upserted {len(chunks)} points into {COLLECTION}; source store: {args.source_db}")


if __name__ == "__main__":
    main()
