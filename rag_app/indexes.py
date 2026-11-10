"""Immutable source/index pairs with one atomic local activation pointer."""

from contextlib import closing, contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import uuid

from qdrant_client import models
from rag_app.storage import SourceStore


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temp.unlink(missing_ok=True)


class IndexManager:
    def __init__(self, root, client):
        self.root, self.client = Path(root).resolve(), client
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with (self.root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def active(self):
        return json.loads((self.root / "active.json").read_text())

    def _directory(self, build_id):
        if (
            not isinstance(build_id, str)
            or not build_id
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in build_id)
        ):
            raise ValueError("Invalid build identity")
        return self.root / build_id

    def manifest(self, build_id):
        return json.loads((self._directory(build_id) / "manifest.json").read_text())

    def source_path(self, manifest):
        path = (self.root / manifest["source"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Source must be inside index root")
        return path

    def _rows(self, collection):
        offset = None
        while True:
            rows, offset = self.client.scroll(
                collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            yield rows
            if offset is None:
                break

    def validate(self, manifest):
        path = self.source_path(manifest)
        if sha256(path) != manifest["source_sha256"]:
            raise ValueError("Source hash does not match index manifest")
        with SourceStore(path, readonly=True) as source:
            if source.db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Source integrity check failed")
            expected = dict(
                source.db.execute("SELECT chunk_id, document_id FROM chunks")
            )
            documents = source.db.execute("SELECT count(*) FROM documents").fetchone()[
                0
            ]
        info = self.client.get_collection(manifest["collection"])
        schema = info.config.params.vectors
        if (
            not isinstance(schema, models.VectorParams)
            or schema.size != 768
            or schema.distance != models.Distance.COSINE
        ):
            raise ValueError("Incompatible vector schema")
        if (
            len(expected) != manifest["counts"]["chunks"]
            or documents != manifest["counts"]["documents"]
            or info.points_count != len(expected)
        ):
            raise ValueError("Source/index count mismatch")
        seen = set()
        for rows in self._rows(manifest["collection"]):
            for point in rows:
                payload = point.payload or {}
                cid = payload.get("chunk_id")
                if (
                    cid in seen
                    or cid not in expected
                    or expected[cid] != payload.get("doc_id")
                    or payload.get("index_build_id") != manifest["build_id"]
                    or payload.get("corpus_version") != "html_v1"
                ):
                    raise ValueError("Source/index membership or version mismatch")
                seen.add(cid)
                vector = point.vector
                if (
                    not isinstance(vector, list)
                    or len(vector) != 768
                    or not all(math.isfinite(x) for x in vector)
                    or not math.isclose(sum(x * x for x in vector), 1.0, abs_tol=0.003)
                ):
                    raise ValueError("Invalid embedding vector")
        if seen != set(expected):
            raise ValueError("Source/index membership mismatch")
        return manifest

    def _record(self, build, collection, encoder, counts, parent=None):
        path = self.root / build / "source.sqlite3"
        value = {
            "build_id": build,
            "collection": collection,
            "source": str(path.relative_to(self.root)),
            "source_sha256": sha256(path),
            "encoder": encoder,
            "counts": counts,
            "parent": parent,
            "corpus_version": "html_v1",
        }
        self.validate(value)
        atomic_json(path.parent / "manifest.json", value)
        return value

    def adopt(self, source_path, collection, build_id, encoder):
        with self.locked():
            if (self.root / "active.json").exists():
                return self.validate(self.active())
            if not Path(source_path).is_file():
                raise FileNotFoundError(source_path)
            directory = self._directory(build_id)
            directory.mkdir(exist_ok=False)
            with closing(sqlite3.connect(source_path)) as src, closing(
                sqlite3.connect(directory / "source.sqlite3")
            ) as dest:
                src.backup(dest)
                counts = {
                    "documents": dest.execute(
                        "SELECT count(*) FROM documents"
                    ).fetchone()[0],
                    "chunks": dest.execute("SELECT count(*) FROM chunks").fetchone()[0],
                }
            value = self._record(build_id, collection, encoder, counts)
            atomic_json(self.root / "active.json", value)
            return value

    def activate(self, build_id):
        with self.locked():
            value = self.validate(self.manifest(build_id))
            if (self.root / "active.json").exists():
                active = self.active()
                if value.get("parent") not in (None, active["build_id"]) and value[
                    "build_id"
                ] != active.get("parent"):
                    raise ValueError(
                        "Stale activation: stage again from the active build"
                    )
            atomic_json(self.root / "active.json", value)
            return value

    def stage(self, updates, deleted, prepare, encode, encoder):
        """Prepare only changed documents; copy unchanged vectors into an isolated build."""
        with self.locked():
            active = self.active()
            self.validate(active)
            if encoder != active["encoder"]:
                raise ValueError("Changed encoder requires a separately rebuilt index")
            if not isinstance(updates, dict) or any(
                not isinstance(k, str) or not k for k in updates
            ):
                raise ValueError("Updates must map nonempty document IDs to documents")
            deleted = set(deleted)
            if deleted & updates.keys():
                raise ValueError("Cannot update and delete the same document")
            with SourceStore(self.source_path(active), readonly=True) as src:
                existing = {
                    r[0]: json.loads(r[1])
                    for r in src.db.execute("SELECT document_id,data FROM documents")
                }
            if deleted - existing.keys():
                raise ValueError("Cannot delete missing documents")
            changed = {
                k: v
                for k, v in updates.items()
                if existing.get(k) != v
                and existing.get(k, {}).get("_input_sha256")
                != hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
            }
            if not changed and not deleted:
                return active
            build = uuid.uuid4().hex
            directory = self.root / build
            directory.mkdir()
            source_path = directory / "source.sqlite3"
            shutil.copyfile(self.source_path(active), source_path)
            new_chunks = []
            with SourceStore(source_path) as src, src.db:
                for docid in deleted | changed.keys():
                    src.db.execute("DELETE FROM chunks WHERE document_id=?", (docid,))
                    src.db.execute(
                        "DELETE FROM documents WHERE document_id=?", (docid,)
                    )
                for docid, doc in changed.items():
                    prepared, chunks = prepare(docid, doc)
                    if (
                        prepared.get("document_id") != docid
                        or not isinstance(prepared.get("text"), str)
                        or not prepared["text"].strip()
                        or not chunks
                    ):
                        raise ValueError(
                            "Prepared document is empty or has an invalid identity"
                        )
                    prepared["raw_source"] = doc
                    prepared["_input_sha256"] = hashlib.sha256(
                        json.dumps(doc, sort_keys=True).encode()
                    ).hexdigest()
                    src.db.execute(
                        "INSERT INTO documents VALUES (?,?)",
                        (docid, json.dumps(prepared)),
                    )
                    for c in chunks:
                        if (
                            c.get("document_id") != docid
                            or not isinstance(c.get("start"), int)
                            or not isinstance(c.get("end"), int)
                            or not 0 <= c["start"] < c["end"] <= len(prepared["text"])
                            or not c.get("search_text")
                        ):
                            raise ValueError("Invalid prepared chunk identity/offsets")
                        src.db.execute(
                            "INSERT INTO chunks VALUES (?,?,?)",
                            (c["chunk_id"], docid, json.dumps(c)),
                        )
                    new_chunks.extend(chunks)
                counts = {
                    "documents": src.db.execute(
                        "SELECT count(*) FROM documents"
                    ).fetchone()[0],
                    "chunks": src.db.execute("SELECT count(*) FROM chunks").fetchone()[
                        0
                    ],
                    "embedded": len(new_chunks),
                    "reused": 0,
                }
            collection = "techqa_" + build
            self.client.create_collection(
                collection,
                vectors_config=models.VectorParams(
                    size=768, distance=models.Distance.COSINE
                ),
            )
            try:
                for rows in self._rows(active["collection"]):
                    kept = [
                        p
                        for p in rows
                        if p.payload["doc_id"] not in deleted | changed.keys()
                    ]
                    if kept:
                        self.client.upsert(
                            collection,
                            points=[
                                models.PointStruct(
                                    id=p.id,
                                    vector=p.vector,
                                    payload={**p.payload, "index_build_id": build},
                                )
                                for p in kept
                            ],
                            wait=True,
                        )
                        counts["reused"] += len(kept)
                for c in new_chunks:
                    vector = [float(x) for x in encode(c["search_text"])]
                    if (
                        len(vector) != 768
                        or not all(math.isfinite(x) for x in vector)
                        or not math.isclose(
                            sum(x * x for x in vector), 1.0, abs_tol=0.003
                        )
                    ):
                        raise ValueError("Invalid embedding vector")
                    self.client.upsert(
                        collection,
                        points=[
                            models.PointStruct(
                                id=str(
                                    uuid.uuid5(
                                        uuid.NAMESPACE_URL, "html_v1:" + c["chunk_id"]
                                    )
                                ),
                                vector=vector,
                                payload={
                                    "chunk_id": c["chunk_id"],
                                    "doc_id": c["document_id"],
                                    "corpus_version": "html_v1",
                                    "index_build_id": build,
                                },
                            )
                        ],
                        wait=True,
                    )
                return self._record(
                    build, collection, encoder, counts, active["build_id"]
                )
            except Exception:
                self.client.delete_collection(collection)
                raise
