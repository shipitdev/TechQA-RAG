"""Gemma dense retrieval, BGE reranking, and audited evidence packing."""
import hashlib
import importlib.metadata
import json
import math
import os
import sqlite3
import time
from pathlib import Path

from scripts.context_selection import is_supporting_metadata, pack_evidence, rerank_candidates
from scripts.eval_embeddings import ROOT, configure_text_input, format_input
from rag_app.storage import SourceStore

RUN = ROOT / "data/retrieval_runs/gemma2_structured_512_html_v1/config.json"
TOKENIZER_DIR = ROOT / "data/tokenizers/bge_wordpiece"
RERANKER_DIR = ROOT / os.getenv("RAG_RERANKER_DIR", "data/models/hub/models--BAAI--bge-reranker-base/snapshots/2cfc18c9415c912f9d8155881c133215df768a70")
RERANKER_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
COLLECTION = "techqa_html_v1_gemma2"
CORPUS_VERSION = "html_v1"


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _versions(names):
    return {name: importlib.metadata.version(name) for name in names}


class Retriever:
    def __init__(self, source_store=None, qdrant_client=None, *, embed_query=None, score_pairs=None,
                 tokenizer=None, cache_path=None, index_manifest=None, index_root=None):
        self.index_root = Path(index_root or ROOT / "data/indexes")
        active_path = self.index_root / "active.json"
        self.index_manifest = index_manifest
        if self.index_manifest is None and source_store is None and active_path.is_file():
            self.index_manifest = json.loads(active_path.read_text())
        self.collection = self.index_manifest['collection'] if self.index_manifest else COLLECTION
        self.corpus_version = self.index_manifest['corpus_version'] if self.index_manifest else CORPUS_VERSION
        source_path = ROOT / "data/source.sqlite3"
        if self.index_manifest:
            from rag_app.indexes import IndexManager
            manager = IndexManager(self.index_root, qdrant_client)
            source_path = manager.source_path(self.index_manifest)
            if _sha256(source_path) != self.index_manifest['source_sha256']:
                raise ValueError("Active source hash does not match index manifest")
        if source_store is None:
            if not source_path.is_file():
                raise FileNotFoundError(f"Source store is missing; run scripts/import_qdrant.py first: {source_path}")
            self.source = SourceStore(source_path, readonly=True)
        else:
            self.source = source_store
        self.qdrant = qdrant_client
        self._owns_qdrant = qdrant_client is None
        self.embed_query = embed_query
        self.score_pairs = score_pairs
        self.tokenizer = tokenizer
        self.cache_path = Path(cache_path or ROOT / "data/app/retrieval_cache.sqlite")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = sqlite3.connect(str(self.cache_path), check_same_thread=False)
        self.cache.executescript("""
            CREATE TABLE IF NOT EXISTS query_vectors (cache_key TEXT PRIMARY KEY, vector TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS pair_scores (cache_key TEXT PRIMARY KEY, score REAL NOT NULL);
        """)
        self.run_config = json.loads(RUN.read_text()) if self.index_manifest is None else {"cache_key": self.index_manifest["build_id"]}
        self.spec = json.loads((ROOT / "models-embeddings.json").read_text())["models"]["gemma2"]
        if self.index_manifest:
            expected = self.index_manifest['encoder'].get('models_sha256')
            if expected and expected != _sha256(ROOT / 'models-embeddings.json'):
                raise ValueError("Active index requires its pinned embedding configuration")
        self._prune_cache()
        self.model = None
        self.native_tokenizer = None
        self.ce_model = None
        self.ce_tokenizer = None
        self._gemma_assets = None
        self._reranker_assets = None
        self._owns_source = source_store is None

    def _prune_cache(self):
        # Fixed-size vectors and hashed keys keep these entry caps bounded on disk.
        for table, cap in (("query_vectors", 2048), ("pair_scores", 50000)):
            self.cache.execute(f"DELETE FROM {table} WHERE rowid NOT IN (SELECT rowid FROM {table} ORDER BY rowid DESC LIMIT ?)", (cap,))
        self.cache.commit()

    def check_assets(self):
        directory = ROOT / "data/models/hub/models--google--embeddinggemma-2/snapshots" / self.spec['revision']
        if any(not (directory / f['rfilename']).is_file() or (directory / f['rfilename']).stat().st_size != f['size'] for f in self.spec['files']):
            raise ValueError("Pinned Gemma assets are missing or incomplete")
        if any(not (RERANKER_DIR / name).is_file() for name in (
                'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json')):
            raise ValueError("Pinned reranker assets are missing")

    def _gemma_policy(self):
        # The input cache identity derives from the pinned file hashes in the run manifest.
        files = [(f["rfilename"], f.get("lfs", {}).get("sha256", f["blobId"])) for f in self.spec["files"]]
        return {"model": self.spec["model_id"], "revision": self.spec["revision"], "assets": files,
                "device": "cpu", "dtype": "float32", "attention": "eager",
                "query_prefix": self.spec["query_prefix"],
                "software": _versions(["torch", "transformers", "sentence-transformers"])}

    def _encode(self, text):
        if self.embed_query is not None:
            return self.embed_query(text)
        if self.model is None:
            os.environ["HF_HUB_CACHE"] = str(ROOT / "data/models/hub")
            import torch
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer
            torch.set_num_threads(4)
            directory = ROOT / "data/models/hub/models--google--embeddinggemma-2/snapshots" / self.spec["revision"]
            self._gemma_assets = {}
            for asset in self.spec["files"]:
                path = directory / asset["rfilename"]
                if not path.is_file() or path.stat().st_size != asset["size"]:
                    raise FileNotFoundError("Verified local Gemma snapshot is missing; restore its pinned assets before serving")
                digest = _sha256(path)
                expected = asset.get("lfs", {}).get("sha256")
                if expected:
                    actual = digest
                else:
                    raw = path.read_bytes()
                    actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                if actual != (expected or asset["blobId"]):
                    raise ValueError("Local Gemma asset hash mismatch")
                self._gemma_assets[asset["rfilename"]] = digest
            self.model = SentenceTransformer(str(directory), device="cpu", trust_remote_code=False,
                local_files_only=True, config_kwargs=self.spec["config_kwargs"],
                model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"})
            self.native_tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True,
                                                                   trust_remote_code=False)
            self.model.float().eval()
            self.model.default_prompt_name = None
            self.model.max_seq_length = self.spec["max_length"]
            configure_text_input(self.model, "gemma2")
            if self.model.get_sentence_embedding_dimension() != 768:
                raise ValueError("Gemma query encoder did not load a 768-dimensional model")
        token_count = len(self.native_tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        if token_count > self.spec["max_length"]:
            raise ValueError(f"Formatted query exceeds Gemma's {self.spec['max_length']}-token limit")
        return self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)[0]

    def _query_vector(self, question, metrics):
        formatted = format_input(self.spec, question, query=True)
        key = _fingerprint([self._gemma_policy(), formatted])
        row = self.cache.execute("SELECT vector FROM query_vectors WHERE cache_key=?", (key,)).fetchone()
        if row:
            metrics["query_cache_hits"] += 1
            vector = json.loads(row[0])
        else:
            metrics["query_cache_misses"] += 1
            vector = self._encode(formatted)
        vector = [float(x) for x in vector]
        norm = math.sqrt(sum(x * x for x in vector))
        if len(vector) != 768 or not all(math.isfinite(x) for x in vector) or not math.isclose(norm, 1, abs_tol=1e-3):
            raise ValueError("Gemma returned an invalid query vector")
        if not row:
            self.cache.execute("INSERT OR REPLACE INTO query_vectors VALUES (?, ?)",
                               (key, json.dumps(vector, separators=(",", ":"))))
            self._prune_cache()
        return vector

    def _client(self):
        if self.qdrant is None:
            from qdrant_client import QdrantClient
            self.qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
                                       api_key=os.getenv("QDRANT_API_KEY"))
        return self.qdrant

    def index_identity(self):
        points, _ = self._client().scroll(self.collection, limit=1, with_payload=True, with_vectors=False)
        active = points[0].payload.get("index_build_id") if points else None
        if active != self.run_config["cache_key"]:
            raise ValueError("Qdrant index version does not match source manifest; run verified importer")
        if self.index_manifest and self._client().get_collection(self.collection).points_count != self.index_manifest['counts']['chunks']:
            raise ValueError("Active index count mismatch")
        return {"collection": self.collection, "build": active}

    def _tokenizer(self):
        if self.tokenizer is None:
            from tokenizers import Tokenizer
            record = json.loads((TOKENIZER_DIR / "source.json").read_text())
            if _sha256(TOKENIZER_DIR / "tokenizer.json") != record["tokenizer_sha256"]:
                raise ValueError("Reference BGE tokenizer hash does not match its manifest")
            self.tokenizer = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
        return self.tokenizer

    def _reranker_policy(self):
        if self.score_pairs is not None:
            assets = {"injected": "test-seam"}
        else:
            if not RERANKER_DIR.is_dir():
                raise FileNotFoundError(f"Pinned local BGE reranker snapshot is missing: {RERANKER_DIR}")
            if self._reranker_assets is None:
                self._reranker_assets = {p.name: _sha256(p) for p in sorted(RERANKER_DIR.iterdir()) if p.is_file()}
                if "model.safetensors" not in self._reranker_assets:
                    raise FileNotFoundError("Pinned local BGE reranker weights are missing")
        return {"model": "BAAI/bge-reranker-base", "revision": RERANKER_REVISION,
                "assets": assets if self.score_pairs is not None else self._reranker_assets,
                "device": "cpu", "dtype": "float32", "max_length": 512,
                "truncation": "longest_first", "batch_size": 16,
                "software": _versions(["torch", "transformers", "tokenizers"])}

    def _score(self, pairs):
        if self.score_pairs is not None:
            return self.score_pairs(pairs)
        if self.ce_model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            torch.set_num_threads(4)
            self.ce_tokenizer = AutoTokenizer.from_pretrained(str(RERANKER_DIR), local_files_only=True,
                                                               trust_remote_code=False)
            self.ce_model = AutoModelForSequenceClassification.from_pretrained(
                str(RERANKER_DIR), local_files_only=True, trust_remote_code=False,
                dtype=torch.float32).cpu().eval()
        import torch
        values = []
        with torch.inference_mode():
            for start in range(0, len(pairs), 16):
                batch = pairs[start:start + 16]
                encoded = self.ce_tokenizer([q for q, _ in batch], [p for _, p in batch],
                    padding=True, truncation="longest_first", max_length=512, return_tensors="pt")
                values.extend(self.ce_model(**encoded).logits.squeeze(-1).tolist())
        return values

    def _rerank_scores(self, question, chunks, docs, candidate_ids, metrics):
        substantive = [cid for cid in candidate_ids if not is_supporting_metadata(chunks[cid], docs)][:20]
        policy = self._reranker_policy()
        result = {}
        missing = []
        for cid in substantive:
            c = chunks[cid]
            passage = docs[c["document_id"]]["text"][c["start"]:c["end"]]
            key = _fingerprint([policy, question, passage])
            cached = self.cache.execute("SELECT score FROM pair_scores WHERE cache_key=?", (key,)).fetchone()
            if cached:
                result[cid] = float(cached[0])
                metrics["pair_cache_hits"] += 1
            else:
                missing.append((cid, key, passage))
        if missing:
            values = self._score([(question, passage) for _, _, passage in missing])
            if len(values) != len(missing):
                raise ValueError("BGE reranker returned a different number of scores than pairs")
            for (cid, key, _), value in zip(missing, values):
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("BGE reranker returned a nonfinite score")
                result[cid] = value
                self.cache.execute("INSERT OR REPLACE INTO pair_scores VALUES (?, ?)", (key, value))
                metrics["pair_cache_misses"] += 1
            self._prune_cache()
        return result

    def retrieve(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        total_started = time.perf_counter()
        metrics = {"query_cache_hits": 0, "query_cache_misses": 0,
                   "pair_cache_hits": 0, "pair_cache_misses": 0,
                   "encoder_revision": self.spec["revision"], "reranker_revision": RERANKER_REVISION,
                   "selector_sha256": _sha256(ROOT / "scripts/context_selection.py")}
        started = time.perf_counter()
        vector = self._query_vector(question, metrics)
        metrics["embed_ms"] = (time.perf_counter() - started) * 1000

        try:
            from qdrant_client import models
            exact_search = models.SearchParams(exact=True)
        except ImportError:
            if self.qdrant is None:
                raise
            exact_search = {"exact": True}
        started = time.perf_counter()
        response = self._client().query_points(collection_name=self.collection, query=vector,
            limit=100, search_params=exact_search, with_payload=True)
        metrics["search_ms"] = (time.perf_counter() - started) * 1000
        points = response.points
        candidate_ids = []
        chunks, docs = {}, {}
        for point in points:
            payload = point.payload or {}
            if payload.get("corpus_version") != self.corpus_version or not payload.get("chunk_id"):
                continue
            if self.index_manifest and payload.get('index_build_id') != self.index_manifest['build_id']:
                raise ValueError("Retrieved point belongs to a different index build")
            cid = payload["chunk_id"]
            chunk = self.source.get_chunk(cid)
            if chunk["document_id"] != payload.get("doc_id"):
                raise ValueError(f"Qdrant payload/source mismatch for chunk {cid}")
            chunks[cid] = chunk
            docid = chunk["document_id"]
            if docid not in docs:
                docs[docid] = self.source.get_document(docid)
            candidate_ids.append(cid)
        if not candidate_ids:
            raise ValueError("Qdrant returned no compatible TechQA candidates")

        started = time.perf_counter()
        scores = self._rerank_scores(question, chunks, docs, candidate_ids, metrics)
        ranked, rank_metrics = rerank_candidates(candidate_ids, chunks, docs, scores, depth=20)
        metrics["rerank_ms"] = (time.perf_counter() - started) * 1000
        metrics.update(rank_metrics)

        started = time.perf_counter()
        packed = pack_evidence(ranked, chunks, docs, self._tokenizer(), budget=4096,
                               per_doc=5, max_docs=8, skip_metadata=True, include_notes=True)
        metrics["pack_ms"] = (time.perf_counter() - started) * 1000
        metrics["dense_candidates"] = len(candidate_ids)
        metrics["context_tokens"] = packed["tokens"]
        metrics["pack_diagnostics"] = packed["diagnostics"]

        selected = []
        sources = []
        for passage in packed["passages"]:
            docid = passage["document_id"]
            overlaps = [cid for cid in ranked if chunks[cid]["document_id"] == docid and
                        chunks[cid]["start"] <= passage["end"] and passage["start"] <= chunks[cid]["end"]]
            cid = overlaps[0] if overlaps else next(cid for cid in ranked if chunks[cid]["document_id"] == docid)
            doc = docs[docid]
            sid = f"S{len(sources) + 1}"
            source = {"id": sid, "document_id": docid, "chunk_id": cid,
                      "title": doc.get("title") or doc.get("metadata", {}).get("title", ""),
                      "url": doc.get("metadata", {}).get("canonicalUrl", ""),
                      "text": doc["text"][passage["start"]:passage["end"]]}
            sources.append(source)
            selected.append({**passage, "source_id": sid, "chunk_ids": overlaps})
        metrics["total_ms"] = (time.perf_counter() - total_started) * 1000
        return {"sources": sources, "context_tokens": packed["tokens"], "context": packed["text"],
                "candidate_chunk_ids": candidate_ids, "ranked_chunk_ids": ranked,
                "candidate_documents": {cid: chunks[cid]["document_id"] for cid in candidate_ids},
                "selected_passages": selected, "metrics": metrics}

    def close(self):
        self.cache.close()
        if self._owns_source:
            self.source.close()
        if self._owns_qdrant and self.qdrant is not None:
            self.qdrant.close()


_default_retriever = None


def retrieve(question):
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever.retrieve(question)
