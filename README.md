# TechQA RAG

A technical-support RAG workspace built around IBM TechQA. It retrieves source passages, writes a cited answer, and checks the answer before showing it. If the evidence is insufficient, it withholds the solution.

The repository contains application code, preparation/evaluation tools and offline tests. Datasets, model weights, vectors, credentials, evaluation outputs and working notes are kept outside Git.

## How it works

```text
Question → Gemma embedding → Qdrant exact search → BGE reranking
         → evidence selection → Groq answer → separate verification
         → cited answer + checked OpenUI components
```

- **Retrieval:** pinned `google/embeddinggemma-2`, cosine search over structured HTML chunks, BGE reranking and reciprocal-rank fusion with the dense ranking.
- **Evidence:** a bounded context pack retains source identities and coordinates. The serving retriever currently uses dense search; BM25/hybrid helpers are available for evaluation, not enabled in the live path.
- **Answers:** Groq `openai/gpt-oss-120b` writes and separately verifies an answer. One repair is allowed; unresolved evidence gaps withhold the answer.
- **UI:** Next.js/React and OpenUI render finding cards, steps, configuration rows and citations. All generated factual fields, including hidden tabs, are checked before release. Invalid layouts fall back to verified text.
- **Sessions:** recent conversation plus a summary, current-session memory, exact answer caching and an explicit regeneration bypass.
- **Evaluation:** an offline RAGAS runner scores saved evidence and released answers. Judge decisions and local query embeddings are cached; withheld responses and evaluation errors stay visible.

## Requirements

Python 3.13, Node.js 22, npm, uv, Docker Compose, a Groq API key, and prepared TechQA/model artifacts. The frontend and API run locally; Qdrant runs in Docker.

### Run the offline checks first

No datasets, model downloads, API keys or Qdrant server are needed for these checks:

```bash
uv venv --python 3.13 .venv-test
uv pip install --python .venv-test/bin/python -r requirements-test.txt
.venv-test/bin/python -m pytest -q

cd frontend
npm ci
npm run test:genui
npm run typecheck
npm run lint
npm run build
```

GitHub Actions runs the same checks. The tests use synthetic data and controlled model outputs; they are not a live answer-quality benchmark.

### Configure the application

```bash
uv venv --python 3.13 .venv-ingest
uv pip install --python .venv-ingest/bin/python -r requirements-app.txt
cp .env.example .env
```

Set `GROQ_API_KEY` in the root `.env`. Existing installations can use the `GROK_API_KEY` alias. Keep the key server-side; do not use a `NEXT_PUBLIC_` variable.

The model namespace starts with `openai/`, but requests go to **Groq's API**. The model is configured centrally in `rag_app/generation.py`.

### Prepare or restore artifacts

This is a code-only checkout, not a bundled dataset or one-command model download. Restore the matching prepared artifacts before serving:

| Location | Purpose |
|---|---|
| `data/runs/html_v1/documents.json` | Prepared document text and metadata |
| `data/runs/html_v1/structured_512/chunks.jsonl` | Ordered document chunks |
| `data/retrieval_runs/gemma2_structured_512_html_v1/config.json` | Source hashes, vector-cache identity and query membership |
| `data/embedding_cache/<cache_key>/state.json` and `vectors.npy` | Complete, verified document/query vector cache |
| `data/tokenizers/bge_wordpiece/` | Pinned reference tokenizer and its `source.json` hash record |
| `data/models/hub/models--google--embeddinggemma-2/snapshots/<revision>/` | Local Gemma snapshot matching `models-embeddings.json` |
| `data/models/hub/models--BAAI--bge-reranker-base/snapshots/2cfc18c9415c912f9d8155881c133215df768a70/` | Local BGE reranker snapshot |

`RAG_RERANKER_DIR` can override the reranker location with an absolute path or a path relative to the project root. Serving validates local assets and does not silently download missing weights.

The current importer deliberately targets the audited snapshot: **28,482 articles, 99,697 document chunks and 120 trailing query vectors**. Query vectors are excluded from the Qdrant collection. A different corpus/chunk build needs a reviewed importer/configuration update, not relaxed integrity checks.

Preparation code is included in `scripts/audit_techqa.py`, `baseline.py`, `chunking.py`, `ingest_html.py` and `eval_embeddings.py`. Rebuilding requires the original release, tokenizer manifests and the pinned model access; embedding preparation uses a CUDA runtime. It is separate from the local serving setup.

Obtain TechQA from the [PrimeQA release](https://huggingface.co/datasets/PrimeQA/TechQA), pinned revision `60437bc79ab217679682217598a3693cab78365b`. Check the release's bundled data terms; the local audit identified CDLA Permissive 1.0 despite different hosting metadata. Dataset/model licenses are separate from this code repository. No dataset text or model weights are redistributed here.

### Import the existing vectors

```bash
docker compose up -d
.venv-ingest/bin/python -m scripts.import_qdrant --dry-run
.venv-ingest/bin/python -m scripts.import_qdrant
```

On a Mac using Colima, start Colima first. If Compose is installed as a standalone executable, use `docker-compose` instead of `docker compose`.

The importer verifies manifests, dimensions, finite/unit vectors and row identities before writing. It builds `data/source.sqlite3` and the `techqa_html_v1_gemma2` collection. Keep source/index backups together. Do not use `docker compose down -v` unless you intend to delete the database volume.

### Start the API and UI

In one terminal:

```bash
.venv-ingest/bin/python -m uvicorn rag_app.api:app --host 127.0.0.1 --port 8000
```

In another:

```bash
cd frontend
npm ci
npm run dev -- --hostname 127.0.0.1
```

Open `http://127.0.0.1:3000`. The example `APP_ORIGIN` matches that address. For another origin, update it explicitly. The frontend proxies to `http://127.0.0.1:8000`; set `RAG_API_URL` in `frontend/.env.local` to change the upstream.

`GET /health/live` checks the API process. `GET /health/ready` checks key/source/index readiness. The first request loads local models, so cold-start latency differs from subsequent requests.

## Run RAGAS offline

```bash
uv venv --python 3.13 .venv-ragas
uv pip install --python .venv-ragas/bin/python -r requirements-ragas.txt
.venv-ingest/bin/python -m scripts.eval_ragas collect --limit 3
.venv-ragas/bin/python -m scripts.eval_ragas run
```

Collection additionally requires `data/splits/techqa_v1.json` and the original training question/document files under `data/raw/techqa/TechQA/training_and_dev/`. References are checked against original answer spans and never passed to inference.

Collection saves standalone canonical answers, without session memory or UI generation. Scoring runs context precision, context recall, faithfulness and answer relevancy sequentially. It reuses local Gemma for relevancy embeddings. Results, per-call marking decisions, usage and caches stay under `data/ragas/`. Use a new output directory for a changed case set; see `--help` for paths and judge/timeout options.

The current runner defaults to all four metrics and has **no hard token/dollar budget**. Use small case limits and inspect quota/usage before expanding. Failed evaluations are not zero scores; answer metrics are skipped on withheld cases. LLM judges can alter facts during claim extraction, so inspect their decisions before using scores as quality gates.

## Operating limits

- Intended for controlled local/demo use. Authentication and a public multi-user deployment are not implemented.
- Sessions and answer caches are process-local. Persistent query/pair caches need an eviction policy before sustained public traffic.
- Requests have queue, rate and deadline controls. Groq quota waits can dominate latency.
- Verified components appear progressively after checking; unverified draft tokens are not streamed to users.
- Backup restoration, broader answer-quality evaluation and production load testing remain separate work.

The initial history is a reconstructed import of local build stages from October 28–November 3, 2026. Author and committer dates identify those reconstructed stages; the repository was first published on November 3, 2026. It is not an archive of the exact working tree at each historical moment.
