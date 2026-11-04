# TechQA RAG

A technical-support RAG workspace built around IBM TechQA. It retrieves source passages, writes a cited answer, and checks the answer before showing it. If the evidence is insufficient, it withholds the solution.

The repository contains application code, preparation/evaluation tools and offline tests. Datasets, model weights, vectors, credentials, raw evaluation outputs and working notes are kept outside Git. A redacted evaluation summary and synthetic examples are published below.

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

## Evaluation suite

![Evaluation layers: components, pipeline, application, regression and planned monitoring](evaluations/overview.svg)

**Evaluate the parts, then their handoff, then the delivered answer.** RAGAS supplies semantic metrics; our Python checks cover source/citation contracts, recorded usage, judge integrity and paired regressions. Unit tests exercise failure handling with controlled fixtures. No single score is called “RAG accuracy.”

| Layer | What we check | Implementation and status |
|---|---|---|
| Components | Retrieval Recall@K, MRR and nDCG against known article labels; writer/verifier behavior with supplied evidence | [Rank metrics](scripts/baseline.py), [retrieval tests](tests/test_app_retriever.py), [generator contract fixtures](tests/test_app_generation.py). Controlled fixtures are not a generator quality benchmark. |
| Pipeline | Selected context precision/recall, answer faithfulness and relevancy | [RAGAS runner](scripts/eval_ragas.py), measured on the pilot below. Reference text stays outside inference. |
| Application | Reference-based correctness and completeness; citation validity; withholding and layout contracts; recorded timing/tokens/cache use | Correctness = RAGAS `FactualCorrectness(mode="f1")`; completeness = its `mode="recall"` proxy. Both are opt-in and not yet benchmarked. [Free saved-case checks](scripts/eval_suite.py), [API contracts](tests/test_app_api.py), [layout checks](frontend/tests/genui-contract.test.mjs) are implemented. |
| Regression | Re-run a fixed case set, then inspect per-case differences | [Comparison CLI](scripts/eval_suite.py) rejects changed cohorts/references. Semantic comparisons also require matching judge settings and exclude scores flagged for review. |
| Online | Sampled live judging, feedback, drift and failure-to-test-set promotion | **Planned.** Content-free operational logs exist; live quality scoring, Langfuse integration and automated feedback loops do not. |

Style rubrics and broad toxicity/PII/jailbreak evaluation are also planned. Existing origin/session, withholding and generated-component checks cover specific application contracts; they are not a comprehensive safety benchmark. Optional semantic metrics compare against available references, which may omit valid alternative answers.

### Component measurements already recorded

A separate [offline retrieval replay](evaluations/results/retrieval_baseline_v1.json) used **97 known-positive training tuning questions**. Article Recall@5 was **83/97 (85.57%)** and Recall@10 **87/97 (89.69%)**. The packed-context reference-word-run proxy passed **80/97 (82.47%)**; that proxy is not semantic context recall.

A 120-query local Qdrant search check recorded **24.41 ms p50 / 38.54 ms p95**, using saved query vectors. Those timings exclude query encoding, reranking and generation. These historical component runs have different conditions and denominators from the three-case pipeline pilot below.

### Measured pilot, with its limitations

The pilot used the **first three sorted answerable IDs** in the saved training tuning split. It produced **one answer and two withheld responses**. Final development was not evaluated.

| Case | Outcome | Context precision | Context recall | Faithfulness | Answer relevancy |
|---|---|---:|---:|---:|---:|
| TRAIN_Q007 | Withheld | 0.756 | 0.000 | Skipped | Skipped |
| TRAIN_Q008 | Withheld | 0.887 | 1.000 | Skipped | Skipped |
| TRAIN_Q013 | Answered | 0.792 | 1.000 | **0.500 — judge error flagged** | 0.979 |
| Raw mean | 1 answered / 3 attempted | **0.812 (n=3)** | **0.667 (n=3)** | **0.500 (n=1)** | **0.979 (n=1)** |

Eight metric results succeeded, four were deliberately skipped, and the completed pass had zero metric errors. These are integration observations, not final accuracy or a measured improvement.

The failure review matters as much as the averages:

- **Q007:** the selected evidence missed the reference information. Related MQ passages could score as useful while context recall remained zero.
- **Q008:** the selected evidence supported the reference, but the pipeline withheld. Inspect the release path, rather than assume retrieval failed.
- **Q013:** the judge changed **“TIP 2.1” to “TIP 1”** while extracting claims, then penalized its altered claim. The raw faithfulness score is retained and flagged; it is not reliable evidence of an unsupported application answer. The new version-preservation heuristic catches this for review without rewriting the score.

The [published, redacted result](evaluations/results/techqa_pilot_v1.json) includes per-case outcomes, denominators, settings, source-artifact hashes and review flags. Questions, references, source passages, credentials and full judge traces remain local. The summary records the November 3 run; today's export and additional contract checks made **zero new live judge calls**.

| Recorded operation | Observation | Scope |
|---|---:|---|
| Answer coverage | 1/3 | Reported separately from conditional answer scores |
| Selected gold-article hit | 2/3 | Final evidence only; **not search Recall@10** or semantic recall |
| Citation contract | 1/1 released answer | IDs/format only; does not prove factual support |
| Completed resumed scoring pass | 48.90 s; 17,135 reported tokens | 25 cached judgments, 8 fresh judgments; earlier passes cost extra |
| Historical cache replay | 33 hits, 0 misses, **0 API calls** | Same raw scores; 0.022 s scoring loop excludes startup/imports |
| Correctness / completeness | Not measured | Available as opt-in RAGAS checks |
| Dollar cost / online drift | Not measured | Pricing and online monitoring not configured |

### Try the suite without a model or dataset

After installing `requirements-test.txt`, run the [synthetic demonstration](evaluations/fixtures/offline_demo.json):

```bash
python -m scripts.eval_suite report \
  --cases evaluations/fixtures/offline_demo.json --output /tmp/eval-demo.json
python -m scripts.eval_suite compare \
  --baseline /tmp/eval-demo.json --candidate /tmp/eval-demo.json \
  --output /tmp/eval-diff.json
```

The report checks selected-document hits, citation contracts and outcomes. It does not run an LLM. The comparison prints paired changes; an unchanged demo should have no regressions. CI exercises this path, the budget controls and actual RAGAS aggregation with mocked judge decisions. Synthetic test results are kept separate from the TechQA pilot.

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
.venv-ragas/bin/python -m scripts.eval_ragas run \
  --metrics context_precision context_recall --limit 3 --max-requests 12
```

Collection additionally requires `data/splits/techqa_v1.json` and the original training question/document files under `data/raw/techqa/TechQA/training_and_dev/`. References are checked against original answer spans and never passed to inference.

Collection saves standalone canonical answers, without session memory or UI generation. Scoring runs only the explicitly selected metrics, sequentially. Only answer relevancy loads local Gemma; the other judge metrics do not require local embedding weights. Results, per-call marking decisions, usage and caches stay under `data/ragas/`. Use a new output directory for a changed case set or serving-code version; see `--help` for paths and judge/timeout options.

Choose metrics for the change you are investigating:

- Retrieval/context change: `--metrics context_precision context_recall`.
- Answer-writing change: `--metrics faithfulness answer_relevancy`.
- Reference-based application review: `--metrics factual_correctness answer_completeness` (opt-in; no pilot scores yet).

The default scoring limit is three cases and **12 HTTP attempts, including retries**. Set `--max-requests 0` for cache-only execution; missing decisions become `budget_exhausted`, not zero quality. Use a fresh `--output` for a comparison; omitted output names are timestamped so earlier reports survive. These are request caps, **not a dollar/token spending guarantee**. Inspect actual usage before raising them; one metric can make several calls.

Create a free report from actual local artifacts:

```bash
python -m scripts.eval_suite report --cases data/ragas/pilot_v1 \
  --ragas data/ragas/results/pilot_v1.json --output /tmp/pilot-review.json
```

Compare two reports with the `compare` command above. Cohort and reference identities must match. Judge/embedding settings must also match before semantic score deltas are reported; flagged scores stay visible but are excluded from those deltas. A negative delta is an observed regression to inspect, not statistical proof.

Failed evaluations are not zero scores; answer metrics are skipped on withheld cases. LLM judges can alter facts during claim extraction, so inspect their decisions before using scores as quality gates.

## Operating limits

- Intended for controlled local/demo use. Authentication and a public multi-user deployment are not implemented.
- Sessions and answer caches are process-local. Persistent query/pair caches need an eviction policy before sustained public traffic.
- Requests have queue, rate and deadline controls. Groq quota waits can dominate latency.
- Verified components appear progressively after checking; unverified draft tokens are not streamed to users.
- Backup restoration, broader answer-quality evaluation and production load testing remain separate work.

The initial history is a reconstructed import of local build stages from October 28–November 3, 2026. Author and committer dates identify those reconstructed stages; the repository was first published on November 3, 2026. It is not an archive of the exact working tree at each historical moment. Subsequent commits record ongoing development normally.
