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
- **Prompts:** `strict_v5_message_roles` puts task instructions in system messages and questions, evidence, drafts and repair findings in JSON user messages. Follow-ups, memory summaries and OpenUI use the same structure. Output schemas remain explicit in the provider request.
- **UI:** Next.js/React and OpenUI render finding cards, steps, configuration rows and citations. All generated factual fields, including hidden tabs, are checked before release. Invalid layouts fall back to verified text.
- **Sessions:** recent conversation plus a summary, current-session memory, exact answer caching and an explicit regeneration bypass.
- **Evaluation:** an offline RAGAS runner scores saved evidence and released answers. Judge decisions and local query embeddings are cached; withheld responses and evaluation errors stay visible.

## Inspect a run in LangSmith

Add `LANGSMITH_API_KEY` to the server's `.env`. The default project is `techqa-rag`. Set `LANGSMITH_TRACING=false` to disable export. Traces contain questions, retrieved passages, private drafts and verification decisions, so keep this project private.

Each question has a parent trace with child steps for query embedding, exact search, reranking, context selection and model calls. Inspect timings, actual token usage, cache hits, source passages, writer/verifier decisions and the released outcome. The answer workspace links to its trace. Local diagnostic logs continue to work when tracing is disabled or export fails.

The workspace streams actual activity and retrieved excerpts. Verified answers and checked OpenUI layouts appear after their release checks. The run history remains available after completion. Desktop uses a compact navigation rail, while smaller screens stack the answer and evidence.

For developers and coding agents, `/ask` remains a server-sent event endpoint. `stage` events describe activity, `evidence` carries retrieved `sources`, `final` carries the approved outcome and `error` carries a safe error category/message. `done` closes the run. Final statuses are `answered`, `needs_clarification` and `insufficient_evidence`. Existing exact-cache, session, cancellation and regeneration behavior remains covered by tests.

Collect a fixed random training sample with a hard HTTP-attempt cap. Provider HTTP 429 responses stop collection and are retried when the same unchanged run resumes. Local per-request quota deadlines are recorded as failed requests while collection continues. Completed cases are saved after each question, with a `completed.json` snapshot for inspection.

```bash
python -m scripts.eval_ragas collect --limit 12 --sample-seed 20261110 \
  --max-requests 18 --output data/ragas/review
python -m scripts.publish_langsmith --cases data/ragas/review/completed.json
```

Publishing replays saved outputs into a LangSmith experiment with response status, selected-reference-article presence and citation-format checks. It makes **zero new model calls**. Add `--ragas path/to/matching-report.json` to publish existing semantic scores too. RAGAS judging remains a separate, explicitly budgeted command below. Reviewed or failed judge results retain their flags.

The [November 10 observability batch](evaluations/results/observability_v1.json) selected **12 training questions**. Retrieval selected the labeled article for **11/12**. **3 answers** were released and passed citation-format checks. **9 provider quota failures** prevented completion. The run used **15 HTTP attempts and 33,230 reported tokens**. This batch establishes dashboard visibility and exposes a quota-handling issue, which is now covered by a stop-on-quota regression test. New semantic RAGAS scores await available provider quota.

## Feedback and future evaluation cases

Every completed response offers Helpful and Not helpful, with an optional explanation. Sending feedback saves the question, resolved follow-up, answer, sources, private diagnostics and pipeline versions in `data/app/feedback/`. These private records persist after the session ends. A stable feedback record also attaches the rating and comment to the LangSmith trace. If cloud synchronization fails, the local record remains available and `python -m scripts.feedback_cases sync` retries it.

Review the saved case against its sources, write a checked reference answer in a local text file, then approve and export it.

```bash
python -m scripts.feedback_cases approve --run-id RUN_UUID \
  --reference-file /path/to/checked-answer.txt --notes "Verified against the original source"
python -m scripts.feedback_cases export --output data/ragas/feedback_review_v1
python -m scripts.eval_ragas run --cases data/ragas/feedback_review_v1 \
  --metrics faithfulness factual_correctness answer_completeness --limit 5 --max-requests 20
```

Approval and review notes are required for feedback-based RAGAS evaluation. Use these cases as a regression set alongside the fixed training sample. Human votes and comments identify examples for review, while checked references support correctness scoring. Human ratings can also help assess and improve judge agreement. Each export uses a new directory to preserve the earlier evaluation version.

## Evaluation suite

![Evaluation layers: components, pipeline, application, regression and planned monitoring](evaluations/overview.svg)

The suite records retrieval quality, generated-answer quality and API workflow behavior. RAGAS supplies semantic judgments. Python checks source/citation contracts, session/cache behavior, recorded usage and paired regressions.

| Layer | What we check | Implemented measurements |
|---|---|---|
| Component | Retriever ranking and generator with supplied original article evidence | Recall@K, MRR and nDCG in the retrieval benchmark. Two supplied-evidence generator cases scored with RAGAS faithfulness, relevancy and reference metrics. |
| Pipeline | Actual retrieved context through writing and verification | Ten fixed training questions. All six metric types scored for eligible cases, with explicit skipped/error states and denominators. |
| Workflow | API conversation turns, follow-up resolution and exact answer caching | Two real API turns scored with RAGAS. Same-session follow-up and repeated-question cache checks also run. [Layer collector](scripts/eval_layers.py). |
| Regression | Matched cases and references | [Comparison CLI](scripts/eval_suite.py) checks cohort and judge identity. The saved v5/v6 outputs were rescored with the same DeepSeek judge. |
| Online | Feedback and run inspection | LangSmith traces, user ratings and reviewed regression exports. Live sampled judging and drift monitoring are planned. |

### Ten-question evaluation audit

The [redacted audit](evaluations/results/layered_audit_10.json) uses the first ten IDs from the fixed training sample. Today's writer, verifier and judge use TokenHarbor `deepseek-v4.1-flash:free`. Historical v5/v6 outputs retain their original Groq model attribution. The judge passed supported/unsupported faithfulness sanity checks at 1.0/0.0.

| Metric | DeepSeek pipeline mean | Scored / eligible |
|---|---:|---:|
| Context recall | 100.0% | 10/10 |
| Context precision | 83.9% | 10/10 |
| Faithfulness | 96.6% | 7/7 |
| Answer relevancy | 88.5% | 7/7 |
| Reference factual correctness | 40.9% | 7/7 |
| Reference-fact recall | 78.6% | 7/7 |

The pipeline released seven answers, withheld two and requested clarification once. It had zero generation errors after the DeepSeek adapter correction. Supplied-evidence component faithfulness was 100% over two cases. Workflow faithfulness was 100% and answer relevancy 94.2% over two turns. Repeated-question caching passed. These small training diagnostics require human review before a release decision.

On the four shared released historical cases, the same DeepSeek judge scored reference correctness **50.0% → 37.8%** from v5 to v6. Q117's broader version assertions contribute to the drop. Q548 also exposes bundled-claim and overly literal error-string judgments. Q376 mentions details and asks where to find them; its reference expects a vulnerability explanation. A reviewed task rubric should resolve that scope difference. The new verifier also incorrectly rejects qualified version differences and demands unrequested remediation. These findings guide the next change.

The legacy `answer_completeness` field uses RAGAS `FactualCorrectness(mode="recall")`. It measures reference agreement. Required-task completeness needs explicit review of the question and requested steps. A separate `reference_claim_coverage` diagnostic reuses cached reference-to-answer judgments. Q425 covers both reference commands even though the original recall-mode score is zero because it rejected bundled answer claims. Both values stay visible and require review. Claim decomposition and coverage settings affect reference scores. [RAGAS metric definitions](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/factual_correctness/)

Each dashboard metric has `ragas_state_*`, `ragas_scored_*` and `ragas_eligible_*` feedback. Skipped answers and failed judgments have reasons. Historical imports include saved duration, model usage and original generation errors. Imported wall clocks anchor those durations at import time. Aggregate usage spans carry recorded token totals. [LangSmith usage instrumentation](https://docs.langchain.com/langsmith/log-llm-trace)

The shared [budget guard](scripts/eval_budget.py) tracks cached inputs, failed preflights and unresolved reservations across generation and judging. The audit stayed at **$0.295/$0.50** of standard-route token value. The selected free route bills zero and consumes its free allowance. Trace names use `Pipeline / ...`, `Component / ...`, `Workflow / ...` and `RAGAS / ...`.

Run the layer collector after collecting the ten pipeline cases. Use the RAGAS runner on each output directory.

```bash
python -m scripts.eval_layers \
  --pipeline data/ragas/ten_questions \
  --output data/ragas/layers \
  --component-ids TRAIN_Q117 TRAIN_Q548 \
  --workflow-root TRAIN_Q013 \
  --followup "Which version must I upgrade it to first?"

python -m scripts.eval_ragas run \
  --provider tokenharbor --judge deepseek-v4.1-flash:free \
  --cases data/ragas/layers/component --limit 2 --max-requests 40 \
  --metrics faithfulness answer_relevancy factual_correctness answer_completeness
```

Set `EVAL_BUDGET_FILE` to a shared ledger path to enforce the $0.50 token-value budget across processes. Set `RAG_LLM_PROVIDER=tokenharbor` for DeepSeek generation. The serving environment and evaluation environment use their pinned requirements.


Style rubrics and broad toxicity/PII/jailbreak evaluation are also planned. Existing origin/session, withholding and generated-component checks cover specific application contracts; they are not a comprehensive safety benchmark. Optional semantic metrics compare against available references, which may omit valid alternative answers.

### Reasoning comparison after trace verification

The [structured recheck](evaluations/results/layered_audit_10.json) evaluates saved generations from the same ten training questions. Evidence, prompts and the 6,000-token output cap were held fixed. Both arms use the same reasoning-disabled judge. The clean pass reused **422 judgments** and made **19 new judge requests**, with zero judge errors.

| Measure | Reasoning off | Reasoning on |
|---|---:|---:|
| Released answers / 10 | 7 | 6 |
| Generation errors / 10 | 0 | 3 |
| Median recorded generation time | 8.73 s | 19.13 s |
| Reported generation tokens | 107,269 | 143,141 |
| Reference F1, five shared released answers | 33.6% | 50.2% |
| Faithfulness, same five answers | 100.0% | 97.1% |

Reasoning stays disabled under the current output cap. All three reasoning-on errors were truncations. The five-case reference improvement requires a larger independent sample before a broader quality claim. Generation timings come from the saved executions. Replay spans report evaluation time and retain original generation durations in their outputs. The provider did not report reasoning-token counts.

The recheck has **24 question chains, 23 threads and 136 metric verdicts**, with zero missing verdict outputs, thread IDs or reported LLM costs. The two workflow turns share a thread. All **144 metric status cells** are present, including eight metrics outside the component/workflow scoring scope. Reference-score review continues around answer scope, version conditions and bundled claims. The combined audit/comparison ledger reached **$0.392/$0.50** of standard-route token value, with one unresolved reservation. The selected free route records zero billed cost.

### Reading evaluation traces

- Each question opens as one chain. Expand the saved pipeline stages, then each RAGAS metric and its judge steps. Conversation turns share a thread ID. Imported child timestamps have zero duration; their outputs retain the original measured duration.
- Each metric reports its score and status. The categorical `ragas_status_*` fields show success, skipped, not run, error, budget exhausted, undefined or needs review. A missing score stays empty. A zero score records a completed judgment with a zero result.
- Context precision judges each passage, so seven passages can require seven judge calls. Faithfulness uses statement extraction and checking. Relevancy uses three generated questions. Reference F1 uses claim extraction and checking in both directions. Cached judgments report zero HTTP calls.
- The TokenHarbor free route reports explicit `$0.00` billed cost. The shared budget ledger separately tracks standard-route token value. Unknown prices stay unknown. Cache and available reasoning-token counts are recorded.
- Completed traces rejected further updates in the current backend. Earlier flat imports remain archived. Replacement imports preserve saved answers and scores, with complete verdicts and parent/child links created together.
- Set `RAG_THINKING=disabled` for the current serving configuration. Use `enabled` only in a separately labeled fixed-evidence comparison. Generation mode is part of the answer-cache identity. The judge stays fixed while generation reasoning changes.

### Bounded generation comparison with prompt version 6

The [saved comparison](evaluations/results/generation_v6.json) keeps the same 30 questions, references and source passages. The updated flow uses one writer, code-only display of valid writer-selected citation IDs and one verifier. A failed factual check withholds the answer. Code formatted citations in **21/30** cases, so prompt changes alone did not solve citation output.

Groq completed 28 cases. At the user's request, the final two used TokenHarbor `deepseek-v4.1-flash:free` with frozen baseline evidence. Those two are excluded from the controlled Groq comparison. One fallback response reached its output limit and was retried with a larger allowance, then withheld. One Groq case retains its provider HTTP429 failure.

| Controlled measurement | Before | After | Cases |
|---|---:|---:|---:|
| Released answers, same Groq model | 14/28 | 18/28 | 28 |
| Citation-format failures | 6 | 0 | 28 |
| Mean model stages, shared released answers | 2.91 | 2.00 | 11 |
| Context recall | 83.3% | 83.3% | 30, unchanged context |
| Context precision | 87.9% | 87.9% | 4, unchanged context |
| Paired faithfulness | 92.8% | 98.3% | 6 |
| Paired reference correctness | 45.0% | 32.3% | 4 |
| Paired completeness | 41.8% | 50.0% | 4 |

Across all 30 outcomes there were **19 answers, 9 withheld responses, 1 clarification and 1 provider failure**. All 19 released answers passed citation-format checks. The paired semantic rows use only shared, unflagged scored cases. Raw candidate faithfulness was **93.0% across 11 answers**. Reference correctness decreased on its four shared cases and requires review. Q117 added version-qualified information beyond the reference, while Q376's bulletin-link reference and Q548's claim alignment remain review concerns. These results do not establish a broad answer-quality improvement.

Collection and diagnostics used **64/66 HTTP attempts and 309,561 reported tokens**, including archived attempts. Recorded judging used **77/120 attempts and 116,119 tokens**, with **172 judge-cache hits** across scoring passes. The judge was stopped when the user ran out of Groq quota, and cache-only assembly made zero model calls. Recorded judge usage may omit in-flight activity at interruption. Answer relevancy was not rescored. Timing differences include quota waits and account changes. Reserved development remains unused.

### Wider training run with prompt version 5

The [saved result](evaluations/results/techqa_wide_v5.json) covers **all 30 planned training questions**. It produced **15 released answers, 6 withheld responses, 1 clarification request and 8 generation failures**. Seven failures omitted inline citations even after repair. One earlier failure reached a local quota-wait deadline. The selected evidence included the labeled article for **25/30** questions. All **15 released answers** passed citation-format checks.

RAGAS used `openai/gpt-oss-20b` after a supported/unsupported faithfulness check scored 1.0/0.0. Answer generation used `openai/gpt-oss-120b`. Samples follow the fixed question order, with answer metrics conditional on released answers.

| RAGAS metric | Raw mean | Scored cases | Review flags |
|---|---:|---:|---:|
| Context recall | 83.3% | 30 | 0 |
| Context precision | 87.9% | 4 | 0 |
| Faithfulness | 94.4% | 10 | 0 |
| Answer relevancy | 93.0% | 3 | 0 |
| Factual correctness against the reference | 42.6% | 5 | 1 |
| Completeness against the reference | 53.4% | 5 | 1 |

Context recall scored the full cohort. Precision scored four of five attempted cases. Expanded correctness/completeness stopped when the provider requested a 527-second wait, beyond the metric deadline. Relevancy retains its earlier three-case sample. Raw means retain flagged values. LangSmith leaves flagged values pending review, so its displayed averages use fewer cases. Source inspection also found a reference-scope issue where a question asks for a bulletin link but its reference contains the full bulletin text. Correctness and completeness need review before use as release criteria.

Generation used **77 HTTP attempts / 396,211 reported tokens**, including the archived rate-limited attempt. Judging used **228 attempts / 384,342 tokens**, plus **4 calibration calls / 3,757 tokens**. The runner reused **108 cached judge decisions**. Median answered-case batch time was **129.7 seconds**, including quota waits and local retrieval. These timings describe a serial batch. Reserved development questions remain available for a later release evaluation.

The collector initially stopped on a local request deadline. That condition was corrected, existing cases were reused, and all 30 questions finished after a provider-key update. Retryable failed attempts are archived before replacement. The pipeline stayed fixed during the evaluation.


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

Standalone generation uses one writer and one verifier, with no factual repair loop. Code can display valid source IDs already selected by the writer while preserving answer text. The verifier then checks support, relevance, completeness and versions. Unknown source IDs fail validation. Rejected answers are withheld or request clarification. This is a two-logical-call limit for standalone generation. Transport retries, follow-up resolution and optional presentation calls are accounted separately.

Generation uses remaining-token/reset headers to pace calls and keeps quota waits within request deadlines. The token estimate is approximate and applies to one provider instance, not a shared multi-worker quota. Offline case collection saves private writer/verifier results (and repair attempts in historical runs) and content-free HTTP statuses for diagnosis; these artifacts stay outside Git and are never returned as released answers.

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

The current importer deliberately targets the audited snapshot: **28,482 articles, 99,697 document chunks and 120 trailing query vectors**. Query vectors are excluded from the Qdrant collection. Use the versioned update workflow below for subsequent document changes. Changes to the embedding or chunking policy require a separately prepared index.

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

Open `http://127.0.0.1:3000`. The example `APP_ORIGIN` matches that address. For another origin, update it explicitly. The frontend proxies to `http://127.0.0.1:8000`; set the full `/ask` URL in `RAG_API_URL` inside `frontend/.env.local` to change the upstream.

`GET /health/live` checks the API process. `GET /health/ready` checks the key, pinned model assets, source hash and active index identity/count. The first request loads local models, so cold-start latency differs from subsequent requests.

## V1 acceptance and operation

The local V1 uses one API worker and Docker Qdrant. Its [redacted acceptance results](evaluations/results/v1_acceptance.json) record the checks and their scope.

| Check | Observation |
|---|---|
| Saved training-context replay | **80/97 → 83/97**, three gains and zero losses, using a reference-word coverage proxy |
| Changed-document drill | **2 chunks embedded, 99,695 vectors reused** across add/update/delete, followed by a zero-embedding repeat and successful rollback |
| Backup recovery | **28,482 documents / 99,697 vectors** validated, with identical exact-search top-ten results for three saved query vectors |
| Verifier calibration | **4/4** targeted canaries passed |
| Four final targeted questions | One answered, one clarification, two withheld |
| Live acceptance usage | **15/15 HTTP 200**, **62,248 reported tokens** across initial checks, canaries and the final generation pass |

The replay uses training tuning questions and a text-coverage proxy. Final generation reused the newly selected contexts. These measurements support the targeted fixes. Full held-out answer quality remains a later milestone. Q046/Q037 still withhold, and historical RAGAS review flags remain visible.

Browser acceptance uses the existing controlled GenUI fixture. It checks rendering, excerpts, structured Run details, regeneration and session clearing. The API suite also covers the storage/search/verification handoff, session isolation, exact cache invalidation after an index change and bounded concurrent admission.

### Update documents and roll back

After the initial import, register the existing index once. Adoption validates source membership, payload versions, vector dimensions and norms.

```bash
.venv-ingest/bin/python -m scripts.manage_index adopt
.venv-ingest/bin/python -m scripts.manage_index status
```

An update file maps document IDs to records with `title`, `text` and `metadata`. Optional `content` supplies original HTML for the existing Docling parser. Keep the original body in `text` when providing HTML so exact source matches can be mapped. Changed documents retain their raw input in the source store.

```json
{
  "your-document-id": {
    "title": "Updated support article",
    "text": "The updated article body.",
    "metadata": {"productName": "Your product"}
  }
}
```

Stage changes into a separate collection, then activate the returned build ID. An unchanged input reuses its existing version. Deletion can be run with `--delete` alone. Policy fingerprints prevent mixing a changed model/parser/chunker configuration with old vectors.

```bash
.venv-ingest/bin/python -m scripts.manage_index stage --updates /path/to/updates.json
.venv-ingest/bin/python -m scripts.manage_index stage --delete document-id-to-remove
.venv-ingest/bin/python -m scripts.manage_index activate BUILD_ID
.venv-ingest/bin/python -m scripts.manage_index validate
.venv-ingest/bin/python -m scripts.manage_index activate PREVIOUS_BUILD_ID
```

Staging copies unchanged vectors to preserve an isolated, recoverable version. It embeds the changed documents only. The serving process picks up activation on its next request. Keep version directories and their Qdrant collections together. Stage each batch from the currently active version. If the activation pointer is lost, serving stops until you explicitly activate a validated build.

### Back up and restore

```bash
.venv-ingest/bin/python -m scripts.manage_index backup /path/to/new-backup-directory
.venv-ingest/bin/python -m scripts.manage_index restore /path/to/backup-directory
.venv-ingest/bin/python -m scripts.manage_index activate RESTORE_ID
```

The backup contains a native Qdrant snapshot, matching SQLite sources and checksums. Restore creates a separate collection and returns `restore_id`. Activation validates the pair before changing the local pointer. This workflow is tested on a single-node Docker Qdrant. See the [Qdrant snapshot documentation](https://qdrant.tech/documentation/snapshots/) for the underlying snapshot operations.

### Inspect runs and operating bounds

The local service admits five requests, processes one at a time, waits up to 30 seconds for the worker and bounds each request at 180 seconds. Provider quota waiting contributes to response time. Controlled concurrency tests exercise these bounds. Public traffic capacity needs a separate load measurement.

Run details displays stage times, model/index/prompt identities, cache hits, token usage and HTTP attempts. Local metrics are content-free, capped at 100 MiB and seven days. Retrieval caches retain up to 2,048 query vectors and 50,000 pair scores. Existing answer-cache/session limits apply as well.

Set `RAG_TRACE_DIR` in the backend environment to opt into private local JSON traces containing the question, candidates, selected passages and generation decisions. Traces use owner-only file permissions and retain up to 50 files, 128 MiB and seven days. Keep that directory outside public hosting and Git. Gold references remain in offline evaluation artifacts.

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
