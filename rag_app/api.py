"""Session-isolated API. Only approved answers leave the worker."""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rag_app.state import SessionStore, AnswerCache
from rag_app.generation import MODEL, GroqREST, AnswerGenerator, GenerationError

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')

class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    regenerate: bool = False


def create_app(retriever=None, generator=None, resolver=None):
    app = FastAPI(title='TechQA evidence workspace')
    serial = threading.Lock()
    admission = threading.BoundedSemaphore(5)
    sessions = SessionStore()
    answers = AnswerCache()
    limits = defaultdict(deque)
    limiter_lock = threading.Lock()
    runtime = {'retriever': retriever, 'generator': generator, 'resolver': resolver}

    def pipeline_version():
        """Invalidate exact answers when data, runtime code, or UI library changes."""
        files = [ROOT / 'data/sources/manifest.json',
                 ROOT / 'data/retrieval_runs/gemma2_structured_512_html_v1/config.json',
                 ROOT / 'data/tokenizers/bge_wordpiece/source.json',
                 ROOT / 'models-embeddings.json']
        files += sorted((ROOT / 'rag_app').glob('*.py'))
        library = ROOT / 'frontend/src/lib'
        if library.is_dir():
            files += sorted(path for path in library.rglob('*') if path.is_file())
        digest = hashlib.sha256()
        for path in files:
            if path.is_file():
                digest.update(path.relative_to(ROOT).as_posix().encode())
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def write_metrics(record):
        try:
            metrics_dir = Path(os.getenv('RAG_METRICS_DIR', str(ROOT / 'data/app/metrics')))
            metrics_dir.mkdir(parents=True, exist_ok=True)
            cutoff = time.time() - 7 * 86400
            for old in metrics_dir.glob('*.jsonl'):
                if old.stat().st_mtime < cutoff:
                    old.unlink()
            with (metrics_dir / (time.strftime('%Y-%m-%d') + '.jsonl')).open('a') as f:
                f.write(json.dumps(record) + '\n')
        except OSError:
            pass

    def dependencies():
        if runtime['retriever'] is None:
            from rag_app.retriever import Retriever
            runtime['retriever'] = Retriever()
        if runtime['generator'] is None:
            provider = GroqREST()
            runtime['provider'] = provider
            runtime['generator'] = AnswerGenerator(provider.json_call)
            from rag_app.generation import ConversationResolver, history_summarizer
            runtime['resolver'] = ConversationResolver(provider.json_call)
            sessions.summarizer = history_summarizer(provider.json_call)
            from tokenizers import Tokenizer
            reference = Tokenizer.from_file(str(ROOT / 'data/tokenizers/bge_wordpiece/tokenizer.json'))
            sessions.token_counter = lambda text: len(reference.encode(text, add_special_tokens=False).ids)
        return runtime['retriever'], runtime['generator']

    @app.get('/health/live')
    def live():
        return {'status': 'ok'}

    @app.get('/health/ready')
    def ready():
        if retriever is not None:
            return {'status': 'ready', 'presentation': 'direct_groq'}
        if not (os.getenv('GROQ_API_KEY') or os.getenv('GROK_API_KEY')):
            raise HTTPException(503, 'Groq API key is not configured')
        try:
            from qdrant_client import QdrantClient
            from scripts.import_qdrant import COLLECTION
            if not (ROOT / 'data/source.sqlite3').exists():
                raise ValueError('source import required')
            info = QdrantClient(url=os.getenv('QDRANT_URL', 'http://127.0.0.1:6333'), timeout=3).get_collection(COLLECTION)
            if info.points_count != 99697:
                raise ValueError('collection count mismatch')
        except Exception:
            raise HTTPException(503, 'Qdrant/source import is not ready') from None
        return {'status': 'ready', 'presentation': 'direct_groq', 'models': 'lazy-loaded on first request'}

    @app.post('/session/end')
    def end(request: Request):
        origin = request.headers.get('origin')
        allowed = os.getenv('APP_ORIGIN', 'http://localhost:3000')
        if origin and origin != allowed:
            raise HTTPException(403, 'Origin not allowed')
        sid = request.cookies.get('techqa_session', '')
        sessions.end(sid); answers.clear_session(sid)
        from fastapi.responses import JSONResponse
        response = JSONResponse({'status': 'cleared'})
        response.delete_cookie('techqa_session')
        return response

    @app.post('/ask')
    async def ask(body: Ask, request: Request):
        if not body.question.strip():
            raise HTTPException(422, 'Question cannot be blank')
        origin = request.headers.get('origin')
        allowed = os.getenv('APP_ORIGIN', 'http://localhost:3000')
        if origin and origin != allowed:
            raise HTTPException(403, 'Origin not allowed')
        session = sessions.get(request.cookies.get('techqa_session', '')) or sessions.create()
        ip = request.client.host if request.client else 'local'
        now = time.monotonic()
        with limiter_lock:
            for old_key, items in list(limits.items()):
                while items and items[0] <= now - 60: items.popleft()
                if not items: del limits[old_key]
            for key in ['session:' + session.id, 'ip:' + ip]:
                items = limits[key]
                if len(items) >= 10: raise HTTPException(429, 'Please wait before asking again')
            for key in ['session:' + session.id, 'ip:' + ip]: limits[key].append(now)
        if not admission.acquire(blocking=False):
            raise HTTPException(429, 'The workspace is busy; please try again shortly')
        events = queue.Queue()
        cancelled = threading.Event()
        rid = str(uuid.uuid4())
        started = time.monotonic()
        deadline = started + 180

        def stopped():
            return cancelled.is_set() or time.monotonic() >= deadline

        def emit(kind, value):
            if not stopped(): events.put((kind, value))

        def worker():
            acquired = False
            result = None
            provider = None
            provider_calls = []
            completed = False
            try:
                emit('stage', {'stage': 'queued'})
                acquired = serial.acquire(timeout=30)
                if not acquired: raise TimeoutError()
                if stopped(): return
                retrieve, generate = dependencies()
                if stopped(): return
                provider = runtime.get('provider')
                if provider:
                    provider.usage.clear()
                    provider.deadline = started + 180
                context = sessions.context(session.id)
                question = body.question
                dependent = False
                regen = body.regenerate or bool(re.fullmatch(r'\s*(?:no[,!.]?\s*)?(?:please\s*)?(?:re-?generate(?:\s+it)?(?:\s+again)?|generate\s+it\s+again)[.!]?\s*', question, re.I))
                if regen:
                    if not context['exchanges']:
                        raise ValueError('There is no previous answer to regenerate')
                    question = context['exchanges'][-1].question
                elif context['exchanges'] and runtime.get('resolver'):
                    if stopped(): return
                    emit('stage', {'stage': 'understanding_followup'})
                    resolved = runtime['resolver'].resolve(question, context)
                    if stopped(): return
                    if resolved.get('needs_clarification'):
                        result = {'status': 'needs_clarification', 'answer': 'Please clarify the product, version, or issue you mean.', 'sources': []}
                    else:
                        dependent = bool(resolved.get('dependent'))
                        question = resolved['question'] if dependent else body.question
                identity = {'question': question,
                            'context': ({'summary': context['summary'], 'exchanges': [vars(x) for x in context['exchanges']]}
                                        if dependent else None),
                            'pipeline': pipeline_version(),
                            'index': retrieve.index_identity() if hasattr(retrieve, 'index_identity') else 'injected_test_index',
                            'model': MODEL,
                            'prompt': 'strict_v1', 'generation': {'provider': 'groq', 'reasoning_effort': 'low', 'max_output_tokens': 1500}}
                cached = None if regen or result else answers.get(session.id, identity, session_expires_at=session.expires_at)
                if cached:
                    result = dict(cached)
                    result['metrics'] = {'answer_cache_hit': True, 'regenerated': False}
                elif result is None:
                    emit('stage', {'stage': 'searching_and_selecting_evidence'})
                    if stopped(): return
                    retrieved = retrieve.retrieve(question)
                    if stopped(): return
                    sources = retrieved['sources']
                    emit('stage', {'stage': 'writing_and_verifying'})
                    if stopped(): return
                    answer = generate.generate(question, sources) if sources else {'status': 'insufficient_evidence', 'answer': 'No supporting evidence was retrieved.', 'citations': []}
                    if stopped(): return
                    answer = {key: answer[key] for key in ('status', 'answer', 'citations') if key in answer}
                    result = {**answer, 'sources': sources, 'metrics': {**retrieved.get('metrics', {}), 'context_tokens': retrieved.get('context_tokens'), 'answer_cache_hit': False, 'regenerated': regen}}
                    if result['status'] == 'answered' and (os.getenv('GROQ_API_KEY') or os.getenv('GROK_API_KEY')) and os.getenv('ENABLE_GENUI', '1') == '1' and runtime.get('provider'):
                        if stopped(): return
                        emit('stage', {'stage': 'validating_presentation'})
                        from rag_app.presentation import generate_presentation
                        result['openui'] = generate_presentation(result, question, generate)
                if stopped() or sessions.get(session.id) is None: return
                sessions.add_exchange(session.id, question, result['answer'], commit_if=lambda: not stopped())
                if stopped() or sessions.get(session.id) is None: return
                if not cached and result['status'] == 'answered':
                    answers.put(session.id, identity, result, session_expires_at=session.expires_at)
                result['request_id'] = rid
                result.setdefault('metrics', {})['total_ms'] = round((time.monotonic() - started) * 1000, 2)
                result['presentation_status'] = 'direct_groq' if result.get('openui') else 'verified_text_fallback'
                provider = runtime.get('provider')
                if provider:
                    provider_calls = list(provider.usage)
                    result['metrics']['provider_calls'] = provider_calls
                    provider.usage.clear()
                emit('final', result)
                write_metrics({'request_id': rid, 'status': result['status'], 'metrics': result['metrics']})
                completed = True
            except Exception as exc:
                message = (('Groq billing or credits need attention. Check the Groq Console before retrying.' if exc.category == 'http_402' else 'Groq did not respond before the time limit. No answer was released.' if exc.category in {'transport_TimeoutError', 'transport_ReadTimeout'} else 'Answer generation failed; please retry.') if isinstance(exc, GenerationError) else
                           str(exc) if isinstance(exc, ValueError) else
                           'A dependency failed; please retry or check local service readiness.')
                if provider:
                    provider_calls = list(provider.usage)
                    provider.usage.clear()
                metrics = {'provider_calls': provider_calls,
                           'failed_provider_calls': int(isinstance(exc, GenerationError)), 'error_category': exc.category if isinstance(exc, GenerationError) else type(exc).__name__}
                emit('error', {'message': message[:220], 'request_id': rid, 'metrics': metrics})
                write_metrics({'request_id': rid, 'status': 'error', 'metrics': metrics})
                completed = True
            finally:
                if provider:
                    if not completed:
                        provider_calls = list(provider.usage)
                    provider.usage.clear()
                if not completed:
                    write_metrics({'request_id': rid, 'status': 'cancelled_or_timed_out',
                                   'metrics': {'provider_calls': provider_calls}})
                if acquired: serial.release()
                admission.release()
                emit('done', {})

        threading.Thread(target=worker, daemon=True).start()
        async def stream():
            try:
                while True:
                    if await request.is_disconnected(): break
                    if time.monotonic() >= deadline:
                        yield 'event: error\ndata: ' + json.dumps({'message': 'Request deadline exceeded', 'request_id': rid}) + '\n\n'
                        break
                    try: kind, payload = events.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(.05); continue
                    if kind == 'done': break
                    yield 'event: ' + kind + '\ndata: ' + json.dumps(payload) + '\n\n'
            finally: cancelled.set()
        response = StreamingResponse(stream(), media_type='text/event-stream', headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})
        response.set_cookie('techqa_session', session.id, httponly=True, samesite='strict', secure=allowed.startswith('https://'))
        return response
    return app

app = create_app()
