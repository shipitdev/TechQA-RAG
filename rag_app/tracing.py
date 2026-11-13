"""Optional LangSmith traces. Export failures never interrupt an answer."""

from contextlib import contextmanager
from functools import lru_cache
import logging
import hashlib
import json
import os

from langsmith import Client, trace, tracing_context, get_current_run_tree


@lru_cache(maxsize=1)
def get_client():
    return Client(timeout_ms=1500)


def evaluation_thread(case):
    """Keep generation, scoring and imports for the same saved answer together."""
    identity = case.get('langsmith_run_id') or hashlib.sha256(json.dumps(
        [case['question_id'], case.get('response'), case.get('retrieved_contexts')],
        sort_keys=True).encode()).hexdigest()[:24]
    return case.get('thread_id') or f"TechQA / {case.get('evaluation_layer', 'pipeline')} / {case['question_id']} / {identity}"


def model_usage(calls, *, provider, model):
    """Normalize reported usage, including explicit zero billing for the free route."""
    inputs = sum(c.get('prompt_tokens') or 0 for c in calls)
    outputs = sum(c.get('completion_tokens', c.get('output_tokens')) or 0 for c in calls)
    cached = sum(c.get('prompt_cache_hit_tokens') or c.get('cached_tokens') or
                 (c.get('prompt_tokens_details') or {}).get('cached_tokens') or 0 for c in calls)
    reasoning = sum(c.get('reasoning_tokens') or
                    (c.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0 for c in calls)
    usage = {'input_tokens': inputs, 'output_tokens': outputs, 'total_tokens': inputs + outputs,
             'input_token_details': {'cache_read': min(inputs, cached)},
             'output_token_details': {'reasoning': reasoning}}
    if not any(c.get('reasoning_tokens') is not None or (c.get('completion_tokens_details') or {}).get('reasoning_tokens') is not None for c in calls):
        usage.pop('output_token_details')
    if calls and provider == 'tokenharbor' and model.endswith(':free'):
        usage.update(input_cost=0.0, output_cost=0.0, total_cost=0.0)
    return usage


@contextmanager
def trace_span(name, *, inputs=None, run_type="chain", **kwargs):
    enabled = (
        bool(os.getenv("LANGSMITH_API_KEY"))
        and os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
    )
    if not enabled:
        yield None
        return
    try:
        context = tracing_context(enabled=True)
        context.__enter__()
        parent = get_current_run_tree()
        inherited = (parent.extra or {}).get('metadata', {}) if parent else {}
        kwargs['metadata'] = {**{k: inherited[k] for k in ('thread_id', 'question_id', 'layer') if k in inherited},
                              **kwargs.get('metadata', {})}
        span = trace(
            name,
            run_type,
            inputs=inputs or {},
            client=get_client(),
            project_name=kwargs.pop("project_name", None) or (parent.session_name if parent else None) or os.getenv("LANGSMITH_PROJECT") or "techqa-rag",
            **kwargs,
        )
        run = span.__enter__()
    except Exception:
        if "context" in locals():
            context.__exit__(None, None, None)
        logging.getLogger(__name__).warning("LangSmith trace setup unavailable")
        yield None
        return
    try:
        yield run
    except BaseException as exc:
        try:
            span.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            logging.getLogger(__name__).warning("LangSmith trace export unavailable")
        raise
    else:
        try:
            span.__exit__(None, None, None)
        except Exception:
            logging.getLogger(__name__).warning("LangSmith trace export unavailable")
    finally:
        context.__exit__(None, None, None)
