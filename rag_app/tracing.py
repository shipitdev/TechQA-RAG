"""Optional LangSmith traces. Export failures never interrupt an answer."""

from contextlib import contextmanager
from functools import lru_cache
import logging
import os

from langsmith import Client, trace, tracing_context


@lru_cache(maxsize=1)
def get_client():
    return Client(timeout_ms=1500)


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
        span = trace(
            name,
            run_type,
            inputs=inputs or {},
            client=get_client(),
            project_name=os.getenv("LANGSMITH_PROJECT", "techqa-rag"),
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
