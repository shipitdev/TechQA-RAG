"""Durable, private user feedback and its associated run snapshot."""

import logging
import os
from pathlib import Path
import time
import uuid

from langsmith.utils import LangSmithNotFoundError
from rag_app.tracing import get_client
from rag_app.indexes import atomic_json


def save_feedback(root, request_id, record):
    directory = Path(root) / "data/app/feedback"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / (str(uuid.UUID(request_id)) + ".json")
    atomic_json(path, record)
    path.chmod(0o600)
    return path


def record_feedback(root, snapshot, helpful, comment):
    request_id = snapshot["result"]["request_id"]
    record = {
        **snapshot,
        "feedback": {
            "helpful": helpful,
            "comment": comment,
            "submitted_at": time.time(),
        },
        "review_status": "pending",
    }
    save_feedback(root, request_id, record)
    sync_feedback(root, record)
    return record


def sync_feedback(root, record):
    if not record["result"].get("langsmith_url") or not os.getenv("LANGSMITH_API_KEY"):
        return
    request_id = record["result"]["request_id"]
    feedback_id = uuid.uuid5(uuid.NAMESPACE_URL, "techqa-feedback/" + request_id)
    values = {
        "score": int(record["feedback"]["helpful"]),
        "comment": record["feedback"]["comment"],
    }
    try:
        client = get_client()
        try:
            client.update_feedback(feedback_id, **values)
        except LangSmithNotFoundError:
            client.create_feedback(
                run_id=request_id, key="user_helpful", feedback_id=feedback_id, **values
            )
        record["langsmith_feedback_synced"] = True
        save_feedback(root, request_id, record)
    except Exception:
        logging.getLogger(__name__).warning(
            "Feedback saved locally; LangSmith synchronization pending"
        )
