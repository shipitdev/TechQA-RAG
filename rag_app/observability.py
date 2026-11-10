"""Local operational logs; evidence and drafts are retained only by explicit opt-in."""

import json
import logging
import os
from pathlib import Path
import time
import uuid


def _prune(directory, pattern, count, max_bytes):
    files = sorted(
        directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
    )
    total = 0
    for i, path in enumerate(files):
        size = path.stat().st_size
        total += size
        if (
            i >= count
            or total > max_bytes
            or path.stat().st_mtime < time.time() - 7 * 86400
        ):
            path.unlink()


def persist_run(root, record, private_trace=None):
    try:
        metrics = Path(
            os.getenv("RAG_METRICS_DIR", str(Path(root) / "data/app/metrics"))
        )
        metrics.mkdir(parents=True, exist_ok=True)
        path = metrics / (time.strftime("%Y-%m-%d") + ".jsonl")
        if path.exists() and path.stat().st_size > 20 * 1024**2:
            path.replace(
                metrics
                / (time.strftime("%Y-%m-%d") + "-" + uuid.uuid4().hex + ".jsonl")
            )
        with path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        _prune(metrics, "*.jsonl", 14, 100 * 1024**2)
        debug = os.getenv("RAG_TRACE_DIR")
        if debug and private_trace is not None:
            traces = Path(debug)
            traces.mkdir(parents=True, exist_ok=True)
            path = traces / (uuid.uuid4().hex + ".json")
            with open(
                path, "w", opener=lambda p, flags: os.open(p, flags, 0o600)
            ) as stream:
                json.dump({"request_id": record["request_id"], **private_trace}, stream)
            _prune(traces, "*.json", 50, 128 * 1024**2)
    except OSError:
        logging.getLogger(__name__).warning("Could not persist local run diagnostics")
