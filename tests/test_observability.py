import os
from rag_app.observability import persist_run


def test_metrics_have_bounded_retention_and_debug_is_opt_in(tmp_path, monkeypatch):
    directory = tmp_path / "data/app/metrics"
    directory.mkdir(parents=True)
    old = directory / "old.jsonl"
    old.write_text("{}\n")
    os.utime(old, (1, 1))
    persist_run(
        tmp_path,
        {"request_id": "abc", "status": "answered", "metrics": {"search_ms": 1}},
        {"question": "private"},
    )
    assert not old.exists()
    assert not (tmp_path / "data/app/traces").exists()
    assert "private" not in "".join(p.read_text() for p in directory.glob("*.jsonl"))
    monkeypatch.setenv("RAG_TRACE_DIR", str(tmp_path / "traces"))
    for i in range(55):
        persist_run(
            tmp_path,
            {"request_id": str(i), "status": "answered", "metrics": {}},
            {"question": "private"},
        )
    assert len(list((tmp_path / "traces").glob("*.json"))) == 50
