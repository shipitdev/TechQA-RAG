from unittest.mock import Mock


def test_tracing_disabled_without_key(monkeypatch):
    from rag_app.tracing import trace_span

    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    with trace_span("disabled") as run:
        assert run is None


def test_nested_spans_keep_parent_and_outputs(monkeypatch):
    from langsmith import Client
    import rag_app.tracing as tracing

    client = Client(
        api_key="test", api_url="http://localhost:1", auto_batch_tracing=False
    )
    create_run = Mock()
    update_run = Mock()
    monkeypatch.setattr(Client, "create_run", create_run)
    monkeypatch.setattr(Client, "update_run", update_run)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(tracing, "get_client", lambda: client)
    with tracing.trace_span("question", inputs={"question": "Example?"}) as root:
        with tracing.trace_span("retrieval", inputs={"question": "Example?"}) as child:
            assert child.parent_run_id == root.id
            child.end(outputs={"candidate_count": 100})
        root.end(outputs={"status": "answered"})
    assert create_run.call_count == 2
    assert update_run.call_count == 2
    assert update_run.call_args.kwargs["outputs"] == {"status": "answered"}


def test_trace_setup_failure_does_not_break_the_pipeline(monkeypatch):
    import rag_app.tracing as tracing

    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(
        tracing, "get_client", Mock(side_effect=RuntimeError("unavailable"))
    )
    with tracing.trace_span("question") as run:
        assert run is None


def test_provider_trace_records_actual_token_usage(monkeypatch):
    from langsmith import Client
    import rag_app.tracing as tracing
    from rag_app.generation import GroqREST

    client = Client(
        api_key="test", api_url="http://localhost:1", auto_batch_tracing=False
    )
    monkeypatch.setattr(Client, "create_run", Mock())
    update = Mock()
    monkeypatch.setattr(Client, "update_run", update)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(tracing, "get_client", lambda: client)
    response = Mock(status_code=200, headers={})
    response.json.return_value = {
        "choices": [
            {"finish_reason": "stop", "message": {"content": '{"answer":"example"}'}}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    provider = GroqREST(api_key="test", post=Mock(return_value=response))
    assert provider.json_call(
        "Synthetic prompt", {"type": "OBJECT", "properties": {}}
    ) == {"answer": "example"}
    assert update.call_args.kwargs["extra"]["metadata"]["usage_metadata"] == {
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
    }
