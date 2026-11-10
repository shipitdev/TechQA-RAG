import json
import uuid

from fastapi.testclient import TestClient
import pytest

import rag_app.api as api
from rag_app.feedback import record_feedback
from scripts.feedback_cases import approve, export_cases


class Search:
    def retrieve(self, question):
        return {
            "sources": [
                {"id": "S1", "text": "Restart.", "title": "Guide", "document_id": "doc"}
            ],
            "metrics": {},
        }


class Writer:
    def generate(self, question, sources):
        return {"status": "answered", "answer": "Restart. [S1]", "citations": ["S1"]}


def final(response):
    return json.loads(
        response.text.split("event: final\ndata: ", 1)[1].split("\n\n", 1)[0]
    )


def test_feedback_belongs_to_the_session_and_persists_after_session_end(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    app = api.create_app(Search(), Writer())
    user, stranger = TestClient(app), TestClient(app)
    result = final(user.post("/ask", json={"question": "Restart?"}))
    body = {
        "request_id": result["request_id"],
        "helpful": False,
        "comment": "Missing version details",
    }
    assert stranger.post("/feedback", json=body).status_code == 404
    assert (
        user.post(
            "/feedback", json=body, headers={"Origin": "https://evil.example"}
        ).status_code
        == 403
    )
    assert user.post("/feedback", json=body).json()["status"] == "saved"
    path = tmp_path / "data/app/feedback" / (result["request_id"] + ".json")
    saved = json.loads(path.read_text())
    assert saved["question"] == "Restart?"
    assert saved["result"]["sources"] == result["sources"]
    assert saved["feedback"]["helpful"] is False
    assert saved["review_status"] == "pending"
    assert path.stat().st_mode & 0o777 == 0o600
    assert user.post("/feedback", json={**body, "helpful": "false"}).status_code == 422
    assert (
        user.post("/feedback", json={**body, "comment": "x" * 2001}).status_code == 422
    )
    user.post("/session/end")
    assert user.post("/feedback", json=body).status_code == 404
    assert path.exists()


def test_feedback_survives_cloud_failure_and_requires_review_before_export(
    tmp_path, monkeypatch
):
    import rag_app.feedback as feedback

    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setattr(
        feedback, "get_client", lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    request_id = str(uuid.uuid4())
    snapshot = {
        "question": "Question?",
        "resolved_question": "Question?",
        "pipeline_version": "test",
        "prompt_version": "test",
        "result": {
            "request_id": request_id,
            "langsmith_url": "https://smith.langchain.com/",
            "status": "answered",
            "answer": "Answer. [S1]",
            "sources": [{"id": "S1", "text": "Answer."}],
            "citations": ["S1"],
        },
    }
    record_feedback(tmp_path, snapshot, False, "Needs detail")
    with pytest.raises(ValueError, match="No approved"):
        export_cases(tmp_path, tmp_path / "pending")
    with pytest.raises(ValueError):
        approve(tmp_path, request_id, "", "notes")
    approve(
        tmp_path,
        request_id,
        "Checked answer from the original source.",
        "Confirmed in S1",
    )
    assert export_cases(tmp_path, tmp_path / "approved") == 1
    case = json.loads((tmp_path / "approved/completed.json").read_text())["cases"][0]
    assert case["review_status"] == "approved"
    assert case["reference_provenance"]["view"] == "human_reviewed_feedback_v1"
    assert case["response"] == snapshot["result"]["answer"]
    from scripts.eval_suite import build_report

    assert build_report([case])["metrics"]["selected_gold_document_hit_rate"] is None
    with pytest.raises(ValueError, match="new export directory"):
        export_cases(tmp_path, tmp_path / "approved")


def test_cloud_feedback_updates_one_stable_record(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from langsmith.utils import LangSmithNotFoundError
    import rag_app.feedback as feedback

    client = Mock()
    client.update_feedback.side_effect = [LangSmithNotFoundError("missing"), None]
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setattr(feedback, "get_client", lambda: client)
    run_id = str(uuid.uuid4())
    snapshot = {
        "result": {
            "request_id": run_id,
            "langsmith_url": "https://smith.langchain.com/",
        }
    }
    first = record_feedback(tmp_path, snapshot, False, "Missing steps")
    second = record_feedback(tmp_path, snapshot, True, "Updated feedback")
    assert first["langsmith_feedback_synced"] is True
    assert second["langsmith_feedback_synced"] is True
    assert client.create_feedback.call_count == 1
    assert (
        client.update_feedback.call_args_list[0].args
        == client.update_feedback.call_args_list[1].args
    )
    assert len(list((tmp_path / "data/app/feedback").glob("*.json"))) == 1
