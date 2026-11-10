"""Publish saved evaluation outcomes to LangSmith without new model calls."""

import argparse
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith.utils import LangSmithNotFoundError

from rag_app.tracing import get_client
from scripts.eval_suite import build_report, load_cases


def publish(directory, client, ragas=None):
    cases = load_cases(directory)
    report = build_report(cases, ragas)
    identity_file = directory / "manifest.json" if directory.is_dir() else directory
    identity = report["cohort_id"]
    artifact_hash = hashlib.sha256(identity_file.read_bytes()).hexdigest()
    name = "techqa-training-" + identity[:12]
    try:
        dataset = client.read_dataset(dataset_name=name)
    except LangSmithNotFoundError:
        dataset = client.create_dataset(
            name,
            description="Fixed training tuning sample. References are used only for scoring.",
        )
    existing = {
        example.inputs["question_id"]
        for example in client.list_examples(dataset_id=dataset.id)
    }
    examples = [
        {
            "inputs": {"question_id": c["question_id"], "question": c["user_input"]},
            "outputs": {"reference": c["reference"]},
        }
        for c in cases
        if c["question_id"] not in existing
    ]
    if examples:
        client.create_examples(dataset_id=dataset.id, examples=examples)
    lookup = {c["question_id"]: c for c in cases}
    rows = {r["question_id"]: r for r in report["cases"]}

    def saved_answer(inputs):
        case = lookup[inputs["question_id"]]
        return {
            "question_id": case["question_id"],
            "response": case["response"],
            "status": case["status"],
            "sources": case["sources"],
            "original_trace_id": case.get("langsmith_run_id"),
        }

    def outcome(run, example):
        return {"key": "response_status", "value": run.outputs["status"]}

    def selected_document(run, example):
        return {
            "key": "selected_gold_document_hit",
            "score": rows[example.inputs["question_id"]]["selected_gold_document_hit"],
        }

    def citations(run, example):
        check = rows[example.inputs["question_id"]]["citation_contract"]
        return {
            "key": "citation_contract",
            "score": int(check["pass"]) if check["state"] == "checked" else None,
            "comment": "Citation format and source IDs only. Semantic support is evaluated separately.",
        }

    evaluators = [outcome, selected_document, citations]
    if ragas is not None:

        def semantic_scores(run, example):
            metrics = rows[example.inputs["question_id"]]["ragas"]
            return {
                "results": [
                    {
                        "key": "ragas_" + key,
                        "score": value["score"]
                        if value["state"] == "success" and not value["needs_review"]
                        else None,
                        "comment": "Judge review required"
                        if value["needs_review"]
                        else value["state"],
                    }
                    for key, value in metrics.items()
                ]
            }

        evaluators.append(semantic_scores)

    results = client.evaluate(
        saved_answer,
        data=name,
        evaluators=evaluators,
        experiment_prefix="techqa-saved-pipeline",
        max_concurrency=1,
        description="Saved canonical pipeline results replayed for inspection. No new inference or semantic judge calls.",
        metadata={
            "cohort_id": identity,
            "collection_artifact_sha256": artifact_hash,
            "mode": "saved_result_replay",
            "split": "training_tuning",
        },
    )
    client.flush()
    return results


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--ragas",
        type=Path,
        help="Optional matching saved RAGAS report; no new judge calls",
    )
    args = parser.parse_args()
    scores = json.loads(args.ragas.read_text()) if args.ragas else None
    publish(args.cases, get_client(), scores)
