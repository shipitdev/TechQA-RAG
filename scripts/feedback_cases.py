"""Review saved feedback and export approved cases for RAGAS."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import uuid

from rag_app.feedback import save_feedback, sync_feedback
from scripts.eval_ragas import ROOT, save


def approve(root, request_id, reference, notes):
    request_id = str(uuid.UUID(request_id))
    if not reference.strip() or not notes.strip():
        raise ValueError("A checked reference answer and review notes are required")
    path = root / "data/app/feedback" / (request_id + ".json")
    record = json.loads(path.read_text())
    record["review_status"] = "approved"
    record["review"] = {
        "reference": reference.strip(),
        "notes": notes.strip(),
        "reviewed_at": time.time(),
    }
    save_feedback(root, request_id, record)


def export_cases(root, output):
    records = [
        json.loads(path.read_text())
        for path in sorted((root / "data/app/feedback").glob("*.json"))
    ]
    cases = []
    for record in records:
        if record.get("review_status") != "approved":
            continue
        result = record["result"]
        reference = record["review"]["reference"]
        if not reference.strip() or not record["review"]["notes"].strip():
            raise ValueError("Incomplete human review")
        sources = result.get("sources", [])
        cases.append(
            {
                "question_id": "FEEDBACK_" + uuid.UUID(result["request_id"]).hex,
                "user_input": record["resolved_question"],
                "original_question": record["question"],
                "reference": reference,
                "reference_provenance": {
                    "view": "human_reviewed_feedback_v1",
                    "reviewed_at": record["review"]["reviewed_at"],
                    "review_notes": record["review"]["notes"],
                },
                "response": result["answer"],
                "status": result["status"],
                "citations": result.get("citations", []),
                "sources": sources,
                "retrieved_contexts": [
                    json.dumps(source, ensure_ascii=False) for source in sources
                ],
                "selected_passages": [],
                "retrieval_metrics": result.get("metrics", {}),
                "pipeline_version": record["pipeline_version"],
                "prompt_version": record["prompt_version"],
                "user_feedback": record["feedback"],
                "review_status": "approved",
                "langsmith_run_id": result["request_id"],
            }
        )
    if not cases:
        raise ValueError("No approved feedback cases. Review pending records first.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "reviewed_user_feedback",
        "question_ids": [c["question_id"] for c in cases],
        "selection": "Human-approved feedback regression cases",
        "cohort_sha256": hashlib.sha256(
            json.dumps(cases, sort_keys=True).encode()
        ).hexdigest(),
    }
    if (output / "manifest.json").exists():
        raise ValueError(
            "Use a new export directory to preserve the saved evaluation version"
        )
    for case in cases:
        save(output / (case["question_id"] + ".json"), case)
    save(output / "manifest.json", manifest)
    save(output / "completed.json", {"cases": cases})
    return len(cases)


def main():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    review = subs.add_parser("approve")
    review.add_argument("--run-id", required=True)
    review.add_argument("--reference-file", type=Path, required=True)
    review.add_argument("--notes", required=True)
    export = subs.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    subs.add_parser("sync")
    args = parser.parse_args()
    if args.command == "approve":
        approve(ROOT, args.run_id, args.reference_file.read_text(), args.notes)
        print("Feedback case approved")
    elif args.command == "export":
        print(f"Exported {export_cases(ROOT, args.output)} reviewed cases")
    else:
        for path in sorted((ROOT / "data/app/feedback").glob("*.json")):
            record = json.loads(path.read_text())
            if not record.get("langsmith_feedback_synced"):
                sync_feedback(ROOT, record)


if __name__ == "__main__":
    main()
