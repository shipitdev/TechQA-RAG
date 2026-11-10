from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.publish_langsmith import publish


def test_saved_experiment_exports_contracts_without_model_calls():
    client = Mock()
    client.read_dataset.return_value = SimpleNamespace(id="dataset")
    client.list_examples.return_value = []

    def evaluate(target, **kwargs):
        example = client.create_examples.call_args.kwargs["examples"][0]
        inputs = example["inputs"]
        output = target(inputs)
        run = SimpleNamespace(outputs=output)
        reference = SimpleNamespace(inputs=inputs)
        feedback = [scorer(run, reference) for scorer in kwargs["evaluators"]]
        assert output["question_id"] == inputs["question_id"]
        assert [f["key"] for f in feedback] == [
            "response_status",
            "selected_gold_document_hit",
            "citation_contract",
        ]
        assert kwargs["max_concurrency"] == 1
        assert kwargs["metadata"]["mode"] == "saved_result_replay"
        return "experiment"

    client.evaluate.side_effect = evaluate
    assert (
        publish(Path("evaluations/fixtures/offline_demo.json"), client) == "experiment"
    )
    client.flush.assert_called_once()


def test_dataset_identity_survives_changed_answers(tmp_path):
    import json

    source = Path("evaluations/fixtures/offline_demo.json")
    payload = json.loads(source.read_text())
    client = Mock()
    client.read_dataset.return_value = SimpleNamespace(id="dataset")
    client.list_examples.return_value = []
    publish(source, client)
    original = client.read_dataset.call_args.kwargs["dataset_name"]
    payload["cases"][0]["response"] = "A changed output. [S1]"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload))
    publish(changed, client)
    assert client.read_dataset.call_args.kwargs["dataset_name"] == original
