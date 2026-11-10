"""Manage local TechQA versions: stage changes, activate, rollback, backup and restore."""

import argparse
import json
import os
from pathlib import Path
import shutil
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from rag_app.indexes import IndexManager, atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def encoder_identity():
    return {
        "models_sha256": sha256(ROOT / "models-embeddings.json"),
        "parser_sha256": sha256(ROOT / "scripts/ingest_html.py"),
        "chunker_sha256": sha256(ROOT / "scripts/chunking.py"),
        "tokenizer_sha256": sha256(
            ROOT / "data/tokenizers/bge_wordpiece/tokenizer.json"
        ),
    }


def prepare_document(docid, raw):
    from scripts.ingest_html import (
        _clean_markdown,
        _convert,
        _repeat_table_headers,
        _structured_chunks,
        _section_at,
        _source_interval,
        _doc_record,
    )
    from tokenizers import Tokenizer

    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("text", ""), str)
        or not isinstance(raw.get("title", ""), str)
        or not isinstance(raw.get("metadata", {}), dict)
    ):
        raise ValueError(
            "Each document requires text/title strings and a metadata object"
        )
    tok = Tokenizer.from_file(
        str(ROOT / "data/tokenizers/bge_wordpiece/tokenizer.json")
    )
    original = raw.get("text", "")
    if "content" in raw:
        if not isinstance(raw["content"], str):
            raise ValueError("HTML content must be a string")
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat

        text = _convert(
            raw["content"], DocumentConverter(allowed_formats=[InputFormat.HTML])
        )
    else:
        text = _clean_markdown(original)
    text = _repeat_table_headers(text, tok)
    record = _doc_record(docid, raw, text)
    chunks = []
    for c in _structured_chunks(text, tok, 512):
        body = text[c["start"] : c["end"]]
        start, end = _source_interval(body, original)
        section = _section_at(text, c["start"])
        chunks.append(
            {
                **c,
                "chunk_id": f'html_v1:{docid}:{c["start"]}',
                "document_id": docid,
                "source_start": start,
                "source_end": end,
                "source_coordinate": "prepared_view",
                "section": section,
                "search_text": "\n".join(
                    filter(
                        None,
                        [
                            record["title"],
                            record["metadata"].get("productName", ""),
                            section,
                            body,
                        ],
                    )
                ),
            }
        )
    return record, chunks


def backup(manager, destination, url, api_key=None):
    """Save a native Qdrant snapshot and its matching immutable source database."""
    import httpx

    with manager.locked():
        active = manager.validate(manager.active())
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        snapshot = manager.client.create_snapshot(active["collection"], wait=True)
        path = destination / "qdrant.snapshot"
        headers = {"api-key": api_key} if api_key else {}
        with httpx.stream(
            "GET",
            f"{url.rstrip('/')}/collections/{active['collection']}/snapshots/{snapshot.name}",
            headers=headers,
            timeout=120,
        ) as response:
            response.raise_for_status()
            with path.open("wb") as stream:
                for block in response.iter_bytes():
                    stream.write(block)
        shutil.copyfile(manager.source_path(active), destination / "source.sqlite3")
        value = {
            "index": active,
            "files": {
                name: sha256(destination / name)
                for name in ["qdrant.snapshot", "source.sqlite3"]
            },
        }
        atomic_json(destination / "backup.json", value)
        manager.client.delete_snapshot(active["collection"], snapshot.name, wait=True)
        return value


def restore(manager, backup_dir):
    """Restore to a new collection; activation is a separate explicit command."""
    backup_dir = Path(backup_dir)
    saved = json.loads((backup_dir / "backup.json").read_text())
    if set(saved["files"]) != {"qdrant.snapshot", "source.sqlite3"}:
        raise ValueError("Invalid backup file list")
    for name, digest in saved["files"].items():
        if sha256(backup_dir / name) != digest:
            raise ValueError("Backup hash mismatch")
    with manager.locked():
        manifest = dict(saved["index"])
        directory = manager.root / ("restore_" + uuid.uuid4().hex)
        directory.mkdir()
        shutil.copyfile(backup_dir / "source.sqlite3", directory / "source.sqlite3")
        collection = "techqa_restore_" + uuid.uuid4().hex
        with (backup_dir / "qdrant.snapshot").open("rb") as stream:
            manager.client.http.snapshots_api.recover_from_uploaded_snapshot(
                collection_name=collection,
                wait=True,
                priority=models.SnapshotPriority.SNAPSHOT,
                checksum=saved["files"]["qdrant.snapshot"],
                snapshot=stream,
            )
        # Keep the original build ID, because every restored point carries that identity.
        manifest.update(
            collection=collection,
            source=str((directory / "source.sqlite3").relative_to(manager.root)),
        )
        manager.validate(manifest)
        atomic_json(directory / "manifest.json", manifest)
        return {"restore_id": directory.name, "index": manifest}


def main(argv=None):
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/indexes")
    parser.add_argument(
        "--url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    )
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("status")
    subs.add_parser("adopt")
    subs.add_parser("validate")
    stage = subs.add_parser("stage")
    stage.add_argument("--updates", type=Path)
    stage.add_argument("--delete", nargs="*", default=[])
    activate = subs.add_parser("activate")
    activate.add_argument("build")
    save = subs.add_parser("backup")
    save.add_argument("destination", type=Path)
    recover = subs.add_parser("restore")
    recover.add_argument("backup", type=Path)
    args = parser.parse_args(argv)
    client = QdrantClient(
        url=args.url, api_key=os.getenv("QDRANT_API_KEY"), timeout=180
    )
    manager = IndexManager(args.root, client)
    if args.command == "adopt":
        from scripts.import_qdrant import CONFIG_PATH, COLLECTION

        result = manager.adopt(
            ROOT / "data/source.sqlite3",
            COLLECTION,
            json.loads(CONFIG_PATH.read_text())["cache_key"],
            encoder_identity(),
        )
    elif args.command == "activate":
        result = manager.activate(args.build)
    elif args.command == "validate":
        result = manager.validate(manager.active())
    elif args.command == "backup":
        result = backup(
            manager, args.destination, args.url, os.getenv("QDRANT_API_KEY")
        )
    elif args.command == "restore":
        result = restore(manager, args.backup)
    elif args.command == "stage":
        from rag_app.retriever import Retriever
        from scripts.eval_embeddings import format_input

        updates = json.loads(args.updates.read_text()) if args.updates else {}
        retriever = Retriever(index_root=args.root)
        try:
            result = manager.stage(
                updates,
                args.delete,
                prepare_document,
                lambda text: retriever._encode(
                    format_input(retriever.spec, text, query=False)
                ),
                encoder_identity(),
            )
        finally:
            retriever.close()
    else:
        result = manager.active()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
