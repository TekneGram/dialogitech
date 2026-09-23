"""Ingest existing classified artifacts without rerunning classification."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _pdf_path_for(classified_json: Path, pdf_root: Path) -> Path | None:
    paper_id = classified_json.name.removesuffix("_classified_chunks.json")
    candidate = pdf_root / f"{paper_id}.pdf"
    return candidate if candidate.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest existing *_classified_chunks.json artifacts into LanceDB "
            "without rerunning chunk classification."
        )
    )
    parser.add_argument(
        "--db-path",
        default="data/lancedb",
        help="Path to the LanceDB database directory.",
    )
    parser.add_argument(
        "--conversion-root",
        default="marker/conversion_results",
        help="Root directory containing per-paper classified artifacts.",
    )
    parser.add_argument(
        "--pdf-root",
        default="pdfs",
        help="Directory containing source PDFs.",
    )
    parser.add_argument(
        "--replace-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace rows for matching paper IDs (default: enabled).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List artifacts that would be ingested without running ingestion.",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    conversion_root = repository_root / args.conversion_root
    pdf_root = repository_root / args.pdf_root
    classified_files = sorted(conversion_root.glob("*/*_classified_chunks.json"))
    if not classified_files:
        raise SystemExit(f"No classified artifacts found under {conversion_root}")

    print(f"Found {len(classified_files)} classified artifact(s).", flush=True)
    if args.dry_run:
        for classified_json in classified_files:
            print(classified_json, flush=True)
        return

    failures: list[tuple[Path, int]] = []
    for index, classified_json in enumerate(classified_files, start=1):
        command = [
            sys.executable,
            "-m",
            "dbinsert.run_ingest",
            "--db-path",
            args.db_path,
            "--classified-json",
            str(classified_json),
        ]
        if args.replace_existing:
            command.append("--replace-existing")
        pdf_path = _pdf_path_for(classified_json, pdf_root)
        if pdf_path is not None:
            command.extend(["--pdf-path", str(pdf_path)])

        print(f"[{index}/{len(classified_files)}] Ingesting {classified_json}", flush=True)
        result = subprocess.run(command, cwd=repository_root, check=False)
        if result.returncode != 0:
            failures.append((classified_json, result.returncode))
            print(
                f"[{index}/{len(classified_files)}] FAILED "
                f"exit_code={result.returncode}",
                flush=True,
            )

    print("Batch summary", flush=True)
    print(f"total_artifacts={len(classified_files)}", flush=True)
    print(f"succeeded={len(classified_files) - len(failures)}", flush=True)
    print(f"failed={len(failures)}", flush=True)
    for classified_json, returncode in failures:
        print(f"FAILED artifact={classified_json} exit_code={returncode}", flush=True)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
