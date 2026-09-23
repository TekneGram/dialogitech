"""Print the current PDF-ingestion stage at regular intervals."""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path


PIPELINE_MARKER = "dbinsert.run_full_pipeline"


def _processes() -> list[str]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,etime=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _pipeline_processes(processes: list[str]) -> list[str]:
    return [
        process
        for process in processes
        if PIPELINE_MARKER in process and "ingestion_monitor.py" not in process
    ]


def _latest_artifact(repository_root: Path) -> tuple[Path | None, set[str]]:
    conversion_root = repository_root / "marker" / "conversion_results"
    if not conversion_root.is_dir():
        return None, set()

    directories = [path for path in conversion_root.iterdir() if path.is_dir()]
    if not directories:
        return None, set()

    latest_directory = max(directories, key=lambda path: path.stat().st_mtime)
    files = {path.name for path in latest_directory.iterdir() if path.is_file()}
    return latest_directory, files


def _paper_from_marker_process(processes: list[str]) -> str | None:
    for process in processes:
        if "convert_single.py" not in process:
            continue
        match = re.search(r"(?:^|\s)([^\s]+\.pdf)(?:\s|$)", process)
        if match:
            return Path(match.group(1)).stem
    return None


def _stage(repository_root: Path, processes: list[str], seen_pipeline: bool) -> str:
    pipeline_processes = _pipeline_processes(processes)
    if not pipeline_processes:
        if seen_pipeline:
            return "pipeline finished"
        return "waiting for the ingestion pipeline to start"

    marker_processes = [process for process in processes if "convert_single.py" in process]
    mlx_processes = [
        process
        for process in processes
        if "unsloth_gemma4_mlx" in process or "mlx_llm_runner.py" in process
    ]
    if marker_processes:
        paper = _paper_from_marker_process(marker_processes)
        suffix = f" ({paper})" if paper else ""
        return f"Marker conversion{suffix}"
    if mlx_processes:
        return "Gemma/MLX classification or metadata extraction"

    artifact_directory, files = _latest_artifact(repository_root)
    paper = f" ({artifact_directory.name})" if artifact_directory else ""
    if artifact_directory is None or not files:
        return "pipeline running; waiting for the first paper artifact"
    if not any(name.endswith(".json") and not name.endswith("_classified_chunks.json") for name in files):
        return f"preparing Marker artifacts{paper}"
    if not any(name.endswith("_filtered.md") for name in files):
        return f"extracting metadata or generating filtered Markdown{paper}"
    if not any(name.endswith("_classified_chunks.json") for name in files):
        return f"Gemma classification{paper}"
    return f"generating embeddings or inserting into LanceDB{paper}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor the PDF ingestion pipeline and print its inferred stage."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between status checks (default: 10).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one status update and exit.",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero.")

    repository_root = Path(__file__).resolve().parents[1]
    seen_pipeline = False
    while True:
        processes = _processes()
        if _pipeline_processes(processes):
            seen_pipeline = True
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {_stage(repository_root, processes, seen_pipeline)}", flush=True)

        if args.once or (seen_pipeline and not _pipeline_processes(processes)):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
