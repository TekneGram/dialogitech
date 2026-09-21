from __future__ import annotations

import argparse
from pathlib import Path

from chunker.llm_metadata_extractor import LLMMetadataExtractor


def resolve_artifact(value: str) -> Path:
    repository_root = Path(__file__).resolve().parents[1]
    supplied_path = Path(value)

    if supplied_path.is_file():
        return supplied_path

    paper_id = supplied_path.stem
    return (
        repository_root
        / "marker"
        / "conversion_results"
        / paper_id
        / f"{paper_id}.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real Gemma metadata integration test."
    )
    parser.add_argument(
        "paper",
        help=(
            "Paper ID, such as 2026_shi_et_al, or a path to its Marker JSON file."
        ),
    )
    args = parser.parse_args()

    artifact_path = resolve_artifact(args.paper)
    if not artifact_path.is_file():
        parser.error(f"Marker JSON artifact not found: {artifact_path}")

    extractor = LLMMetadataExtractor(event_logger=print)

    try:
        result = extractor.extract_metadata(artifact_path)
    finally:
        extractor.close()

    print(f"Artifact: {artifact_path}")
    for component in ("title", "journal", "authors"):
        decision = getattr(result, component)
        print(
            f"Validated Gemma {component} response:",
            {
                "value": decision.value,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "provenance": decision.provenance,
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
