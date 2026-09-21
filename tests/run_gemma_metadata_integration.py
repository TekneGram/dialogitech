from __future__ import annotations

import argparse
from pathlib import Path

from chunker.llm_metadata_extractor import LLMMetadataExtractor
from chunker.llm_metadata_extractor_helpers.response_parsing_validation import (
    MetadataResponseValidator,
)


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
    parser.add_argument(
        "--component",
        choices=("title", "journal", "authors"),
        default="title",
        help="Metadata component to extract (default: title).",
    )
    args = parser.parse_args()

    artifact_path = resolve_artifact(args.paper)
    if not artifact_path.is_file():
        parser.error(f"Marker JSON artifact not found: {artifact_path}")

    extractor = LLMMetadataExtractor()
    validator = MetadataResponseValidator()

    try:
        document = extractor.load_marker_json(artifact_path)
        pages = extractor.select_pages(document, [0, 1])
        compact_json = extractor.compact_page_json(pages)
        prompt_builder = {
            "title": extractor.build_title_prompt,
            "journal": extractor.build_journal_prompt,
            "authors": extractor.build_authors_prompt,
        }[args.component]
        prompt = prompt_builder(compact_json)

        raw_response = extractor._generate([
            {"role": "user", "content": prompt}
        ])
        payload = validator.parse_json_response(raw_response)
        decision = validator.validate_decision(payload, args.component)
    finally:
        extractor.close()

    print(f"Artifact: {artifact_path}")
    print(
        f"Validated Gemma {args.component} response:",
        {
            "value": decision.value,
            "confidence": decision.confidence,
            "reason": decision.reason,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
