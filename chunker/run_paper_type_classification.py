from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .llm_paper_type_classifier import PaperTypeClassificationLLM
from .llm_metadata_extractor import LLMMetadataExtractor
from .llm_paper_type_classifier_helpers.evidence_builder import (
    PaperTypeEvidenceBuilder,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify the type of an academic paper from filtered Markdown.")
    parser.add_argument("markdown_path", help="Path to filtered Markdown.")
    parser.add_argument("--marker-json", help="Optional Marker JSON used to provide title metadata.")
    parser.add_argument("--model-path", required=True, help="LLM path for classification.")
    parser.add_argument("--python-executable", help="External Python executable for MLX inference.")
    parser.add_argument("--output-path", help="Optional JSON sidecar output path.")
    args = parser.parse_args()

    markdown = Path(args.markdown_path).read_text(encoding="utf-8")
    metadata = {}
    if args.marker_json:
        with LLMMetadataExtractor(
            model_path=args.model_path,
            python_executable=args.python_executable,
            event_logger=lambda message: print(
                f"metadata: {message}",
                flush=True,
            )
        ) as metadata_extractor:
            metadata = metadata_extractor.extract_all(
                args.marker_json,
                allow_manual=True # Ensures interactive prompting for missing fields such as journal name; can be set to False in tests so that tests don't hang.
            )

    evidence = PaperTypeEvidenceBuilder().build(
        markdown,
        metadata=metadata
    )

    with PaperTypeClassificationLLM(
        model_path=args.model_path,
        python_executable=args.python_executable,
        event_logger=lambda message: print(message, flush=True)
    ) as paper_type_classifier:
        result = paper_type_classifier.classify(
            filtered_markdown=markdown,
            metadata=metadata
        )

    payload = {
        "source_markdown": str(Path(args.markdown_path)),
        "marker_json_path": args.marker_json,
        "paper_type": asdict(result),
        "evidence": asdict(evidence),
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output_path:
        Path(args.output_path).write_text(rendered + "\n", encoding="utf-8")
        print(f"output_path={args.output_path}")
    print(rendered)


if __name__ == "__main__":
    main()
