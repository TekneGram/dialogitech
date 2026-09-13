from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .llm_paper_type_classifier import PaperTypeClassificationLLM
from .metadata_extractor import MetadataExtractor
from .paper_type_classifier import build_paper_type_evidence, classify_paper_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify the type of an academic paper from filtered Markdown.")
    parser.add_argument("markdown_path", help="Path to filtered Markdown.")
    parser.add_argument("--marker-json", help="Optional Marker JSON used to provide title metadata.")
    parser.add_argument("--model-path", help="Gemma model path for LLM fallback classification.")
    parser.add_argument("--python-executable", help="External Python executable for MLX inference.")
    parser.add_argument("--force-llm", action="store_true", help="Classify with Gemma even when deterministic evidence resolves the type.")
    parser.add_argument("--output-path", help="Optional JSON sidecar output path.")
    args = parser.parse_args()

    markdown = Path(args.markdown_path).read_text(encoding="utf-8")
    metadata = MetadataExtractor().extract_all(args.marker_json) if args.marker_json else {}
    evidence = build_paper_type_evidence(markdown, metadata=metadata)
    llm_classifier = None
    if args.model_path:
        llm_classifier = PaperTypeClassificationLLM(
            model_path=args.model_path,
            python_executable=args.python_executable,
            event_logger=lambda message: print(message, flush=True),
        )
    try:
        result = classify_paper_type(
            markdown,
            metadata=metadata,
            llm_classifier=llm_classifier,
            force_llm=args.force_llm,
        )
    finally:
        if llm_classifier is not None:
            llm_classifier.close()

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
