from __future__ import annotations

import argparse
from pathlib import Path

from .llm_paper_type_classifier_helpers.paper_type_models import PAPER_TYPES
from .llm_section_classifier import SectionClassificationLLM
from .markdown_section_chunker import MarkdownSectionChunker
from .chunk_models import ClassifiedHeadingSplit, ClassifiedSectionChunk


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM section classification on filtered markdown.")
    parser.add_argument("markdown_path", help="Path to a filtered markdown file.")
    parser.add_argument(
        "--paper-type",
        choices=PAPER_TYPES,
        default="empirical_research",
        help="Resolved document type used to constrain the allowed section labels.",
    )
    parser.add_argument("--model-path", required=True, help="Path to a local MLX model for LLM classification.")
    parser.add_argument(
        "--python-executable",
        help="Optional Python executable to use for MLX inference in a separate environment.",
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=float,
        default=180.0,
        help="Maximum time to wait for each external Gemma request.",
    )
    parser.add_argument("--min-words", type=int, default=200, help="Minimum chunk size in words.")
    parser.add_argument("--overlap-words", type=int, default=50, help="Chunk overlap size in words.")
    args = parser.parse_args()

    markdown = Path(args.markdown_path).read_text(encoding="utf-8")
    heading_splits = MarkdownSectionChunker(
        min_words=args.min_words,
        overlap_words=args.overlap_words,
    ).process(markdown)

    with SectionClassificationLLM(
        filtered_markdown=markdown,
        heading_splits=heading_splits,
        model_path=args.model_path,
        python_executable=args.python_executable,
        request_timeout_seconds=args.llm_timeout_seconds,
        event_logger=lambda message: print(message, flush=True),
    ) as section_classifier:
        previous_resolved_label = None
        classified: list[ClassifiedHeadingSplit] = []
        for heading_split in heading_splits:
            classified_chunks: list[ClassifiedSectionChunk] = []
            for section_chunk in heading_split.chunks:
                classification = section_classifier.classify_section_chunk(
                    section_chunk=section_chunk,
                    heading_split=heading_split,
                    paper_type=args.paper_type,
                    previous_label=previous_resolved_label,
                )
                classified_chunks.append(
                    ClassifiedSectionChunk(
                        title=section_chunk.title,
                        heading_level=section_chunk.heading_level,
                        chunk_index=section_chunk.chunk_index,
                        text=section_chunk.text,
                        word_count=section_chunk.word_count,
                        classification=classification,
                    )
                )
                if classification.label is not None:
                    previous_resolved_label = classification.label
            classified.append(
                ClassifiedHeadingSplit(
                    title=heading_split.title,
                    heading_level=heading_split.heading_level,
                    raw_heading=heading_split.raw_heading,
                    content=heading_split.content,
                    chunks=classified_chunks,
                )
            )

    total = sum(len(split.chunks) for split in classified)
    llm_count = sum(
        1
        for split in classified
        for chunk in split.chunks
        if chunk.classification.source in {"llm", "llm_fallback"}
    )
    print(f"total_chunks={total}")
    print(f"paper_type={args.paper_type}")
    print(f"llm_classified_chunks={llm_count}")
    for split in classified:
        for chunk in split.chunks:
            print(
                "\t".join(
                    [
                        split.title,
                        str(chunk.chunk_index),
                        str(chunk.word_count),
                        str(chunk.classification.label),
                        chunk.classification.source,
                        str(chunk.classification.confidence),
                        str(chunk.classification.used_context),
                        chunk.classification.reason,
                    ]
                )
            )


if __name__ == "__main__":
    main()
