from __future__ import annotations

import argparse
from pathlib import Path

from .llm_section_classifier import ChunkClassificationLLM
from .section_classifier import classify_filtered_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Run section classification on filtered markdown.")
    parser.add_argument("markdown_path", help="Path to a filtered markdown file.")
    parser.add_argument("--model-path", help="Path to a local MLX model for LLM classification.")
    parser.add_argument(
        "--python-executable",
        help="Optional Python executable to use for MLX inference in a separate environment.",
    )
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Classify all chunks with the LLM instead of only unresolved chunks.",
    )
    parser.add_argument("--min-words", type=int, default=200, help="Minimum chunk size in words.")
    parser.add_argument("--overlap-words", type=int, default=50, help="Chunk overlap size in words.")
    args = parser.parse_args()

    markdown = Path(args.markdown_path).read_text(encoding="utf-8")
    llm_classifier = None

    if args.model_path:
        from .markdown_section_chunker import MarkdownSectionChunker

        heading_splits = MarkdownSectionChunker(
            min_words=args.min_words,
            overlap_words=args.overlap_words,
        ).process(markdown)
        llm_classifier = ChunkClassificationLLM(
            filtered_markdown=markdown,
            heading_splits=heading_splits,
            model_path=args.model_path,
            python_executable=args.python_executable,
        )

    classified = classify_filtered_markdown(
        markdown,
        min_words=args.min_words,
        overlap_words=args.overlap_words,
        llm_classifier=llm_classifier,
        force_llm=args.force_llm,
    )

    total = sum(len(split.chunks) for split in classified)
    llm_count = sum(
        1 for split in classified for chunk in split.chunks if chunk.classification.source == "llm"
    )

    print(f"total_chunks={total}")
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
