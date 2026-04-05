from __future__ import annotations

import argparse
from pathlib import Path

from .embedding_service import DeterministicHashEmbeddingService, OllamaEmbeddingService
from .full_pipeline_service import (
    DEFAULT_GEMMA_MODEL_PATH,
    DEFAULT_GEMMA_PYTHON,
    PdfToLancePipeline,
)
from .index_manager import LanceIndexManager
from .ingest_service import ChunkIngestionService
from .lancedb_client import LanceChunkStore
from .lancedb_schema import DEFAULT_TABLE_NAME
from .paper_chunk_serializer import PaperChunkSerializer


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a PDF end to end and insert its chunks into LanceDB.")
    parser.add_argument("pdf_path", help="Path to a source PDF.")
    parser.add_argument("--db-path", required=True, help="Path to the LanceDB database directory.")
    parser.add_argument(
        "--table-name",
        default=DEFAULT_TABLE_NAME,
        help="Name of the LanceDB table to create or update.",
    )
    parser.add_argument(
        "--conversion-root",
        default="marker/conversion_results",
        help="Directory where Marker outputs and derived artifacts should be stored.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=("ollama", "hash"),
        default="ollama",
        help="Embedding backend. Use 'hash' only for local smoke tests.",
    )
    parser.add_argument(
        "--embedding-model",
        default="qwen3-embedding:0.6b",
        help="Embedding model name for the selected backend.",
    )
    parser.add_argument("--embedding-dimensions", type=int, help="Optional embedding dimension override.")
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/embed",
        help="Ollama embeddings endpoint URL.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_GEMMA_MODEL_PATH,
        help="Gemma model path for LLM fallback classification.",
    )
    parser.add_argument(
        "--python-executable",
        default=DEFAULT_GEMMA_PYTHON,
        help="Optional Python executable to use for MLX inference in a separate environment.",
    )
    parser.add_argument("--force-llm", action="store_true", help="Classify all chunks with the LLM.")
    parser.add_argument("--min-words", type=int, default=200, help="Minimum chunk size in words.")
    parser.add_argument("--overlap-words", type=int, default=50, help="Chunk overlap size in words.")
    parser.add_argument("--skip-indexes", action="store_true", help="Insert rows without building indexes.")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete existing rows for the same paper_id before inserting.",
    )
    parser.add_argument(
        "--rerun-marker",
        action="store_true",
        help="Force rerunning Marker even if the JSON output already exists.",
    )
    parser.add_argument(
        "--rerun-filtered-markdown",
        action="store_true",
        help="Force regenerating filtered markdown even if it already exists.",
    )
    parser.add_argument(
        "--rerun-classification",
        action="store_true",
        help="Force re-chunking and reclassification even if classified chunks already exist.",
    )
    args = parser.parse_args()

    if args.embedding_provider == "ollama":
        embedding_service = OllamaEmbeddingService(
            model=args.embedding_model,
            base_url=args.ollama_url,
            dimensions=args.embedding_dimensions,
        )
    else:
        embedding_service = DeterministicHashEmbeddingService(dimensions=args.embedding_dimensions or 256)

    store = LanceChunkStore(db_path=args.db_path, table_name=args.table_name)
    index_manager = None if args.skip_indexes else LanceIndexManager(store)
    ingestion_service = ChunkIngestionService(
        serializer=PaperChunkSerializer(),
        embedding_service=embedding_service,
        store=store,
        index_manager=index_manager,
    )
    pipeline = PdfToLancePipeline(
        ingestion_service=ingestion_service,
        conversion_root=Path(args.conversion_root),
        min_words=args.min_words,
        overlap_words=args.overlap_words,
    )

    result = pipeline.process_pdf(
        args.pdf_path,
        model_path=args.model_path,
        python_executable=args.python_executable,
        force_llm=args.force_llm,
        replace_existing=args.replace_existing,
        create_indexes=not args.skip_indexes,
        rerun_marker=args.rerun_marker,
        rerun_filtered_markdown=args.rerun_filtered_markdown,
        rerun_classification=args.rerun_classification,
    )

    print(f"paper_id={result['paper_id']}")
    print(f"paper_title={result['paper_title']}")
    print(f"marker_json_path={result['marker_json_path']}")
    print(f"filtered_markdown_path={result['filtered_markdown_path']}")
    print(f"classified_json_path={result['classified_json_path']}")
    print(f"marker_log_path={result['marker_log_path']}")
    print(f"classification_log_path={result['classification_log_path']}")
    print(f"inserted_chunks={result['inserted_chunks']}")


if __name__ == "__main__":
    main()
