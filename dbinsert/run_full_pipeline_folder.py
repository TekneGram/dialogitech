from __future__ import annotations

import argparse
import traceback
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


def _discover_pdfs(pdf_dir: Path, *, pattern: str, recursive: bool) -> list[Path]:
    iterator = pdf_dir.rglob(pattern) if recursive else pdf_dir.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a folder of PDFs end to end and insert their chunks into LanceDB."
    )
    parser.add_argument("pdf_dir", help="Path to a directory containing source PDFs.")
    parser.add_argument(
        "--glob",
        default="*.pdf",
        help="Glob pattern used to find PDF files within the directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories for matching PDF files.",
    )
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

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"PDF directory is not a directory: {pdf_dir}")

    pdf_paths = _discover_pdfs(pdf_dir, pattern=args.glob, recursive=args.recursive)
    if not pdf_paths:
        raise RuntimeError(
            f"No PDF files matched pattern {args.glob!r} in directory: {pdf_dir}"
        )

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

    total_pdfs = len(pdf_paths)
    successes: list[dict[str, str | int]] = []
    failures: list[dict[str, str]] = []

    print(f"Discovered {total_pdfs} PDF file(s) in {pdf_dir}", flush=True)
    for index, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[{index}/{total_pdfs}] Processing {pdf_path}", flush=True)
        try:
            result = pipeline.process_pdf(
                pdf_path,
                model_path=args.model_path,
                python_executable=args.python_executable,
                force_llm=args.force_llm,
                replace_existing=args.replace_existing,
                create_indexes=False,
                rerun_marker=args.rerun_marker,
                rerun_filtered_markdown=args.rerun_filtered_markdown,
                rerun_classification=args.rerun_classification,
            )
        except Exception as exc:
            failures.append(
                {
                    "pdf_path": str(pdf_path),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[{index}/{total_pdfs}] FAILED {pdf_path}: {type(exc).__name__}: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            continue

        successes.append(
            {
                "paper_id": str(result["paper_id"]),
                "pdf_path": str(pdf_path),
                "inserted_chunks": int(result["inserted_chunks"]),
            }
        )
        print(
            f"[{index}/{total_pdfs}] Completed {pdf_path} "
            f"(paper_id={result['paper_id']}, inserted_chunks={result['inserted_chunks']})",
            flush=True,
        )

    if index_manager is not None and successes:
        print("Building LanceDB indexes once after batch ingestion.", flush=True)
        index_manager.ensure_all_indexes()

    print("", flush=True)
    print("Batch summary", flush=True)
    print(f"total_pdfs={total_pdfs}", flush=True)
    print(f"succeeded={len(successes)}", flush=True)
    print(f"failed={len(failures)}", flush=True)

    for success in successes:
        print(
            f"SUCCESS paper_id={success['paper_id']} "
            f"inserted_chunks={success['inserted_chunks']} "
            f"pdf_path={success['pdf_path']}",
            flush=True,
        )

    for failure in failures:
        print(
            f"FAILED pdf_path={failure['pdf_path']} error={failure['error']}",
            flush=True,
        )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
