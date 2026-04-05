from __future__ import annotations

import argparse

from .embedding_service import DeterministicHashEmbeddingService, OllamaEmbeddingService
from .index_manager import LanceIndexManager
from .ingest_service import ChunkIngestionService
from .lancedb_client import LanceChunkStore
from .lancedb_schema import DEFAULT_TABLE_NAME
from .paper_chunk_serializer import PaperChunkSerializer
from .pipeline_loader import build_paper_metadata, load_classified_heading_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest processed paper chunks into LanceDB.")
    parser.add_argument("--db-path", required=True, help="Path to the LanceDB database directory.")
    parser.add_argument(
        "--classified-json",
        required=True,
        help="Path to a *_classified_chunks.json file produced by the chunk classification pipeline.",
    )
    parser.add_argument(
        "--marker-json",
        help="Optional path to the Marker JSON file used to extract paper title, authors, journal, year, and DOI.",
    )
    parser.add_argument(
        "--table-name",
        default=DEFAULT_TABLE_NAME,
        help="Name of the LanceDB table to create or update.",
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
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        help="Optional embedding dimension override. Usually not needed for Ollama.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/embed",
        help="Ollama embeddings endpoint URL.",
    )
    parser.add_argument(
        "--paper-id",
        help="Optional explicit paper ID. Defaults to the base filename.",
    )
    parser.add_argument(
        "--pdf-path",
        help="Optional source PDF path to store with each chunk row.",
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Insert rows without building indexes.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete existing rows for the same paper_id before inserting.",
    )
    args = parser.parse_args()

    classified_splits, source_markdown = load_classified_heading_splits(args.classified_json)
    paper_metadata = build_paper_metadata(
        classified_json_path=args.classified_json,
        marker_json_path=args.marker_json,
        paper_id=args.paper_id,
        pdf_path=args.pdf_path,
        markdown_path=source_markdown,
    )

    if args.embedding_provider == "ollama":
        embedding_service = OllamaEmbeddingService(
            model=args.embedding_model,
            base_url=args.ollama_url,
            dimensions=args.embedding_dimensions,
        )
    else:
        embedding_service = DeterministicHashEmbeddingService(
            dimensions=args.embedding_dimensions or 256
        )

    store = LanceChunkStore(db_path=args.db_path, table_name=args.table_name)
    index_manager = None if args.skip_indexes else LanceIndexManager(store)
    ingestion_service = ChunkIngestionService(
        serializer=PaperChunkSerializer(),
        embedding_service=embedding_service,
        store=store,
        index_manager=index_manager,
    )

    inserted_count = ingestion_service.ingest_paper(
        paper_metadata,
        classified_splits,
        create_indexes=not args.skip_indexes,
        replace_existing=args.replace_existing,
    )

    print(f"paper_id={paper_metadata.paper_id}")
    print(f"paper_title={paper_metadata.paper_title}")
    print(f"inserted_chunks={inserted_count}")


if __name__ == "__main__":
    main()
