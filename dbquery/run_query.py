from __future__ import annotations

import argparse
from pathlib import Path

from dbinsert.embedding_service import OllamaEmbeddingService
from dbinsert.lancedb_schema import DEFAULT_TABLE_NAME

from .chunk_batcher import ChunkBatcher
from .citation_formatter import CitationFormatter
from .gemma_client import GemmaClient
from .gemma_summarizer import GemmaBatchSummarizer
from .hyde_generator import GemmaHyDEGenerator
from .lancedb_retriever import LanceDBRetriever
from .models import QueryRequest
from .output_writer import QueryOutputWriter
from .query_embedder import QueryEmbedder
from .query_pipeline import QueryPipeline
from .query_rewriter import GemmaQueryRewriter
from .result_fuser import ReciprocalRankFuser
from .synthesis_summarizer import GemmaSynthesisSummarizer

DEFAULT_GEMMA_MODEL = "unsloth/gemma-4-E4B-it-UD-MLX-4bit"
DEFAULT_GEMMA_PYTHON = "/Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python"


def main() -> None:
    parser = argparse.ArgumentParser(description="Query paper chunks from LanceDB and summarize the results.")
    parser.add_argument("query", help="User query.")
    parser.add_argument("--db-path", required=True, help="Path to the LanceDB database directory.")
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME, help="Name of the LanceDB table.")
    parser.add_argument(
        "--retrieval-mode",
        choices=("full", "fts_only", "hybrid_only", "fts_hybrid"),
        default="full",
        help="Select which retrieval branches to run.",
    )
    parser.add_argument("--model-path", default=DEFAULT_GEMMA_MODEL, help="Gemma model path.")
    parser.add_argument(
        "--python-executable",
        default=DEFAULT_GEMMA_PYTHON,
        help="Python executable for the external Gemma MLX runtime.",
    )
    parser.add_argument("--retrieval-depth", type=int, default=10, help="Results to fetch per retrieval list.")
    parser.add_argument("--final-top-k", type=int, default=15, help="Final fused result count.")
    parser.add_argument("--batch-size", type=int, default=5, help="Chunks per summary batch.")
    parser.add_argument("--rewrite-count", type=int, default=2, help="Number of Gemma rewrites to generate.")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant.")
    parser.add_argument("--min-rrf-lists", type=int, default=1, help="Minimum contributing lists per chunk.")
    parser.add_argument(
        "--min-relevance-score",
        type=float,
        help="Optional minimum LanceDB hybrid relevance score filter.",
    )
    parser.add_argument(
        "--output-path",
        help="Optional path to write ranked chunks and summaries for inspection.",
    )
    parser.add_argument("--no-hyde", action="store_true", help="Disable HyDE retrieval.")
    args = parser.parse_args()

    gemma_client = GemmaClient(
        model_path=args.model_path,
        python_executable=args.python_executable,
    )
    pipeline = QueryPipeline(
        query_rewriter=GemmaQueryRewriter(gemma_client),
        hyde_generator=GemmaHyDEGenerator(gemma_client),
        query_embedder=QueryEmbedder(OllamaEmbeddingService()),
        retriever=LanceDBRetriever(db_path=args.db_path, table_name=args.table_name),
        result_fuser=ReciprocalRankFuser(k=args.rrf_k, min_lists=args.min_rrf_lists),
        chunk_batcher=ChunkBatcher(),
        summarizer=GemmaBatchSummarizer(gemma_client, CitationFormatter()),
        synthesis_summarizer=GemmaSynthesisSummarizer(gemma_client),
    )
    result = pipeline.run(
        QueryRequest(
            query=args.query,
            retrieval_mode=args.retrieval_mode,
            retrieval_depth=args.retrieval_depth,
            final_top_k=args.final_top_k,
            batch_size=args.batch_size,
            rewrite_count=args.rewrite_count,
            include_hyde=not args.no_hyde,
            min_rrf_lists=args.min_rrf_lists,
            rrf_k=args.rrf_k,
            min_relevance_score=args.min_relevance_score,
        )
    )

    output_path: Path | None = None
    if args.output_path:
        output_path = QueryOutputWriter().write(result, args.output_path)

    print(f"query={result.request.query}")
    print(f"retrieval_mode={result.request.retrieval_mode}")
    print(f"rewrites={result.rewrites}")
    print(f"hyde_present={result.hyde_text is not None}")
    if result.hyde_text is not None:
        print(f"hyde_text={result.hyde_text}")
    print(f"raw_hits={len(result.retrieved_chunks)}")
    print(f"fused_hits={len(result.fused_results)}")
    if output_path is not None:
        print(f"output_path={output_path}")

    for index, chunk in enumerate(result.fused_results, start=1):
        print(f"-- fused hit {index} --")
        print(f"chunk_id={chunk.chunk_id}")
        print(f"paper_id={chunk.paper_id}")
        print(f"paper_title={chunk.paper_title}")
        print(f"section_title={chunk.section_title}")
        print(f"chunk_index={chunk.chunk_index}")
        print(f"rrf_score={chunk.rrf_score:.6f}")
        print(f"retrieval_keys={','.join(chunk.retrieval_keys)}")
        print(f"text_preview={chunk.text[:240].replace(chr(10), ' ')}")

    for summary in result.summaries:
        print(f"-- summary batch {summary.batch_index} --")
        print(summary.summary_text)

    if result.synthesized_summary is not None:
        print("-- synthesized summary --")
        print(result.synthesized_summary)


if __name__ == "__main__":
    main()
