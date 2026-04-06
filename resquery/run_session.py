from __future__ import annotations

import argparse

from dbinsert.embedding_service import OllamaEmbeddingService
from dbinsert.lancedb_schema import DEFAULT_TABLE_NAME
from dbquery.chunk_batcher import ChunkBatcher
from dbquery.citation_formatter import CitationFormatter
from dbquery.gemma_client import GemmaClient
from dbquery.gemma_summarizer import GemmaBatchSummarizer
from dbquery.hyde_generator import GemmaHyDEGenerator
from dbquery.lancedb_retriever import LanceDBRetriever
from dbquery.models import QueryFilters, QueryRequest
from dbquery.output_writer import QueryOutputWriter
from dbquery.query_embedder import QueryEmbedder
from dbquery.query_pipeline import QueryPipeline
from dbquery.query_rewriter import GemmaQueryRewriter
from dbquery.result_fuser import ReciprocalRankFuser
from dbquery.synthesis_summarizer import GemmaSynthesisSummarizer

from .batch_summary_claims_extractor import GemmaBatchSummaryClaimsExtractor
from .claim_deduper import ClaimDeduper
from .claim_relevance_ranker import ClaimRelevanceRanker
from .followup_suggester import GemmaFollowupSuggester
from .models import ResQueryRequest
from .output_writer import ResQueryOutputWriter
from .pipeline import ResQueryPipeline
from .query_context_builder import QueryContextBuilder
from .session_initializer import ResearchSessionInitializer
from .state_merger import ResearchStateMerger
from .state_selector import ResearchStateSelector
from .state_store import ResearchStateStore
from .state_updater import ResearchStateUpdater

DEFAULT_GEMMA_MODEL = "unsloth/gemma-4-E4B-it-UD-MLX-4bit"
DEFAULT_GEMMA_PYTHON = "/Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an iterative research session over LanceDB.")
    parser.add_argument("user_question", help="The current research question.")
    parser.add_argument("--session-path", required=True, help="Path to the persisted research session JSON.")
    parser.add_argument("--db-path", required=True, help="Path to the LanceDB database directory.")
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME, help="Name of the LanceDB table.")
    parser.add_argument("--output-path", help="Optional path to write a human-readable resquery turn artifact.")
    parser.add_argument("--query-output-path", help="Optional path to write the underlying dbquery artifact.")
    parser.add_argument("--model-path", default=DEFAULT_GEMMA_MODEL, help="Gemma model path.")
    parser.add_argument(
        "--python-executable",
        default=DEFAULT_GEMMA_PYTHON,
        help="Python executable for the external Gemma MLX runtime.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("full", "fts_only", "hybrid_only", "fts_hybrid"),
        default="full",
        help="Select which retrieval branches to run.",
    )
    parser.add_argument("--retrieval-depth", type=int, default=10, help="Results to fetch per retrieval list.")
    parser.add_argument("--final-top-k", type=int, default=15, help="Final fused result count.")
    parser.add_argument("--batch-size", type=int, default=5, help="Chunks per summary batch.")
    parser.add_argument("--rewrite-count", type=int, default=2, help="Number of Gemma rewrites to generate.")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant.")
    parser.add_argument("--min-rrf-lists", type=int, default=1, help="Minimum contributing lists per chunk.")
    parser.add_argument(
        "--run-mode",
        choices=("new", "continue", "deepen", "expand"),
        default="continue",
        help="How to handle branch context for this turn.",
    )
    parser.add_argument("--branch-id", help="Branch to continue/deepen/expand, or ID to create for --run-mode new.")
    parser.add_argument("--branch-label", help="Optional label when creating a new branch.")
    parser.add_argument(
        "--candidate-pool-k",
        type=int,
        help="Optional retrieval pool size before exclusions and fusion. deepen/expand choose a larger default if unset.",
    )
    parser.add_argument("--min-relevance-score", type=float, help="Optional LanceDB relevance score cutoff.")
    parser.add_argument("--paper-id", help="Restrict retrieval to a single paper ID.")
    parser.add_argument("--year", type=int, help="Restrict retrieval to a specific publication year.")
    parser.add_argument("--paper-title-contains", help="Restrict retrieval to papers with a title substring.")
    parser.add_argument("--classification-label", help="Restrict retrieval to a specific classification label.")
    parser.add_argument(
        "--author",
        "--authors",
        dest="author",
        help="Restrict retrieval to chunks whose author list contains this string.",
    )
    parser.add_argument("--no-hyde", action="store_true", help="Disable HyDE retrieval.")
    parser.add_argument(
        "--claim-similarity-threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for filtering semantically similar claims.",
    )
    args = parser.parse_args()

    gemma_client = GemmaClient(
        model_path=args.model_path,
        python_executable=args.python_executable,
    )
    embedding_service = OllamaEmbeddingService()
    query_pipeline = QueryPipeline(
        query_rewriter=GemmaQueryRewriter(gemma_client),
        hyde_generator=GemmaHyDEGenerator(gemma_client),
        query_embedder=QueryEmbedder(embedding_service),
        retriever=LanceDBRetriever(db_path=args.db_path, table_name=args.table_name),
        result_fuser=ReciprocalRankFuser(k=args.rrf_k, min_lists=args.min_rrf_lists),
        chunk_batcher=ChunkBatcher(),
        summarizer=GemmaBatchSummarizer(gemma_client, CitationFormatter()),
        synthesis_summarizer=GemmaSynthesisSummarizer(gemma_client),
    )
    pipeline = ResQueryPipeline(
        state_store=ResearchStateStore(),
        session_initializer=ResearchSessionInitializer(),
        state_selector=ResearchStateSelector(
            claim_relevance_ranker=ClaimRelevanceRanker(embedding_service=embedding_service),
        ),
        query_context_builder=QueryContextBuilder(),
        query_pipeline=query_pipeline,
        state_updater=ResearchStateUpdater(
            claims_extractor=GemmaBatchSummaryClaimsExtractor(gemma_client),
            claim_deduper=ClaimDeduper(
                embedding_service=embedding_service,
                similarity_threshold=args.claim_similarity_threshold,
            ),
            followup_suggester=GemmaFollowupSuggester(gemma_client),
        ),
        state_merger=ResearchStateMerger(),
    )
    query_request = QueryRequest(
        query=args.user_question,
        retrieval_mode=args.retrieval_mode,
        retrieval_depth=args.retrieval_depth,
        candidate_pool_k=args.candidate_pool_k,
        final_top_k=args.final_top_k,
        batch_size=args.batch_size,
        rewrite_count=args.rewrite_count,
        include_hyde=not args.no_hyde,
        min_rrf_lists=args.min_rrf_lists,
        rrf_k=args.rrf_k,
        min_relevance_score=args.min_relevance_score,
        filters=QueryFilters(
            paper_id=args.paper_id,
            year=args.year,
            paper_title_contains=args.paper_title_contains,
            classification_label=args.classification_label,
            author=args.author,
        ),
    )
    result = pipeline.run(
        ResQueryRequest(
            session_path=args.session_path,
            user_question=args.user_question,
            query_request=query_request,
            run_mode=args.run_mode,
            branch_id=args.branch_id,
            branch_label=args.branch_label,
            query_output_path=args.query_output_path,
        )
    )

    if args.query_output_path:
        QueryOutputWriter().write(result.query_result, args.query_output_path)

    turn_output_path = None
    if args.output_path:
        turn_output_path = ResQueryOutputWriter().write(result, args.output_path)

    print(f"session_id={result.session_state.session_id}")
    print(f"turn_id={result.turn.turn_id}")
    print(f"branch_id={result.turn.branch_id}")
    print(f"run_mode={result.turn.run_mode}")
    print(f"session_path={result.session_path}")
    print(f"query_output_path={args.query_output_path or '[not written]'}")
    if turn_output_path is not None:
        print(f"output_path={turn_output_path}")
    print("-- synthesized summary --")
    print(result.turn.synthesized_summary)


if __name__ == "__main__":
    main()
