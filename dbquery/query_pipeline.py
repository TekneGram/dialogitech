from __future__ import annotations

from .chunk_batcher import ChunkBatcher
from .gemma_summarizer import GemmaBatchSummarizer
from .hyde_generator import GemmaHyDEGenerator
from .lancedb_retriever import LanceDBRetriever
from .models import QueryPipelineResult, QueryRequest, RetrievalMode, SearchVariant
from .query_embedder import QueryEmbedder
from .query_rewriter import GemmaQueryRewriter
from .result_fuser import ReciprocalRankFuser
from .synthesis_summarizer import GemmaSynthesisSummarizer


class QueryPipeline:
    def __init__(
        self,
        *,
        query_rewriter: GemmaQueryRewriter,
        hyde_generator: GemmaHyDEGenerator,
        query_embedder: QueryEmbedder,
        retriever: LanceDBRetriever,
        result_fuser: ReciprocalRankFuser,
        chunk_batcher: ChunkBatcher,
        summarizer: GemmaBatchSummarizer,
        synthesis_summarizer: GemmaSynthesisSummarizer | None = None,
    ) -> None:
        self.query_rewriter = query_rewriter
        self.hyde_generator = hyde_generator
        self.query_embedder = query_embedder
        self.retriever = retriever
        self.result_fuser = result_fuser
        self.chunk_batcher = chunk_batcher
        self.summarizer = summarizer
        self.synthesis_summarizer = synthesis_summarizer

    def run(self, request: QueryRequest) -> QueryPipelineResult:
        rewrites: list[str] = []
        if self._should_generate_rewrites(request.retrieval_mode):
            rewrites = self.query_rewriter.rewrite(request.query, rewrite_count=request.rewrite_count)

        hyde_text = None
        if self._should_generate_hyde(request.retrieval_mode, request.include_hyde):
            hyde_text = self.hyde_generator.generate(request.query)

        variants = self._build_variants(
            query=request.query,
            rewrites=rewrites,
            hyde_text=hyde_text,
            retrieval_mode=request.retrieval_mode,
        )

        embedded_variants = self.query_embedder.embed_variants(
            [variant for variant in variants if variant.retrieval_kind in {"hybrid", "vector"}]
        )

        retrieved_chunks = []
        for variant in variants:
            retrieved_chunks.extend(
                self.retriever.retrieve(
                    variant=variant,
                    vector=embedded_variants.get(variant.key),
                    limit=request.retrieval_depth,
                    min_relevance_score=request.min_relevance_score,
                    filters=request.filters,
                )
            )

        fused_results = self.result_fuser.fuse(
            retrieved_chunks,
            final_top_k=request.final_top_k,
        )
        summary_batches = self.chunk_batcher.batch(fused_results, batch_size=request.batch_size)
        summaries = [
            self.summarizer.summarize_batch(batch, question=request.query)
            for batch in summary_batches
        ]
        synthesized_summary = None
        if self.synthesis_summarizer is not None:
            synthesized_summary = self.synthesis_summarizer.synthesize(
                summaries,
                question=request.query,
            )

        return QueryPipelineResult(
            request=request,
            rewrites=rewrites,
            hyde_text=hyde_text,
            variants=variants,
            retrieved_chunks=retrieved_chunks,
            fused_results=fused_results,
            summary_batches=summary_batches,
            summaries=summaries,
            synthesized_summary=synthesized_summary,
        )

    def _build_variants(
        self,
        *,
        query: str,
        rewrites: list[str],
        hyde_text: str | None,
        retrieval_mode: RetrievalMode,
    ) -> list[SearchVariant]:
        variants: list[SearchVariant] = []

        if retrieval_mode in {"full", "hybrid_only", "fts_hybrid"}:
            variants.append(
                SearchVariant(
                    key="original_hybrid",
                    text=query,
                    kind="original",
                    retrieval_kind="hybrid",
                )
            )
        if retrieval_mode in {"full", "fts_only", "fts_hybrid"}:
            variants.append(
                SearchVariant(
                    key="original_fts",
                    text=query,
                    kind="original",
                    retrieval_kind="fts",
                )
            )

        if retrieval_mode != "full":
            return variants

        for index, rewrite in enumerate(rewrites, start=1):
            variants.append(
                SearchVariant(
                    key=f"rewrite_{index}_vector",
                    text=rewrite,
                    kind="rewrite",
                    retrieval_kind="vector",
                )
            )
        if hyde_text is not None:
            variants.append(
                SearchVariant(
                    key="hyde_vector",
                    text=hyde_text,
                    kind="hyde",
                    retrieval_kind="vector",
                )
            )
        return variants

    def _should_generate_rewrites(self, retrieval_mode: RetrievalMode) -> bool:
        return retrieval_mode == "full"

    def _should_generate_hyde(self, retrieval_mode: RetrievalMode, include_hyde: bool) -> bool:
        return retrieval_mode == "full" and include_hyde
