from __future__ import annotations

from dbinsert.embedding_service import OllamaEmbeddingService

from dbquery.lancedb_retriever import LanceDBRetriever
from dbquery.models import QueryFilters, SearchVariant
from dbquery.query_embedder import QueryEmbedder
from dbquery.result_fuser import ReciprocalRankFuser

from .models import EvidenceFilters, EvidenceResult, EvidenceToolCall


class EvidenceRetriever:
    def __init__(
        self,
        *,
        db_path: str,
        table_name: str,
        query_embedder: QueryEmbedder | None = None,
        retriever: LanceDBRetriever | None = None,
        result_fuser: ReciprocalRankFuser | None = None,
    ) -> None:
        self.query_embedder = query_embedder or QueryEmbedder(OllamaEmbeddingService())
        self.retriever = retriever or LanceDBRetriever(db_path=db_path, table_name=table_name)
        self.result_fuser = result_fuser or ReciprocalRankFuser(k=60, min_lists=1)

    def retrieve(self, tool_call: EvidenceToolCall) -> EvidenceResult:
        variants = self._build_variants(tool_call.query)
        embedded_variants = self.query_embedder.embed_variants(
            [variant for variant in variants if variant.retrieval_kind == "hybrid"]
        )
        query_filters = self._to_query_filters(tool_call.filters)

        retrieved_chunks = []
        for variant in variants:
            retrieved_chunks.extend(
                self.retriever.retrieve(
                    variant=variant,
                    vector=embedded_variants.get(variant.key),
                    limit=tool_call.top_k,
                    filters=query_filters,
                )
            )

        return EvidenceResult(
            tool_call=tool_call,
            fused_results=self.result_fuser.fuse(retrieved_chunks, final_top_k=tool_call.top_k),
        )

    def _build_variants(self, query_text: str) -> list[SearchVariant]:
        return [
            SearchVariant(
                key="followup_hybrid",
                text=query_text,
                kind="original",
                retrieval_kind="hybrid",
            ),
            SearchVariant(
                key="followup_fts",
                text=query_text,
                kind="original",
                retrieval_kind="fts",
            ),
        ]

    def _to_query_filters(self, filters: EvidenceFilters) -> QueryFilters:
        return QueryFilters(
            paper_id_in=list(filters.paper_id_in),
            year_min=filters.year_min,
            year_max=filters.year_max,
            classification_label_in=list(filters.classification_label_in),
            section_title_contains=filters.section_title_contains,
            authors_any=list(filters.authors_any),
        )
