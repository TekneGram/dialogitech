from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


VariantKind = Literal["original", "rewrite", "hyde"]
RetrievalKind = Literal["hybrid", "fts", "vector"]


@dataclass(slots=True)
class QueryRequest:
    query: str
    retrieval_depth: int = 10
    final_top_k: int = 15
    batch_size: int = 5
    rewrite_count: int = 2
    include_hyde: bool = True
    min_rrf_lists: int = 1
    rrf_k: int = 60
    min_relevance_score: float | None = None


@dataclass(slots=True)
class SearchVariant:
    key: str
    text: str
    kind: VariantKind
    retrieval_kind: RetrievalKind


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    paper_id: str
    paper_title: str
    authors: list[str]
    year: int | None
    section_title: str
    chunk_index: int
    text: str
    classification_label: str | None
    classification_source: str
    markdown_path: str | None
    marker_json_path: str | None
    pdf_path: str | None
    retrieval_key: str
    retrieval_kind: RetrievalKind
    rank: int
    score: float | None = None
    relevance_score: float | None = None
    distance: float | None = None
    fts_score: float | None = None
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)


@dataclass(slots=True)
class FusedChunkResult:
    chunk_id: str
    paper_id: str
    paper_title: str
    authors: list[str]
    year: int | None
    section_title: str
    chunk_index: int
    text: str
    classification_label: str | None
    classification_source: str
    markdown_path: str | None
    marker_json_path: str | None
    pdf_path: str | None
    rrf_score: float
    retrieval_keys: list[str]
    retrieval_kinds: list[RetrievalKind]
    contributing_ranks: dict[str, int]
    relevance_score: float | None = None
    distance: float | None = None
    fts_score: float | None = None


@dataclass(slots=True)
class SummaryBatch:
    batch_index: int
    chunks: list[FusedChunkResult]


@dataclass(slots=True)
class BatchSummary:
    batch_index: int
    summary_text: str
    citation_labels: list[str]
    chunk_ids: list[str]


@dataclass(slots=True)
class QueryPipelineResult:
    request: QueryRequest
    rewrites: list[str]
    hyde_text: str | None
    variants: list[SearchVariant]
    retrieved_chunks: list[RetrievedChunk]
    fused_results: list[FusedChunkResult]
    summary_batches: list[SummaryBatch]
    summaries: list[BatchSummary]
