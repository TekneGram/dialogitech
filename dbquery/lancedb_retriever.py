from __future__ import annotations

from pathlib import Path
from typing import Any

import lancedb
from lancedb.rerankers import RRFReranker

from .models import QueryFilters, RetrievedChunk, RetrievalKind, SearchVariant
from .utils import build_where_clause, expanded_limit_for_filters, row_matches_filters


class LanceDBRetriever:
    def __init__(
        self,
        *,
        db_path: str | Path,
        table_name: str,
    ) -> None:
        self.db_path = str(db_path)
        self.table_name = table_name
        self._db: Any = None
        self._table: Any = None

    @property
    def table(self) -> Any:
        if self._table is None:
            self._db = lancedb.connect(self.db_path)
            self._table = self._db.open_table(self.table_name)
        return self._table

    def retrieve(
        self,
        *,
        variant: SearchVariant,
        limit: int,
        vector: list[float] | None = None,
        min_relevance_score: float | None = None,
        filters: QueryFilters | None = None,
    ) -> list[RetrievedChunk]:
        expanded_limit = expanded_limit_for_filters(limit, filters)
        where_clause = build_where_clause(filters)
        if variant.retrieval_kind == "hybrid":
            if vector is None:
                raise RuntimeError("Hybrid retrieval requires an embedded query vector.")
            query = (
                self.table.search(query_type="hybrid", vector_column_name="embedding", fts_columns="text")
                .vector(vector)
                .text(variant.text)
                .rerank(RRFReranker())
                .limit(expanded_limit)
            )
        elif variant.retrieval_kind == "fts":
            query = self.table.search(variant.text, query_type="fts", fts_columns="text").limit(expanded_limit)
        elif variant.retrieval_kind == "vector":
            if vector is None:
                raise RuntimeError("Vector retrieval requires an embedded query vector.")
            query = self.table.search(vector, query_type="vector", vector_column_name="embedding").limit(
                expanded_limit
            )
        else:
            raise RuntimeError(f"Unsupported retrieval kind: {variant.retrieval_kind}")

        if where_clause:
            query = query.where(where_clause)

        rows = query.to_list()
        results: list[RetrievedChunk] = []
        for row in rows:
            if not row_matches_filters(row, filters):
                continue
            relevance_score = _as_float(row.get("_relevance_score"))
            if min_relevance_score is not None and relevance_score is not None:
                if relevance_score < min_relevance_score:
                    continue
            rank = len(results) + 1
            results.append(self._row_to_retrieved_chunk(row=row, variant=variant, rank=rank))
            if len(results) >= limit:
                break
        return results

    def _row_to_retrieved_chunk(
        self,
        *,
        row: dict[str, Any],
        variant: SearchVariant,
        rank: int,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=str(row["chunk_id"]),
            paper_id=str(row["paper_id"]),
            paper_title=str(row["paper_title"]),
            authors=[str(author) for author in row.get("authors", [])],
            year=_as_int(row.get("year")),
            section_title=str(row["section_title"]),
            chunk_index=int(row["chunk_index"]),
            text=str(row["text"]),
            classification_label=_as_optional_string(row.get("classification_label")),
            classification_source=str(row["classification_source"]),
            markdown_path=_as_optional_string(row.get("markdown_path")),
            marker_json_path=_as_optional_string(row.get("marker_json_path")),
            pdf_path=_as_optional_string(row.get("pdf_path")),
            retrieval_key=variant.key,
            retrieval_kind=variant.retrieval_kind,
            rank=rank,
            score=_as_float(row.get("score")),
            relevance_score=_as_float(row.get("_relevance_score")),
            distance=_as_float(row.get("_distance")),
            fts_score=_as_float(row.get("score")),
        )


def _as_optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
