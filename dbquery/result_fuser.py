from __future__ import annotations

from .models import FusedChunkResult, RetrievedChunk


class ReciprocalRankFuser:
    def __init__(self, *, k: int = 60, min_lists: int = 1) -> None:
        if k <= 0:
            raise ValueError("k must be positive.")
        if min_lists <= 0:
            raise ValueError("min_lists must be positive.")
        self.k = k
        self.min_lists = min_lists

    def fuse(self, retrieved_chunks: list[RetrievedChunk], *, final_top_k: int) -> list[FusedChunkResult]:
        grouped: dict[str, list[RetrievedChunk]] = {}
        for chunk in retrieved_chunks:
            grouped.setdefault(chunk.chunk_id, []).append(chunk)

        fused: list[FusedChunkResult] = []
        for chunk_id, hits in grouped.items():
            retrieval_keys = sorted({hit.retrieval_key for hit in hits})
            if len(retrieval_keys) < self.min_lists:
                continue

            exemplar = hits[0]
            rrf_score = sum(1.0 / (self.k + hit.rank) for hit in hits)
            contributing_ranks = {hit.retrieval_key: hit.rank for hit in hits}
            fused.append(
                FusedChunkResult(
                    chunk_id=chunk_id,
                    paper_id=exemplar.paper_id,
                    paper_title=exemplar.paper_title,
                    authors=list(exemplar.authors),
                    year=exemplar.year,
                    section_title=exemplar.section_title,
                    chunk_index=exemplar.chunk_index,
                    text=exemplar.text,
                    classification_label=exemplar.classification_label,
                    classification_source=exemplar.classification_source,
                    markdown_path=exemplar.markdown_path,
                    marker_json_path=exemplar.marker_json_path,
                    pdf_path=exemplar.pdf_path,
                    rrf_score=rrf_score,
                    retrieval_keys=retrieval_keys,
                    retrieval_kinds=[hit.retrieval_kind for hit in hits],
                    contributing_ranks=contributing_ranks,
                    relevance_score=_best_descending([hit.relevance_score for hit in hits]),
                    distance=_best_ascending([hit.distance for hit in hits]),
                    fts_score=_best_descending([hit.fts_score for hit in hits]),
                )
            )

        fused.sort(key=lambda item: (-item.rrf_score, item.paper_id, item.chunk_index))
        return fused[:final_top_k]


def _best_descending(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    return max(filtered) if filtered else None


def _best_ascending(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    return min(filtered) if filtered else None
