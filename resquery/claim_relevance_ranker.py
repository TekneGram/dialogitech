from __future__ import annotations

import math
from collections.abc import Callable

from dbinsert.embedding_service import EmbeddingService

from .models import ResearchClaim


class ClaimRelevanceRanker:
    def __init__(self, *, embedding_service: EmbeddingService | None = None) -> None:
        self.embedding_service = embedding_service

    def rank(
        self,
        *,
        user_question: str,
        claims: list[ResearchClaim],
        fallback_sort_key: Callable[[ResearchClaim], tuple[int, str]],
    ) -> list[ResearchClaim]:
        if not claims:
            return []

        question = user_question.strip()
        if not question or self.embedding_service is None:
            return self._fallback_rank(claims, fallback_sort_key=fallback_sort_key)

        try:
            vectors = self.embedding_service.embed_texts([question, *[claim.text for claim in claims]])
        except RuntimeError:
            return self._fallback_rank(claims, fallback_sort_key=fallback_sort_key)

        if len(vectors) != len(claims) + 1:
            return self._fallback_rank(claims, fallback_sort_key=fallback_sort_key)

        question_vector = vectors[0]
        scored_claims: list[tuple[float, tuple[int, str], ResearchClaim]] = []
        for claim, vector in zip(claims, vectors[1:], strict=True):
            scored_claims.append(
                (
                    self._cosine_similarity(question_vector, vector),
                    fallback_sort_key(claim),
                    claim,
                )
            )

        scored_claims.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored_claims]

    def _fallback_rank(
        self,
        claims: list[ResearchClaim],
        *,
        fallback_sort_key: Callable[[ResearchClaim], tuple[int, str]],
    ) -> list[ResearchClaim]:
        return sorted(claims, key=fallback_sort_key, reverse=True)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise RuntimeError("Claim similarity embedding dimension mismatch.")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
        return dot_product / (left_norm * right_norm)
