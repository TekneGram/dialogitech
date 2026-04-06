from __future__ import annotations

import math
import re

from dbinsert.embedding_service import EmbeddingService

from .models import ResearchSessionState, StateUpdateClaim


class ClaimDeduper:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService | None = None,
        similarity_threshold: float = 0.85,
    ) -> None:
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold

    def dedupe(
        self,
        *,
        existing_state: ResearchSessionState,
        claims: list[StateUpdateClaim],
    ) -> list[StateUpdateClaim]:
        existing_texts = {self._normalize(claim.text) for claim in existing_state.claims.values()}
        deduped_exact: list[StateUpdateClaim] = []
        seen_new: set[str] = set()
        for claim in claims:
            normalized = self._normalize(claim.text)
            if not normalized or normalized in existing_texts or normalized in seen_new:
                continue
            deduped_exact.append(claim)
            seen_new.add(normalized)
        return self._dedupe_semantic(existing_state=existing_state, claims=deduped_exact)

    def _dedupe_semantic(
        self,
        *,
        existing_state: ResearchSessionState,
        claims: list[StateUpdateClaim],
    ) -> list[StateUpdateClaim]:
        if self.embedding_service is None or len(claims) <= 1 and not existing_state.claims:
            return claims

        existing_claims = [claim for claim in existing_state.claims.values() if self._normalize(claim.text)]
        texts_to_embed = [claim.text for claim in existing_claims]
        texts_to_embed.extend(claim.text for claim in claims)
        vectors = self.embedding_service.embed_texts(texts_to_embed)

        existing_vectors = vectors[: len(existing_claims)]
        candidate_vectors = vectors[len(existing_claims) :]
        accepted_claims: list[StateUpdateClaim] = []
        accepted_vectors = list(existing_vectors)

        for claim, vector in zip(claims, candidate_vectors, strict=True):
            if self._is_semantic_duplicate(vector, accepted_vectors):
                continue
            accepted_claims.append(claim)
            accepted_vectors.append(vector)

        return accepted_claims

    def _is_semantic_duplicate(
        self,
        candidate_vector: list[float],
        comparison_vectors: list[list[float]],
    ) -> bool:
        for comparison_vector in comparison_vectors:
            if self._cosine_similarity(candidate_vector, comparison_vector) >= self.similarity_threshold:
                return True
        return False

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise RuntimeError("Claim embedding dimension mismatch during semantic dedupe.")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
        return dot_product / (left_norm * right_norm)

    def _normalize(self, text: str) -> str:
        stripped = re.sub(r"[^\w\s]", " ", text.lower())
        return " ".join(stripped.split())
