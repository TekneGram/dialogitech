from __future__ import annotations

from dbinsert.embedding_service import EmbeddingService

from .models import SearchVariant


class QueryEmbedder:
    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    def embed_variants(self, variants: list[SearchVariant]) -> dict[str, list[float]]:
        if not variants:
            return {}
        vectors = self.embedding_service.embed_texts([variant.text for variant in variants])
        return {
            variant.key: vector
            for variant, vector in zip(variants, vectors, strict=True)
        }
