from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from .models import ChunkRecord, EmbeddedChunkRecord


class EmbeddingService(ABC):
    @abstractmethod
    def embedding_dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_chunks(self, chunks: list[ChunkRecord]) -> list[EmbeddedChunkRecord]:
        embeddings = self.embed_texts([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise RuntimeError("Embedding service returned a different number of embeddings than input chunks.")

        embedded_chunks: list[EmbeddedChunkRecord] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if len(embedding) != self.embedding_dimension():
                raise RuntimeError("Embedding dimension mismatch while building embedded chunk records.")

            embedded_chunks.append(
                EmbeddedChunkRecord(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    paper_title=chunk.paper_title,
                    authors=list(chunk.authors),
                    journal=chunk.journal,
                    year=chunk.year,
                    doi=chunk.doi,
                    section_title=chunk.section_title,
                    heading_level=chunk.heading_level,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    word_count=chunk.word_count,
                    classification_label=chunk.classification_label,
                    classification_source=chunk.classification_source,
                    classification_confidence=chunk.classification_confidence,
                    used_context=chunk.used_context,
                    reason=chunk.reason,
                    embedding=list(embedding),
                    markdown_path=chunk.markdown_path,
                    pdf_path=chunk.pdf_path,
                )
            )

        return embedded_chunks


class OpenAIEmbeddingService(EmbeddingService):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1/embeddings",
        dimensions: int | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.dimensions = dimensions

        if not self.api_key:
            raise RuntimeError("OpenAIEmbeddingService requires an API key via argument or OPENAI_API_KEY.")

        if self.dimensions is None:
            if model == "text-embedding-3-small":
                self.dimensions = 1536
            elif model == "text-embedding-3-large":
                self.dimensions = 3072
            else:
                raise RuntimeError("Provide dimensions explicitly for embedding models with unknown output size.")

    def embedding_dimension(self) -> int:
        if self.dimensions is None:
            raise RuntimeError("Embedding dimension is not configured.")
        return self.dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload: dict[str, object] = {
            "model": self.model,
            "input": texts,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Embedding request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Embedding request failed: {exc.reason}") from exc

        data = response_payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Embedding response did not contain a data list.")

        embeddings: list[list[float]] = []
        for item in data:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list):
                raise RuntimeError("Embedding response item was missing an embedding vector.")
            embeddings.append([float(value) for value in embedding])

        return embeddings


class OllamaEmbeddingService(EmbeddingService):
    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        base_url: str = "http://localhost:11434/api/embed",
        dimensions: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.dimensions = dimensions

    def embedding_dimension(self) -> int:
        if self.dimensions is None:
            raise RuntimeError("Embedding dimension is not known yet. Run one embedding request first.")
        return self.dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {
            "model": self.model,
            "input": texts,
        }

        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama embedding request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama embedding request failed: {exc.reason}") from exc

        embeddings_payload = response_payload.get("embeddings")
        if not isinstance(embeddings_payload, list):
            raise RuntimeError("Ollama embedding response did not contain an embeddings list.")

        embeddings: list[list[float]] = []
        for embedding in embeddings_payload:
            if not isinstance(embedding, list):
                raise RuntimeError("Ollama embedding response contained a non-vector embedding item.")
            embeddings.append([float(value) for value in embedding])

        if not embeddings:
            raise RuntimeError("Ollama embedding response returned no embeddings.")

        vector_dim = len(embeddings[0])
        if vector_dim <= 0:
            raise RuntimeError("Ollama embedding response returned an empty embedding vector.")

        if self.dimensions is None:
            self.dimensions = vector_dim
        elif self.dimensions != vector_dim:
            raise RuntimeError(
                f"Ollama embedding dimension changed from {self.dimensions} to {vector_dim}."
            )

        return embeddings


class DeterministicHashEmbeddingService(EmbeddingService):
    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")
        self.dimensions = dimensions

    def embedding_dimension(self) -> int:
        return self.dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]
