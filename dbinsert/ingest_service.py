from __future__ import annotations

from chunker.section_classifier import ClassifiedHeadingSplit

from .embedding_service import EmbeddingService
from .index_manager import LanceIndexManager
from .lancedb_client import LanceChunkStore
from .models import PaperMetadataRecord
from .paper_chunk_serializer import PaperChunkSerializer


class ChunkIngestionService:
    def __init__(
        self,
        serializer: PaperChunkSerializer,
        embedding_service: EmbeddingService,
        store: LanceChunkStore,
        index_manager: LanceIndexManager | None = None,
    ) -> None:
        self.serializer = serializer
        self.embedding_service = embedding_service
        self.store = store
        self.index_manager = index_manager

    def ingest_paper(
        self,
        paper_metadata: PaperMetadataRecord,
        classified_splits: list[ClassifiedHeadingSplit],
        *,
        create_indexes: bool = True,
        replace_existing: bool = False,
    ) -> int:
        chunk_records = self.serializer.serialize_paper(paper_metadata, classified_splits)
        if not chunk_records:
            return 0

        embedded_chunks = self.embedding_service.embed_chunks(chunk_records)
        self.store.ensure_table(vector_dim=self.embedding_service.embedding_dimension())

        if replace_existing:
            self.store.delete_paper(paper_metadata.paper_id)

        inserted_count = self.store.add_chunks(embedded_chunks)

        if create_indexes:
            if self.index_manager is None:
                raise RuntimeError("create_indexes=True requires an index_manager.")
            self.index_manager.ensure_all_indexes()

        return inserted_count
