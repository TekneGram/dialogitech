from __future__ import annotations

from .lancedb_client import LanceChunkStore


class LanceIndexManager:
    def __init__(self, store: LanceChunkStore) -> None:
        self.store = store

    def ensure_vector_index(self) -> None:
        table = self.store.table
        try:
            table.create_index(vector_column_name="embedding", replace=False)
        except RuntimeError as exc:
            if "Not enough rows to train PQ" in str(exc):
                return
            raise

    def ensure_full_text_index(self) -> None:
        table = self.store.table
        table.create_fts_index("text", replace=False)

    def ensure_scalar_indexes(self) -> None:
        table = self.store.table
        for column_name in ("paper_id", "classification_label", "classification_source"):
            table.create_scalar_index(column_name, replace=False)

    def ensure_all_indexes(self) -> None:
        self.ensure_vector_index()
        self.ensure_full_text_index()
        self.ensure_scalar_indexes()
