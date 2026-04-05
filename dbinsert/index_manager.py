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
            if self._is_existing_index_error(exc) or "Not enough rows to train PQ" in str(exc):
                return
            raise

    def ensure_full_text_index(self) -> None:
        table = self.store.table
        try:
            table.create_fts_index("text", replace=False)
        except RuntimeError as exc:
            if self._is_existing_index_error(exc):
                return
            raise

    def ensure_scalar_indexes(self) -> None:
        table = self.store.table
        for column_name in ("paper_id", "classification_label", "classification_source"):
            try:
                table.create_scalar_index(column_name, replace=False)
            except RuntimeError as exc:
                if self._is_existing_index_error(exc):
                    continue
                raise

    def ensure_all_indexes(self) -> None:
        self.ensure_vector_index()
        self.ensure_full_text_index()
        self.ensure_scalar_indexes()

    def _is_existing_index_error(self, exc: RuntimeError) -> bool:
        message = str(exc)
        return "already exists" in message or "Duplicate key name" in message
