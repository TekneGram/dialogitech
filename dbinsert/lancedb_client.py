from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from .lancedb_schema import chunk_table_schema
from .models import EmbeddedChunkRecord


class LanceChunkStore:
    def __init__(
        self,
        db_path: str | Path,
        table_name: str,
    ) -> None:
        self.db_path = str(db_path)
        self.table_name = table_name
        self._db: Any = None
        self._table: Any = None

    def connect(self) -> None:
        if self._db is None:
            self._db = lancedb.connect(self.db_path)

    def ensure_table(self, vector_dim: int) -> None:
        self.connect()
        table_names = set(self._db.table_names())

        if self.table_name in table_names:
            self._table = self._db.open_table(self.table_name)
            self._ensure_schema_columns(chunk_table_schema(vector_dim))
            return

        self._table = self._db.create_table(
            self.table_name,
            schema=chunk_table_schema(vector_dim),
        )

    def add_chunks(self, chunks: list[EmbeddedChunkRecord]) -> int:
        if not chunks:
            return 0
        if self._table is None:
            raise RuntimeError("Table is not initialized. Call ensure_table() before add_chunks().")

        self._table.add([self._record_to_row(chunk) for chunk in chunks])
        return len(chunks)

    def has_paper(self, paper_id: str) -> bool:
        if self._table is None:
            raise RuntimeError("Table is not initialized. Call ensure_table() before has_paper().")

        matches = self._table.search().where(f"paper_id = '{self._escape_sql_string(paper_id)}'").limit(1).to_list()
        return bool(matches)

    def delete_paper(self, paper_id: str) -> int:
        if self._table is None:
            raise RuntimeError("Table is not initialized. Call ensure_table() before delete_paper().")

        if not self.has_paper(paper_id):
            return 0

        self._table.delete(f"paper_id = '{self._escape_sql_string(paper_id)}'")
        return 1

    @property
    def table(self) -> Any:
        if self._table is None:
            raise RuntimeError("Table is not initialized. Call ensure_table() first.")
        return self._table

    def _record_to_row(self, chunk: EmbeddedChunkRecord) -> dict[str, Any]:
        row = asdict(chunk)
        row["embedding"] = [float(value) for value in chunk.embedding]
        return row

    def _escape_sql_string(self, value: str) -> str:
        return value.replace("'", "''")

    def _ensure_schema_columns(self, expected_schema: pa.Schema) -> None:
        """Add non-breaking columns when opening a database created by an older schema."""
        assert self._table is not None
        existing_names = set(self._table.schema.names)
        missing = [field for field in expected_schema if field.name not in existing_names]
        if not missing:
            return
        # Existing rows receive nulls; newly ingested rows always provide values.
        nullable_fields = [
            pa.field(field.name, self._all_nullable_type(field.type), nullable=True)
            for field in missing
        ]
        self._table.add_columns(pa.schema(nullable_fields))

    def _all_nullable_type(self, data_type: pa.DataType) -> pa.DataType:
        if pa.types.is_struct(data_type):
            return pa.struct(
                [
                    pa.field(child.name, self._all_nullable_type(child.type), nullable=True)
                    for child in data_type
                ]
            )
        if pa.types.is_list(data_type):
            value_field = data_type.value_field
            return pa.list_(
                pa.field(value_field.name, self._all_nullable_type(value_field.type), nullable=True)
            )
        return data_type
