from __future__ import annotations

import pyarrow as pa

DEFAULT_TABLE_NAME = "paper_chunks"


def chunk_table_schema(vector_dim: int) -> pa.Schema:
    if vector_dim <= 0:
        raise ValueError("vector_dim must be positive.")

    return pa.schema(
        [
            pa.field("chunk_id", pa.string(), nullable=False),
            pa.field("paper_id", pa.string(), nullable=False),
            pa.field("paper_title", pa.string(), nullable=False),
            pa.field("authors", pa.list_(pa.string()), nullable=False),
            pa.field("journal", pa.string()),
            pa.field("volume", pa.string()),
            pa.field("issue", pa.string()),
            pa.field("year", pa.int32()),
            pa.field("doi", pa.string()),
            pa.field("issn", pa.string()),
            pa.field("references", pa.list_(pa.string()), nullable=False),
            pa.field("section_title", pa.string(), nullable=False),
            pa.field("heading_level", pa.int32(), nullable=False),
            pa.field("chunk_index", pa.int32(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("word_count", pa.int32(), nullable=False),
            pa.field("classification_label", pa.string()),
            pa.field("classification_source", pa.string(), nullable=False),
            pa.field("classification_confidence", pa.string()),
            pa.field("used_context", pa.bool_(), nullable=False),
            pa.field("reason", pa.string(), nullable=False),
            pa.field("paper_type", pa.string(), nullable=False),
            pa.field("paper_type_source", pa.string(), nullable=False),
            pa.field("paper_type_confidence", pa.string()),
            pa.field("paper_type_used_context", pa.bool_(), nullable=False),
            pa.field("paper_type_reason", pa.string(), nullable=False),
            pa.field(
                "rhetorical_moves",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("label", pa.string(), nullable=False),
                            pa.field("confidence", pa.string(), nullable=False),
                            pa.field("reason", pa.string(), nullable=False),
                        ]
                    )
                ),
                nullable=False,
            ),
            pa.field("rhetorical_move_source", pa.string()),
            pa.field("rhetorical_move_used_context", pa.bool_(), nullable=False),
            pa.field("rhetorical_move_reason", pa.string()),
            pa.field("markdown_path", pa.string()),
            pa.field("marker_json_path", pa.string()),
            pa.field("pdf_path", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), vector_dim), nullable=False),
        ]
    )
