from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SectionLabel = Literal["abstract", "introduction", "method", "results", "discussion"]
ClassificationSource = Literal["deterministic", "llm"]
ClassificationConfidence = Literal["low", "medium", "high"]


@dataclass(slots=True)
class PaperMetadataRecord:
    paper_id: str
    paper_title: str
    authors: list[str]
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None
    year: int | None = None
    doi: str | None = None
    issn: str | None = None
    references: list[str] | None = None
    markdown_path: str | None = None
    marker_json_path: str | None = None
    pdf_path: str | None = None


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    paper_id: str
    paper_title: str
    authors: list[str]
    journal: str | None
    volume: str | None
    issue: str | None
    year: int | None
    doi: str | None
    issn: str | None
    references: list[str]
    section_title: str
    heading_level: int
    chunk_index: int
    text: str
    word_count: int
    classification_label: SectionLabel | None
    classification_source: ClassificationSource
    classification_confidence: ClassificationConfidence | None
    used_context: bool
    reason: str
    markdown_path: str | None = None
    marker_json_path: str | None = None
    pdf_path: str | None = None


@dataclass(slots=True)
class EmbeddedChunkRecord:
    chunk_id: str
    paper_id: str
    paper_title: str
    authors: list[str]
    journal: str | None
    volume: str | None
    issue: str | None
    year: int | None
    doi: str | None
    issn: str | None
    references: list[str]
    section_title: str
    heading_level: int
    chunk_index: int
    text: str
    word_count: int
    classification_label: SectionLabel | None
    classification_source: ClassificationSource
    classification_confidence: ClassificationConfidence | None
    used_context: bool
    reason: str
    embedding: list[float]
    markdown_path: str | None = None
    marker_json_path: str | None = None
    pdf_path: str | None = None
