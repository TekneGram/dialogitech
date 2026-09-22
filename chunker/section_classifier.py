from __future__ import annotations

from dataclasses import dataclass, field

from .llm_section_type_classifier_helpers.section_type_models import ChunkClassification
from .rhetorical_move_classifier import RhetoricalMoveResult


@dataclass(slots=True)
class ClassifiedSectionChunk:
    title: str
    heading_level: int
    chunk_index: int
    text: str
    word_count: int
    classification: ChunkClassification
    rhetorical_move_result: RhetoricalMoveResult | None = None


@dataclass(slots=True)
class ClassifiedHeadingSplit:
    title: str
    heading_level: int
    raw_heading: str
    content: str
    chunks: list[ClassifiedSectionChunk] = field(default_factory=list)
