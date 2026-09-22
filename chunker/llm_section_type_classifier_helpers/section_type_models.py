from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .section_taxonomy import SectionLabel

ClassificationConfidence = Literal["low", "medium", "high"]
FallbackLabel = Literal["unclassified"]
ClassificationLabel = SectionLabel | FallbackLabel
ArticleQuintile = Literal[
    "first 20%",
    "second 20%",
    "third 20%",
    "fourth 20%",
    "last 20%",
]
LLMAction = Literal["classify", "request_context"]


@dataclass(slots=True)
class ChunkClassification:
    label: ClassificationLabel | None
    source: Literal["llm", "llm_fallback"]
    reason: str
    confidence: ClassificationConfidence | None = None
    used_context: bool = False


@dataclass(slots=True)
class ChunkLocation:
    article_start: int
    article_end: int
    quintile: ArticleQuintile
    section_index: int


@dataclass(slots=True)
class ChunkContext:
    previous_section: str | None
    current_chunk: str
    next_section: str | None


@dataclass(slots=True)
class LLMDecision:
    action: LLMAction
    label: SectionLabel | None = None
    confidence: ClassificationConfidence | None = None
    reason: str = ""
