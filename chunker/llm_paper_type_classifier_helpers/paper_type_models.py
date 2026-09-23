from dataclasses import dataclass
from typing import Literal

PaperType = Literal[
  "empirical_research",
  "literature_review",
  "systematic_review_or_meta_analysis",
  "theoretical_or_conceptual",
  "position_or_discussion_paper",
  "methodological_paper",
  "pedagogical_or_practice_paper",
  "argumentative_essay",
  "other_or_unclear"
]

PAPER_TYPES: tuple[PaperType, ...] = (
  "empirical_research",
  "literature_review",
  "systematic_review_or_meta_analysis",
  "theoretical_or_conceptual",
  "position_or_discussion_paper",
  "methodological_paper",
  "pedagogical_or_practice_paper",
  "argumentative_essay",
  "other_or_unclear"
)

@dataclass(slots=True)
class PaperTypeEvidence:
  title: str | None
  abstract: str | None
  headings: list[str]
  opening_excerpt: str
  closing_excerpt: str

@dataclass(slots=True)
class PaperTypeClassification:
  label: PaperType
  source: Literal["llm"]
  reason: str
  confidence: Literal["low", "medium", "high"] | None
  used_context: bool = False
