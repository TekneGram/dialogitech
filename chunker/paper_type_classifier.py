from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from .markdown_section_chunker import MarkdownSectionChunker

PaperType = Literal[
    "empirical_research",
    "literature_review",
    "systematic_review_or_meta_analysis",
    "theoretical_or_conceptual",
    "position_or_discussion_paper",
    "methodological_paper",
    "pedagogical_or_practice_paper",
    "argumentative_essay",
    "other_or_unclear",
]
ClassificationSource = Literal["deterministic", "llm"]
ClassificationConfidence = Literal["low", "medium", "high"]

PAPER_TYPES: tuple[PaperType, ...] = (
    "empirical_research",
    "literature_review",
    "systematic_review_or_meta_analysis",
    "theoretical_or_conceptual",
    "position_or_discussion_paper",
    "methodological_paper",
    "pedagogical_or_practice_paper",
    "argumentative_essay",
    "other_or_unclear",
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
    label: PaperType | None
    source: ClassificationSource
    reason: str
    confidence: ClassificationConfidence | None = None
    used_context: bool = False
    needs_llm: bool = False


class DeterministicPaperTypeClassifier:
    """Resolve only document types supported by strong title, abstract, or heading evidence."""

    def classify(self, evidence: PaperTypeEvidence) -> PaperTypeClassification:
        combined = "\n".join(
            part for part in (evidence.title, evidence.abstract, "\n".join(evidence.headings)) if part
        ).lower()
        identity_text = "\n".join(part for part in (evidence.title, evidence.abstract) if part).lower()
        heading_text = "\n".join(evidence.headings).lower()

        if self._contains_any(identity_text, ("systematic review", "scoping review", "meta-analysis", "meta analysis")):
            return self._resolved(
                "systematic_review_or_meta_analysis",
                "title, abstract, or headings explicitly identify a systematic review or meta-analysis",
            )
        if self._contains_any(identity_text, ("methodological paper", "instrument development", "validation study", "study protocol")):
            return self._resolved(
                "methodological_paper",
                "title, abstract, or headings explicitly identify methodological development or validation",
            )
        if self._contains_any(identity_text, ("position paper", "commentary", "response to", "discussion paper")):
            return self._resolved(
                "position_or_discussion_paper",
                "title, abstract, or headings explicitly identify a position, commentary, response, or discussion paper",
            )
        if self._contains_any(identity_text, ("conceptual paper", "theoretical paper", "conceptual framework")) and not self._has_empirical_structure(heading_text):
            return self._resolved(
                "theoretical_or_conceptual",
                "title, abstract, or headings explicitly identify conceptual or theoretical work without empirical structure",
            )
        if self._contains_any(identity_text, ("literature review", "narrative review", "integrative review")):
            return self._resolved(
                "literature_review",
                "title, abstract, or headings explicitly identify a literature review",
            )
        if self._has_empirical_structure(heading_text):
            return self._resolved(
                "empirical_research",
                "headings contain both a method family and a results family",
            )
        if self._contains_any(combined, ("pedagogical implications", "teaching practice", "classroom practice")) and not self._has_empirical_structure(heading_text):
            return self._resolved(
                "pedagogical_or_practice_paper",
                "document centers pedagogy or practice without an empirical method-and-results structure",
            )

        return PaperTypeClassification(
            label=None,
            source="llm",
            reason="deterministic paper-type evidence was insufficient",
            needs_llm=True,
        )

    def _has_empirical_structure(self, heading_text: str) -> bool:
        has_method = self._contains_any(
            heading_text,
            ("method", "methods", "methodology", "participants", "materials", "procedure", "data collection", "research design"),
        )
        has_results = self._contains_any(
            heading_text,
            ("results", "findings", "quantitative results", "qualitative results"),
        )
        return has_method and has_results

    def _contains_any(self, text: str, phrases: tuple[str, ...]) -> bool:
        return any(re.search(rf"\b{re.escape(phrase)}\b", text) for phrase in phrases)

    def _resolved(self, label: PaperType, reason: str) -> PaperTypeClassification:
        return PaperTypeClassification(
            label=label,
            source="deterministic",
            reason=reason,
            confidence="high",
        )


class PaperTypeClassificationEnricher:
    def __init__(
        self,
        deterministic_classifier: DeterministicPaperTypeClassifier | None = None,
        llm_classifier: Any | None = None,
        force_llm: bool = False,
    ) -> None:
        self.deterministic_classifier = deterministic_classifier or DeterministicPaperTypeClassifier()
        self.llm_classifier = llm_classifier
        self.force_llm = force_llm

    def classify(self, evidence: PaperTypeEvidence) -> PaperTypeClassification:
        deterministic = self.deterministic_classifier.classify(evidence)
        if self.force_llm:
            if self.llm_classifier is None:
                raise RuntimeError("force_llm=True requires an llm_classifier.")
            return self.llm_classifier.classify(evidence)
        if deterministic.needs_llm and self.llm_classifier is not None:
            return self.llm_classifier.classify(evidence)
        return deterministic


def build_paper_type_evidence(
    filtered_markdown: str,
    metadata: dict[str, Any] | None = None,
) -> PaperTypeEvidence:
    metadata = metadata or {}
    headings = [heading.title for heading in MarkdownSectionChunker().extract_headings(filtered_markdown)]
    abstract = _abstract_from_markdown(filtered_markdown)
    return PaperTypeEvidence(
        title=_optional_string(metadata.get("title")),
        abstract=abstract,
        headings=headings,
        opening_excerpt=filtered_markdown[:1800].strip(),
        closing_excerpt=filtered_markdown[-900:].strip(),
    )


def classify_paper_type(
    filtered_markdown: str,
    *,
    metadata: dict[str, Any] | None = None,
    llm_classifier: Any | None = None,
    force_llm: bool = False,
) -> PaperTypeClassification:
    evidence = build_paper_type_evidence(filtered_markdown, metadata=metadata)
    return PaperTypeClassificationEnricher(
        llm_classifier=llm_classifier,
        force_llm=force_llm,
    ).classify(evidence)


def _abstract_from_markdown(markdown: str) -> str | None:
    splits = MarkdownSectionChunker().split_by_headings(markdown)
    for split in splits:
        if split.title.strip().lower() == "abstract":
            return split.content[:1800].strip() or None
    return None


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None
