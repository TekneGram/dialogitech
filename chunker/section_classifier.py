from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .llm_section_classifier import ChunkClassificationLLM
from .markdown_section_chunker import HeadingSplit, MarkdownSectionChunker, SectionChunk

SectionLabel = Literal["abstract", "introduction", "method", "results", "discussion"]
ClassificationSource = Literal["deterministic", "llm"]
ClassificationConfidence = Literal["low", "medium", "high"]


@dataclass(slots=True)
class ChunkClassification:
    label: SectionLabel | None
    source: ClassificationSource
    reason: str
    confidence: ClassificationConfidence | None = None
    used_context: bool = False
    needs_llm: bool = False


@dataclass(slots=True)
class ClassifiedSectionChunk:
    title: str
    heading_level: int
    chunk_index: int
    text: str
    word_count: int
    classification: ChunkClassification


@dataclass(slots=True)
class ClassifiedHeadingSplit:
    title: str
    heading_level: int
    raw_heading: str
    content: str
    chunks: list[ClassifiedSectionChunk] = field(default_factory=list)


class DeterministicSectionClassifier:
    TITLE_NORMALIZE_PATTERN = re.compile(r"[*_`#]+")
    SPACE_PATTERN = re.compile(r"\s+")
    CEFR_LEVEL_TITLE_PATTERN = re.compile(r"^[abc]\d(?:\.\d+)?$", re.IGNORECASE)

    TITLE_RULES: tuple[tuple[set[str], SectionLabel, str], ...] = (
        ({"abstract"}, "abstract", "heading matched abstract"),
        (
            {"introduction", "background", "related work", "literature review"},
            "introduction",
            "heading matched introduction/background family",
        ),
        (
            {
                "method",
                "methods",
                "methodology",
                "materials and methods",
                "experimental setup",
                "experiment",
                "experiments",
                "procedure",
                "procedures",
                "participants",
                "materials",
                "data",
                "data collection",
                "corpus",
                "text generation using chatgpt",
                "vocabulary level analysis using software",
                "topic analysis method",
                "seed demonstration collection",
            },
            "method",
            "heading matched method/experiment family",
        ),
        (
            {
                "result",
                "results",
                "finding",
                "findings",
                "analysis",
                "analyses",
                "evaluation",
                "evaluations",
                "topic frequency",
            },
            "results",
            "heading matched results/analysis/evaluation family",
        ),
        (
            {
                "discussion",
                "conclusion",
                "conclusions",
                "limitations",
                "implications",
                "future directions",
            },
            "discussion",
            "heading matched discussion/conclusion family",
        ),
    )

    def classify(
        self,
        chunk: SectionChunk,
        heading_split: HeadingSplit,
        parent_label: SectionLabel | None = None,
        previous_label: SectionLabel | None = None,
    ) -> ChunkClassification:
        normalized_title = self._normalize_title(heading_split.title or chunk.title)

        for titles, label, reason in self.TITLE_RULES:
            if normalized_title in titles:
                return ChunkClassification(
                    label=label,
                    source="deterministic",
                    reason=reason,
                    confidence="high",
                    used_context=False,
                    needs_llm=False,
                )

        if heading_split.heading_level > 1 and parent_label is not None:
            return ChunkClassification(
                label=parent_label,
                source="deterministic",
                reason="inherited classification from parent heading",
                confidence="medium",
                used_context=False,
                needs_llm=False,
            )

        if previous_label is not None and self._should_inherit_from_previous(normalized_title):
            return ChunkClassification(
                label=previous_label,
                source="deterministic",
                reason="inherited classification from previous resolved heading",
                confidence="medium",
                used_context=False,
                needs_llm=False,
            )

        return ChunkClassification(
            label=None,
            source="llm",
            reason="deterministic heading-based rules could not resolve the section label",
            confidence=None,
            used_context=False,
            needs_llm=True,
        )

    def _normalize_title(self, title: str) -> str:
        cleaned = self.TITLE_NORMALIZE_PATTERN.sub("", title.strip().lower())
        cleaned = self.SPACE_PATTERN.sub(" ", cleaned)
        return cleaned.strip()

    def _should_inherit_from_previous(self, normalized_title: str) -> bool:
        return bool(self.CEFR_LEVEL_TITLE_PATTERN.fullmatch(normalized_title))


class ChunkClassificationEnricher:
    def __init__(
        self,
        deterministic_classifier: DeterministicSectionClassifier | None = None,
        llm_classifier: ChunkClassificationLLM | None = None,
        force_llm: bool = False,
    ) -> None:
        self.deterministic_classifier = deterministic_classifier or DeterministicSectionClassifier()
        self.llm_classifier = llm_classifier
        self.force_llm = force_llm

    def enrich_heading_splits(self, heading_splits: list[HeadingSplit]) -> list[ClassifiedHeadingSplit]:
        enriched: list[ClassifiedHeadingSplit] = []
        current_top_level_label: SectionLabel | None = None
        previous_resolved_label: SectionLabel | None = None

        for heading_split in heading_splits:
            parent_label = current_top_level_label if heading_split.heading_level > 1 else None
            classified_chunks: list[ClassifiedSectionChunk] = []
            for chunk in heading_split.chunks:
                if self.force_llm:
                    if self.llm_classifier is None:
                        raise RuntimeError("force_llm=True requires an llm_classifier.")
                    classification = self.llm_classifier.classify_with_previous_label(
                        chunk=chunk,
                        heading_split=heading_split,
                        previous_label=previous_resolved_label,
                    )
                else:
                    classification = self.deterministic_classifier.classify(
                        chunk=chunk,
                        heading_split=heading_split,
                        parent_label=parent_label,
                        previous_label=previous_resolved_label,
                    )
                    if classification.needs_llm and self.llm_classifier is not None:
                        classification = self.llm_classifier.classify_with_previous_label(
                            chunk=chunk,
                            heading_split=heading_split,
                            previous_label=previous_resolved_label,
                        )

                classified_chunks.append(
                    ClassifiedSectionChunk(
                        title=chunk.title,
                        heading_level=chunk.heading_level,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        word_count=chunk.word_count,
                        classification=classification,
                    )
                )

            if heading_split.heading_level == 1 and classified_chunks:
                top_level_label = classified_chunks[0].classification.label
                if top_level_label is not None:
                    current_top_level_label = top_level_label
                    previous_resolved_label = top_level_label
            elif classified_chunks:
                for classified_chunk in classified_chunks:
                    if classified_chunk.classification.label is not None:
                        previous_resolved_label = classified_chunk.classification.label
                        break

            enriched.append(
                ClassifiedHeadingSplit(
                    title=heading_split.title,
                    heading_level=heading_split.heading_level,
                    raw_heading=heading_split.raw_heading,
                    content=heading_split.content,
                    chunks=classified_chunks,
                )
            )

        return enriched


def classify_filtered_markdown(
    filtered_markdown: str,
    *,
    min_words: int = 200,
    overlap_words: int = 50,
    llm_classifier: ChunkClassificationLLM | None = None,
    force_llm: bool = False,
) -> list[ClassifiedHeadingSplit]:
    heading_splits = MarkdownSectionChunker(min_words=min_words, overlap_words=overlap_words).process(filtered_markdown)
    enricher = ChunkClassificationEnricher(llm_classifier=llm_classifier, force_llm=force_llm)
    return enricher.enrich_heading_splits(heading_splits)
