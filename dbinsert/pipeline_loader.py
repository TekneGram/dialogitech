from __future__ import annotations

import json
import re
from pathlib import Path

from chunker.llm_metadata_extractor import LLMMetadataExtractor
from chunker.llm_section_type_classifier_helpers.section_type_models import ChunkClassification
from chunker.section_classifier import (
    ClassifiedHeadingSplit,
    ClassifiedSectionChunk,
)
from chunker.rhetorical_move_classifier import RhetoricalMoveClassification, RhetoricalMoveResult

from .models import PaperMetadataRecord


def load_classified_heading_splits(path: str | Path) -> tuple[list[ClassifiedHeadingSplit], str | None]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    headings = payload.get("headings")
    if not isinstance(headings, list):
        raise RuntimeError("Classified chunk JSON is missing a headings list.")

    classified_splits: list[ClassifiedHeadingSplit] = []
    for heading_payload in headings:
        chunks_payload = heading_payload.get("chunks", [])
        classified_chunks: list[ClassifiedSectionChunk] = []

        for chunk_payload in chunks_payload:
            classification_payload = chunk_payload.get("classification", {})
            classification = ChunkClassification(
                label=classification_payload.get("label"),
                source=classification_payload["source"],
                reason=classification_payload["reason"],
                confidence=classification_payload.get("confidence"),
                used_context=bool(classification_payload.get("used_context", False)),
            )
            rhetorical_payload = chunk_payload.get("rhetorical_move_result")
            rhetorical_move_result = None
            if isinstance(rhetorical_payload, dict):
                moves_payload = rhetorical_payload.get("moves", [])
                if not isinstance(moves_payload, list):
                    raise RuntimeError("Rhetorical move result has an invalid moves list.")
                rhetorical_move_result = RhetoricalMoveResult(
                    moves=[
                        RhetoricalMoveClassification(
                            label=move["label"],
                            confidence=move["confidence"],
                            reason=move["reason"],
                        )
                        for move in moves_payload
                    ],
                    source=rhetorical_payload.get("source", "llm"),
                    used_context=bool(rhetorical_payload.get("used_context", False)),
                    reason=str(rhetorical_payload.get("reason", "")),
                )

            classified_chunks.append(
                ClassifiedSectionChunk(
                    title=chunk_payload["title"],
                    heading_level=int(chunk_payload["heading_level"]),
                    chunk_index=int(chunk_payload["chunk_index"]),
                    text=chunk_payload["text"],
                    word_count=int(chunk_payload["word_count"]),
                    classification=classification,
                    rhetorical_move_result=rhetorical_move_result,
                )
            )

        classified_splits.append(
            ClassifiedHeadingSplit(
                title=heading_payload["title"],
                heading_level=int(heading_payload["heading_level"]),
                raw_heading=heading_payload["raw_heading"],
                content=heading_payload["content"],
                chunks=classified_chunks,
            )
        )

    source_markdown = payload.get("source_markdown")
    return classified_splits, source_markdown if isinstance(source_markdown, str) else None


def build_paper_metadata(
    *,
    classified_json_path: str | Path,
    marker_json_path: str | Path | None = None,
    paper_id: str | None = None,
    pdf_path: str | None = None,
    markdown_path: str | None = None,
) -> PaperMetadataRecord:
    classified_json_path = Path(classified_json_path)
    marker_json_path = _resolve_marker_json_path(classified_json_path, marker_json_path)

    title: str | None = None
    authors: list[str] = []
    journal_name: str | None = None
    volume: str | None = None
    issue: str | None = None
    year: int | None = None
    doi: str | None = None
    issn: str | None = None
    references: list[str] = []
    classified_payload = json.loads(classified_json_path.read_text(encoding="utf-8"))
    paper_type_payload = classified_payload.get("paper_type")
    if not isinstance(paper_type_payload, dict) or paper_type_payload.get("label") is None:
        raise RuntimeError("Classified chunk JSON is missing a resolved paper_type.")

    if marker_json_path is not None:
        metadata_extractor = LLMMetadataExtractor()
        try:
            metadata = metadata_extractor.extract_all(marker_json_path, allow_manual=False)
        finally:
            metadata_extractor.close()
        title = metadata.get("title")
        authors = list(metadata.get("authors") or [])

        journal_payload = metadata.get("journal") or {}
        if isinstance(journal_payload, dict):
            journal_name = journal_payload.get("name")
            volume = journal_payload.get("volume")
            issue = journal_payload.get("issue")
            doi = journal_payload.get("doi")
            issn = journal_payload.get("issn")
            raw_year = journal_payload.get("year")
            if isinstance(raw_year, str) and raw_year.isdigit():
                year = int(raw_year)
        references = list(metadata.get("references") or [])

    resolved_markdown_path = markdown_path or _source_markdown_from_classified_json(classified_json_path)
    resolved_paper_id = paper_id or _default_paper_id(classified_json_path, marker_json_path)
    resolved_title = title or _fallback_title_from_paths(classified_json_path, resolved_markdown_path)

    return PaperMetadataRecord(
        paper_id=resolved_paper_id,
        paper_title=resolved_title,
        authors=authors,
        journal=journal_name,
        volume=volume,
        issue=issue,
        year=year,
        doi=doi,
        issn=issn,
        references=references,
        markdown_path=resolved_markdown_path,
        marker_json_path=str(marker_json_path) if marker_json_path is not None else None,
        pdf_path=pdf_path,
        paper_type=str(paper_type_payload["label"]),
        paper_type_source=paper_type_payload["source"],
        paper_type_confidence=paper_type_payload.get("confidence"),
        paper_type_used_context=bool(paper_type_payload.get("used_context", False)),
        paper_type_reason=str(paper_type_payload["reason"]),
    )


def _resolve_marker_json_path(
    classified_json_path: Path,
    marker_json_path: str | Path | None,
) -> Path | None:
    if marker_json_path is not None:
        return Path(marker_json_path)

    inferred_name = classified_json_path.name.replace("_classified_chunks.json", ".json")
    inferred_path = classified_json_path.with_name(inferred_name)
    return inferred_path if inferred_path.exists() else None


def _source_markdown_from_classified_json(classified_json_path: Path) -> str | None:
    payload = json.loads(classified_json_path.read_text(encoding="utf-8"))
    source_markdown = payload.get("source_markdown")
    return source_markdown if isinstance(source_markdown, str) else None


def _default_paper_id(classified_json_path: Path, marker_json_path: Path | None) -> str:
    source_name = marker_json_path.stem if marker_json_path is not None else classified_json_path.stem
    source_name = re.sub(r"_classified_chunks$", "", source_name)
    return source_name


def _fallback_title_from_paths(classified_json_path: Path, markdown_path: str | None) -> str:
    if markdown_path:
        return Path(markdown_path).stem.replace("_filtered", "")
    return classified_json_path.stem.replace("_classified_chunks", "")
