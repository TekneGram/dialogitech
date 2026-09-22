from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .llm_section_classifier import SectionClassificationLLM
from chunker.llm_paper_type_classifier_helpers.paper_type_models import PaperType
from .llm_section_type_classifier_helpers.section_type_models import (
    ChunkContext,
    ChunkLocation,
    ClassificationConfidence,
)
from .rhetorical_move_classifier import (
    RhetoricalMoveClassification,
    RhetoricalMoveEnricher,
    RhetoricalMoveResult,
    SectionLabel,
)


class RhetoricalMoveClassificationLLM(SectionClassificationLLM):
    """Gemma-backed, section-constrained rhetorical-move classifier."""

    MODEL_MAX_TOKENS = 360
    JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
    CONFIDENCE_LEVELS: tuple[ClassificationConfidence, ...] = ("low", "medium", "high")
    SYSTEM_PROMPT = """You identify rhetorical moves in chunks from academic articles.

Return JSON only. Select zero to three move labels from the allowed labels supplied by the user.
Order moves from primary to secondary. Do not invent labels. A chunk may have no applicable move.
Every move needs a confidence of low, medium, or high and a short reason.

JSON schema:
{"moves":[{"label":"present_results","confidence":"high","reason":"reports a measured outcome"}],"reason":"short overall explanation"}
"""

    def __init__(
        self,
        *,
        filtered_markdown: str,
        heading_splits: list[Any],
        model_path: str | Path,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        python_executable: str | Path | None = None,
        request_timeout_seconds: float = 180.0,
        event_logger: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(
            filtered_markdown=filtered_markdown,
            heading_splits=heading_splits,
            model_path=model_path,
            max_tokens=max_tokens or self.MODEL_MAX_TOKENS,
            temperature=temperature,
            python_executable=python_executable,
            request_timeout_seconds=request_timeout_seconds,
            event_logger=event_logger,
        )

    def classify(
        self,
        *,
        chunk: Any,
        heading_split: Any,
        section_label: SectionLabel,
        paper_type: PaperType = "empirical_research",
    ) -> RhetoricalMoveResult:
        RhetoricalMoveEnricher.validate_section_for_paper_type(section_label, paper_type=paper_type)
        allowed_moves = RhetoricalMoveEnricher.allowed_moves(section_label)
        location = self._chunk_location(chunk, heading_split)
        chunk_ref = self._chunk_ref(chunk=chunk, heading_split=heading_split)
        initial_prompt = self._initial_prompt(
            chunk, heading_split, paper_type, section_label, location, allowed_moves
        )
        used_context = False
        try:
            self._log_event(f"{chunk_ref} sending rhetorical-move request (section={section_label}).")
            response = self._generate([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": initial_prompt},
            ])
            self._log_event(f"{chunk_ref} received rhetorical-move response: {self._truncate_for_log(response)}")
            try:
                payload = self._parse_payload(response)
            except RuntimeError as exc:
                return self._correct_invalid_response(
                    chunk_ref=chunk_ref,
                    invalid_response=response,
                    failure=exc,
                    section_label=section_label,
                    paper_type=paper_type,
                    allowed_moves=allowed_moves,
                    used_context=used_context,
                )
            if payload.get("action") == "request_context":
                used_context = True
                context = self._context_for_chunk(chunk, heading_split)
                context_prompt = self._context_prompt(
                    chunk, heading_split, paper_type, section_label, location, allowed_moves, context
                )
                self._log_event(f"{chunk_ref} requested rhetorical-move context; sending one context round.")
                response = self._generate([
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": initial_prompt},
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": context_prompt},
                ])
                self._log_event(
                    f"{chunk_ref} received rhetorical-move context response: {self._truncate_for_log(response)}"
                )
                try:
                    payload = self._parse_payload(response)
                except RuntimeError as exc:
                    return self._correct_invalid_response(
                        chunk_ref=chunk_ref,
                        invalid_response=response,
                        failure=exc,
                        section_label=section_label,
                        paper_type=paper_type,
                        allowed_moves=allowed_moves,
                        used_context=used_context,
                    )
                if payload.get("action") == "request_context":
                    raise RuntimeError("Model requested rhetorical-move context after the single allowed context round.")
            try:
                return self._result_from_payload(
                    payload,
                    section_label=section_label,
                    paper_type=paper_type,
                    used_context=used_context,
                )
            except RuntimeError as exc:
                return self._correct_invalid_response(
                    chunk_ref=chunk_ref,
                    invalid_response=response,
                    failure=exc,
                    section_label=section_label,
                    paper_type=paper_type,
                    allowed_moves=allowed_moves,
                    used_context=used_context,
                )
        except RuntimeError as exc:
            self._log_event(f"{chunk_ref} rhetorical-move request failed: {exc}")
            raise RuntimeError(f"{chunk_ref} Gemma failed to classify rhetorical moves: {exc}") from exc

    def _initial_prompt(
        self, chunk: Any, heading_split: Any, paper_type: PaperType, section_label: SectionLabel,
        location: ChunkLocation, allowed_moves: tuple[str, ...],
    ) -> str:
        return "\n".join([
            "This chunk has already been classified into a paper section.",
            f"Paper type: {paper_type}",
            f"Section classification: {section_label}",
            f"Heading: {heading_split.title}",
            f"Article position: {location.quintile}",
            "Allowed rhetorical moves:",
            *[f"- {label}" for label in allowed_moves],
            "Return zero to three applicable moves from that exact list.",
            "If necessary context is missing, return only {\"action\":\"request_context\",\"reason\":\"...\"}.",
            "Chunk:", chunk.text,
        ])

    def _context_prompt(
        self, chunk: Any, heading_split: Any, paper_type: PaperType, section_label: SectionLabel,
        location: ChunkLocation, allowed_moves: tuple[str, ...], context: ChunkContext,
    ) -> str:
        return "\n".join([
            "Context requested. You must now return the final rhetorical-move JSON object.",
            f"Paper type: {paper_type}",
            f"Section classification: {section_label}", f"Heading: {heading_split.title}",
            f"Article position: {location.quintile}", "Allowed rhetorical moves:",
            *[f"- {label}" for label in allowed_moves],
            "Previous heading section:", context.previous_section or "[none]",
            "Current chunk:", context.current_chunk,
            "Next heading section:", context.next_section or "[none]",
        ])

    def _parse_payload(self, raw_response: str) -> dict[str, Any]:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if "\n" in cleaned:
                cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.removesuffix("```").strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            match = self.JSON_PATTERN.search(cleaned)
            if match is None:
                raise RuntimeError(f"Failed to parse rhetorical-move JSON: {raw_response}")
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse rhetorical-move JSON: {raw_response}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Rhetorical-move response must be a JSON object.")
        return payload

    def _correct_invalid_response(
        self,
        *,
        chunk_ref: str,
        invalid_response: str,
        failure: RuntimeError,
        section_label: SectionLabel,
        paper_type: PaperType,
        allowed_moves: tuple[str, ...],
        used_context: bool,
    ) -> RhetoricalMoveResult:
        self._log_event(f"{chunk_ref} invalid rhetorical-move response; sending one correction prompt: {failure}")
        correction_prompt = "\n".join([
            "Your previous rhetorical-move response was invalid.",
            f"Paper type: {paper_type}",
            f"Section classification: {section_label}",
            "Allowed rhetorical moves:",
            *[f"- {label}" for label in allowed_moves],
            "Return JSON only using the required moves-list schema. Do not request context.",
            "Previous invalid response:", invalid_response,
        ])
        corrected_response = self._generate([
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": correction_prompt},
        ])
        self._log_event(
            f"{chunk_ref} received rhetorical-move correction: {self._truncate_for_log(corrected_response)}"
        )
        payload = self._parse_payload(corrected_response)
        if payload.get("action") == "request_context":
            raise RuntimeError("Model requested context in its rhetorical-move correction response.")
        return self._result_from_payload(
            payload,
            section_label=section_label,
            paper_type=paper_type,
            used_context=used_context,
        )

    def _result_from_payload(
        self,
        payload: dict[str, Any],
        *,
        section_label: SectionLabel,
        paper_type: PaperType = "empirical_research",
        used_context: bool,
    ) -> RhetoricalMoveResult:
        if payload.get("action") == "request_context":
            return RhetoricalMoveResult(reason=str(payload.get("reason") or "context requested"))
        moves_payload = payload.get("moves")
        if not isinstance(moves_payload, list):
            raise RuntimeError("Rhetorical-move response must contain a moves list.")
        if len(moves_payload) > 3:
            raise RuntimeError("Rhetorical-move response contains more than three moves.")
        RhetoricalMoveEnricher.validate_section_for_paper_type(section_label, paper_type=paper_type)
        allowed = set(RhetoricalMoveEnricher.allowed_moves(section_label))
        moves: list[RhetoricalMoveClassification] = []
        for item in moves_payload:
            if not isinstance(item, dict):
                raise RuntimeError("Each rhetorical move must be a JSON object.")
            label = item.get("label")
            confidence = item.get("confidence")
            reason = str(item.get("reason") or "no reason provided").strip()
            if label not in allowed:
                raise RuntimeError(f"Unsupported rhetorical move for {section_label}: {label!r}")
            if confidence not in self.CONFIDENCE_LEVELS:
                raise RuntimeError(f"Unsupported rhetorical-move confidence: {confidence!r}")
            moves.append(RhetoricalMoveClassification(label=label, confidence=confidence, reason=reason))
        if len({move.label for move in moves}) != len(moves):
            raise RuntimeError("Rhetorical-move response contains duplicate labels.")
        return RhetoricalMoveResult(
            moves=moves,
            used_context=used_context,
            reason=str(payload.get("reason") or "no applicable rhetorical moves" if not moves else "classified rhetorical moves").strip(),
        )
