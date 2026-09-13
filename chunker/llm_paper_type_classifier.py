from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .llm_section_classifier import ChunkClassificationLLM
from .paper_type_classifier import (
    ClassificationConfidence,
    PAPER_TYPES,
    PaperType,
    PaperTypeClassification,
    PaperTypeEvidence,
)


class PaperTypeClassificationLLM(ChunkClassificationLLM):
    """Gemma-backed document-level classifier using compact paper evidence."""

    MODEL_MAX_TOKENS = 220
    JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
    SYSTEM_PROMPT = """You classify the genre of academic papers.

Return JSON only. Choose exactly one allowed paper type. Do not infer an empirical study merely because a paper discusses research. Use other_or_unclear when the supplied evidence does not justify a more specific type.
Confidence must be low, medium, or high.

JSON schema:
{"label":"literature_review","confidence":"high","reason":"short explanation"}
"""

    def __init__(
        self,
        *,
        model_path: str | Path,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        python_executable: str | Path | None = None,
        request_timeout_seconds: float = 180.0,
        event_logger: Callable[[str], None] | None = None,
    ) -> None:
        # The inherited runtime and JSON-safe subprocess worker do not require document anchors.
        super().__init__(
            filtered_markdown="",
            heading_splits=[],
            model_path=model_path,
            max_tokens=max_tokens or self.MODEL_MAX_TOKENS,
            temperature=temperature,
            python_executable=python_executable,
            request_timeout_seconds=request_timeout_seconds,
            event_logger=event_logger,
        )

    def classify(self, evidence: PaperTypeEvidence) -> PaperTypeClassification:
        prompt = self._prompt(evidence)
        try:
            self._log_event("sending paper-type classification request to Gemma 4.")
            response = self._generate([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            self._log_event(f"received paper-type response: {self._truncate_for_log(response)}")
            try:
                return self._result_from_response(response)
            except RuntimeError as exc:
                return self._correct_invalid_response(response, exc)
        except RuntimeError as exc:
            self._log_event(f"paper-type classification failed: {exc}")
            raise RuntimeError(f"Gemma failed to classify paper type: {exc}") from exc

    def _prompt(self, evidence: PaperTypeEvidence) -> str:
        return "\n".join([
            "Allowed paper types:",
            *[f"- {label}" for label in PAPER_TYPES],
            f"Title: {evidence.title or '[unavailable]'}",
            "Abstract:", evidence.abstract or "[unavailable]",
            "Headings:", "\n".join(f"- {heading}" for heading in evidence.headings) or "[none]",
            "Opening excerpt:", evidence.opening_excerpt or "[unavailable]",
            "Closing excerpt:", evidence.closing_excerpt or "[unavailable]",
        ])

    def _correct_invalid_response(self, response: str, failure: RuntimeError) -> PaperTypeClassification:
        self._log_event(f"invalid paper-type response; sending one correction prompt: {failure}")
        correction = "\n".join([
            "Your previous paper-type response was invalid.",
            "Allowed paper types:",
            *[f"- {label}" for label in PAPER_TYPES],
            "Return JSON only using the required label, confidence, and reason schema.",
            "Previous invalid response:", response,
        ])
        corrected = self._generate([
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": correction},
        ])
        self._log_event(f"received paper-type correction: {self._truncate_for_log(corrected)}")
        return self._result_from_response(corrected)

    def _result_from_response(self, response: str) -> PaperTypeClassification:
        payload = self._parse_payload(response)
        label = payload.get("label")
        confidence = payload.get("confidence")
        if label not in PAPER_TYPES:
            raise RuntimeError(f"Unsupported paper type: {label!r}")
        if confidence not in {"low", "medium", "high"}:
            raise RuntimeError(f"Unsupported paper-type confidence: {confidence!r}")
        return PaperTypeClassification(
            label=label,
            source="llm",
            confidence=confidence,
            reason=str(payload.get("reason") or "no reason provided").strip(),
        )

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
                raise RuntimeError(f"Failed to parse paper-type JSON: {raw_response}")
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse paper-type JSON: {raw_response}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Paper-type response must be a JSON object.")
        return payload
