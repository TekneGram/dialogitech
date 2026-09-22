from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .llm_rhetorical_move_classifier_helpers import (
    ClassificationConfidence,
    RhetoricalMoveClassification,
    RhetoricalMoveResult,
    allowed_moves,
    validate_rhetorical_move_result,
)
from .llm_section_type_classifier_helpers.section_type_models import ClassificationLabel
from .llm_section_type_classifier_helpers.location_builder import SectionLocationBuilder
from .llm_worker import LLMWorker
from .markdown_section_chunker import HeadingSplit, SectionChunk


class RhetoricalMoveClassificationLLM:
    """Gemma-backed rhetorical-move classifier for one section chunk."""

    MODEL_MAX_TOKENS = 360
    JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
    CONFIDENCE_LEVELS: tuple[ClassificationConfidence, ...] = ("low", "medium", "high")
    SYSTEM_PROMPT = """You identify rhetorical moves in chunks from academic articles.

      Return JSON only. Select one to three move labels from the allowed labels supplied by the user.
      Order moves from primary to secondary. Do not invent labels.
      Every move needs a confidence of low, medium, or high and a short reason.

      JSON schema:
      {"moves":[{"label":"present_results","confidence":"high","reason":"reports a measured outcome"}],"reason":"short overall explanation"}
    """

    def __init__(
        self,
        *,
        filtered_markdown: str,
        heading_splits: list[HeadingSplit],
        model_path: str | Path,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        python_executable: str | Path | None = None,
        request_timeout_seconds: float = 180.0,
        event_logger: Callable[[str], None] | None = None,
    ) -> None:
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive.")
        self.filtered_markdown = filtered_markdown
        self.heading_splits = heading_splits
        self.model_path = str(model_path)
        self.max_tokens = max_tokens or self.MODEL_MAX_TOKENS
        self.temperature = temperature
        self.python_executable = str(
            python_executable or Path.home() / ".unsloth" / "unsloth_gemma4_mlx" / "bin" / "python"
        )
        self.request_timeout_seconds = request_timeout_seconds
        self.event_logger = event_logger
        self._worker: LLMWorker | None = None
        self._location_builder = SectionLocationBuilder(
            filtered_markdown=filtered_markdown,
            heading_splits=heading_splits,
            event_logger=self._log_event,
        )

    def classify(
        self,
        *,
        chunk: SectionChunk,
        heading_split: HeadingSplit,
        section_label: ClassificationLabel,
    ) -> RhetoricalMoveResult:
        allowed = allowed_moves(section_label)
        location = self._chunk_location(chunk, heading_split)
        chunk_ref = self._chunk_ref(chunk=chunk, heading_split=heading_split)
        initial_prompt = self._initial_prompt(
            chunk, heading_split, section_label, location, allowed
        )
        initial_failure: Exception | None = None
        try:
            self._log_event(f"{chunk_ref} sending rhetorical-move request (section={section_label}).")
            response = self._generate([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": initial_prompt},
            ])
            self._log_event(
                f"{chunk_ref} received rhetorical-move response: {self._truncate_for_log(response)}"
            )
            payload = self._parse_payload(response)
            if payload.get("action") != "request_context":
                return self._result_from_payload(
                    payload,
                    section_label=section_label,
                    used_context=False,
                )
            self._log_event(f"{chunk_ref} requested rhetorical-move context.")
        except Exception as exc:
            initial_failure = exc
            self._log_event(
                f"{chunk_ref} initial rhetorical-move request failed; moving to context prompt: "
                f"{type(exc).__name__}: {exc}"
            )

        used_context = True
        try:
            context = self._context_for_chunk(chunk, heading_split)
            context_prompt = self._second_prompt(
                chunk, heading_split, section_label, location, allowed, context
            )
            self._log_event(f"{chunk_ref} sending rhetorical-move context request.")
            response = self._generate([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": context_prompt},
            ])
            self._log_event(
                f"{chunk_ref} received rhetorical-move context response: "
                f"{self._truncate_for_log(response)}"
            )
            payload = self._parse_payload(response)
            if payload.get("action") == "request_context":
                raise RuntimeError(
                    "Model requested rhetorical-move context after the single allowed context round."
                )
            return self._result_from_payload(
                payload,
                section_label=section_label,
                used_context=used_context,
            )
        except Exception as exc:
            initial_detail = f" Initial failure: {initial_failure}." if initial_failure else ""
            reason = (
                "Rhetorical-move classification produced no moves after the context attempt."
                f"{initial_detail} Context failure: {type(exc).__name__}: {exc}"
            )
            self._log_event(f"{chunk_ref} {reason}")
            return RhetoricalMoveResult(
                moves=[
                    RhetoricalMoveClassification(
                        label="unclassified",
                        confidence="low",
                        reason=reason,
                    )
                ],
                used_context=True,
                reason=reason,
            )

    def _initial_prompt(
        self,
        chunk: SectionChunk,
        heading_split: HeadingSplit,
        section_label: ClassificationLabel,
        location: Any,
        allowed: tuple[str, ...],
    ) -> str:
        section_description = section_label if section_label != "unclassified" else "[unavailable]"
        return "\n".join([
            "This following chunk is classified as followed.",
            f"Section classification: {section_description}",
            f"Heading: {heading_split.title}",
            f"Position with the academic article: {location.quintile}",
            "The allowed rhetorical moves are:",
            *[f"- {label}" for label in allowed],
            "Return one to three applicable moves from that exact list.",
            "If necessary context is missing, return only {\"action\":\"request_context\",\"reason\":\"...\"}.",
            "Chunk:", chunk.text,
        ])

    def _second_prompt(
        self,
        chunk: SectionChunk,
        heading_split: HeadingSplit,
        section_label: ClassificationLabel,
        location: Any,
        allowed: tuple[str, ...],
        context: Any,
    ) -> str:
        section_description = section_label if section_label != "unclassified" else "[unavailable]"
        return "\n".join([
            "Here is the information about the chunk that you must classified and its surrounding chunks",
            f"Section classification: {section_description}",
            f"Heading: {heading_split.title}",
            f"Position with the academic article: {location.quintile}",
            "You are allowed to use only these rhetorical moves:",
            *[f"- {label}" for label in allowed],
            "The previous chunk was this:", context.previous_section or "[none]",
            "This is the current chunk that you must classify:", context.current_chunk,
            "The chunk that follows it is this:", context.next_section or "[none]",
            "Classify only the chunk that you must classify."
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

    def _result_from_payload(
        self,
        payload: dict[str, Any],
        *,
        section_label: ClassificationLabel,
        used_context: bool,
    ) -> RhetoricalMoveResult:
        moves_payload = payload.get("moves")
        if not isinstance(moves_payload, list):
            raise RuntimeError("Rhetorical-move response must contain a moves list.")
        moves: list[RhetoricalMoveClassification] = []
        for item in moves_payload:
            if not isinstance(item, dict):
                raise RuntimeError("Each rhetorical move must be a JSON object.")
            label = item.get("label")
            confidence = item.get("confidence")
            reason = str(item.get("reason") or "no reason provided").strip()
            if label not in allowed_moves(section_label):
                raise RuntimeError(f"Unsupported rhetorical move for {section_label}: {label!r}")
            if confidence not in self.CONFIDENCE_LEVELS:
                raise RuntimeError(f"Unsupported rhetorical-move confidence: {confidence!r}")
            moves.append(RhetoricalMoveClassification(label=label, confidence=confidence, reason=reason))
        if not moves:
            raise RuntimeError("Rhetorical-move response contained no classifications.")
        result = RhetoricalMoveResult(
            moves=moves,
            used_context=used_context,
            reason=str(payload.get("reason") or "no applicable rhetorical moves" if not moves else "classified rhetorical moves").strip(),
        )
        validate_rhetorical_move_result(result, section_label=section_label)
        return result

    def _generate(self, messages: list[dict[str, str]]) -> str:
        if self._worker is None:
            self._worker = LLMWorker(
                python_executable=self.python_executable,
                runner_path=Path(__file__).with_name("mlx_llm_runner.py"),
                model_path=self.model_path,
                request_timeout_seconds=self.request_timeout_seconds,
                event_logger=self._log_event,
            )
        try:
            return self._worker.generate(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except RuntimeError:
            self._worker.close()
            self._worker = None
            raise

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def __enter__(self) -> "RhetoricalMoveClassificationLLM":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _chunk_location(self, chunk: SectionChunk, heading_split: HeadingSplit) -> Any:
        return self._location_builder.location_for(chunk=chunk, heading_split=heading_split)

    def _context_for_chunk(self, chunk: SectionChunk, heading_split: HeadingSplit) -> Any:
        return self._location_builder.context_for(chunk=chunk, heading_split=heading_split)

    def _chunk_ref(self, *, chunk: SectionChunk, heading_split: HeadingSplit) -> str:
        return f"[heading={heading_split.title!r} chunk={chunk.chunk_index}]"

    def _log_event(self, message: str) -> None:
        if self.event_logger is not None:
            self.event_logger(message)

    def _truncate_for_log(self, response: str, max_chars: int = 240) -> str:
        collapsed = " ".join(response.split())
        if len(collapsed) <= max_chars:
            return collapsed
        return f"{collapsed[: max_chars - 3]}..."
