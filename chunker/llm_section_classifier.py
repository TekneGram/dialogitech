from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .llm_worker import LLMWorker
from .markdown_section_chunker import HeadingSplit, SectionChunk
from .llm_paper_type_classifier_helpers.paper_type_models import PaperType
from .llm_section_type_classifier_helpers.location_builder import SectionLocationBuilder
from .llm_section_type_classifier_helpers.response_parser import (
    SectionClassificationResponseParser,
)
from .llm_section_type_classifier_helpers.section_type_models import (
    ChunkClassification,
    ChunkContext,
    ChunkLocation,
    ClassificationConfidence,
)
from .llm_section_type_classifier_helpers.section_taxonomy import (
    SECTION_LABEL_DESCRIPTIONS,
    SectionLabel,
    allowed_sections,
)


class SectionClassificationLLM:
    MODEL_MAX_TOKENS = 220
    SYSTEM_PROMPT = """You classify chunks from academic papers into the allowed section labels supplied by the user.

      Rules:
      - Return JSON only.
      - Use exactly one of the allowed labels when classifying.
      - If the chunk is ambiguous or you are not confident enough, request context instead of guessing.
      - Confidence must be one of: low, medium, high.

      JSON schema:
      {"action":"classify","label":"results","confidence":"medium","reason":"short explanation"}
      or
      {"action":"request_context","reason":"short explanation"}
    """

    CONFIDENCE_LEVELS: tuple[ClassificationConfidence, ...] = ("low", "medium", "high")

    def __init__(
        self,
        filtered_markdown: str,
        heading_splits: list[HeadingSplit],
        model_path: str | Path,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        python_executable: str | Path | None = None,
        request_timeout_seconds: float = 180.0,
        event_logger: Callable[[str], None] | None = None,
    ) -> None:
        self.filtered_markdown = filtered_markdown
        self.heading_splits = heading_splits
        self.model_path = str(model_path)
        self.max_tokens = max_tokens or self.MODEL_MAX_TOKENS
        self.temperature = temperature
        self.python_executable = str(
            python_executable or Path.home() / ".unsloth" / "unsloth_gemma4_mlx" / "bin" / "python"
        )
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive.")
        self.request_timeout_seconds = request_timeout_seconds
        self.event_logger = event_logger
        self._worker: LLMWorker | None = None
        self._location_builder = SectionLocationBuilder(
            filtered_markdown=filtered_markdown,
            heading_splits=heading_splits,
            event_logger=self._log_event,
        )
        self._response_parser = SectionClassificationResponseParser(
            confidence_levels=self.CONFIDENCE_LEVELS,
        )

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def __enter__(self) -> "SectionClassificationLLM":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def classify_section_chunk(
        self,
        *,
        section_chunk: SectionChunk,
        heading_split: HeadingSplit,
        paper_type: PaperType,
        previous_label: SectionLabel | None = None,
    ) -> ChunkClassification:
        chunk_location = self._chunk_location(section_chunk, heading_split)
        chunk_ref = self._chunk_ref(chunk=section_chunk, heading_split=heading_split)
        allowed_labels = allowed_sections(paper_type)
        failure_reasons: list[str] = []
        initial_prompt = self._initial_user_prompt(
            section_chunk,
            heading_split,
            chunk_location,
            paper_type,
            allowed_labels,
        )
        used_context = False

        try:
            self._log_event(
                f"{chunk_ref} sending initial classification request to LLM "
                f"(position={chunk_location.quintile})."
            )
            initial_response = self._generate(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": initial_prompt},
                ]
            )
            self._log_event(
                f"{chunk_ref} received initial response: {self._truncate_for_log(initial_response)}"
            )
            initial_decision = self._response_parser.parse_decision(
                initial_response,
                allow_request_context=True,
                allowed_labels=allowed_labels,
            )
            if initial_decision.action == "classify":
                return self._classification_from_decision(initial_decision, used_context=False)
        except Exception as exc:
            failure_reasons.append(self._record_stage_failure(chunk_ref, "initial", exc))

        used_context = True
        try:
            context = self._context_for_chunk(section_chunk, heading_split)
            second_prompt = self._second_prompt(
                heading_split,
                chunk_location,
                context,
                paper_type,
                allowed_labels,
            )
            self._log_event(f"{chunk_ref} moving to second classification request.")
            second_response = self._generate(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": second_prompt},
                ]
            )
            self._log_event(
                f"{chunk_ref} received second response: "
                f"{self._truncate_for_log(second_response)}"
            )
            second_decision = self._response_parser.parse_decision(
                second_response,
                allow_request_context=True,
                allowed_labels=allowed_labels,
            )
            if second_decision.action == "classify":
                return self._classification_from_decision(second_decision, used_context=True)
        except Exception as exc:
            failure_reasons.append(self._record_stage_failure(chunk_ref, "second", exc))

        try:
            final_system_prompt = self._final_system_prompt(previous_label)
            final_user_prompt = self._final_user_prompt(
                chunk=section_chunk,
                heading_split=heading_split,
                chunk_location=chunk_location,
                paper_type=paper_type,
                allowed_labels=allowed_labels,
            )
            self._log_event(f"{chunk_ref} moving to final classification request.")
            final_response = self._generate(
                messages=[
                    {"role": "system", "content": final_system_prompt},
                    {"role": "user", "content": final_user_prompt},
                ]
            )
            self._log_event(
                f"{chunk_ref} received final response: "
                f"{self._truncate_for_log(final_response)}"
            )
            final_decision = self._response_parser.parse_decision(
                final_response,
                allow_request_context=False,
                allowed_labels=allowed_labels,
            )
            return self._classification_from_decision(final_decision, used_context=True)
        except Exception as exc:
            failure_reasons.append(self._record_stage_failure(chunk_ref, "final", exc))

        return self._fallback_to_unclassified(
            chunk_ref=chunk_ref,
            used_context=used_context,
            failure_reasons=failure_reasons,
        )

    def _classification_from_decision(self, decision: Any, *, used_context: bool) -> ChunkClassification:
        return ChunkClassification(
            label=decision.label,
            source="llm",
            reason=decision.reason,
            confidence=decision.confidence,
            used_context=used_context,
        )

    def _record_stage_failure(self, chunk_ref: str, stage: str, exc: Exception) -> str:
        failure = f"{stage} classification failed with {type(exc).__name__}: {exc}"
        self._log_event(f"{chunk_ref} {failure}")
        return failure

    def _initial_user_prompt(
        self,
        chunk: SectionChunk,
        heading_split: HeadingSplit,
        chunk_location: ChunkLocation,
        paper_type: PaperType,
        allowed_labels: tuple[SectionLabel, ...],
    ) -> str:
        return "\n".join(
            [
                "This chunk is from an academic article.",
                f"Paper type: {paper_type}",
                "Allowed section labels:",
                *[f"- {label}: {SECTION_LABEL_DESCRIPTIONS[label]}" for label in allowed_labels],
                f"Heading: {heading_split.title}",
                f"It's position in the article is: {chunk_location.quintile}",
                "Decide whether the chunk should be labeled as one of the allowed labels.",
                "If the chunk is ambiguous or you are not confident, return action=request_context.",
                "Chunk:",
                chunk.text,
            ]
        )

    def _second_prompt(
        self,
        heading_split: HeadingSplit,
        chunk_location: ChunkLocation,
        context: ChunkContext,
        paper_type: PaperType,
        allowed_labels: tuple[SectionLabel, ...],
    ) -> str:
        previous_section = context.previous_section or ""
        next_section = context.next_section or ""
        return "\n".join(
            [
                "This chunk is from an academic article",
                f"Paper type: {paper_type}",
                "Allowed section labels:",
                *[f"- {label}: {SECTION_LABEL_DESCRIPTIONS[label]}" for label in allowed_labels],
                f"Heading: {heading_split.title}",
                f"It's position in the article is: {chunk_location.quintile}",
                "Decide whether the chunk should be labeled as one of the allowed labels."
                "Chunk:",
                previous_section,
                context.current_chunk,
                next_section,
                "Return JSON only with action=classify. Do not request more context.",
            ]
        )

    def _final_system_prompt(self, previous_label: SectionLabel | None) -> str:
        if previous_label is None:
            return """You classify chunks from academic papers into the allowed section labels supplied by the user. You must do your best to classify this chunk using its position information within the academic article as your way to make a good guess.

              Rules:
              - Return JSON only.
              - Use exactly one of the allowed labels when classifying.
              - You *must* provide a classification.
              - Confidence must be one of: low, medium, high.

              JSON schema:
              {{"action":"classify","label":"results","confidence":"medium","reason":"short explanation"}}"""

        return f"""You classify chunks from academic papers into the allowed section labels supplied by the user. A nearby chunk was classified as {previous_label}

          Rules:
          - Return JSON only.
          - Use exactly one of the allowed labels when classifying.
          - You *must* provide a classification.
          - Confidence must be one of: low, medium, high.

          JSON schema:
          {{"action":"classify","label":"results","confidence":"medium","reason":"short explanation"}}"""

    def _final_user_prompt(
        self,
        *,
        chunk: SectionChunk,
        heading_split: HeadingSplit,
        chunk_location: ChunkLocation,
        paper_type: PaperType,
        allowed_labels: tuple[SectionLabel, ...],
    ) -> str:
        return "\n".join(
            [
                f"Paper type: {paper_type}",
                "Allowed section labels:",
                *[f"- {label}: {SECTION_LABEL_DESCRIPTIONS[label]}" for label in allowed_labels],
                f"Heading: {heading_split.title}",
                f"Article position: {chunk_location.quintile}",
                "Chunk:",
                chunk.text,
            ]
        )

    def _generate(self, messages: list[dict[str, str]]) -> str:
        if self._worker is None:
            runner_path = Path(__file__).with_name("mlx_llm_runner.py")
            self._worker = LLMWorker(
                python_executable=self.python_executable,
                runner_path=runner_path,
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

    def _fallback_to_unclassified(
        self,
        *,
        chunk_ref: str,
        used_context: bool,
        failure_reasons: list[str],
    ) -> ChunkClassification:
        fallback_reason = "LLM classification failed; assigned fallback label 'unclassified'. " + " | ".join(
            failure_reasons
        )
        self._log_event(
            f"{chunk_ref} LLM failed at every classification stage; using fallback label "
            f"'unclassified'. Failures: {' | '.join(failure_reasons)}"
        )
        return ChunkClassification(
            label="unclassified",
            source="llm_fallback",
            reason=fallback_reason,
            confidence="low",
            used_context=used_context,
        )

    def _chunk_location(self, chunk: SectionChunk, heading_split: HeadingSplit) -> ChunkLocation:
        return self._location_builder.location_for(chunk=chunk, heading_split=heading_split)

    def _context_for_chunk(self, chunk: SectionChunk, heading_split: HeadingSplit) -> ChunkContext:
        return self._location_builder.context_for(chunk=chunk, heading_split=heading_split)
