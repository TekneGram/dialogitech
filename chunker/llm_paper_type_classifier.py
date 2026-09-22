from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .llm_worker import LLMWorker

from .llm_paper_type_classifier_helpers.evidence_builder import PaperTypeEvidenceBuilder
from .llm_paper_type_classifier_helpers.paper_type_models import (
    PAPER_TYPES,
    PaperTypeClassification,
    PaperTypeEvidence,
)


class PaperTypeClassificationLLM:
    """
      Gemma-backed document-level classifier using compact paper evidence.
    """

    MODEL_MAX_TOKENS = 220
    JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
    SYSTEM_PROMPT = """You classify the genre of academic papers.

      Return JSON only. Choose exactly one allowed paper type. Do not infer an empirical study merely because a paper discusses research.
      Use other_or_unclear when the supplied evidence does not justify a more specific type.
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
        paper_type_evidence_builder: PaperTypeEvidenceBuilder | None = None,
    ) -> None:
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive.")

        self.model_path = str(model_path)
        self.python_executable = str(
            python_executable or Path.home() / ".unsloth" / "unsloth_gemma4_mlx" / "bin" /"python"
        )
        self.max_tokens = max_tokens or self.MODEL_MAX_TOKENS
        self.temperature = temperature
        self.request_timeout_seconds = request_timeout_seconds
        self.event_logger = event_logger

        self.paper_type_evidence_builder = paper_type_evidence_builder or PaperTypeEvidenceBuilder()
        self._worker: LLMWorker | None = None

    def classify(self, filtered_markdown: str, *, metadata: dict[str, Any] | None = None) -> PaperTypeClassification:
        evidence = self.paper_type_evidence_builder.build(
            filtered_markdown=filtered_markdown,
            metadata=metadata,
        )
        return self._classify_evidence(evidence)

    def _classify_evidence(self, evidence: PaperTypeEvidence) -> PaperTypeClassification:
        prompt = self._prompt(evidence)

        try:
            self._log_event(
                "sending paper-type classification request to LLM"
            )
            response = self._generate([
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ])
            self._log_event(
                f"received paper-type response: "
                f"{self._truncate_for_log(response)}"
            )

            try:
                return self._result_from_response(response)
            except RuntimeError as exc:
                return self._correct_invalid_response(response, exc)

        except RuntimeError as exc:
            self._log_event(f"paper-type classification failed: {exc}")

            fallback_reason = (
                "LLM did not produce a valid paper-type classification; "
                f"assigned other_or_unclear. Original error: {exc}"
            )

            self._log_event(
                f"paper-type fallback applied: {fallback_reason}"
            )

            return PaperTypeClassification(
                label="other_or_unclear",
                source="llm",
                reason=fallback_reason,
                confidence="low",
            )

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

    def _log_event(self, message: str) -> None:
        if self.event_logger is not None:
            self.event_logger(message)

    def _truncate_for_log(self, response: str, max_chars: int = 240) -> str:
        collapsed = " ".join(response.split())
        if len(collapsed) <= max_chars:
            return collapsed
        return f"{collapsed[:max_chars - 3]}..."

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    # Make the class usable with Python's "with" statement
    # e.g., with PaperTypeClassificationLLM(...) as classifier
    #         result = classifier.classify(evidence)
    def __enter__(self) -> "PaperTypeClassificationLLM":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
