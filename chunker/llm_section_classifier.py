from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .markdown_section_chunker import HeadingSplit, SectionChunk

SectionLabel = Literal["abstract", "introduction", "method", "results", "discussion"]
ClassificationConfidence = Literal["low", "medium", "high"]
ArticleQuintile = Literal["first 20%", "second 20%", "third 20%", "fourth 20%", "last 20%"]
LLMAction = Literal["classify", "request_context"]


@dataclass(slots=True)
class ChunkClassification:
    label: SectionLabel | None
    source: Literal["llm"]
    reason: str
    confidence: ClassificationConfidence | None = None
    used_context: bool = False
    needs_llm: bool = False


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


def article_quintile(article_start: int, article_end: int, article_length: int) -> ArticleQuintile:
    if article_length <= 0:
        raise ValueError("article_length must be positive.")

    midpoint = (article_start + article_end) / 2
    ratio = midpoint / article_length

    if ratio < 0.2:
        return "first 20%"
    if ratio < 0.4:
        return "second 20%"
    if ratio < 0.6:
        return "third 20%"
    if ratio < 0.8:
        return "fourth 20%"
    return "last 20%"


class ChunkClassificationLLM:
    MODEL_MAX_TOKENS = 220
    SYSTEM_PROMPT = """You classify chunks from academic articles.

Allowed labels:
- abstract
- introduction
- method
- results
- discussion

Rules:
- Return JSON only.
- Use exactly one of the allowed labels when classifying.
- If the chunk is ambiguous or you are not confident enough, request context instead of guessing.
- After context is provided, make your best final classification from the allowed labels.
- Confidence must be one of: low, medium, high.

JSON schema:
{"action":"classify","label":"results","confidence":"medium","reason":"short explanation"}
or
{"action":"request_context","reason":"short explanation"}
"""

    JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
    LABELS: tuple[SectionLabel, ...] = ("abstract", "introduction", "method", "results", "discussion")
    CONFIDENCE_LEVELS: tuple[ClassificationConfidence, ...] = ("low", "medium", "high")

    def __init__(
        self,
        filtered_markdown: str,
        heading_splits: list[HeadingSplit],
        model_path: str | Path,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        python_executable: str | Path | None = None,
    ) -> None:
        self.filtered_markdown = filtered_markdown
        self.heading_splits = heading_splits
        self.model_path = str(model_path)
        self.max_tokens = max_tokens or self.MODEL_MAX_TOKENS
        self.temperature = temperature
        self.python_executable = str(python_executable) if python_executable is not None else None
        self._model: Any = None
        self._tokenizer: Any = None
        self._section_ranges = self._build_section_ranges(filtered_markdown, heading_splits)
        self._chunk_locations = self._build_chunk_locations(filtered_markdown, heading_splits, self._section_ranges)

    def classify(self, chunk: SectionChunk, heading_split: HeadingSplit) -> ChunkClassification:
        chunk_location = self._chunk_location(chunk, heading_split)
        initial_prompt = self._initial_user_prompt(chunk, heading_split, chunk_location)
        initial_response = self._generate(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": initial_prompt},
            ]
        )
        initial_decision = self._parse_decision(initial_response, allow_request_context=True)

        if initial_decision.action == "request_context":
            context = self._context_for_chunk(chunk, heading_split)
            context_prompt = self._context_user_prompt(chunk, heading_split, chunk_location, context)
            final_response = self._generate(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": initial_prompt},
                    {"role": "assistant", "content": initial_response},
                    {"role": "user", "content": context_prompt},
                ]
            )
            final_decision = self._parse_decision(final_response, allow_request_context=False)
            return ChunkClassification(
                label=final_decision.label,
                source="llm",
                reason=final_decision.reason,
                confidence=final_decision.confidence,
                used_context=True,
                needs_llm=False,
            )

        return ChunkClassification(
            label=initial_decision.label,
            source="llm",
            reason=initial_decision.reason,
            confidence=initial_decision.confidence,
            used_context=False,
            needs_llm=False,
        )

    def _initial_user_prompt(
        self,
        chunk: SectionChunk,
        heading_split: HeadingSplit,
        chunk_location: ChunkLocation,
    ) -> str:
        return "\n".join(
            [
                "This chunk is from an academic article.",
                f"Heading: {heading_split.title}",
                f"Article position: {chunk_location.quintile}",
                "Decide whether the chunk should be labeled as one of the allowed labels.",
                "If the chunk is ambiguous or you are not confident, return action=request_context.",
                "Chunk:",
                chunk.text,
            ]
        )

    def _context_user_prompt(
        self,
        chunk: SectionChunk,
        heading_split: HeadingSplit,
        chunk_location: ChunkLocation,
        context: ChunkContext,
    ) -> str:
        previous_section = context.previous_section or "[no previous heading section]"
        next_section = context.next_section or "[no next heading section]"
        return "\n".join(
            [
                "Context requested. You must now make a final classification.",
                f"Heading: {heading_split.title}",
                f"Article position: {chunk_location.quintile}",
                "Previous heading section:",
                previous_section,
                "Current chunk:",
                context.current_chunk,
                "Next heading section:",
                next_section,
                "Return JSON only with action=classify.",
            ]
        )

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        try:
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError(
                "mlx_lm is not installed. Install the requirements into .venv before using the LLM classifier."
            ) from exc

        self._model, self._tokenizer = load(self.model_path)
        return self._model, self._tokenizer

    def _generate(self, messages: list[dict[str, str]]) -> str:
        if self.python_executable is not None:
            return self._generate_via_subprocess(messages)

        model, tokenizer = self._load_model()

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except AttributeError:
            prompt = self._fallback_chat_prompt(messages)

        try:
            from mlx_lm import generate
        except ImportError as exc:
            raise RuntimeError(
                "mlx_lm is not installed. Install the requirements into .venv before using the LLM classifier."
            ) from exc

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            verbose=False,
        )
        return response.strip()

    def _generate_via_subprocess(self, messages: list[dict[str, str]]) -> str:
        runner_path = Path(__file__).with_name("mlx_llm_runner.py")
        payload = {
            "model_path": self.model_path,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        completed = subprocess.run(
            [self.python_executable, str(runner_path)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(
                f"External MLX runner failed with exit code {completed.returncode}: {stderr or 'no stderr output'}"
            )

        response = completed.stdout.strip()
        if not response:
            raise RuntimeError("External MLX runner returned empty output.")
        return response

    def _fallback_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        rendered: list[str] = []
        for message in messages:
            rendered.append(f"{message['role'].upper()}:\n{message['content']}")
        rendered.append("ASSISTANT:")
        return "\n\n".join(rendered)

    def _parse_decision(self, raw_response: str, allow_request_context: bool) -> LLMDecision:
        try:
            payload = self._parse_json_object(raw_response)
        except ValueError as exc:
            raise RuntimeError(f"Failed to parse model response as JSON: {raw_response}") from exc

        action = self._normalize_optional_string(payload.get("action"))
        label = self._normalize_optional_string(payload.get("label"))
        confidence = self._normalize_optional_string(payload.get("confidence"))

        if action in self.LABELS and label is None:
            label = action
            action = "classify"

        if action not in {"classify", "request_context"}:
            raise RuntimeError(f"Model returned unsupported action: {action!r}")

        reason = str(payload.get("reason", "")).strip() or "no reason provided"

        if action == "request_context":
            if not allow_request_context:
                raise RuntimeError("Model requested context after the single allowed retrieval round.")
            return LLMDecision(action="request_context", reason=reason)

        if label not in self.LABELS:
            raise RuntimeError(f"Model returned unsupported label: {label!r}")

        if confidence not in self.CONFIDENCE_LEVELS:
            raise RuntimeError(f"Model returned unsupported confidence: {confidence!r}")

        return LLMDecision(action="classify", label=label, confidence=confidence, reason=reason)

    def _parse_json_object(self, raw_response: str) -> dict[str, Any]:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if "\n" in cleaned:
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = self.JSON_PATTERN.search(raw_response)
            if not match:
                raise ValueError("No JSON object found in model response.")
            return json.loads(match.group(0))

    def _normalize_optional_string(self, value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return normalized or None

    def _build_section_ranges(
        self,
        markdown: str,
        heading_splits: list[HeadingSplit],
    ) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        search_start = 0

        for index, heading_split in enumerate(heading_splits):
            heading_start = markdown.find(heading_split.raw_heading, search_start)
            if heading_start < 0:
                raise RuntimeError(f"Could not locate heading {heading_split.raw_heading!r} in filtered markdown.")

            if index + 1 < len(heading_splits):
                next_heading = heading_splits[index + 1].raw_heading
                section_end = markdown.find(next_heading, heading_start + len(heading_split.raw_heading))
                if section_end < 0:
                    raise RuntimeError(f"Could not locate next heading {next_heading!r} in filtered markdown.")
            else:
                section_end = len(markdown)

            ranges.append((heading_start, section_end))
            search_start = section_end

        return ranges

    def _build_chunk_locations(
        self,
        markdown: str,
        heading_splits: list[HeadingSplit],
        section_ranges: list[tuple[int, int]],
    ) -> dict[tuple[str, int], ChunkLocation]:
        locations: dict[tuple[str, int], ChunkLocation] = {}

        for section_index, heading_split in enumerate(heading_splits):
            section_start, section_end = section_ranges[section_index]
            section_text = markdown[section_start:section_end]
            section_search_start = 0

            for chunk in heading_split.chunks:
                local_start = section_text.find(chunk.text, section_search_start)
                if local_start < 0:
                    raise RuntimeError(
                        f"Could not anchor chunk {chunk.chunk_index} under heading {heading_split.title!r} in filtered markdown."
                    )

                local_end = local_start + len(chunk.text)
                article_start = section_start + local_start
                article_end = section_start + local_end
                section_search_start = local_start

                locations[(heading_split.title, chunk.chunk_index)] = ChunkLocation(
                    article_start=article_start,
                    article_end=article_end,
                    quintile=article_quintile(article_start, article_end, len(markdown)),
                    section_index=section_index,
                )

        return locations

    def _chunk_location(self, chunk: SectionChunk, heading_split: HeadingSplit) -> ChunkLocation:
        key = (heading_split.title, chunk.chunk_index)
        try:
            return self._chunk_locations[key]
        except KeyError as exc:
            raise RuntimeError(
                f"Missing chunk location for heading {heading_split.title!r}, chunk {chunk.chunk_index}."
            ) from exc

    def _context_for_chunk(self, chunk: SectionChunk, heading_split: HeadingSplit) -> ChunkContext:
        location = self._chunk_location(chunk, heading_split)
        previous_section = None
        next_section = None

        if location.section_index > 0:
            prev_start, prev_end = self._section_ranges[location.section_index - 1]
            previous_section = self.filtered_markdown[prev_start:prev_end].strip()

        if location.section_index + 1 < len(self._section_ranges):
            next_start, next_end = self._section_ranges[location.section_index + 1]
            next_section = self.filtered_markdown[next_start:next_end].strip()

        return ChunkContext(
            previous_section=previous_section,
            current_chunk=chunk.text,
            next_section=next_section,
        )
