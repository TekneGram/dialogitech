from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from .markdown_section_chunker import HeadingSplit, SectionChunk
from .paper_type_classifier import PaperType
from .section_taxonomy import SECTION_LABEL_DESCRIPTIONS, SectionLabel, allowed_sections

ClassificationConfidence = Literal["low", "medium", "high"]
ArticleQuintile = Literal["first 20%", "second 20%", "third 20%", "fourth 20%", "last 20%"]
LLMAction = Literal["classify", "request_context"]


class _GemmaSubprocessWorker:
    def __init__(
        self,
        *,
        python_executable: str,
        runner_path: Path,
        model_path: str,
        request_timeout_seconds: float,
        event_logger: Callable[[str], None],
    ) -> None:
        self.request_timeout_seconds = request_timeout_seconds
        self.event_logger = event_logger
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self.process = subprocess.Popen(
            [python_executable, str(runner_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(self.process.stdout,),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(self.process.stderr,),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._send_payload(
            {
                "model_path": model_path,
                "messages": [],
                "max_tokens": 1,
                "temperature": 0.0,
            },
            initialization=True,
        )

    def generate(self, *, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
        self._send_payload(
            {
                "model_path": "",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        try:
            response_line = self._responses.get(timeout=self.request_timeout_seconds)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError(
                f"Gemma request timed out after {self.request_timeout_seconds:g} seconds. "
                f"Recent stderr: {self._recent_stderr()}"
            ) from exc

        if response_line is None:
            returncode = self.process.poll()
            self.close()
            raise RuntimeError(
                f"External MLX runner exited before responding (exit code {returncode}). "
                f"Recent stderr: {self._recent_stderr()}"
            )

        try:
            payload = json.loads(response_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"External MLX runner returned invalid JSON: {response_line!r}") from exc
        if payload.get("error"):
            raise RuntimeError(f"External MLX runner error: {payload['error']}")
        response = payload.get("response")
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError("External MLX runner returned an empty response.")
        return response.strip()

    def _send_payload(self, payload: dict[str, object], *, initialization: bool = False) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"External MLX runner exited with code {self.process.returncode}. "
                f"Recent stderr: {self._recent_stderr()}"
            )
        assert self.process.stdin is not None
        if initialization:
            # The first request makes the worker load the model. It intentionally
            # performs no generation; the worker consumes it before normal calls.
            payload["initialize_only"] = True
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

    def _read_stdout(self, stream: Any) -> None:
        for line in stream:
            self._responses.put(line.rstrip("\n"))
        self._responses.put(None)

    def _read_stderr(self, stream: Any) -> None:
        for line in stream:
            clean_line = line.rstrip("\n")
            with self._stderr_lock:
                self._stderr_lines.append(clean_line)
                del self._stderr_lines[:-20]
            self.event_logger(f"Gemma worker: {clean_line}")

    def _recent_stderr(self) -> str:
        with self._stderr_lock:
            return " | ".join(self._stderr_lines[-5:]) or "no stderr output"

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process.stdin is not None:
            self.process.stdin.close()


@dataclass(slots=True)
class ChunkClassification:
    label: SectionLabel | None
    source: Literal["llm", "deterministic"]
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
    SYSTEM_PROMPT = """You classify chunks from academic papers into the allowed section labels supplied by the user.

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
        self.python_executable = str(python_executable) if python_executable is not None else None
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive.")
        self.request_timeout_seconds = request_timeout_seconds
        self.event_logger = event_logger
        self._model: Any = None
        self._tokenizer: Any = None
        self._worker: _GemmaSubprocessWorker | None = None
        self._section_ranges = self._build_section_ranges(filtered_markdown, heading_splits)
        self._chunk_locations = self._build_chunk_locations(filtered_markdown, heading_splits, self._section_ranges)

    def classify(
        self,
        chunk: SectionChunk,
        heading_split: HeadingSplit,
        paper_type: PaperType = "empirical_research",
    ) -> ChunkClassification:
        return self.classify_with_previous_label(
            chunk=chunk,
            heading_split=heading_split,
            previous_label=None,
            paper_type=paper_type,
        )

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def __enter__(self) -> "ChunkClassificationLLM":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def classify_with_previous_label(
        self,
        *,
        chunk: SectionChunk,
        heading_split: HeadingSplit,
        previous_label: SectionLabel | None,
        paper_type: PaperType = "empirical_research",
    ) -> ChunkClassification:
        chunk_location = self._chunk_location(chunk, heading_split)
        chunk_ref = self._chunk_ref(chunk=chunk, heading_split=heading_split)
        allowed_labels = allowed_sections(paper_type)
        initial_prompt = self._initial_user_prompt(
            chunk,
            heading_split,
            chunk_location,
            paper_type,
            allowed_labels,
        )
        used_context = False

        try:
            self._log_event(
                f"{chunk_ref} sending initial classification request to Gemma 4 "
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
            initial_decision = self._parse_decision(
                initial_response,
                allow_request_context=True,
                allowed_labels=allowed_labels,
            )

            if initial_decision.action == "request_context":
                used_context = True
                context = self._context_for_chunk(chunk, heading_split)
                context_prompt = self._context_user_prompt(
                    chunk,
                    heading_split,
                    chunk_location,
                    context,
                    paper_type,
                    allowed_labels,
                )
                self._log_event(f"{chunk_ref} requested context; sending one context round.")
                final_response = self._generate(
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": initial_prompt},
                        {"role": "assistant", "content": initial_response},
                        {"role": "user", "content": context_prompt},
                    ]
                )
                self._log_event(
                    f"{chunk_ref} received final response after context: "
                    f"{self._truncate_for_log(final_response)}"
                )
                final_decision = self._parse_decision(
                    final_response,
                    allow_request_context=True,
                    allowed_labels=allowed_labels,
                )
                if final_decision.action == "request_context":
                    if previous_label is None:
                        raise RuntimeError("Model requested context after the single allowed retrieval round.")
                    confirmation_prompt = self._previous_label_confirmation_prompt(
                        chunk=chunk,
                        heading_split=heading_split,
                        chunk_location=chunk_location,
                        previous_label=previous_label,
                        paper_type=paper_type,
                        allowed_labels=allowed_labels,
                    )
                    self._log_event(
                        f"{chunk_ref} requested context again; sending previous-label confirmation "
                        f"prompt with prior label={previous_label}."
                    )
                    confirmation_response = self._generate(
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": initial_prompt},
                            {"role": "assistant", "content": initial_response},
                            {"role": "user", "content": context_prompt},
                            {"role": "assistant", "content": final_response},
                            {"role": "user", "content": confirmation_prompt},
                        ]
                    )
                    self._log_event(
                        f"{chunk_ref} received previous-label confirmation response: "
                        f"{self._truncate_for_log(confirmation_response)}"
                    )
                    final_decision = self._parse_decision(
                        confirmation_response,
                        allow_request_context=False,
                        allowed_labels=allowed_labels,
                    )
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
        except RuntimeError as exc:
            return self._fallback_to_previous_label(
                chunk_ref=chunk_ref,
                previous_label=previous_label,
                used_context=used_context,
                failure_reason=str(exc),
            )

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
        paper_type: PaperType,
        allowed_labels: tuple[SectionLabel, ...],
    ) -> str:
        previous_section = context.previous_section or "[no previous heading section]"
        next_section = context.next_section or "[no next heading section]"
        return "\n".join(
            [
                "Context requested. You must now make a final classification.",
                f"Paper type: {paper_type}",
                "Allowed section labels:",
                *[f"- {label}: {SECTION_LABEL_DESCRIPTIONS[label]}" for label in allowed_labels],
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

    def _previous_label_confirmation_prompt(
        self,
        *,
        chunk: SectionChunk,
        heading_split: HeadingSplit,
        chunk_location: ChunkLocation,
        previous_label: SectionLabel,
        paper_type: PaperType,
        allowed_labels: tuple[SectionLabel, ...],
    ) -> str:
        return "\n".join(
            [
                "You must now return a final classification.",
                f"Paper type: {paper_type}",
                "Allowed section labels:",
                *[f"- {label}: {SECTION_LABEL_DESCRIPTIONS[label]}" for label in allowed_labels],
                f"Heading: {heading_split.title}",
                f"Article position: {chunk_location.quintile}",
                f"The previous chunk was classified as: {previous_label}",
                "This chunk is very likely to have the same label unless the content clearly indicates otherwise.",
                "Confirm the same label or override it with a different allowed label.",
                "Do not request more context.",
                "Return JSON only with action=classify.",
                "Chunk:",
                chunk.text,
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
        if self._worker is None:
            runner_path = Path(__file__).with_name("mlx_llm_runner.py")
            self._worker = _GemmaSubprocessWorker(
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

    def _fallback_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        rendered: list[str] = []
        for message in messages:
            rendered.append(f"{message['role'].upper()}:\n{message['content']}")
        rendered.append("ASSISTANT:")
        return "\n\n".join(rendered)

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

    def _fallback_to_previous_label(
        self,
        *,
        chunk_ref: str,
        previous_label: SectionLabel | None,
        used_context: bool,
        failure_reason: str,
    ) -> ChunkClassification:
        if previous_label is None:
            raise RuntimeError(failure_reason)

        fallback_reason = (
            f"LLM failed to provide a concrete label; inherited previous resolved label "
            f"'{previous_label}'. Gemma failure: {failure_reason}"
        )
        self._log_event(
            f"{chunk_ref} Gemma failed to classify chunk; using deterministic fallback "
            f"with previous label={previous_label}. Failure: {failure_reason}"
        )
        return ChunkClassification(
            label=previous_label,
            source="deterministic",
            reason=fallback_reason,
            confidence="low",
            used_context=used_context,
            needs_llm=False,
        )

    def _parse_decision(
        self,
        raw_response: str,
        allow_request_context: bool,
        allowed_labels: tuple[SectionLabel, ...],
    ) -> LLMDecision:
        try:
            payload = self._parse_json_object(raw_response)
        except ValueError as exc:
            raise RuntimeError(f"Failed to parse model response as JSON: {raw_response}") from exc

        action = self._normalize_optional_string(payload.get("action"))
        label = self._normalize_optional_string(payload.get("label"))
        confidence = self._normalize_optional_string(payload.get("confidence"))

        if action in allowed_labels:
            if label is None:
                label = action
            action = "classify"

        if action not in {"classify", "request_context"}:
            raise RuntimeError(f"Model returned unsupported action: {action!r}")

        reason = str(payload.get("reason", "")).strip() or "no reason provided"

        if action == "request_context":
            if not allow_request_context:
                raise RuntimeError("Model requested context after the single allowed retrieval round.")
            return LLMDecision(action="request_context", reason=reason)

        if label not in allowed_labels:
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
                return self._recover_json_like_payload(raw_response)
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return self._recover_json_like_payload(match.group(0))

    def _recover_json_like_payload(self, raw_response: str) -> dict[str, Any]:
        action = self._extract_json_string_field(raw_response, "action")
        reason = self._extract_json_string_field(raw_response, "reason")
        label = self._extract_json_string_field(raw_response, "label")
        confidence = self._extract_json_string_field(raw_response, "confidence")

        if action is None:
            raise ValueError("Could not recover action field from model response.")

        payload: dict[str, Any] = {"action": action}
        if label is not None:
            payload["label"] = label
        if confidence is not None:
            payload["confidence"] = confidence
        if reason is not None:
            payload["reason"] = reason
        return payload

    def _extract_json_string_field(self, raw_response: str, field_name: str) -> str | None:
        field_token = f'"{field_name}"'
        field_index = raw_response.find(field_token)
        if field_index < 0:
            return None

        colon_index = raw_response.find(":", field_index + len(field_token))
        if colon_index < 0:
            return None

        value_start = raw_response.find('"', colon_index)
        if value_start < 0:
            return None

        scan_index = value_start + 1
        while scan_index < len(raw_response):
            char = raw_response[scan_index]
            if char == '"' and raw_response[scan_index - 1] != "\\":
                remainder = raw_response[scan_index + 1 :]
                if re.match(r'\s*(?:,|\})', remainder):
                    return raw_response[value_start + 1 : scan_index]
            scan_index += 1

        return raw_response[value_start + 1 :].rstrip().rstrip("}").strip()

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
