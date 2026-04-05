from __future__ import annotations

from dbquery.gemma_client import GemmaClient

from .models import EvidenceFilters, EvidencePlan, EvidenceToolCall
from .utils.json_parser import JsonResponseParser
from .utils.prompt_rendering import FollowupPromptRenderer


class GemmaEvidencePlanner:
    SYSTEM_PROMPT = """You plan evidence retrieval for an academic follow-up question.

Rules:
- Use the prior summary and the user question together.
- Output JSON only.
- The top-level JSON object must contain: intent, answer_mode, reason, tool_calls.
- tool_calls must be a list of 1 to 3 retrieve_evidence calls.
- Each tool call must contain: tool, query, filters, top_k.
- Allowed filters: paper_id_in, authors_any, year_min, year_max, classification_label_in, section_title_contains.
- Keep queries short and evidence-seeking.
- Prefer separate calls when the user asks for comparisons between papers or authors.
"""

    def __init__(
        self,
        gemma_client: GemmaClient,
        prompt_renderer: FollowupPromptRenderer | None = None,
        json_parser: JsonResponseParser | None = None,
    ) -> None:
        self.gemma_client = gemma_client
        self.prompt_renderer = prompt_renderer or FollowupPromptRenderer()
        self.json_parser = json_parser or JsonResponseParser()

    def create_plan(self, *, prior_summary: str, user_question: str) -> EvidencePlan:
        prompt = self.prompt_renderer.render_planner_prompt(
            prior_summary=prior_summary,
            user_question=user_question,
        )
        response = self.gemma_client.generate(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
        )
        payload = self.json_parser.parse_object(response)
        return self._parse_plan(payload)

    def _parse_plan(self, payload: dict[str, object]) -> EvidencePlan:
        tool_calls_payload = payload.get("tool_calls")
        if not isinstance(tool_calls_payload, list):
            raise ValueError("tool_calls must be a list.")

        return EvidencePlan(
            intent=str(payload.get("intent", "")).strip(),
            answer_mode=str(payload.get("answer_mode", "")).strip(),
            reason=str(payload.get("reason", "")).strip(),
            tool_calls=[self._parse_tool_call(item) for item in tool_calls_payload],
        )

    def _parse_tool_call(self, payload: object) -> EvidenceToolCall:
        if not isinstance(payload, dict):
            raise ValueError("tool_call entries must be objects.")
        filters_payload = payload.get("filters", {})
        if filters_payload is None:
            filters_payload = {}
        if not isinstance(filters_payload, dict):
            raise ValueError("filters must be an object.")
        return EvidenceToolCall(
            tool=str(payload.get("tool", "")).strip(),
            query=str(payload.get("query", "")).strip(),
            filters=self._parse_filters(filters_payload),
            top_k=int(payload.get("top_k", 6)),
        )

    def _parse_filters(self, payload: dict[str, object]) -> EvidenceFilters:
        return EvidenceFilters(
            paper_id_in=self._as_string_list(payload.get("paper_id_in")),
            authors_any=self._as_string_list(payload.get("authors_any")),
            year_min=self._as_optional_int(payload.get("year_min")),
            year_max=self._as_optional_int(payload.get("year_max")),
            classification_label_in=self._as_string_list(payload.get("classification_label_in")),
            section_title_contains=self._as_optional_string(payload.get("section_title_contains")),
        )

    def _as_string_list(self, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if not isinstance(value, list):
            raise ValueError("Expected a list of strings.")
        return [str(item).strip() for item in value if str(item).strip()]

    def _as_optional_int(self, value: object) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    def _as_optional_string(self, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
