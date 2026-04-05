from __future__ import annotations

from .models import EvidenceFilters, EvidencePlan, EvidenceToolCall
from .utils.plan_defaults import ALLOWED_ANSWER_MODES, ALLOWED_TOOLS, DEFAULT_TOP_K, MAX_TOOL_CALLS, MAX_TOP_K


class EvidencePlanValidator:
    def validate(self, plan: EvidencePlan) -> EvidencePlan:
        if not plan.intent:
            raise ValueError("Plan intent must not be empty.")
        normalized_answer_mode = self._normalize_answer_mode(plan.answer_mode)
        if normalized_answer_mode not in ALLOWED_ANSWER_MODES:
            raise ValueError(f"Unsupported answer_mode: {plan.answer_mode}")
        if not plan.reason:
            raise ValueError("Plan reason must not be empty.")
        if not plan.tool_calls:
            raise ValueError("Plan must contain at least one tool call.")
        if len(plan.tool_calls) > MAX_TOOL_CALLS:
            raise ValueError(f"Plan has too many tool calls: {len(plan.tool_calls)}")

        return EvidencePlan(
            intent=plan.intent,
            answer_mode=normalized_answer_mode,
            reason=plan.reason,
            tool_calls=[self._validate_tool_call(tool_call) for tool_call in plan.tool_calls],
        )

    def _validate_tool_call(self, tool_call: EvidenceToolCall) -> EvidenceToolCall:
        if tool_call.tool not in ALLOWED_TOOLS:
            raise ValueError(f"Unsupported tool: {tool_call.tool}")
        if not tool_call.query:
            raise ValueError("Tool call query must not be empty.")
        normalized_top_k = tool_call.top_k or DEFAULT_TOP_K
        if normalized_top_k <= 0:
            raise ValueError("Tool call top_k must be positive.")
        normalized_top_k = min(normalized_top_k, MAX_TOP_K)
        return EvidenceToolCall(
            tool=tool_call.tool,
            query=tool_call.query,
            filters=self._validate_filters(tool_call.filters),
            top_k=normalized_top_k,
        )

    def _validate_filters(self, filters: EvidenceFilters) -> EvidenceFilters:
        if filters.year_min is not None and filters.year_max is not None:
            if filters.year_min > filters.year_max:
                raise ValueError("year_min cannot be greater than year_max.")
        return EvidenceFilters(
            paper_id_in=[value for value in filters.paper_id_in if value],
            authors_any=[value for value in filters.authors_any if value],
            year_min=filters.year_min,
            year_max=filters.year_max,
            classification_label_in=[value for value in filters.classification_label_in if value],
            section_title_contains=filters.section_title_contains,
        )

    def _normalize_answer_mode(self, answer_mode: str) -> str:
        normalized = answer_mode.strip().lower()
        aliases = {
            "evidence_retrieval": "direct_answer",
            "synthesize": "direct_answer",
            "synthesis": "direct_answer",
            "summarize": "direct_answer",
            "summary": "direct_answer",
            "comparison": "compare",
            "compare_approaches": "compare",
            "recommendation": "recommend",
        }
        return aliases.get(normalized, normalized)
