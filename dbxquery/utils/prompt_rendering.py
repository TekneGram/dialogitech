from __future__ import annotations

from dbquery.models import FusedChunkResult

from dbxquery.models import EvidenceFilters, EvidencePlan, EvidenceResult


class FollowupPromptRenderer:
    def render_planner_prompt(self, *, prior_summary: str, user_question: str) -> str:
        return "\n\n".join(
            [
                "Prior summary:",
                prior_summary.strip(),
                "User follow-up question:",
                user_question.strip(),
                "Plan evidence retrieval for this follow-up and return JSON only.",
            ]
        )

    def render_grounded_answer_prompt(
        self,
        *,
        plan: EvidencePlan,
        prior_summary: str,
        user_question: str,
        evidence_results: list[EvidenceResult],
    ) -> str:
        return "\n\n".join(
            [
                f"Answer mode: {plan.answer_mode}",
                "Prior summary:",
                prior_summary.strip(),
                "User follow-up question:",
                user_question.strip(),
                "Retrieved evidence:",
                self._render_evidence_results(evidence_results),
            ]
        )

    def _render_evidence_results(self, evidence_results: list[EvidenceResult]) -> str:
        if not evidence_results:
            return "[no evidence retrieved]"

        groups: list[str] = []
        for index, evidence_result in enumerate(evidence_results, start=1):
            lines = [
                f"Evidence call {index}",
                f"query: {evidence_result.tool_call.query}",
                f"filters: {self._format_filters(evidence_result.tool_call.filters)}",
            ]
            if not evidence_result.fused_results:
                lines.append("results: [none]")
            else:
                lines.extend(self._render_chunks(evidence_result.fused_results))
            groups.append("\n".join(lines))
        return "\n\n".join(groups)

    def _render_chunks(self, chunks: list[FusedChunkResult]) -> list[str]:
        lines: list[str] = []
        for rank, chunk in enumerate(chunks, start=1):
            lines.extend(
                [
                    f"- Rank {rank}",
                    f"  chunk_id: {chunk.chunk_id}",
                    f"  paper_id: {chunk.paper_id}",
                    f"  paper_title: {chunk.paper_title}",
                    f"  authors: {', '.join(chunk.authors) if chunk.authors else '[missing]'}",
                    f"  year: {chunk.year if chunk.year is not None else '[missing]'}",
                    f"  section_title: {chunk.section_title}",
                    f"  classification_label: {chunk.classification_label or '[none]'}",
                    "  text:",
                    self._indent_block(chunk.text.strip()),
                ]
            )
        return lines

    def _format_filters(self, filters: EvidenceFilters) -> str:
        return str(filters)

    def _indent_block(self, text: str) -> str:
        if not text:
            return "    [empty]"
        return "\n".join(f"    {line}" for line in text.splitlines())
