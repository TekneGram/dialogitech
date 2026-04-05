from __future__ import annotations

from pathlib import Path

from .models import FollowupPipelineResult


class FollowupOutputWriter:
    def write(self, result: FollowupPipelineResult, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._render(result), encoding="utf-8")
        return path

    def _render(self, result: FollowupPipelineResult) -> str:
        lines: list[str] = []
        lines.append("# Follow-up Query Output")
        lines.append("")
        lines.append("## Request")
        lines.append("")
        lines.append(f"User question: {result.request.user_question}")
        lines.append("")
        lines.append("### Prior Summary")
        lines.append("")
        lines.append(result.request.prior_summary.strip())
        lines.append("")
        lines.append("## Evidence Plan")
        lines.append("")
        lines.append(f"intent: {result.plan.intent}")
        lines.append(f"answer_mode: {result.plan.answer_mode}")
        lines.append(f"reason: {result.plan.reason}")
        lines.append("")
        lines.append("## Tool Calls")
        lines.append("")
        for index, evidence_result in enumerate(result.evidence_results, start=1):
            tool_call = evidence_result.tool_call
            lines.append(f"### Tool Call {index}")
            lines.append(f"tool: {tool_call.tool}")
            lines.append(f"query: {tool_call.query}")
            lines.append(f"filters: {tool_call.filters}")
            lines.append(f"top_k: {tool_call.top_k}")
            lines.append("")
        lines.append("## Evidence")
        lines.append("")
        for index, evidence_result in enumerate(result.evidence_results, start=1):
            lines.append(f"### Evidence Result {index}")
            lines.append(f"query: {evidence_result.tool_call.query}")
            lines.append(f"filters: {evidence_result.tool_call.filters}")
            lines.append(f"hits: {len(evidence_result.fused_results)}")
            lines.append("")
            for rank, chunk in enumerate(evidence_result.fused_results, start=1):
                lines.append(f"#### Rank {rank}")
                lines.append(f"chunk_id: {chunk.chunk_id}")
                lines.append(f"paper_id: {chunk.paper_id}")
                lines.append(f"paper_title: {chunk.paper_title}")
                lines.append(f"authors: {', '.join(chunk.authors) if chunk.authors else '[missing]'}")
                lines.append(f"year: {chunk.year if chunk.year is not None else '[missing]'}")
                lines.append(f"section_title: {chunk.section_title}")
                lines.append(f"classification_label: {chunk.classification_label}")
                lines.append(f"rrf_score: {chunk.rrf_score:.6f}")
                lines.append("text:")
                lines.append("```text")
                lines.append(chunk.text)
                lines.append("```")
                lines.append("")
        lines.append("## Final Answer")
        lines.append("")
        lines.append(result.grounded_answer.answer_text)
        lines.append("")
        lines.append("## Answer Metadata")
        lines.append("")
        lines.append(f"used_chunk_ids: {', '.join(result.grounded_answer.used_chunk_ids)}")
        lines.append(f"used_paper_ids: {', '.join(result.grounded_answer.used_paper_ids)}")
        lines.append(f"insufficient_evidence: {result.grounded_answer.insufficient_evidence}")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"
