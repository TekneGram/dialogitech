from __future__ import annotations

from pathlib import Path

from .models import ResQueryResult
from .utils import SessionSummaryRenderer


class ResQueryOutputWriter:
    def __init__(self, session_summary_renderer: SessionSummaryRenderer | None = None) -> None:
        self.session_summary_renderer = session_summary_renderer or SessionSummaryRenderer()

    def write(self, result: ResQueryResult, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._render(result), encoding="utf-8")
        return path

    def _render(self, result: ResQueryResult) -> str:
        lines: list[str] = []
        lines.append("# Research Session Turn")
        lines.append("")
        lines.append(f"Session path: {result.session_path}")
        lines.append(f"Turn: {result.turn.turn_id}")
        lines.append(f"Branch: {result.turn.branch_id}")
        lines.append(f"Run mode: {result.turn.run_mode}")
        lines.append(f"User question: {result.turn.user_question}")
        lines.append("")
        lines.append("## Selected Prior State")
        lines.append("")
        lines.append(f"Root query: {result.selected_state_view.root_query}")
        lines.append(f"Selected branch: {result.selected_state_view.branch_id or '[none]'}")
        lines.append("")
        lines.append("### Prior Claims")
        lines.append("")
        if not result.selected_state_view.prior_claims:
            lines.append("[none]")
        else:
            for claim in result.selected_state_view.prior_claims:
                lines.append(
                    f"- {claim.claim_id} | status={claim.status} | confidence={claim.confidence} | {claim.text}"
                )
        lines.append("")
        lines.append("### Follow-up Suggestions")
        lines.append("")
        if not result.selected_state_view.followup_suggestions:
            lines.append("[none]")
        else:
            for followup in result.selected_state_view.followup_suggestions:
                lines.append(f"- {followup.question_id} | {followup.text}")
        lines.append("")
        lines.append("### Recent Evidence")
        lines.append("")
        if not result.selected_state_view.recent_evidence:
            lines.append("[none]")
        else:
            for evidence in result.selected_state_view.recent_evidence:
                lines.append(
                    f"- {evidence.chunk_id} | paper_id={evidence.paper_id} | "
                    f"section_label={evidence.section_label or '[none]'} | section_title={evidence.section_title}"
                )
        lines.append("")
        lines.append("## Query Context")
        lines.append("")
        lines.append("Text sent to dbquery:")
        lines.append("```text")
        lines.append(result.turn.contextualized_query)
        lines.append("```")
        lines.append("")
        lines.append(
            f"Excluded chunk_ids: {', '.join(result.turn.query_request.exclude_chunk_ids) or '[none]'}"
        )
        lines.append(
            f"Excluded paper_ids: {', '.join(result.turn.query_request.exclude_paper_ids) or '[none]'}"
        )
        lines.append(
            f"Candidate pool k: {result.turn.query_request.candidate_pool_k or '[default]'}"
        )
        lines.append("")
        lines.append("## dbquery Artifact")
        lines.append("")
        lines.append(f"Query artifact path: {result.query_output_path or '[not written]'}")
        lines.append("")
        lines.append("## Synthesized Summary")
        lines.append("")
        lines.append(result.turn.synthesized_summary)
        lines.append("")
        lines.append("## State Update")
        lines.append("")
        lines.append("### Claims Added")
        lines.append("")
        if not result.state_update.claims_added:
            lines.append("[none]")
        else:
            for claim in result.state_update.claims_added:
                lines.append(
                    f"- status={claim.status} | confidence={claim.confidence} | "
                    f"chunk_ids={', '.join(claim.evidence_chunk_ids)} | {claim.text}"
                )
        lines.append("")
        lines.append("### Follow-up Suggestions Added")
        lines.append("")
        if not result.state_update.followups_added:
            lines.append("[none]")
        else:
            for followup in result.state_update.followups_added:
                lines.append(f"- {followup.text}")
        lines.append("")
        lines.append("## Session Snapshot")
        lines.append("")
        lines.append(self.session_summary_renderer.render(result.session_state))
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"
