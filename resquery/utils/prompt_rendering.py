from __future__ import annotations

from dbquery.models import BatchSummary

from resquery.models import SelectedStateView


class ResearchStatePromptRenderer:
    def render_batch_summary_claims_prompt(
        self,
        *,
        user_question: str,
        summary: BatchSummary,
    ) -> str:
        return "\n\n".join(
            [
                f"User question:\n{user_question.strip()}",
                f"Batch summary index: {summary.batch_index}",
                f"Batch summary chunk_ids: {', '.join(summary.chunk_ids)}",
                "Batch summary:",
                summary.summary_text.strip(),
                "Return only the claims_added JSON object.",
            ]
        )

    def render_followup_suggestions_prompt(
        self,
        *,
        user_question: str,
        synthesized_summary: str,
    ) -> str:
        return "\n\n".join(
            [
                f"User question:\n{user_question.strip()}",
                "Synthesized summary:",
                synthesized_summary.strip(),
                "Return only the followups_added JSON object.",
            ]
        )

    def render_selected_state_view(self, view: SelectedStateView) -> str:
        lines = [
            f"root_query: {view.root_query}",
            "prior_claims:",
        ]
        if not view.prior_claims:
            lines.append("- [none]")
        else:
            for claim in view.prior_claims:
                lines.append(
                    f"- {claim.claim_id} | status={claim.status} | confidence={claim.confidence} | text={claim.text}"
                )
        lines.append("followup_suggestions:")
        if not view.followup_suggestions:
            lines.append("- [none]")
        else:
            for followup in view.followup_suggestions:
                lines.append(f"- {followup.question_id} | text={followup.text}")
        lines.append("recent_evidence:")
        if not view.recent_evidence:
            lines.append("- [none]")
        else:
            for evidence in view.recent_evidence:
                lines.append(
                    f"- {evidence.chunk_id} | paper_id={evidence.paper_id} | "
                    f"section_label={evidence.section_label or '[none]'} | section_title={evidence.section_title}"
                )
        return "\n".join(lines)
