from __future__ import annotations

from resquery.models import ResearchSessionState


class SessionSummaryRenderer:
    def render(self, state: ResearchSessionState) -> str:
        lines = [
            f"session_id: {state.session_id}",
            f"root_query: {state.root_query}",
            f"active_branch_id: {state.active_branch_id}",
            f"branch_count: {len(state.branch_order)}",
            f"turn_count: {len(state.turn_order)}",
            f"claim_count: {len(state.claims)}",
            f"followup_count: {len(state.followup_suggestions)}",
        ]
        return "\n".join(lines)
