from __future__ import annotations

from .models import (
    ResearchSessionState,
    SelectedClaim,
    SelectedEvidence,
    SelectedFollowup,
    SelectedStateView,
)


class ResearchStateSelector:
    def __init__(self, *, max_claims: int = 8, max_followups: int = 6, max_recent_evidence: int = 8) -> None:
        self.max_claims = max_claims
        self.max_followups = max_followups
        self.max_recent_evidence = max_recent_evidence

    def select(self, state: ResearchSessionState) -> SelectedStateView:
        selected_turn_ids = list(reversed(state.turn_order[-2:]))
        recent_chunk_ids: list[str] = []
        for turn_id in selected_turn_ids:
            turn = state.turns.get(turn_id)
            if turn is None:
                continue
            recent_chunk_ids.extend(reversed(turn.retrieved_chunk_ids))

        unique_recent_chunk_ids = list(dict.fromkeys(recent_chunk_ids))
        return SelectedStateView(
            root_query=state.root_query,
            prior_claims=self._select_claims(state),
            followup_suggestions=self._select_followups(state),
            recent_evidence=self._select_recent_evidence(state, unique_recent_chunk_ids),
        )

    def _select_claims(self, state: ResearchSessionState) -> list[SelectedClaim]:
        ordered_claims = [
            state.claims[claim_id]
            for claim_id in sorted(
                state.claims.keys(),
                key=lambda claim_id: self._claim_sort_key(state.claims[claim_id]),
                reverse=True,
            )
        ]
        return [
            SelectedClaim(
                claim_id=claim.claim_id,
                text=claim.text,
                status=claim.status,
                confidence=claim.confidence,
            )
            for claim in ordered_claims[: self.max_claims]
        ]

    def _select_followups(self, state: ResearchSessionState) -> list[SelectedFollowup]:
        followups = list(state.followup_suggestions.values())
        followups.sort(key=lambda followup: followup.created_in_turn, reverse=True)
        return [
            SelectedFollowup(question_id=followup.question_id, text=followup.text)
            for followup in followups[: self.max_followups]
        ]

    def _select_recent_evidence(
        self,
        state: ResearchSessionState,
        recent_chunk_ids: list[str],
    ) -> list[SelectedEvidence]:
        evidence: list[SelectedEvidence] = []
        for chunk_id in recent_chunk_ids:
            entry = state.evidence_index.get(chunk_id)
            if entry is None:
                continue
            evidence.append(
                SelectedEvidence(
                    chunk_id=entry.chunk_id,
                    paper_id=entry.paper_id,
                    section_label=entry.section_label,
                    section_title=entry.section_title,
                )
            )
            if len(evidence) >= self.max_recent_evidence:
                break
        return evidence

    def _claim_sort_key(self, claim: object) -> tuple[int, str]:
        if not hasattr(claim, "status") or not hasattr(claim, "created_in_turn"):
            return (0, "")
        confidence_value = {"high": 2, "medium": 1, "low": 0}.get(getattr(claim, "confidence", ""), 0)
        status_value = 1 if getattr(claim, "status", "") == "supported" else 0
        return (status_value * 10 + confidence_value, getattr(claim, "created_in_turn", ""))
