from __future__ import annotations

from datetime import datetime, timezone

from dbquery.models import FusedChunkResult

from .models import (
    EvidenceIndexEntry,
    ResearchBranch,
    ResearchClaim,
    ResearchSessionState,
    ResearchTurn,
    StateUpdate,
    SuggestedFollowup,
)


class ResearchStateMerger:
    def merge(
        self,
        *,
        state: ResearchSessionState,
        turn: ResearchTurn,
        state_update: StateUpdate,
        fused_results: list[FusedChunkResult],
    ) -> ResearchSessionState:
        branch = state.branches.get(turn.branch_id)
        if branch is None:
            branch = ResearchBranch(
                branch_id=turn.branch_id,
                label=turn.branch_id,
                created_at=turn.timestamp,
            )
            state.branches[turn.branch_id] = branch
            state.branch_order.append(turn.branch_id)
        state.turn_order.append(turn.turn_id)
        branch.turn_order.append(turn.turn_id)
        state.turns[turn.turn_id] = turn
        self._merge_claims(state, turn, state_update)
        self._merge_followups(state, turn, state_update)
        self._merge_evidence_index(state, fused_results)
        self._merge_branch_history(branch, fused_results)
        state.active_branch_id = turn.branch_id
        state.updated_at = self._utc_now()
        return state

    def _merge_claims(
        self,
        state: ResearchSessionState,
        turn: ResearchTurn,
        state_update: StateUpdate,
    ) -> None:
        for claim in state_update.claims_added:
            claim_id = f"c{len(state.claims) + 1}"
            state.claims[claim_id] = ResearchClaim(
                claim_id=claim_id,
                text=claim.text,
                status=claim.status,
                confidence=claim.confidence,
                evidence_chunk_ids=claim.evidence_chunk_ids,
                created_in_turn=turn.turn_id,
                branch_id=turn.branch_id,
            )
            turn.claims_added.append(claim_id)

    def _merge_followups(
        self,
        state: ResearchSessionState,
        turn: ResearchTurn,
        state_update: StateUpdate,
    ) -> None:
        for followup in state_update.followups_added:
            question_id = f"q{len(state.followup_suggestions) + 1}"
            state.followup_suggestions[question_id] = SuggestedFollowup(
                question_id=question_id,
                text=followup.text,
                created_in_turn=turn.turn_id,
                branch_id=turn.branch_id,
            )
            turn.followups_added.append(question_id)

    def _merge_evidence_index(
        self,
        state: ResearchSessionState,
        fused_results: list[FusedChunkResult],
    ) -> None:
        for chunk in fused_results:
            state.evidence_index[chunk.chunk_id] = EvidenceIndexEntry(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                section_label=chunk.classification_label,
                section_title=chunk.section_title,
            )

    def _merge_branch_history(
        self,
        branch: ResearchBranch,
        fused_results: list[FusedChunkResult],
    ) -> None:
        seen_chunk_ids = set(branch.seen_chunk_ids)
        seen_paper_ids = set(branch.seen_paper_ids)
        for chunk in fused_results:
            if chunk.chunk_id not in seen_chunk_ids:
                branch.seen_chunk_ids.append(chunk.chunk_id)
                seen_chunk_ids.add(chunk.chunk_id)
            if chunk.paper_id not in seen_paper_ids:
                branch.seen_paper_ids.append(chunk.paper_id)
                seen_paper_ids.add(chunk.paper_id)

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
