from __future__ import annotations

from .models import (
    BranchClaimProvenance,
    BranchSynthesisInput,
    BranchTurnSynthesisInput,
    ResearchSessionState,
)


class BranchSynthesisContextBuilder:
    def build(self, state: ResearchSessionState, *, branch_id: str) -> BranchSynthesisInput:
        branch = state.branches.get(branch_id)
        if branch is None:
            raise ValueError(f"Unknown branch_id: {branch_id}")

        turns = [
            BranchTurnSynthesisInput(
                turn_id=turn.turn_id,
                user_question=turn.user_question,
                synthesized_summary=turn.synthesized_summary,
            )
            for turn_id in branch.turn_order
            for turn in [state.turns.get(turn_id)]
            if turn is not None
        ]

        claim_provenance = self._build_claim_provenance(state, branch_id=branch_id)
        return BranchSynthesisInput(
            branch_id=branch.branch_id,
            branch_label=branch.label,
            root_query=state.root_query,
            turns=turns,
            claim_provenance=claim_provenance,
        )

    def render(self, synthesis_input: BranchSynthesisInput) -> str:
        lines: list[str] = []
        lines.append(f"Branch ID: {synthesis_input.branch_id}")
        lines.append(f"Branch label: {synthesis_input.branch_label}")
        lines.append(f"Root query: {synthesis_input.root_query}")
        lines.append("")
        lines.append("Turn summaries:")
        if not synthesis_input.turns:
            lines.append("[none]")
        else:
            for turn in synthesis_input.turns:
                lines.append(f"- {turn.turn_id} | question: {turn.user_question}")
                lines.append(f"  summary: {turn.synthesized_summary}")
        lines.append("")
        lines.append("Evidence provenance:")
        if not synthesis_input.claim_provenance:
            lines.append("[none]")
        else:
            for claim in synthesis_input.claim_provenance:
                paper_ids = ", ".join(claim.paper_ids) or "[none]"
                chunk_ids = ", ".join(claim.evidence_chunk_ids) or "[none]"
                lines.append(
                    f"- {claim.provenance_label} | {claim.claim_id} | turn={claim.created_in_turn} | "
                    f"status={claim.status} | confidence={claim.confidence}"
                )
                lines.append(f"  claim: {claim.text}")
                lines.append(f"  paper_ids: {paper_ids}")
                lines.append(f"  chunk_ids: {chunk_ids}")
        return "\n".join(lines).strip()

    def _build_claim_provenance(
        self,
        state: ResearchSessionState,
        *,
        branch_id: str,
    ) -> list[BranchClaimProvenance]:
        branch_claims = [claim for claim in state.claims.values() if claim.branch_id == branch_id]
        branch_claims.sort(key=lambda item: item.created_in_turn)

        provenance: list[BranchClaimProvenance] = []
        for index, claim in enumerate(branch_claims, start=1):
            paper_ids = [
                entry.paper_id
                for chunk_id in claim.evidence_chunk_ids
                for entry in [state.evidence_index.get(chunk_id)]
                if entry is not None
            ]
            unique_paper_ids = list(dict.fromkeys(paper_ids))
            provenance.append(
                BranchClaimProvenance(
                    provenance_label=f"P{index}",
                    claim_id=claim.claim_id,
                    text=claim.text,
                    status=claim.status,
                    confidence=claim.confidence,
                    created_in_turn=claim.created_in_turn,
                    paper_ids=unique_paper_ids,
                    evidence_chunk_ids=list(claim.evidence_chunk_ids),
                )
            )
        return provenance
