from __future__ import annotations

from itertools import combinations

from .branch_synthesis_context_builder import BranchSynthesisContextBuilder
from .branch_synthesizer import BranchSynthesizer
from .branch_comparison_synthesizer import BranchComparisonSynthesizer
from .models import SessionSynthesisResult
from .state_store import ResearchStateStore


class SessionSynthesizer:
    def __init__(
        self,
        *,
        state_store: ResearchStateStore,
        branch_context_builder: BranchSynthesisContextBuilder,
        branch_synthesizer: BranchSynthesizer,
        branch_comparison_synthesizer: BranchComparisonSynthesizer,
    ) -> None:
        self.state_store = state_store
        self.branch_context_builder = branch_context_builder
        self.branch_synthesizer = branch_synthesizer
        self.branch_comparison_synthesizer = branch_comparison_synthesizer

    def synthesize(
        self,
        *,
        session_path: str,
        branch_ids: list[str] | None = None,
    ) -> SessionSynthesisResult:
        state = self.state_store.load(session_path)
        selected_branch_ids = branch_ids or list(state.branch_order)

        branch_summaries = []
        for branch_id in selected_branch_ids:
            branch_input = self.branch_context_builder.build(state, branch_id=branch_id)
            branch_summaries.append(self.branch_synthesizer.synthesize(branch_input))

        branch_comparisons = [
            self.branch_comparison_synthesizer.compare(left, right)
            for left, right in combinations(branch_summaries, 2)
        ]
        return SessionSynthesisResult(
            session_state=state,
            branch_summaries=branch_summaries,
            branch_comparisons=branch_comparisons,
        )
