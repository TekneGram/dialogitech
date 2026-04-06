from __future__ import annotations

from dbquery.models import QueryRequest

from .models import RunMode, SelectedStateView


class QueryContextBuilder:
    def __init__(self, *, max_context_claims: int = 3) -> None:
        self.max_context_claims = max_context_claims

    def build_query_request(
        self,
        *,
        base_request: QueryRequest,
        selected_state_view: SelectedStateView,
        run_mode: RunMode,
    ) -> QueryRequest:
        query_text = self._build_query_text(
            user_question=base_request.query,
            selected_state_view=selected_state_view,
            run_mode=run_mode,
        )
        return QueryRequest(
            query=query_text,
            retrieval_mode=base_request.retrieval_mode,
            retrieval_depth=base_request.retrieval_depth,
            candidate_pool_k=base_request.candidate_pool_k,
            final_top_k=base_request.final_top_k,
            batch_size=base_request.batch_size,
            rewrite_count=base_request.rewrite_count,
            include_hyde=base_request.include_hyde,
            min_rrf_lists=base_request.min_rrf_lists,
            rrf_k=base_request.rrf_k,
            min_relevance_score=base_request.min_relevance_score,
            filters=base_request.filters,
            exclude_chunk_ids=list(base_request.exclude_chunk_ids),
            exclude_paper_ids=list(base_request.exclude_paper_ids),
        )

    def _build_query_text(
        self,
        *,
        user_question: str,
        selected_state_view: SelectedStateView,
        run_mode: RunMode,
    ) -> str:
        question = user_question.strip()
        if run_mode not in {"deepen", "expand"}:
            return question

        context_sections = self._build_context_sections(selected_state_view)
        if not context_sections:
            return question

        return "\n\n".join(
            [
                *context_sections,
                "New request:",
                question,
            ]
        )

    def _build_context_sections(self, selected_state_view: SelectedStateView) -> list[str]:
        sections: list[str] = []
        root_query = selected_state_view.root_query.strip()
        if root_query:
            sections.append("\n".join(["Current research focus:", root_query]))

        claim_lines = [
            f"- {claim.text}"
            for claim in selected_state_view.prior_claims[: self.max_context_claims]
            if claim.text.strip()
        ]
        if claim_lines:
            sections.append("\n".join(["Prior findings:", *claim_lines]))
        return sections
