from __future__ import annotations

from dbquery.models import QueryRequest

from .models import SelectedStateView


class QueryContextBuilder:
    def build_query_request(
        self,
        *,
        base_request: QueryRequest,
        selected_state_view: SelectedStateView,
    ) -> QueryRequest:
        contextualized_query = self._build_contextualized_query(
            user_question=base_request.query,
            selected_state_view=selected_state_view,
        )
        return QueryRequest(
            query=contextualized_query,
            retrieval_mode=base_request.retrieval_mode,
            retrieval_depth=base_request.retrieval_depth,
            final_top_k=base_request.final_top_k,
            batch_size=base_request.batch_size,
            rewrite_count=base_request.rewrite_count,
            include_hyde=base_request.include_hyde,
            min_rrf_lists=base_request.min_rrf_lists,
            rrf_k=base_request.rrf_k,
            min_relevance_score=base_request.min_relevance_score,
            filters=base_request.filters,
        )

    def _build_contextualized_query(
        self,
        *,
        user_question: str,
        selected_state_view: SelectedStateView,
    ) -> str:
        if not selected_state_view.prior_claims:
            return user_question.strip()

        claim_lines = [f"- {claim.text}" for claim in selected_state_view.prior_claims]
        return "\n\n".join(
            [
                "This is what we know so far:",
                "\n".join(claim_lines),
                "New question:",
                user_question.strip(),
            ]
        )
