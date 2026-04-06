from __future__ import annotations

from datetime import datetime, timezone

from dbquery.models import QueryRequest

from .models import ResearchBranch, ResearchTurn, ResQueryRequest, ResQueryResult
from .query_context_builder import QueryContextBuilder
from .session_initializer import ResearchSessionInitializer
from .state_merger import ResearchStateMerger
from .state_selector import ResearchStateSelector
from .state_store import ResearchStateStore
from .state_updater import ResearchStateUpdater


class ResQueryPipeline:
    def __init__(
        self,
        *,
        state_store: ResearchStateStore,
        session_initializer: ResearchSessionInitializer,
        state_selector: ResearchStateSelector,
        query_context_builder: QueryContextBuilder,
        query_pipeline: object,
        state_updater: ResearchStateUpdater,
        state_merger: ResearchStateMerger,
    ) -> None:
        self.state_store = state_store
        self.session_initializer = session_initializer
        self.state_selector = state_selector
        self.query_context_builder = query_context_builder
        self.query_pipeline = query_pipeline
        self.state_updater = state_updater
        self.state_merger = state_merger

    def run(self, request: ResQueryRequest) -> ResQueryResult:
        state = self._load_or_initialize_state(request)
        branch = self._resolve_branch(state, request)
        selected_state_view = self.state_selector.select(
            state,
            branch_id=branch.branch_id,
            user_question=request.user_question,
        )
        base_query_request = self._build_mode_query_request(branch=branch, request=request)
        contextualized_request = self.query_context_builder.build_query_request(
            base_request=base_query_request,
            selected_state_view=selected_state_view,
            run_mode=request.run_mode,
        )
        query_result = self.query_pipeline.run(contextualized_request)
        synthesized_summary = (query_result.synthesized_summary or "").strip()
        if not synthesized_summary:
            raise RuntimeError("dbquery did not produce a synthesized summary for resquery.")

        turn = ResearchTurn(
            turn_id=f"t{len(state.turn_order) + 1}",
            branch_id=branch.branch_id,
            run_mode=request.run_mode,
            user_question=request.user_question,
            timestamp=self._utc_now(),
            selected_state_view=selected_state_view,
            query_request=contextualized_request,
            contextualized_query=contextualized_request.query,
            query_output_path=request.query_output_path,
            synthesized_summary=synthesized_summary,
            retrieved_chunk_ids=[chunk.chunk_id for chunk in query_result.fused_results],
        )
        state_update = self.state_updater.update_state(
            existing_state=state,
            user_question=request.user_question,
            synthesized_summary=synthesized_summary,
            summaries=query_result.summaries,
        )
        merged_state = self.state_merger.merge(
            state=state,
            turn=turn,
            state_update=state_update,
            fused_results=query_result.fused_results,
        )
        self.state_store.save(merged_state, request.session_path)
        return ResQueryResult(
            session_state=merged_state,
            selected_state_view=selected_state_view,
            query_result=query_result,
            query_output_path=request.query_output_path,
            turn=turn,
            state_update=state_update,
            session_path=request.session_path,
        )

    def _load_or_initialize_state(self, request: ResQueryRequest):
        if self.state_store.exists(request.session_path):
            return self.state_store.load(request.session_path)
        return self.session_initializer.initialize(
            session_path=request.session_path,
            root_query=request.user_question,
        )

    def _resolve_branch(self, state, request: ResQueryRequest) -> ResearchBranch:
        if request.run_mode == "new":
            return self._create_branch(state, request)

        branch_id = request.branch_id or state.active_branch_id
        branch = state.branches.get(branch_id)
        if branch is None:
            raise ValueError(f"Unknown branch_id: {branch_id}")
        return branch

    def _create_branch(self, state, request: ResQueryRequest) -> ResearchBranch:
        branch_id = request.branch_id or f"b{len(state.branch_order) + 1}"
        if branch_id in state.branches:
            raise ValueError(f"Branch already exists: {branch_id}")

        branch = ResearchBranch(
            branch_id=branch_id,
            label=request.branch_label or request.user_question.strip()[:60] or branch_id,
            created_at=self._utc_now(),
        )
        state.branches[branch_id] = branch
        state.branch_order.append(branch_id)
        state.active_branch_id = branch_id
        return branch

    def _build_mode_query_request(self, *, branch: ResearchBranch, request: ResQueryRequest) -> QueryRequest:
        base_request = request.query_request
        candidate_pool_k = base_request.candidate_pool_k
        exclude_chunk_ids: list[str] = []
        exclude_paper_ids: list[str] = []

        if request.run_mode in {"deepen", "expand"}:
            candidate_pool_k = candidate_pool_k or max(base_request.final_top_k * 4, base_request.retrieval_depth)
        if request.run_mode == "deepen":
            exclude_chunk_ids = list(branch.seen_chunk_ids)
        if request.run_mode == "expand":
            exclude_paper_ids = list(branch.seen_paper_ids)

        return QueryRequest(
            query=request.user_question,
            retrieval_mode=base_request.retrieval_mode,
            retrieval_depth=base_request.retrieval_depth,
            candidate_pool_k=candidate_pool_k,
            final_top_k=base_request.final_top_k,
            batch_size=base_request.batch_size,
            rewrite_count=base_request.rewrite_count,
            include_hyde=base_request.include_hyde,
            min_rrf_lists=base_request.min_rrf_lists,
            rrf_k=base_request.rrf_k,
            min_relevance_score=base_request.min_relevance_score,
            filters=base_request.filters,
            exclude_chunk_ids=exclude_chunk_ids,
            exclude_paper_ids=exclude_paper_ids,
        )

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
