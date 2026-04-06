from __future__ import annotations

from datetime import datetime, timezone

from .models import ResearchTurn, ResQueryRequest, ResQueryResult
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
        selected_state_view = self.state_selector.select(state)
        contextualized_request = self.query_context_builder.build_query_request(
            base_request=request.query_request,
            selected_state_view=selected_state_view,
        )
        query_result = self.query_pipeline.run(contextualized_request)
        synthesized_summary = (query_result.synthesized_summary or "").strip()
        if not synthesized_summary:
            raise RuntimeError("dbquery did not produce a synthesized summary for resquery.")

        turn = ResearchTurn(
            turn_id=f"t{len(state.turn_order) + 1}",
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

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
