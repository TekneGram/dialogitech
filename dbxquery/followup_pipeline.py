from __future__ import annotations

from .evidence_planner import GemmaEvidencePlanner
from .evidence_retriever import EvidenceRetriever
from .grounded_writer import GemmaGroundedWriter
from .models import FollowupPipelineResult, FollowupRequest
from .plan_validator import EvidencePlanValidator


class FollowupPipeline:
    def __init__(
        self,
        *,
        planner: GemmaEvidencePlanner,
        validator: EvidencePlanValidator,
        retriever: EvidenceRetriever,
        grounded_writer: GemmaGroundedWriter,
    ) -> None:
        self.planner = planner
        self.validator = validator
        self.retriever = retriever
        self.grounded_writer = grounded_writer

    def run(self, request: FollowupRequest) -> FollowupPipelineResult:
        plan = self.planner.create_plan(
            prior_summary=request.prior_summary,
            user_question=request.user_question,
        )
        validated_plan = self.validator.validate(plan)
        evidence_results = [self.retriever.retrieve(tool_call) for tool_call in validated_plan.tool_calls]
        grounded_answer = self.grounded_writer.write_answer(
            plan=validated_plan,
            prior_summary=request.prior_summary,
            user_question=request.user_question,
            evidence_results=evidence_results,
        )
        return FollowupPipelineResult(
            request=request,
            plan=validated_plan,
            evidence_results=evidence_results,
            grounded_answer=grounded_answer,
        )
