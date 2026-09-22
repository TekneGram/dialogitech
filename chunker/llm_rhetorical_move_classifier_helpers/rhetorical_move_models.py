from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RhetoricalMoveLabel = Literal[
    "criticize_prior_research",
    "identify_literature_gap",
    "state_gap_response",
    "state_limitations",
    "make_claim",
    "present_results",
    "interpret_results",
    "state_research_questions",
    "establish_topic_importance",
    "provide_background",
    "review_prior_research",
    "state_objective_or_aim",
    "state_hypothesis_or_prediction",
    "describe_research_design",
    "describe_data_analysis",
    "justify_methodological_choice",
    "report_evidence",
    "compare_with_prior_research",
    "explain_causal_mechanism",
    "state_implications",
    "recommend_action_or_practice",
    "propose_future_research",
    "summarize_contribution",
    "define_scope_or_boundary",
]
RhetoricalMoveClassificationLabel = RhetoricalMoveLabel | Literal["unclassified"]

ClassificationConfidence = Literal["low", "medium", "high"]


@dataclass(slots=True)
class RhetoricalMoveClassification:
    label: RhetoricalMoveClassificationLabel
    confidence: ClassificationConfidence
    reason: str


@dataclass(slots=True)
class RhetoricalMoveResult:
    moves: list[RhetoricalMoveClassification] = field(default_factory=list)
    source: Literal["llm"] = "llm"
    used_context: bool = False
    reason: str = ""
