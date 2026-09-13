from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .section_classifier import ClassifiedHeadingSplit

from .section_taxonomy import SectionLabel
from .paper_type_classifier import PaperType
from .section_taxonomy import allowed_sections

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
ClassificationConfidence = Literal["low", "medium", "high"]

ALL_RHETORICAL_MOVES: tuple[RhetoricalMoveLabel, ...] = (
    "criticize_prior_research", "identify_literature_gap", "state_gap_response",
    "state_limitations", "make_claim", "present_results", "interpret_results",
    "state_research_questions", "establish_topic_importance", "provide_background",
    "review_prior_research", "state_objective_or_aim", "state_hypothesis_or_prediction",
    "describe_research_design", "describe_data_analysis", "justify_methodological_choice",
    "report_evidence", "compare_with_prior_research", "explain_causal_mechanism",
    "state_implications", "recommend_action_or_practice", "propose_future_research",
    "summarize_contribution", "define_scope_or_boundary",
)

SECTION_ALLOWED_MOVES: dict[SectionLabel, tuple[RhetoricalMoveLabel, ...]] = {
    "abstract": (
        "establish_topic_importance", "provide_background", "identify_literature_gap",
        "state_gap_response", "state_objective_or_aim", "state_research_questions",
        "state_hypothesis_or_prediction", "describe_research_design", "present_results",
        "interpret_results", "summarize_contribution", "state_implications",
        "state_limitations", "define_scope_or_boundary",
    ),
    "introduction": (
        "establish_topic_importance", "provide_background", "review_prior_research",
        "criticize_prior_research", "identify_literature_gap", "state_gap_response",
        "state_objective_or_aim", "state_research_questions", "state_hypothesis_or_prediction",
        "make_claim", "define_scope_or_boundary", "summarize_contribution",
    ),
    "method": (
        "describe_research_design", "describe_data_analysis", "justify_methodological_choice",
        "define_scope_or_boundary", "report_evidence", "make_claim",
    ),
    "results": (
        "present_results", "report_evidence", "make_claim", "interpret_results",
        "compare_with_prior_research", "explain_causal_mechanism", "state_limitations",
    ),
    "discussion": (
        "interpret_results", "compare_with_prior_research", "criticize_prior_research",
        "make_claim", "explain_causal_mechanism", "state_limitations",
        "state_implications", "recommend_action_or_practice", "propose_future_research",
        "summarize_contribution", "state_gap_response", "define_scope_or_boundary",
    ),
    "review_method": (
        "describe_research_design", "describe_data_analysis", "justify_methodological_choice",
        "define_scope_or_boundary", "report_evidence",
    ),
    "literature_synthesis": (
        "review_prior_research", "criticize_prior_research", "identify_literature_gap",
        "make_claim", "compare_with_prior_research", "summarize_contribution",
    ),
    "conceptual_framework": (
        "provide_background", "make_claim", "define_scope_or_boundary", "summarize_contribution",
        "review_prior_research",
    ),
    "argument_or_analysis": (
        "make_claim", "report_evidence", "criticize_prior_research", "explain_causal_mechanism",
        "compare_with_prior_research", "interpret_results",
    ),
    "evaluation_or_example": (
        "report_evidence", "present_results", "interpret_results", "make_claim",
        "justify_methodological_choice",
    ),
    "practice_description": (
        "describe_research_design", "justify_methodological_choice", "recommend_action_or_practice",
        "define_scope_or_boundary", "make_claim",
    ),
    "background_or_review": (
        "establish_topic_importance", "provide_background", "review_prior_research",
        "criticize_prior_research", "identify_literature_gap", "make_claim",
    ),
    "method_or_approach": (
        "describe_research_design", "describe_data_analysis", "justify_methodological_choice",
        "define_scope_or_boundary", "report_evidence",
    ),
    "analysis_or_argument": (
        "make_claim", "report_evidence", "interpret_results", "criticize_prior_research",
        "compare_with_prior_research", "explain_causal_mechanism",
    ),
    "discussion_or_conclusion": (
        "interpret_results", "compare_with_prior_research", "make_claim", "state_limitations",
        "state_implications", "recommend_action_or_practice", "propose_future_research",
        "summarize_contribution", "state_gap_response", "define_scope_or_boundary",
    ),
}


@dataclass(slots=True)
class RhetoricalMoveClassification:
    label: RhetoricalMoveLabel
    confidence: ClassificationConfidence
    reason: str


@dataclass(slots=True)
class RhetoricalMoveResult:
    moves: list[RhetoricalMoveClassification] = field(default_factory=list)
    source: Literal["llm"] = "llm"
    used_context: bool = False
    reason: str = ""


class RhetoricalMoveEnricher:
    def __init__(self, llm_classifier: Any) -> None:
        self.llm_classifier = llm_classifier

    def enrich_heading_splits(
        self,
        classified_splits: list[ClassifiedHeadingSplit],
        *,
        paper_type: PaperType = "empirical_research",
    ) -> list[ClassifiedHeadingSplit]:
        for split in classified_splits:
            for chunk in split.chunks:
                section_label = chunk.classification.label
                if section_label is None:
                    raise RuntimeError("Cannot classify rhetorical moves for an unresolved section chunk.")
                self.validate_section_for_paper_type(section_label, paper_type=paper_type)
                result = self.llm_classifier.classify(
                    chunk=chunk,
                    heading_split=split,
                    section_label=section_label,
                    paper_type=paper_type,
                )
                self.validate_result(result, section_label=section_label, paper_type=paper_type)
                chunk.rhetorical_move_result = result
        return classified_splits

    @staticmethod
    def allowed_moves(section_label: SectionLabel) -> tuple[RhetoricalMoveLabel, ...]:
        return SECTION_ALLOWED_MOVES[section_label]

    @staticmethod
    def validate_section_for_paper_type(section_label: SectionLabel, *, paper_type: PaperType) -> None:
        if section_label not in allowed_sections(paper_type):
            raise RuntimeError(
                f"Section label {section_label!r} is not allowed for paper type {paper_type!r}."
            )

    @classmethod
    def validate_result(
        cls,
        result: RhetoricalMoveResult,
        *,
        section_label: SectionLabel,
        paper_type: PaperType = "empirical_research",
    ) -> None:
        cls.validate_section_for_paper_type(section_label, paper_type=paper_type)
        if len(result.moves) > 3:
            raise RuntimeError("Rhetorical move classification returned more than three moves.")
        allowed = set(cls.allowed_moves(section_label))
        labels = [move.label for move in result.moves]
        if len(labels) != len(set(labels)):
            raise RuntimeError("Rhetorical move classification returned duplicate move labels.")
        invalid = set(labels) - allowed
        if invalid:
            raise RuntimeError(
                f"Rhetorical move classification returned moves not allowed for {section_label}: {sorted(invalid)}"
            )
