from __future__ import annotations

from chunker.llm_section_type_classifier_helpers.section_taxonomy import SectionLabel

from .rhetorical_move_models import RhetoricalMoveLabel

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
    )
}


def allowed_moves(section_label: SectionLabel | str) -> tuple[RhetoricalMoveLabel, ...]:
    if section_label == "unclassified":
        return ALL_RHETORICAL_MOVES
    return SECTION_ALLOWED_MOVES[section_label]  # type: ignore[index]
