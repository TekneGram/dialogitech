from __future__ import annotations

from typing import Literal

from chunker.llm_paper_type_classifier_helpers.paper_type_models import PaperType

SectionLabel = Literal[
    "abstract",
    "introduction",
    "method",
    "results",
    "discussion",
    "review_method",
    "literature_synthesis",
    "conceptual_framework",
    "argument_or_analysis",
    "evaluation_or_example",
    "practice_description",
    "background_or_review",
    "method_or_approach",
    "analysis_or_argument",
    "discussion_or_conclusion",
]

PAPER_TYPE_ALLOWED_SECTIONS: dict[PaperType, tuple[SectionLabel, ...]] = {
    "empirical_research": ("abstract", "introduction", "method", "results", "discussion"),
    "literature_review": ("abstract", "introduction", "review_method", "literature_synthesis", "discussion"),
    "systematic_review_or_meta_analysis": ("abstract", "introduction", "review_method", "results", "discussion"),
    "theoretical_or_conceptual": ("abstract", "introduction", "conceptual_framework", "argument_or_analysis", "discussion"),
    "position_or_discussion_paper": ("abstract", "introduction", "argument_or_analysis", "discussion"),
    "methodological_paper": ("abstract", "introduction", "method", "evaluation_or_example", "discussion"),
    "pedagogical_or_practice_paper": ("abstract", "introduction", "practice_description", "evaluation_or_example", "discussion"),
    "argumentative_essay": ("abstract", "introduction", "argument_or_analysis", "discussion"),
    "other_or_unclear": (
        "abstract", "introduction", "background_or_review", "method_or_approach",
        "analysis_or_argument", "discussion_or_conclusion",
    ),
}

SECTION_LABEL_DESCRIPTIONS: dict[SectionLabel, str] = {
    "abstract": "compact overview of the paper",
    "introduction": "background, literature positioning, gap, aims, or research questions",
    "method": "original-study design, data, materials, procedures, or analysis",
    "results": "original-study findings or reported outcomes",
    "discussion": "interpretation, implications, limitations, or conclusion",
    "review_method": "search, screening, selection, coding, or synthesis method for a review",
    "literature_synthesis": "organized synthesis or comparison of prior literature",
    "conceptual_framework": "definitions, theory, model, or conceptual framework",
    "argument_or_analysis": "the paper's substantive argument, analysis, or line of reasoning",
    "evaluation_or_example": "evaluation, worked example, demonstration, or illustrative application",
    "practice_description": "description of teaching, intervention, implementation, or professional practice",
    "background_or_review": "background or prior-work review in a genre-unclear paper",
    "method_or_approach": "method, approach, or procedure in a genre-unclear paper",
    "analysis_or_argument": "analysis, evidence, or argument in a genre-unclear paper",
    "discussion_or_conclusion": "discussion, implications, limitations, or conclusion in a genre-unclear paper",
}

COMMON_HEADING_LABELS: dict[SectionLabel, set[str]] = {
    "abstract": {"abstract"},
    "introduction": {"introduction"},
    "discussion": {"discussion", "conclusion", "conclusions", "limitations", "implications", "future directions"},
    "discussion_or_conclusion": {"discussion", "conclusion", "conclusions", "limitations", "implications", "future directions"},
}

PAPER_TYPE_HEADING_LABELS: dict[PaperType, dict[SectionLabel, set[str]]] = {
    "empirical_research": {
        "introduction": {"background", "related work", "literature review"},
        "method": {"method", "methods", "methodology", "materials and methods", "experimental setup", "experiment", "experiments", "procedure", "procedures", "participants", "materials", "data", "data collection", "corpus", "research design"},
        "results": {"result", "results", "finding", "findings", "analysis", "analyses", "evaluation", "evaluations", "topic frequency"},
    },
    "literature_review": {
        "introduction": {"background", "related work"},
        "review_method": {"method", "methods", "methodology", "search strategy", "study selection", "inclusion criteria", "data extraction", "quality appraisal", "review protocol"},
        "literature_synthesis": {"literature review", "findings", "themes", "synthesis", "evidence synthesis", "review findings"},
    },
    "systematic_review_or_meta_analysis": {
        "introduction": {"background", "related work"},
        "review_method": {"method", "methods", "methodology", "search strategy", "study selection", "inclusion criteria", "data extraction", "quality appraisal", "review protocol", "prisma"},
        "results": {"result", "results", "findings", "meta-analysis results", "study characteristics", "evidence synthesis"},
    },
    "theoretical_or_conceptual": {
        "introduction": {"background", "related work", "literature review"},
        "conceptual_framework": {"conceptual framework", "theoretical framework", "framework", "theory", "definitions"},
        "argument_or_analysis": {"analysis", "argument", "arguments", "conceptual analysis", "implications of the framework"},
    },
    "position_or_discussion_paper": {
        "introduction": {"background", "related work", "literature review"},
        "argument_or_analysis": {"analysis", "argument", "arguments", "position", "commentary", "recommendations"},
    },
    "methodological_paper": {
        "introduction": {"background", "related work", "literature review"},
        "method": {"method", "methods", "methodology", "procedure", "procedures", "instrument development", "validation"},
        "evaluation_or_example": {"evaluation", "evaluations", "example", "examples", "demonstration", "application", "case study"},
    },
    "pedagogical_or_practice_paper": {
        "introduction": {"background", "related work", "literature review"},
        "practice_description": {"teaching practice", "classroom practice", "implementation", "intervention", "pedagogical approach", "instructional design"},
        "evaluation_or_example": {"evaluation", "evaluations", "example", "examples", "demonstration", "application", "case study"},
    },
    "argumentative_essay": {
        "introduction": {"background", "related work", "literature review"},
        "argument_or_analysis": {"analysis", "argument", "arguments", "position", "commentary"},
    },
    "other_or_unclear": {
        "background_or_review": {"background", "related work", "literature review", "theoretical framework"},
        "method_or_approach": {"method", "methods", "methodology", "approach", "procedure", "procedures"},
        "analysis_or_argument": {"analysis", "analyses", "argument", "arguments", "results", "findings", "evaluation"},
    },
}


def allowed_sections(paper_type: PaperType) -> tuple[SectionLabel, ...]:
    return PAPER_TYPE_ALLOWED_SECTIONS[paper_type]


def heading_label(paper_type: PaperType, normalized_heading: str) -> SectionLabel | None:
    allowed = set(allowed_sections(paper_type))
    for label, titles in COMMON_HEADING_LABELS.items():
        if label in allowed and normalized_heading in titles:
            return label
    for label, titles in PAPER_TYPE_HEADING_LABELS[paper_type].items():
        if label in allowed and normalized_heading in titles:
            return label
    return None
