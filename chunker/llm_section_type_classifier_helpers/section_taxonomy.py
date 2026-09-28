from __future__ import annotations

from typing import Literal

from chunker.llm_paper_type_classifier_helpers.paper_type_models import PaperType

SectionLabel = Literal[
    "front_matter",
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
    "empirical_research": ("front_matter", "abstract", "introduction", "method", "results", "discussion"),
    "literature_review": ("front_matter", "abstract", "introduction", "review_method", "literature_synthesis", "discussion"),
    "systematic_review_or_meta_analysis": ("front_matter", "abstract", "introduction", "review_method", "results", "discussion"),
    "theoretical_or_conceptual": ("front_matter", "abstract", "introduction", "conceptual_framework", "argument_or_analysis", "discussion"),
    "position_or_discussion_paper": ("front_matter", "abstract", "introduction", "argument_or_analysis", "discussion"),
    "methodological_paper": ("front_matter", "abstract", "introduction", "method", "evaluation_or_example", "discussion"),
    "pedagogical_or_practice_paper": ("front_matter", "abstract", "introduction", "practice_description", "evaluation_or_example", "discussion"),
    "argumentative_essay": ("front_matter", "abstract", "introduction", "argument_or_analysis", "discussion"),
    "other_or_unclear": (
        "front_matter", "abstract", "introduction", "background_or_review", "method_or_approach",
        "analysis_or_argument", "discussion_or_conclusion",
    ),
}

SECTION_LABEL_DESCRIPTIONS: dict[SectionLabel, str] = {
    "front_matter": (
        "publication, editorial, conference, author-information, copyright, preface, "
        "acknowledgement, keyword, or other non-substantive material before the paper body"
    ),
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


def allowed_sections(paper_type: PaperType) -> tuple[SectionLabel, ...]:
    return PAPER_TYPE_ALLOWED_SECTIONS[paper_type]
