import unittest

from chunker.llm_rhetorical_move_classifier_helpers.rhetorical_move_taxonomy import allowed_moves
from chunker.llm_section_type_classifier_helpers.section_taxonomy import (
    SECTION_LABEL_DESCRIPTIONS,
    allowed_sections,
)


class FrontMatterClassificationTests(unittest.TestCase):
    def test_front_matter_is_allowed_for_every_paper_type(self) -> None:
        for paper_type in (
            "empirical_research",
            "literature_review",
            "systematic_review_or_meta_analysis",
            "theoretical_or_conceptual",
            "position_or_discussion_paper",
            "methodological_paper",
            "pedagogical_or_practice_paper",
            "argumentative_essay",
            "other_or_unclear",
        ):
            self.assertIn("front_matter", allowed_sections(paper_type))

    def test_front_matter_has_no_rhetorical_moves(self) -> None:
        self.assertEqual(allowed_moves("front_matter"), ())
        self.assertIn("publication", SECTION_LABEL_DESCRIPTIONS["front_matter"])


if __name__ == "__main__":
    unittest.main()
