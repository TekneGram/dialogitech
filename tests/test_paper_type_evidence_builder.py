from __future__ import annotations

import unittest

from chunker.llm_paper_type_classifier_helpers.evidence_builder import (
    PaperTypeEvidenceBuilder,
)


class TestPaperTypeEvidenceBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = PaperTypeEvidenceBuilder()

    def test_builds_title_abstract_headings_and_excerpts(self) -> None:
        markdown = (
            "# ABSTRACT\n"
            "This is the abstract.\n\n"
            "# Introduction\n"
            "This is the introduction.\n\n"
            "# Results\n"
            "These are the results."
        )

        evidence = self.builder.build(
            markdown,
            metadata={"title": "Example paper"},
        )

        self.assertEqual(evidence.title, "Example paper")
        self.assertEqual(evidence.abstract, "This is the abstract.")
        self.assertEqual(evidence.headings, ["ABSTRACT", "Introduction", "Results"])
        self.assertIn("This is the abstract.", evidence.opening_excerpt)
        self.assertIn("These are the results.", evidence.closing_excerpt)

    def test_missing_metadata_and_abstract_are_none(self) -> None:
        evidence = self.builder.build("# Introduction\nBody text.")

        self.assertIsNone(evidence.title)
        self.assertIsNone(evidence.abstract)
        self.assertEqual(evidence.headings, ["Introduction"])

    def test_excerpts_are_bounded(self) -> None:
        opening = "# Introduction\n" + ("open " * 700)
        closing = "\n# Discussion\n" + ("close " * 400)

        evidence = self.builder.build(opening + closing)

        self.assertLessEqual(len(evidence.opening_excerpt), 1800)
        self.assertLessEqual(len(evidence.closing_excerpt), 900)


if __name__ == "__main__":
    unittest.main()
