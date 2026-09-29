from __future__ import annotations

import unittest

from dbinsert.metadata_checker import MetadataCompletenessChecker
from dbinsert.metadata_year import apply_year_fallback, infer_year_from_filename


class TestMetadataYearFallback(unittest.TestCase):
    def test_infers_year_from_filename_prefix(self) -> None:
        self.assertEqual(
            infer_year_from_filename("pdfs/2025 Prompt Taxonomy.pdf"),
            "2025",
        )

    def test_returns_unknown_without_filename_year(self) -> None:
        metadata = {"journal": {"name": "Example Journal", "year": None}}

        self.assertEqual(apply_year_fallback(metadata, "Prompt Taxonomy.pdf"), "unknown")
        self.assertEqual(metadata["journal"]["year"], "unknown")

    def test_preserves_extracted_year(self) -> None:
        metadata = {"journal": {"name": "Example Journal", "year": "2024"}}

        self.assertEqual(apply_year_fallback(metadata, "2025 Prompt Taxonomy.pdf"), "2024")

    def test_unknown_year_is_not_a_missing_prompt_field(self) -> None:
        issues = MetadataCompletenessChecker().find_missing_fields({
            "title": "Example",
            "authors": ["Author"],
            "journal": {"name": "Example Journal", "year": "unknown"},
        })

        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
