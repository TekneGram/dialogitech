from __future__ import annotations

import unittest

from chunker.llm_metadata_extractor import LLMMetadataExtractor


class TestBuildTitlePrompt(unittest.TestCase):
    COMPACT_JSON = {
        "pages": [
            {
                "page_number": 0,
                "blocks": [
                    {
                        "block_type": "SectionHeader",
                        "html": "<h1>Example paper title</h1>",
                    },
                    {
                        "block_type": "Text",
                        "html": "<p>Author One; Journal Name</p>",
                    },
                ],
            }
        ]
    }

    def test_prompt_contains_evidence_and_required_response_schema(self) -> None:
        prompt = LLMMetadataExtractor().build_title_prompt(self.COMPACT_JSON)

        self.assertIn("Example paper title", prompt)
        self.assertIn('"value"', prompt)
        self.assertIn('"confidence"', prompt)
        self.assertIn('"reason"', prompt)
        self.assertIn('"high"', prompt)
        self.assertIn('"medium"', prompt)
        self.assertIn('"low"', prompt)
        self.assertIn("Return JSON only", prompt)

    def test_journal_prompt_contains_required_schema(self) -> None:
        prompt = LLMMetadataExtractor().build_journal_prompt(self.COMPACT_JSON)

        for field in ("name", "volume", "issue", "year", "doi", "issn"):
            self.assertIn(f'"{field}"', prompt)
        self.assertIn("Do not infer values", prompt)
        self.assertIn("Return JSON only", prompt)

    def test_authors_prompt_contains_required_schema(self) -> None:
        prompt = LLMMetadataExtractor().build_authors_prompt(self.COMPACT_JSON)

        self.assertIn("Author One", prompt)
        self.assertIn("Return authors in their displayed order", prompt)
        self.assertIn('"confidence"', prompt)
        self.assertIn('"reason"', prompt)
        self.assertIn("Return JSON only", prompt)

if __name__ == "__main__":
    unittest.main()
