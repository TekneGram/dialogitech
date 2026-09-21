from __future__ import annotations

import unittest
from pathlib import Path

from chunker.llm_metadata_extractor import LLMMetadataExtractor
from chunker.llm_metadata_extractor_helpers.doi_candidate_extractor import (
    DoiCandidateExtractor,
)


class TestDoiCandidateExtractor(unittest.TestCase):
    def test_previous_and_next_links_are_not_selected(self) -> None:
        compact = {
            "pages": [
                {
                    "page_number": 0,
                    "blocks": [
                        {
                            "block_type": "Text",
                            "html": (
                                '<a href="https://doi.org/10.1080/2331186X.2025.2543113">'
                                "Cite this article</a>"
                            ),
                        },
                        {
                            "block_type": "Text",
                            "html": (
                                '<a href="https://doi.org/10.1080/2331186X.2025.2551175">'
                                "Previous article</a>"
                            ),
                        },
                    ],
                }
            ]
        }

        candidates = DoiCandidateExtractor().extract(compact)

        self.assertEqual(candidates[0].doi, "10.1080/2331186x.2025.2543113")
        self.assertEqual(
            DoiCandidateExtractor().canonical(compact).doi,
            "10.1080/2331186x.2025.2543113",
        )

    def test_real_marker_artifact_selects_article_doi(self) -> None:
        path = Path(
            "marker/conversion_results/An autoethnographic study of ESL academic writing with ChatGPT/"
            "An autoethnographic study of ESL academic writing with ChatGPT.json"
        )
        if not path.is_file():
            self.skipTest(f"Local artifact is not available: {path}")

        extractor = LLMMetadataExtractor()
        document = extractor.load_marker_json(path)
        compact = extractor.compact_page_json(
            extractor.select_pages(document, [0, 1])
        )

        candidate = DoiCandidateExtractor().canonical(compact)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.doi, "10.1080/2331186x.2025.2543113")


if __name__ == "__main__":
    unittest.main()
