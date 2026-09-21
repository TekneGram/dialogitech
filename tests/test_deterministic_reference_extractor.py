from __future__ import annotations

import unittest

from chunker.llm_metadata_extractor_helpers.reference_extractor import (
    DeterministicReferenceExtractor,
)


class TestDeterministicReferenceExtractor(unittest.TestCase):
    def test_extracts_list_and_text_references_and_stops_at_next_heading(self) -> None:
        document = {
            "children": [
                {
                    "children": [
                        {"block_type": "SectionHeader", "html": "<h1>References</h1>"},
                        {
                            "block_type": "ListGroup",
                            "children": [
                                {"html": "<li>Smith, J. (2020). Journal of Testing, 1, 2-3.</li>"},
                                {"html": "<li>Smith, J. (2020). Journal of Testing, 1, 2-3.</li>"},
                            ],
                        },
                        {
                            "block_type": "Text",
                            "html": "<p>Jones, A. (2021). Conference proceedings, 4, 5-6.</p>",
                        },
                        {"block_type": "SectionHeader", "html": "<h1>Appendix</h1>"},
                        {
                            "block_type": "Text",
                            "html": "<p>Brown, B. (2022). This must not be included.</p>",
                        },
                    ]
                }
            ]
        }

        self.assertEqual(
            DeterministicReferenceExtractor().extract(document),
            [
                "Smith, J. (2020). Journal of Testing, 1, 2-3.",
                "Jones, A. (2021). Conference proceedings, 4, 5-6.",
            ],
        )


if __name__ == "__main__":
    unittest.main()
