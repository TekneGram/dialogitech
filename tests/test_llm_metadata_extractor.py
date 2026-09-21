from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chunker.llm_metadata_extractor import LLMMetadataExtractor
from chunker.llm_metadata_extractor_helpers.metadata_models import MetadataDecision


class TestLLMMetadataExtractorUnit(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = LLMMetadataExtractor()
        self.document = {
            "children": [
                {
                    "id": "/page/0/Page/100",
                    "children": [
                        {
                            "id": "/page/0/SectionHeader/1",
                            "block_type": "SectionHeader",
                            "html": "<h1>Example title</h1>",
                            "bbox": [1, 2, 3, 4],
                        },
                        {
                            "id": "/page/0/Text/2",
                            "block_type": "Text",
                            "html": "<p>Author One</p>",
                            "polygon": [[1, 2]],
                        },
                    ],
                },
                {
                    "id": "/page/1/Page/101",
                    "children": [
                        {
                            "id": "/page/1/Text/1",
                            "block_type": "Text",
                            "html": "<p>Abstract text</p>",
                        }
                    ],
                },
            ]
        }

    def test_load_marker_json_accepts_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "marker.json"
            path.write_text(json.dumps(self.document), encoding="utf-8")

            loaded = self.extractor.load_marker_json(path)

        self.assertEqual(loaded["children"], self.document["children"])
        self.assertEqual(loaded["__source_path__"], str(path))

    def test_select_pages_preserves_requested_order_and_deduplicates(self) -> None:
        selected = self.extractor.select_pages(self.document, [1, 0, 1])

        self.assertEqual(
            [page["id"] for page in selected],
            ["/page/1/Page/101", "/page/0/Page/100"],
        )

    def test_compact_page_json_keeps_only_metadata_relevant_fields(self) -> None:
        selected = self.extractor.select_pages(self.document, [0, 1])

        compact = self.extractor.compact_page_json(selected)

        self.assertEqual([page["page_number"] for page in compact["pages"]], [0, 1])
        self.assertEqual(
            compact["pages"][0]["blocks"],
            [
                {"block_type": "SectionHeader", "html": "<h1>Example title</h1>"},
                {"block_type": "Text", "html": "<p>Author One</p>"},
            ],
        )
        self.assertTrue(
            all(
                set(block) == {"block_type", "html"}
                for page in compact["pages"]
                for block in page["blocks"]
            )
        )

    def test_extract_all_includes_deterministic_references(self) -> None:
        document = {
            "children": [
                {
                    "id": "/page/0/Page/0",
                    "children": [
                        {"block_type": "SectionHeader", "html": "<h1>References</h1>"},
                        {
                            "block_type": "ListGroup",
                            "children": [
                                {"html": "<li>Smith, J. (2020). Journal of Testing, 1, 2-3.</li>"}
                            ],
                        },
                    ],
                }
            ]
        }

        class CompleteExtractor(LLMMetadataExtractor):
            def _extract_component(self, component, compact_json):
                if component == "journal":
                    value = {"name": "Journal", "year": "2020"}
                elif component == "authors":
                    value = ["Smith, J."]
                else:
                    value = "Example title"
                return MetadataDecision(
                    value=value,
                    confidence="high",
                    reason="Test response",
                    source_pages=[0],
                )

        result = CompleteExtractor().extract_all(document, allow_manual=False)
        self.assertEqual(result["references"], ["Smith, J. (2020). Journal of Testing, 1, 2-3."])


class TestLLMMetadataExtractorIntegration(unittest.TestCase):
    ARTIFACT_PATH = Path(__file__).resolve().parents[1] / (
        "marker/conversion_results/2026_shi_et_al/2026_shi_et_al.json"
    )

    def test_real_marker_artifact_pages_zero_and_one(self) -> None:
        self.assertTrue(self.ARTIFACT_PATH.is_file(), self.ARTIFACT_PATH)

        extractor = LLMMetadataExtractor()
        document = extractor.load_marker_json(self.ARTIFACT_PATH)
        selected = extractor.select_pages(document, [0, 1])
        compact = extractor.compact_page_json(selected)

        self.assertEqual(len(selected), 2)
        self.assertEqual([page["page_number"] for page in compact["pages"]], [0, 1])
        self.assertTrue(all(page["blocks"] for page in compact["pages"]))
        self.assertTrue(
            all(
                set(block) == {"block_type", "html"}
                for page in compact["pages"]
                for block in page["blocks"]
            )
        )


if __name__ == "__main__":
    unittest.main()
