from __future__ import annotations

import unittest

from chunker.llm_metadata_extractor import LLMMetadataExtractor
from chunker.llm_metadata_extractor_helpers.metadata_models import MetadataDecision


def marker_document(page_count: int = 4) -> dict:
    return {
        "children": [
            {
                "id": f"/page/{page_number}/Page/{page_number}",
                "children": [
                    {
                        "block_type": "Text",
                        "html": f"<p>Page {page_number}</p>",
                    }
                ],
            }
            for page_number in range(page_count)
        ]
    }


class FakeMetadataExtractor(LLMMetadataExtractor):
    def __init__(self, input_fn=None) -> None:
        super().__init__(input_fn=input_fn)
        self.calls: list[list[int]] = []

    def _extract_component(self, component, compact_json):
        pages = [page["page_number"] for page in compact_json["pages"]]
        self.calls.append(pages)

        if pages == [0, 1]:
            value = {
                "name": None,
                "volume": None,
                "issue": None,
                "year": "2025",
                "doi": "10.1234/example",
                "issn": None,
            }
        else:
            value = {
                "name": "Example Journal",
                "volume": "12",
                "issue": "2",
                "year": "2025",
                "doi": "10.1234/example",
                "issn": "1234-5678",
            }

        return MetadataDecision(
            value=value,
            confidence="medium",
            reason="Test response",
            source_pages=pages,
        )


class AlwaysMissingExtractor(LLMMetadataExtractor):
    def _extract_component(self, component, compact_json):
        return MetadataDecision(
            value={"name": None, "year": None}
            if component == "journal"
            else None,
            confidence="medium",
            reason="Test response",
            source_pages=[page["page_number"] for page in compact_json["pages"]],
        )


class TestMetadataFlow(unittest.TestCase):
    def test_missing_values_trigger_pages_two_and_three(self) -> None:
        extractor = FakeMetadataExtractor()

        decision = extractor.extract_component(
            marker_document(),
            "journal",
            allow_manual=False,
        )

        self.assertEqual(decision.value["name"], "Example Journal")
        self.assertEqual(decision.value["issn"], "1234-5678")
        self.assertEqual(extractor.calls, [[0, 1], [0, 1, 2, 3]])

    def test_remaining_missing_values_use_manual_input(self) -> None:
        answers = iter(
            [
                "Manual Journal",
                "7",
                "1",
                "2024",
                "10.9999/manual",
                "9999-9999",
            ]
        )
        extractor = AlwaysMissingExtractor(input_fn=lambda prompt: next(answers))

        decision = extractor.extract_component(
            marker_document(),
            "journal",
            allow_manual=True,
        )

        self.assertEqual(decision.confidence, "high")
        self.assertEqual(decision.value["name"], "Manual Journal")
        self.assertEqual(decision.value["issn"], "9999-9999")


if __name__ == "__main__":
    unittest.main()
