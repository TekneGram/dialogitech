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
    def __init__(self, input_fn=None, doi_metadata_client=None) -> None:
        super().__init__(
            input_fn=input_fn,
            doi_metadata_client=doi_metadata_client,
        )

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
                "unknown",
                "Manual Journal",
                "7",
                "1",
                "2024",
            ]
        )
        extractor = AlwaysMissingExtractor(input_fn=lambda prompt: next(answers))

        decision = extractor.extract_component(
            marker_document(),
            "journal",
            allow_manual=True,
        )

        self.assertEqual(decision.confidence, "high")
        self.assertEqual(decision.value["doi"], "unknown")
        self.assertEqual(decision.value["name"], "Manual Journal")
        self.assertIsNone(decision.value.get("issn"))

    def test_optional_journal_fields_do_not_require_manual_input(self) -> None:
        class CompleteRequiredExtractor(LLMMetadataExtractor):
            def _extract_component(self, component, compact_json):
                return MetadataDecision(
                    value={"name": "Example Journal", "year": "2025"}
                    if component == "journal"
                    else None,
                    confidence="medium",
                    reason="Test response",
                    source_pages=[page["page_number"] for page in compact_json["pages"]],
                )

        extractor = CompleteRequiredExtractor(
            input_fn=lambda prompt: (_ for _ in ()).throw(
                AssertionError("Manual input should not be requested")
            )
        )
        decision = extractor.extract_component(
            marker_document(),
            "journal",
            allow_manual=True,
        )

        self.assertEqual(decision.value, {"name": "Example Journal", "year": "2025"})

    def test_supplied_manual_doi_is_looked_up_before_remaining_manual_fields(self) -> None:
        class FakeDoiClient:
            def __init__(self) -> None:
                self.dois: list[str] = []

            def lookup(self, doi: str) -> dict:
                self.dois.append(doi)
                return {"doi": doi, "journal": "Lookup Journal"}

        doi_client = FakeDoiClient()
        answers = iter(
            [
                "10.1234/manual",
                "Manual Journal",
                "7",
                "1",
                "2024",
            ]
        )
        extractor = AlwaysMissingExtractor(
            input_fn=lambda prompt: next(answers),
            doi_metadata_client=doi_client,
        )

        decision = extractor.extract_component(
            marker_document(),
            "journal",
            allow_manual=True,
        )

        self.assertEqual(doi_client.dois, ["10.1234/manual"])
        self.assertEqual(decision.value["doi"], "10.1234/manual")


if __name__ == "__main__":
    unittest.main()
