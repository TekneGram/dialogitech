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
    def test_crossref_search_fills_missing_doi_before_manual_input(self) -> None:
        class SearchDoiClient:
            def search(self, *, title: str, authors) -> dict:
                self.search_args = (title, authors)
                return {
                    "doi": "10.1234/found",
                    "title": title,
                    "authors": authors,
                    "journal": "Found Journal",
                    "year": "2025",
                }

        doi_client = SearchDoiClient()

        class SearchExtractor(LLMMetadataExtractor):
            def _extract_component(self, component, compact_json):
                if component == "title":
                    value = "A paper title"
                elif component == "authors":
                    value = ["Ada Lovelace"]
                elif "doi_metadata" in compact_json:
                    value = {
                        "name": compact_json["doi_metadata"]["journal"],
                        "year": compact_json["doi_metadata"]["year"],
                        "doi": compact_json["doi_metadata"]["doi"],
                    }
                else:
                    value = {"name": None, "year": "2025", "doi": "unknown"}
                return MetadataDecision(
                    value=value,
                    confidence="high",
                    reason="Test response",
                    source_pages=[0, 1],
                )

        extractor = SearchExtractor(
            input_fn=lambda prompt: (_ for _ in ()).throw(
                AssertionError("Manual input should not be requested")
            ),
            doi_metadata_client=doi_client,
        )

        result = extractor.extract_metadata(marker_document())

        self.assertEqual(doi_client.search_args, ("A paper title", ["Ada Lovelace"]))
        self.assertEqual(result.journal.value["doi"], "10.1234/found")
        self.assertEqual(result.journal.value["name"], "Found Journal")

    def test_arxiv_search_fills_preprint_metadata_after_crossref_miss(self) -> None:
        class MissingCrossrefClient:
            def search(self, *, title: str, authors):
                return None

        class FoundArxivClient:
            def search(self, *, title: str, authors):
                return {
                    "doi": "unknown",
                    "title": title,
                    "authors": authors,
                    "journal": "arXiv-preprint",
                    "year": "2026",
                    "arxiv_url": "https://arxiv.org/abs/2609.09425",
                }

        class ArxivExtractor(LLMMetadataExtractor):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def _extract_component(self, component, compact_json):
                if component == "title":
                    value = "Edu-QuRating"
                elif component == "authors":
                    value = ["Oliver Garrod"]
                elif "doi_metadata" in compact_json:
                    value = {
                        "name": "arXiv-preprint",
                        "year": "2026",
                        "doi": "unknown",
                    }
                else:
                    value = {"name": None, "year": None, "doi": "unknown"}
                return MetadataDecision(
                    value=value,
                    confidence="high",
                    reason="Test response",
                    source_pages=[0, 1],
                )

        extractor = ArxivExtractor(
            doi_metadata_client=MissingCrossrefClient(),
            arxiv_metadata_client=FoundArxivClient(),
            input_fn=lambda prompt: (_ for _ in ()).throw(
                AssertionError("Manual metadata input should not be requested")
            ),
        )

        result = extractor.extract_metadata(marker_document())

        self.assertEqual(
            result.journal.value["arxiv_url"],
            "https://arxiv.org/abs/2609.09425",
        )
        self.assertEqual(result.journal.value["name"], "arXiv-preprint")
        self.assertEqual(result.journal.value["doi"], "unknown")

    def test_marker_html_rejects_doi_from_navigation_link(self) -> None:
        document = {
            "children": [
                {
                    "id": "/page/0/Page/0",
                    "children": [
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

        class WrongDoiExtractor(LLMMetadataExtractor):
            def _extract_component(self, component, compact_json):
                return MetadataDecision(
                    value={
                        "name": "Example Journal",
                        "year": "2025",
                        "doi": "10.1080/2331186X.2025.2551175",
                    },
                    confidence="high",
                    reason="Test response",
                    source_pages=[0],
                )

        messages: list[str] = []
        extractor = WrongDoiExtractor(event_logger=messages.append)

        decision = extractor.extract_component(document, "journal", allow_manual=False)

        self.assertTrue(any("does not occur" in message for message in messages))
        self.assertIsNone(decision.value["doi"])
        self.assertEqual(decision.provenance["doi"], ["unresolved"])

    def test_crossref_is_retried_after_page_expansion_finds_doi(self) -> None:
        document = {
            "children": [
                {
                    "id": f"/page/{page_number}/Page/{page_number}",
                    "children": [
                        {
                            "block_type": "Text",
                            "html": (
                                '<a href="https://doi.org/10.1080/2331186X.2025.2551175">'
                                "Previous article</a>"
                                if page_number < 2
                                else '<a href="https://doi.org/10.1080/2331186X.2025.2543113">'
                                "Cite this article</a>"
                            ),
                        }
                    ],
                }
                for page_number in range(4)
            ]
        }

        class FakeDoiClient:
            def __init__(self) -> None:
                self.lookups: list[str] = []

            def lookup(self, doi: str) -> dict:
                self.lookups.append(doi)
                return {
                    "doi": doi,
                    "title": "Example Paper",
                    "authors": ["Example Author"],
                    "journal": "Example Journal",
                    "year": "2025",
                }

        doi_client = FakeDoiClient()

        class ExpandingExtractor(LLMMetadataExtractor):
            def _extract_component(self, component, compact_json):
                if "doi_metadata" in compact_json:
                    value = {
                        "name": "Example Journal",
                        "year": "2025",
                        "doi": compact_json["doi_metadata"]["doi"],
                    }
                elif len(compact_json["pages"]) == 2:
                    value = {
                        "name": None,
                        "year": "2025",
                        "doi": "10.1080/2331186X.2025.2551175",
                    }
                else:
                    value = {
                        "name": None,
                        "year": "2025",
                        "doi": "10.1080/2331186X.2025.2543113",
                    }
                return MetadataDecision(
                    value=value,
                    confidence="high",
                    reason="Test response",
                    source_pages=[page["page_number"] for page in compact_json["pages"]],
                )

        extractor = ExpandingExtractor(
            input_fn=lambda prompt: "y",
            doi_metadata_client=doi_client,
        )
        decision = extractor.extract_component(document, "journal", allow_manual=False)

        self.assertEqual(doi_client.lookups, ["10.1080/2331186x.2025.2543113"])
        self.assertEqual(decision.value["name"], "Example Journal")

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

    def test_missing_doi_and_journal_are_marked_unknown_without_manual_input(self) -> None:
        extractor = AlwaysMissingExtractor(
            input_fn=lambda prompt: (_ for _ in ()).throw(
                AssertionError("Manual metadata input should not be requested")
            )
        )

        decision = extractor.extract_component(
            marker_document(),
            "journal",
            allow_manual=True,
        )

        self.assertEqual(decision.confidence, "low")
        self.assertEqual(decision.value["doi"], "unknown")
        self.assertEqual(decision.value["name"], "unknown")
        self.assertEqual(decision.value["year"], "unknown")
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

    def test_missing_doi_does_not_prompt_for_manual_doi(self) -> None:
        class FakeDoiClient:
            def __init__(self) -> None:
                self.dois: list[str] = []

            def lookup(self, doi: str) -> dict:
                self.dois.append(doi)
                return {"doi": doi, "journal": "Lookup Journal"}

        doi_client = FakeDoiClient()
        extractor = AlwaysMissingExtractor(
            input_fn=lambda prompt: (_ for _ in ()).throw(
                AssertionError("Manual DOI input should not be requested")
            ),
            doi_metadata_client=doi_client,
        )

        decision = extractor.extract_component(
            marker_document(),
            "journal",
            allow_manual=True,
        )

        self.assertEqual(doi_client.dois, [])
        self.assertEqual(decision.value["doi"], "unknown")

    def test_unverifiable_crossref_data_does_not_trigger_manual_fields(self) -> None:
        class FakeDoiClient:
            def __init__(self) -> None:
                self.lookups = 0

            def lookup(self, doi: str) -> dict:
                self.lookups += 1
                return {"doi": doi, "journal": "Unconfirmed Journal", "year": "2025"}

        doi_client = FakeDoiClient()
        extractor = AlwaysMissingExtractor(
            input_fn=lambda prompt: (_ for _ in ()).throw(
                AssertionError("Manual metadata input should not be requested")
            ),
            doi_metadata_client=doi_client,
        )

        decision = extractor.extract_component(
            marker_document(),
            "journal",
            allow_manual=True,
        )

        self.assertEqual(doi_client.lookups, 0)
        self.assertEqual(decision.value["doi"], "unknown")
        self.assertEqual(decision.value["name"], "unknown")


if __name__ == "__main__":
    unittest.main()
