from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chunker.llm_metadata_extractor_helpers.doi_metadata_client import (
    DoiMetadataClient,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class TestDoiMetadataClient(unittest.TestCase):
    def test_normalize_doi(self) -> None:
        client = DoiMetadataClient()

        self.assertEqual(
            client.normalize_doi("https://doi.org/10.1234/ABC."),
            "10.1234/ABC",
        )

    def test_lookup_normalizes_crossref_record_and_caches_it(self) -> None:
        calls: list[str] = []

        def urlopen(request, timeout):
            calls.append(request.full_url)
            return FakeResponse(
                {
                    "message": {
                        "title": ["Example title"],
                        "author": [
                            {"given": "Ada", "family": "Lovelace"},
                        ],
                        "container-title": ["Example Journal"],
                        "published-print": {"date-parts": [[2025, 2]]},
                        "volume": "12",
                        "issue": "2",
                        "ISSN": ["1234-5678"],
                    }
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            client = DoiMetadataClient(
                cache_dir=Path(directory),
                urlopen=urlopen,
            )
            first = client.lookup("doi:10.1234/example")
            second = client.lookup("10.1234/example")

        self.assertEqual(first, second)
        self.assertEqual(first["title"], "Example title")
        self.assertEqual(first["authors"], ["Ada Lovelace"])
        self.assertEqual(first["journal"], "Example Journal")
        self.assertEqual(first["year"], "2025")
        self.assertEqual(first["volume"], "12")
        self.assertEqual(first["issn"], "1234-5678")
        self.assertEqual(len(calls), 1)

    def test_search_prefers_title_and_author_match(self) -> None:
        queries: list[str] = []

        def urlopen(request, timeout):
            queries.append(request.full_url)
            return FakeResponse(
                {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1234/wrong",
                                "title": ["A different paper"],
                                "author": [{"family": "Other"}],
                            },
                            {
                                "DOI": "10.1234/right",
                                "title": ["A paper about language models"],
                                "author": [{"family": "Lovelace"}],
                                "container-title": ["Example Journal"],
                                "published-print": {"date-parts": [[2025]]},
                            },
                        ]
                    }
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            client = DoiMetadataClient(
                cache_dir=Path(directory),
                urlopen=urlopen,
            )
            result = client.search(
                title="A paper about language models",
                authors=["Ada Lovelace"],
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["doi"], "10.1234/right")
        self.assertEqual(len(queries), 1)
        self.assertIn("Ada+Lovelace", queries[0])

    def test_search_falls_back_to_title_only(self) -> None:
        queries: list[str] = []

        def urlopen(request, timeout):
            queries.append(request.full_url)
            if len(queries) == 1:
                items = [{
                    "DOI": "10.1234/wrong",
                    "title": ["A paper about language models"],
                    "author": [{"family": "Other"}],
                }]
            else:
                items = [{
                    "DOI": "10.1234/right",
                    "title": ["A paper about language models"],
                    "author": [{"family": "Lovelace"}],
                }]
            return FakeResponse({"message": {"items": items}})

        with tempfile.TemporaryDirectory() as directory:
            client = DoiMetadataClient(
                cache_dir=Path(directory),
                urlopen=urlopen,
            )
            result = client.search(
                title="A paper about language models",
                authors=["Ada Lovelace"],
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["doi"], "10.1234/right")
        self.assertEqual(len(queries), 2)
        self.assertIn("query.bibliographic=A+paper+about+language+models+Ada+Lovelace", queries[0])
        self.assertIn("query.bibliographic=A+paper+about+language+models", queries[1])


if __name__ == "__main__":
    unittest.main()
