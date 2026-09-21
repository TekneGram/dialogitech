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


if __name__ == "__main__":
    unittest.main()
