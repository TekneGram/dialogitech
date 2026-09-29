from __future__ import annotations

import unittest

from chunker.llm_metadata_extractor_helpers.arxiv_metadata_client import (
    ArxivMetadataClient,
)


class FakeResponse:
    def __init__(self, payload: str) -> None:
        self.payload = payload.encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


ATOM = "http://www.w3.org/2005/Atom"


def feed(title: str, authors: list[str], arxiv_id: str = "2609.09425") -> str:
    authors_xml = "".join(f"<author><name>{author}</name></author>" for author in authors)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="{ATOM}">
      <entry>
        <id>http://arxiv.org/abs/{arxiv_id}v2</id>
        <title>{title}</title>
        <published>2026-09-08T00:00:00Z</published>
        {authors_xml}
      </entry>
    </feed>"""


class TestArxivMetadataClient(unittest.TestCase):
    def test_search_uses_author_route_first(self) -> None:
        urls: list[str] = []

        def urlopen(request, timeout):
            urls.append(request.full_url)
            return FakeResponse(feed(
                "Edu-QuRating: Multi-Dimensional Educational Data Curation",
                ["Oliver G. B. Garrod"],
            ))

        client = ArxivMetadataClient(urlopen=urlopen)
        result = client.search(
            title="Edu-QuRating: Multi-Dimensional Educational Data Curation",
            authors=["Oliver G. B. Garrod"],
        )

        self.assertEqual(result["arxiv_url"], "https://arxiv.org/abs/2609.09425")
        self.assertEqual(result["journal"], "arXiv-preprint")
        self.assertEqual(len(urls), 1)
        self.assertIn("au%3A%22garrod%22", urls[0])

    def test_search_falls_back_to_title_only(self) -> None:
        urls: list[str] = []

        def urlopen(request, timeout):
            urls.append(request.full_url)
            if len(urls) == 1:
                return FakeResponse(feed("A different title", ["Other Author"], "1.2.3"))
            return FakeResponse(feed(
                "Edu-QuRating: Multi-Dimensional Educational Data Curation",
                ["Oliver G. B. Garrod"],
            ))

        client = ArxivMetadataClient(urlopen=urlopen)
        result = client.search(
            title="Edu-QuRating: Multi-Dimensional Educational Data Curation",
            authors=["Oliver G. B. Garrod"],
        )

        self.assertEqual(result["arxiv_url"], "https://arxiv.org/abs/2609.09425")
        self.assertEqual(len(urls), 2)
        self.assertNotIn("au%3A%22garrod%22", urls[1])


if __name__ == "__main__":
    unittest.main()
