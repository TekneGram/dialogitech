import unittest

from chunker.llm_metadata_extractor_helpers.doi_metadata_client import (
    DoiMetadataClient,
)


class TestDoiMetadataMatching(unittest.TestCase):
    def setUp(self) -> None:
        self.client = DoiMetadataClient()
        self.paper_title = (
            "An autoethnographic study of ESL academic writing with ChatGPT"
        )
        self.paper_authors = ["Francis Group"]

    def test_matching_title_and_author_is_accepted(self) -> None:
        metadata = {
            "title": self.paper_title,
            "authors": ["Francis Group"],
        }

        self.assertTrue(
            self.client.matches_paper(
                metadata,
                title=self.paper_title,
                authors=self.paper_authors,
            )
        )

    def test_wrong_crossref_record_is_rejected(self) -> None:
        metadata = {
            "title": "Recognizing and managing disinformation in the digital age",
            "authors": ["Different Author"],
        }

        self.assertFalse(
            self.client.matches_paper(
                metadata,
                title=self.paper_title,
                authors=self.paper_authors,
            )
        )


if __name__ == "__main__":
    unittest.main()
