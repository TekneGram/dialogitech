from __future__ import annotations

import unittest

from chunker.llm_section_type_classifier_helpers.location_builder import SectionLocationBuilder
from chunker.markdown_section_chunker import MarkdownSectionChunker
from dbinsert.models import PaperMetadataRecord
from dbinsert.paper_chunk_serializer import PaperChunkSerializer
from chunker.chunk_models import ClassifiedHeadingSplit, ClassifiedSectionChunk
from chunker.llm_section_type_classifier_helpers.section_type_models import ChunkClassification


class TestChunkLocationAndIdentity(unittest.TestCase):
    def test_overlapping_chunks_are_anchored_without_fallbacks(self) -> None:
        content = " ".join(f"word{i}." for i in range(700))
        markdown = f"# First section\n{content}\n# Second section\n{content}"
        heading_splits = MarkdownSectionChunker(min_words=200, overlap_words=50).process(markdown)
        events: list[str] = []

        builder = SectionLocationBuilder(
            filtered_markdown=markdown,
            heading_splits=heading_splits,
            event_logger=events.append,
        )

        self.assertGreater(len(heading_splits[0].chunks), 1)
        self.assertFalse(any("could not be anchored" in event for event in events))

        locations = [
            builder.location_for(chunk=chunk, heading_split=heading_splits[0])
            for chunk in heading_splits[0].chunks
        ]
        self.assertEqual(
            locations,
            sorted(locations, key=lambda location: location.article_start),
        )

    def test_repeated_headings_have_distinct_locations(self) -> None:
        body = " ".join(f"term{i}." for i in range(240))
        markdown = f"# Repeated\n{body}\n# Middle\nshort\n# Repeated\n{body}"
        heading_splits = MarkdownSectionChunker(min_words=200, overlap_words=50).process(markdown)
        builder = SectionLocationBuilder(
            filtered_markdown=markdown,
            heading_splits=heading_splits,
        )

        repeated = [split for split in heading_splits if split.title == "Repeated"]
        self.assertEqual(len(repeated), 2)
        first_location = builder.location_for(
            chunk=repeated[0].chunks[0],
            heading_split=repeated[0],
        )
        second_location = builder.location_for(
            chunk=repeated[1].chunks[0],
            heading_split=repeated[1],
        )
        self.assertNotEqual(first_location.section_index, second_location.section_index)
        self.assertLess(first_location.article_start, second_location.article_start)

    def test_repeated_headings_have_distinct_chunk_ids(self) -> None:
        classification = ChunkClassification(
            label="introduction",
            source="llm",
            reason="test",
            confidence="high",
        )
        splits = [
            ClassifiedHeadingSplit(
                title="Repeated",
                heading_level=1,
                raw_heading="# Repeated",
                content="first",
                chunks=[
                    ClassifiedSectionChunk(
                        title="Repeated",
                        heading_level=1,
                        chunk_index=0,
                        text="first",
                        word_count=1,
                        classification=classification,
                    )
                ],
            ),
            ClassifiedHeadingSplit(
                title="Repeated",
                heading_level=1,
                raw_heading="# Repeated",
                content="second",
                chunks=[
                    ClassifiedSectionChunk(
                        title="Repeated",
                        heading_level=1,
                        chunk_index=0,
                        text="second",
                        word_count=1,
                        classification=classification,
                    )
                ],
            ),
        ]
        metadata = PaperMetadataRecord(
            paper_id="paper",
            paper_title="Paper",
            authors=[],
            paper_type="empirical_research",
            paper_type_source="llm",
            paper_type_reason="test",
        )

        records = PaperChunkSerializer().serialize_paper(metadata, splits)

        self.assertEqual(len(records), 2)
        self.assertEqual(len({record.chunk_id for record in records}), 2)


if __name__ == "__main__":
    unittest.main()
