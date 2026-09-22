from __future__ import annotations

from .section_type_models import ArticleQuintile, ChunkContext, ChunkLocation
from ..markdown_section_chunker import HeadingSplit, SectionChunk


class SectionLocationBuilder:
    def __init__(self, *, filtered_markdown: str, heading_splits: list[HeadingSplit]) -> None:
        self.filtered_markdown = filtered_markdown
        self.heading_splits = heading_splits
        self.section_ranges = self.build_section_ranges()
        self.chunk_locations = self.build_chunk_locations()

    @staticmethod
    def article_quintile(
        article_start: int,
        article_end: int,
        article_length: int,
    ) -> ArticleQuintile:
        if article_length <= 0:
            raise ValueError("article_length must be positive.")

        midpoint = (article_start + article_end) / 2
        ratio = midpoint / article_length
        if ratio < 0.2:
            return "first 20%"
        if ratio < 0.4:
            return "second 20%"
        if ratio < 0.6:
            return "third 20%"
        if ratio < 0.8:
            return "fourth 20%"
        return "last 20%"

    def build_section_ranges(self) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        search_start = 0
        for index, heading_split in enumerate(self.heading_splits):
            heading_start = self.filtered_markdown.find(heading_split.raw_heading, search_start)
            if heading_start < 0:
                raise RuntimeError(
                    f"Could not locate heading {heading_split.raw_heading!r} in filtered markdown."
                )

            if index + 1 < len(self.heading_splits):
                next_heading = self.heading_splits[index + 1].raw_heading
                section_end = self.filtered_markdown.find(
                    next_heading,
                    heading_start + len(heading_split.raw_heading),
                )
                if section_end < 0:
                    raise RuntimeError(
                        f"Could not locate next heading {next_heading!r} in filtered markdown."
                    )
            else:
                section_end = len(self.filtered_markdown)

            ranges.append((heading_start, section_end))
            search_start = section_end
        return ranges

    def build_chunk_locations(self) -> dict[tuple[str, int], ChunkLocation]:
        locations: dict[tuple[str, int], ChunkLocation] = {}
        for section_index, heading_split in enumerate(self.heading_splits):
            section_start, section_end = self.section_ranges[section_index]
            section_text = self.filtered_markdown[section_start:section_end]
            section_search_start = 0

            for chunk in heading_split.chunks:
                local_start = section_text.find(chunk.text, section_search_start)
                if local_start < 0:
                    raise RuntimeError(
                        f"Could not anchor chunk {chunk.chunk_index} under heading "
                        f"{heading_split.title!r} in filtered markdown."
                    )

                local_end = local_start + len(chunk.text)
                article_start = section_start + local_start
                article_end = section_start + local_end
                section_search_start = local_start
                locations[(heading_split.title, chunk.chunk_index)] = ChunkLocation(
                    article_start=article_start,
                    article_end=article_end,
                    quintile=self.article_quintile(
                        article_start,
                        article_end,
                        len(self.filtered_markdown),
                    ),
                    section_index=section_index,
                )
        return locations

    def location_for(self, *, chunk: SectionChunk, heading_split: HeadingSplit) -> ChunkLocation:
        key = (heading_split.title, chunk.chunk_index)
        try:
            return self.chunk_locations[key]
        except KeyError as exc:
            raise RuntimeError(
                f"Missing chunk location for heading {heading_split.title!r}, "
                f"chunk {chunk.chunk_index}."
            ) from exc

    def context_for(self, *, chunk: SectionChunk, heading_split: HeadingSplit) -> ChunkContext:
        location = self.location_for(chunk=chunk, heading_split=heading_split)
        previous_section = None
        next_section = None
        if location.section_index > 0:
            previous_start, previous_end = self.section_ranges[location.section_index - 1]
            previous_section = self.filtered_markdown[previous_start:previous_end].strip()
        if location.section_index + 1 < len(self.section_ranges):
            next_start, next_end = self.section_ranges[location.section_index + 1]
            next_section = self.filtered_markdown[next_start:next_end].strip()
        return ChunkContext(
            previous_section=previous_section,
            current_chunk=chunk.text,
            next_section=next_section,
        )
