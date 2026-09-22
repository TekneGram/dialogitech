from __future__ import annotations

import re
from typing import Callable

from .section_type_models import ArticleQuintile, ChunkContext, ChunkLocation
from ..markdown_section_chunker import HeadingSplit, SectionChunk


class SectionLocationBuilder:
    def __init__(
        self,
        *,
        filtered_markdown: str,
        heading_splits: list[HeadingSplit],
        event_logger: Callable[[str], None] | None = None,
    ) -> None:
        self.filtered_markdown = filtered_markdown
        self.heading_splits = heading_splits
        self.event_logger = event_logger
        self.section_ranges = self.build_section_ranges()
        self.chunk_locations = self.build_chunk_locations()

    @staticmethod
    def article_quintile(
        article_start: int,
        article_end: int,
        article_length: int,
    ) -> ArticleQuintile:
        if article_length <= 0:
            return "third 20%"

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
            heading_start = self._find_heading_start(
                heading_split,
                search_start=search_start,
            )
            if heading_start < 0:
                heading_start = search_start
                self._log_fallback(
                    f"heading {heading_split.title!r} was not found; using approximate section start "
                    f"at character {heading_start}."
                )

            if index + 1 < len(self.heading_splits):
                next_heading_split = self.heading_splits[index + 1]
                section_end = self._find_heading_start(
                    next_heading_split,
                    search_start=max(heading_start + 1, search_start),
                )
                if section_end < 0:
                    section_end = len(self.filtered_markdown)
                    self._log_fallback(
                        f"next heading {next_heading_split.title!r} was not found; using end of filtered Markdown "
                        f"at character {section_end}."
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
                local_start = self._find_chunk_start(
                    section_text,
                    chunk.text,
                    search_start=section_search_start,
                )
                if local_start < 0:
                    article_start = section_start
                    article_end = section_end
                    self._log_fallback(
                        f"chunk {chunk.chunk_index} under heading {heading_split.title!r} could not be anchored; "
                        f"using approximate section range {article_start}:{article_end}."
                    )
                else:
                    local_end = local_start + len(chunk.text)
                    article_start = section_start + local_start
                    article_end = section_start + local_end
                    section_search_start = max(local_end, local_start + 1)
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
        location = self.chunk_locations.get(key)
        if location is not None:
            return location

        if not self.section_ranges:
            self._log_fallback(
                f"location for heading {heading_split.title!r}, chunk {chunk.chunk_index} was missing "
                "and no section ranges exist; using synthetic zero-length location."
            )
            return self._location_from_range(
                section_start=0,
                section_end=0,
                section_index=0,
            )

        section_index = self._section_index_for(heading_split)
        section_start, section_end = self.section_ranges[section_index]
        self._log_fallback(
            f"location for heading {heading_split.title!r}, chunk {chunk.chunk_index} was missing; "
            f"using approximate section range {section_start}:{section_end}."
        )
        return self._location_from_range(
            section_start=section_start,
            section_end=section_end,
            section_index=section_index,
        )

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

    def _find_heading_start(self, heading_split: HeadingSplit, *, search_start: int) -> int:
        exact_start = self.filtered_markdown.find(heading_split.raw_heading, search_start)
        if exact_start >= 0:
            return exact_start

        wanted = self._normalize_heading(heading_split.title)
        for match in re.finditer(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$", self.filtered_markdown[search_start:]):
            if self._normalize_heading(match.group(1)) == wanted:
                matched_start = search_start + match.start()
                self._log_fallback(
                    f"heading {heading_split.raw_heading!r} matched normalized title at character {matched_start}."
                )
                return matched_start
        return -1

    def _find_chunk_start(self, section_text: str, chunk_text: str, *, search_start: int) -> int:
        exact_start = section_text.find(chunk_text, search_start)
        if exact_start >= 0:
            return exact_start

        normalized_chunk = re.sub(r"\s+", " ", chunk_text).strip()
        if not normalized_chunk:
            return -1
        escaped_words = [re.escape(word) for word in normalized_chunk.split(" ")]
        pattern = r"\s+".join(escaped_words)
        match = re.search(pattern, section_text[search_start:], flags=re.IGNORECASE)
        if match is None:
            return -1
        matched_start = search_start + match.start()
        self._log_fallback(
            f"chunk anchor used normalized-whitespace matching at local character {matched_start}."
        )
        return matched_start

    def _section_index_for(self, heading_split: HeadingSplit) -> int:
        for index, candidate in enumerate(self.heading_splits):
            if candidate is heading_split or candidate.title == heading_split.title:
                return index
        return 0

    def _location_from_range(
        self,
        *,
        section_start: int,
        section_end: int,
        section_index: int,
    ) -> ChunkLocation:
        if not self.filtered_markdown:
            self._log_fallback("filtered Markdown is empty; using synthetic zero-length location.")
            section_start = 0
            section_end = 0
        return ChunkLocation(
            article_start=section_start,
            article_end=section_end,
            quintile=self.article_quintile(
                section_start,
                section_end,
                len(self.filtered_markdown),
            ),
            section_index=section_index,
        )

    def _normalize_heading(self, heading: str) -> str:
        normalized = re.sub(r"^\s*#{1,6}\s*", "", heading)
        normalized = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s+", "", normalized)
        return re.sub(r"\s+", " ", normalized).strip().casefold()

    def _log_fallback(self, message: str) -> None:
        if self.event_logger is not None:
            self.event_logger(f"location fallback: {message}")
