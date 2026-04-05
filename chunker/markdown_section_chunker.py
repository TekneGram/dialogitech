from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(slots=True)
class MarkdownHeading:
    title: str
    level: int
    raw_heading: str


@dataclass(slots=True)
class SectionChunk:
    title: str
    heading_level: int
    chunk_index: int
    text: str
    word_count: int


@dataclass(slots=True)
class HeadingSplit:
    title: str
    heading_level: int
    raw_heading: str
    content: str
    chunks: list[SectionChunk] = field(default_factory=list)


class MarkdownSectionChunker:
    HEADING_PATTERN = re.compile(r"^(#{1,2})\s+(.+?)\s*$", re.MULTILINE)
    WORD_PATTERN = re.compile(r"\S+")
    SENTENCE_END_PATTERN = re.compile(r'[.!?][\'")\]]*(?:\s+|$)')

    def __init__(self, min_words: int = 200, overlap_words: int = 50) -> None:
        self.min_words = min_words
        self.overlap_words = overlap_words

    def extract_headings(self, markdown: str) -> list[MarkdownHeading]:
        headings: list[MarkdownHeading] = []
        for match in self.HEADING_PATTERN.finditer(markdown):
            level = len(match.group(1))
            title = self._normalize_heading_title(match.group(2))
            headings.append(
                MarkdownHeading(
                    title=title,
                    level=level,
                    raw_heading=match.group(0),
                )
            )
        return headings

    def split_by_headings(self, markdown: str) -> list[HeadingSplit]:
        matches = list(self.HEADING_PATTERN.finditer(markdown))
        heading_splits: list[HeadingSplit] = []

        for index, match in enumerate(matches):
            next_start = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
            content = markdown[match.end():next_start].strip()
            heading_splits.append(
                HeadingSplit(
                    title=self._normalize_heading_title(match.group(2)),
                    heading_level=len(match.group(1)),
                    raw_heading=match.group(0),
                    content=content,
                )
            )

        return heading_splits

    def chunk_heading_splits(self, heading_splits: list[HeadingSplit]) -> list[HeadingSplit]:
        chunked_splits: list[HeadingSplit] = []

        for heading_split in heading_splits:
            chunked_splits.append(
                HeadingSplit(
                    title=heading_split.title,
                    heading_level=heading_split.heading_level,
                    raw_heading=heading_split.raw_heading,
                    content=heading_split.content,
                    chunks=self._build_chunks_for_section(
                        title=heading_split.title,
                        heading_level=heading_split.heading_level,
                        content=heading_split.content,
                    ),
                )
            )

        return chunked_splits

    def process(self, markdown: str) -> list[HeadingSplit]:
        heading_splits = self.split_by_headings(markdown)
        return self.chunk_heading_splits(heading_splits)

    def _build_chunks_for_section(
        self,
        title: str,
        heading_level: int,
        content: str,
    ) -> list[SectionChunk]:
        stripped_content = content.strip()
        if not stripped_content:
            return []

        words = list(self.WORD_PATTERN.finditer(stripped_content))
        if not words:
            return []

        chunks: list[SectionChunk] = []
        total_words = len(words)
        start_word_index = 0

        while start_word_index < total_words:
            remaining_words = total_words - start_word_index
            if chunks and remaining_words < self.min_words:
                if remaining_words < 100:
                    start_word_index = max(0, start_word_index - self.overlap_words)
                chunks.append(
                    self._make_chunk(
                        title=title,
                        heading_level=heading_level,
                        content=stripped_content,
                        words=words,
                        start_word_index=start_word_index,
                        end_char_index=len(stripped_content),
                        chunk_index=len(chunks),
                    )
                )
                break

            if remaining_words <= self.min_words:
                chunks.append(
                    self._make_chunk(
                        title=title,
                        heading_level=heading_level,
                        content=stripped_content,
                        words=words,
                        start_word_index=start_word_index,
                        end_char_index=len(stripped_content),
                        chunk_index=len(chunks),
                    )
                )
                break

            target_word_index = min(total_words - 1, start_word_index + self.min_words - 1)
            target_end_char = words[target_word_index].end()
            end_char_index = self._find_sentence_end(stripped_content, target_end_char)
            end_word_index = self._find_last_word_index_within_char(words, end_char_index)

            chunks.append(
                self._make_chunk(
                    title=title,
                    heading_level=heading_level,
                    content=stripped_content,
                    words=words,
                    start_word_index=start_word_index,
                    end_char_index=end_char_index,
                    chunk_index=len(chunks),
                )
            )

            if end_word_index >= total_words - 1:
                break

            next_start = self._next_chunk_start_word_index(
                content=stripped_content,
                words=words,
                current_start_word_index=start_word_index,
                end_word_index=end_word_index,
            )
            if next_start <= start_word_index:
                next_start = min(total_words, start_word_index + self.min_words)
            start_word_index = next_start

        return chunks

    def _make_chunk(
        self,
        title: str,
        heading_level: int,
        content: str,
        words: list[re.Match[str]],
        start_word_index: int,
        end_char_index: int,
        chunk_index: int,
    ) -> SectionChunk:
        start_char_index = words[start_word_index].start()
        text = content[start_char_index:end_char_index].strip()
        word_count = len(self.WORD_PATTERN.findall(text))
        return SectionChunk(
            title=title,
            heading_level=heading_level,
            chunk_index=chunk_index,
            text=text,
            word_count=word_count,
        )

    def _find_sentence_end(self, content: str, min_end_char: int) -> int:
        for match in self.SENTENCE_END_PATTERN.finditer(content):
            if match.end() >= min_end_char:
                return match.end()
        return len(content)

    def _find_last_word_index_within_char(
        self,
        words: list[re.Match[str]],
        end_char_index: int,
    ) -> int:
        last_index = 0
        for index, match in enumerate(words):
            if match.end() <= end_char_index:
                last_index = index
            else:
                break
        return last_index

    def _next_chunk_start_word_index(
        self,
        content: str,
        words: list[re.Match[str]],
        current_start_word_index: int,
        end_word_index: int,
    ) -> int:
        tentative_start = max(0, end_word_index + 1 - self.overlap_words)
        if tentative_start <= current_start_word_index:
            return tentative_start

        sentence_start_char = self._find_sentence_start_before_char(
            content=content,
            char_index=words[tentative_start].start(),
        )
        if sentence_start_char is None:
            return tentative_start

        adjusted_start = self._find_first_word_index_at_or_after_char(words, sentence_start_char)
        return adjusted_start if adjusted_start > current_start_word_index else tentative_start

    def _find_sentence_start_before_char(self, content: str, char_index: int) -> int | None:
        sentence_start = 0
        found_boundary = False

        for match in self.SENTENCE_END_PATTERN.finditer(content):
            if match.end() >= char_index:
                break
            sentence_start = match.end()
            found_boundary = True

        return sentence_start if found_boundary else None

    def _find_first_word_index_at_or_after_char(
        self,
        words: list[re.Match[str]],
        char_index: int,
    ) -> int:
        for index, match in enumerate(words):
            if match.start() >= char_index:
                return index
        return len(words) - 1

    def _normalize_heading_title(self, title: str) -> str:
        cleaned = title.strip()
        cleaned = re.sub(r"[*_`]+", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()
