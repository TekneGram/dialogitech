from __future__ import annotations

from .models import FusedChunkResult, SummaryBatch


class CitationFormatter:
    def citation_label(self, chunk: FusedChunkResult) -> str:
        authors = chunk.authors
        year = str(chunk.year) if chunk.year is not None else "n.d."
        if not authors:
            author_label = chunk.paper_id
        elif len(authors) == 1:
            author_label = self._surname(authors[0])
        else:
            author_label = f"{self._surname(authors[0])} et al."
        return f"{author_label}, {year}"

    def format_batch(self, batch: SummaryBatch) -> str:
        blocks: list[str] = []
        for chunk in batch.chunks:
            citation = self.citation_label(chunk)
            blocks.append(
                "\n".join(
                    [
                        f"Citation: ({citation})",
                        f"Paper title: {chunk.paper_title}",
                        f"Section title: {chunk.section_title}",
                        f"Chunk index: {chunk.chunk_index}",
                        "Chunk text:",
                        chunk.text,
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _surname(self, author_name: str) -> str:
        parts = [part.strip(",") for part in author_name.split() if part.strip(",")]
        if not parts:
            return author_name
        return parts[-1]
