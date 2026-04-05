from __future__ import annotations

from .citation_formatter import CitationFormatter
from .gemma_client import GemmaClient
from .models import BatchSummary, SummaryBatch


class GemmaBatchSummarizer:
    SYSTEM_PROMPT = """You summarize retrieved academic evidence.

Rules:
- Use only the supplied chunk text and metadata.
- Cite claims using the supplied author-year labels exactly, for example (Smith, 2024) or (Smith et al., 2024).
- Do not invent citations.
- If sources disagree, say so.
- Write a compact synthesis in 1 to 3 short paragraphs.
"""

    def __init__(
        self,
        gemma_client: GemmaClient,
        citation_formatter: CitationFormatter | None = None,
    ) -> None:
        self.gemma_client = gemma_client
        self.citation_formatter = citation_formatter or CitationFormatter()

    def summarize_batch(self, batch: SummaryBatch, *, question: str) -> BatchSummary:
        prompt = "\n\n".join(
            [
                f"Question: {question}",
                "Summarize the following evidence and cite claims with the provided author-year labels.",
                self.citation_formatter.format_batch(batch),
            ]
        )
        summary = self.gemma_client.generate(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=520,
        ).strip()
        return BatchSummary(
            batch_index=batch.batch_index,
            summary_text=summary,
            citation_labels=[self.citation_formatter.citation_label(chunk) for chunk in batch.chunks],
            chunk_ids=[chunk.chunk_id for chunk in batch.chunks],
        )
