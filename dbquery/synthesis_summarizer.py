from __future__ import annotations

from .gemma_client import GemmaClient
from .models import BatchSummary


class GemmaSynthesisSummarizer:
    SYSTEM_PROMPT = """You synthesize previously generated academic evidence summaries.

Rules:
- Use only the supplied batch summaries.
- Preserve the substance of the evidence; do not add new claims.
- Keep citations that already appear in the supplied summaries.
- Write one concise synthesis in 1 to 3 short paragraphs.
- If the batch summaries disagree or emphasize different findings, note that.
"""

    def __init__(self, gemma_client: GemmaClient) -> None:
        self.gemma_client = gemma_client

    def synthesize(self, summaries: list[BatchSummary], *, question: str) -> str | None:
        if not summaries:
            return None

        rendered_summaries = "\n\n".join(
            [
                "\n".join(
                    [
                        f"Batch {summary.batch_index}",
                        f"Citation labels: {', '.join(summary.citation_labels)}",
                        summary.summary_text,
                    ]
                )
                for summary in summaries
            ]
        )
        return self.gemma_client.generate(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"Question: {question}",
                            "Synthesize the following batch summaries into one final summary.",
                            rendered_summaries,
                        ]
                    ),
                },
            ],
            max_tokens=520,
        ).strip()
