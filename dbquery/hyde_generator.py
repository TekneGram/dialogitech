from __future__ import annotations

from .gemma_client import GemmaClient


class GemmaHyDEGenerator:
    SYSTEM_PROMPT = """You generate a short hypothetical answer passage for retrieval.

Rules:
- Write one compact paragraph of 80 to 140 words.
- Focus on the type of evidence likely to appear in academic papers.
- Do not mention that the answer is hypothetical.
- Do not use bullet points.
"""

    def __init__(self, gemma_client: GemmaClient) -> None:
        self.gemma_client = gemma_client

    def generate(self, query: str) -> str:
        response = self.gemma_client.generate(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}"},
            ],
            max_tokens=220,
        )
        return response.strip()
