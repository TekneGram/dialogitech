from __future__ import annotations

from dbquery.gemma_client import GemmaClient
from dbquery.models import BatchSummary

from .models import StateUpdateClaim
from .utils import JsonResponseParser, ResearchStatePromptRenderer


class GemmaBatchSummaryClaimsExtractor:
    SYSTEM_PROMPT = """You extract compact research claims from one evidence summary.

Rules:
- Use only the supplied summary text.
- Return JSON only in this shape:
{
  "claims_added": [
    {
      "text": "string",
      "status": "supported|tentative|insufficient_evidence",
      "confidence": "high|medium|low"
    }
  ]
}
- Each claim must be one or two sentences only.
- Return at most 2 claims.
- If no claim should be added, return {"claims_added": []}.
"""

    def __init__(
        self,
        gemma_client: GemmaClient,
        prompt_renderer: ResearchStatePromptRenderer | None = None,
        json_parser: JsonResponseParser | None = None,
    ) -> None:
        self.gemma_client = gemma_client
        self.prompt_renderer = prompt_renderer or ResearchStatePromptRenderer()
        self.json_parser = json_parser or JsonResponseParser()

    def extract(
        self,
        *,
        user_question: str,
        summary: BatchSummary,
    ) -> list[StateUpdateClaim]:
        prompt = self.prompt_renderer.render_batch_summary_claims_prompt(
            user_question=user_question,
            summary=summary,
        )
        response = self.gemma_client.generate(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=260,
        )
        payload = self._parse_payload(response)
        claims: list[StateUpdateClaim] = []
        for item in payload.get("claims_added", []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            claims.append(
                StateUpdateClaim(
                    text=text,
                    status=str(item.get("status", "supported")).strip(),
                    confidence=str(item.get("confidence", "medium")).strip(),
                    evidence_chunk_ids=list(summary.chunk_ids),
                )
            )
        return claims

    def _parse_payload(self, response: str) -> dict[str, object]:
        try:
            return self.json_parser.parse_object(response)
        except ValueError:
            return {"claims_added": []}
