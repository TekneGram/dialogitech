from __future__ import annotations

from dbquery.gemma_client import GemmaClient

from .models import StateUpdateFollowup
from .utils import JsonResponseParser, ResearchStatePromptRenderer


class GemmaFollowupSuggester:
    SYSTEM_PROMPT = """You suggest user follow-up research questions.

Rules:
- Use only the supplied synthesized summary and current user question.
- Return JSON only in this shape:
{
  "followups_added": [
    {
      "text": "string"
    }
  ]
}
- Suggest 2 to 4 concise follow-up questions.
- These are suggestions for the user, not state-machine tasks.
- If none are appropriate, return {"followups_added": []}.
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

    def suggest(
        self,
        *,
        user_question: str,
        synthesized_summary: str,
    ) -> list[StateUpdateFollowup]:
        prompt = self.prompt_renderer.render_followup_suggestions_prompt(
            user_question=user_question,
            synthesized_summary=synthesized_summary,
        )
        response = self.gemma_client.generate(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=220,
        )
        payload = self._parse_payload(response)
        suggestions: list[StateUpdateFollowup] = []
        for item in payload.get("followups_added", []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            suggestions.append(StateUpdateFollowup(text=text))
        return suggestions

    def _parse_payload(self, response: str) -> dict[str, object]:
        try:
            return self.json_parser.parse_object(response)
        except ValueError:
            return {"followups_added": []}
