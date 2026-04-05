from __future__ import annotations

import json
import re

from .gemma_client import GemmaClient


class GemmaQueryRewriter:
    SYSTEM_PROMPT = """You rewrite academic retrieval queries.

Return JSON only with this schema:
{"rewrites":["rewrite one","rewrite two"]}

Rules:
- Return concise retrieval-oriented phrases, not sentences.
- Preserve the user's intent.
- Avoid adding facts not present in the question.
- Return exactly the requested number of rewrites.
"""

    JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

    def __init__(self, gemma_client: GemmaClient) -> None:
        self.gemma_client = gemma_client

    def rewrite(self, query: str, *, rewrite_count: int) -> list[str]:
        response = self.gemma_client.generate(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"User query: {query}\n"
                        f"Number of rewrites required: {rewrite_count}\n"
                        "Return JSON only."
                    ),
                },
            ],
            max_tokens=220,
        )
        payload = self._parse_json_object(response)
        rewrites = payload.get("rewrites")
        if not isinstance(rewrites, list):
            raise RuntimeError("Gemma query rewrite response did not contain a rewrites list.")

        cleaned = [str(item).strip() for item in rewrites if str(item).strip()]
        if len(cleaned) < rewrite_count:
            raise RuntimeError(
                f"Gemma returned {len(cleaned)} rewrites, expected at least {rewrite_count}."
            )
        return cleaned[:rewrite_count]

    def _parse_json_object(self, raw_response: str) -> dict[str, object]:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if "\n" in cleaned:
                cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = self.JSON_PATTERN.search(raw_response)
            if match is None:
                raise RuntimeError(f"Failed to parse rewrite response as JSON: {raw_response}")
            return json.loads(match.group(0))
