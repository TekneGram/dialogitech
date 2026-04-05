from __future__ import annotations

import json
import re


class JsonResponseParser:
    def parse_object(self, response: str) -> dict[str, object]:
        candidate = self._extract_candidate(response)
        payload = json.loads(candidate)
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        return payload

    def _extract_candidate(self, response: str) -> str:
        stripped = response.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped

        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response, flags=re.DOTALL)
        if fenced_match is not None:
            return fenced_match.group(1).strip()

        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            return response[start : end + 1].strip()

        raise ValueError("No JSON object found in model response.")
