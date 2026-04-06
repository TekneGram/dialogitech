from __future__ import annotations

import ast
import json
import re


class JsonResponseParser:
    def parse_object(self, response: str) -> dict[str, object]:
        candidate = self._extract_candidate(response)
        for parser in (self._parse_json, self._parse_python_literal):
            try:
                payload = parser(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        raise ValueError("Expected a JSON object.")

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

    def _parse_json(self, candidate: str) -> object:
        return json.loads(candidate)

    def _parse_python_literal(self, candidate: str) -> object:
        normalized = self._normalize_for_python_literal(candidate)
        return ast.literal_eval(normalized)

    def _normalize_for_python_literal(self, candidate: str) -> str:
        normalized = re.sub(r",(\s*[}\]])", r"\1", candidate)
        normalized = re.sub(r"\btrue\b", "True", normalized)
        normalized = re.sub(r"\bfalse\b", "False", normalized)
        normalized = re.sub(r"\bnull\b", "None", normalized)
        return normalized


__all__ = ["JsonResponseParser"]
