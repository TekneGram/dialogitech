from __future__ import annotations

import json
import re
from typing import Any

from .section_type_models import ClassificationConfidence, LLMDecision
from .section_taxonomy import SectionLabel


class SectionClassificationResponseParser:
    JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

    def __init__(self, *, confidence_levels: tuple[ClassificationConfidence, ...]) -> None:
        self.confidence_levels = confidence_levels

    def parse_decision(
        self,
        raw_response: str,
        *,
        allow_request_context: bool,
        allowed_labels: tuple[SectionLabel, ...],
    ) -> LLMDecision:
        try:
            payload = self.parse_json_object(raw_response)
        except ValueError as exc:
            raise RuntimeError(f"Failed to parse model response as JSON: {raw_response}") from exc

        action = self.normalize_optional_string(payload.get("action"))
        label = self.normalize_optional_string(payload.get("label"))
        confidence = self.normalize_optional_string(payload.get("confidence"))
        if action in allowed_labels:
            label = label or action
            action = "classify"
        if action not in {"classify", "request_context"}:
            raise RuntimeError(f"Model returned unsupported action: {action!r}")

        reason = str(payload.get("reason", "")).strip() or "no reason provided"
        if action == "request_context":
            if not allow_request_context:
                raise RuntimeError("Model requested context after the single allowed retrieval round.")
            return LLMDecision(action="request_context", reason=reason)

        if label not in allowed_labels:
            raise RuntimeError(f"Model returned unsupported label: {label!r}")
        if confidence not in self.confidence_levels:
            raise RuntimeError(f"Model returned unsupported confidence: {confidence!r}")
        return LLMDecision(
            action="classify",
            label=label,
            confidence=confidence,
            reason=reason,
        )

    def parse_json_object(self, raw_response: str) -> dict[str, Any]:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if "\n" in cleaned:
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            match = self.JSON_PATTERN.search(raw_response)
            if not match:
                return self.recover_json_like_payload(raw_response)
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return self.recover_json_like_payload(match.group(0))
        if not isinstance(payload, dict):
            raise ValueError("Model response must be a JSON object.")
        return payload

    def recover_json_like_payload(self, raw_response: str) -> dict[str, Any]:
        action = self.extract_json_string_field(raw_response, "action")
        reason = self.extract_json_string_field(raw_response, "reason")
        label = self.extract_json_string_field(raw_response, "label")
        confidence = self.extract_json_string_field(raw_response, "confidence")
        if action is None:
            raise ValueError("Could not recover action field from model response.")

        payload: dict[str, Any] = {"action": action}
        if label is not None:
            payload["label"] = label
        if confidence is not None:
            payload["confidence"] = confidence
        if reason is not None:
            payload["reason"] = reason
        return payload

    def extract_json_string_field(self, raw_response: str, field_name: str) -> str | None:
        field_token = f'"{field_name}"'
        field_index = raw_response.find(field_token)
        if field_index < 0:
            return None
        colon_index = raw_response.find(":", field_index + len(field_token))
        if colon_index < 0:
            return None
        value_start = raw_response.find('"', colon_index)
        if value_start < 0:
            return None

        scan_index = value_start + 1
        while scan_index < len(raw_response):
            char = raw_response[scan_index]
            if char == '"' and raw_response[scan_index - 1] != "\\":
                remainder = raw_response[scan_index + 1 :]
                if re.match(r"\s*(?:,|\})", remainder):
                    return raw_response[value_start + 1 : scan_index]
            scan_index += 1
        return raw_response[value_start + 1 :].rstrip().rstrip("}").strip()

    @staticmethod
    def normalize_optional_string(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return normalized or None
