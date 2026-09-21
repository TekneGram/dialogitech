import json
from typing import Any

from .metadata_models import MetadataDecision


class MetadataResponseValidator:

  def __init__(self) -> None:
    return

  # Response parsing and validation
  # Handle fenched JSON, extra explanatory text, malformed JSON recovery,
  # missing value
  # invalid confidence values
  # wrong data types, such as a string instead of an author list.
  # Invalid responses should raise an error, log the error, and call a correction prompt to the LLM
  # Continuous errors (say 3 consecutive fails) should result prompting the user for manual entry
  def parse_json_response(self, raw_response: str) -> dict[str, Any]:
    cleaned = raw_response.strip()

    if cleaned.startswith("```"):
      lines = cleaned.splitlines()

      if lines and lines[0].startswith("```"):
        lines = lines[1:]

      if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

      cleaned = "\n".join(lines).strip()

    try:
      payload = json.loads(cleaned)
    except json.JSONDecodeError:
      start = cleaned.find("{")
      if start == -1:
        raise ValueError("No JSON object found in Gemma response.")

      try:
        payload, _ = json.JSONDecoder().raw_decode(cleaned[start:])
      except json.JSONDecodeError as exc:
        raise ValueError(
          "Could not parse Gemma response as JSON."
        ) from exc

    if not isinstance(payload, dict):
      raise ValueError("Gemma response must be a JSON object.")

    return payload

  def validate_decision(
      self,
      payload: dict[str, Any],
      component: str,
  ) -> MetadataDecision:
    confidence = str(payload.get("confidence", "")).lower()

    if confidence not in {"high", "medium", "low"}:
      raise ValueError(f"Invalid confidence value: {confidence!r}")

    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
      raise ValueError("Gemma response requires a non-empty reason.")

    value = payload.get("value")

    if component == "title":
      if value is not None and (
        not isinstance(value, str) or not value.strip()
      ):
        raise ValueError("Title must be a string or null.")

    elif component == "authors":
      if value is not None and (
        not isinstance(value, list)
        or not all(
          isinstance(author, str) and author.strip()
          for author in value
        )
      ):
        raise ValueError(
          "Authors must be a list of strings or null."
        )

    elif component == "journal":
      if value is not None and not isinstance(value, dict):
        raise ValueError("Journal must be an object or null.")

    else:
      raise ValueError(f"Unsupported metadata component: {component}")

    return MetadataDecision(
      value=value,
      confidence=confidence,
      reason=reason.strip(),
      source_pages=[],
    )
