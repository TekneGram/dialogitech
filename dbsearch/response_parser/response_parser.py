from __future__ import annotations

import json
from typing import Any

class ResponseParser:
  def __init__(self) -> None:
    pass

  def question_search_response(self, response: str) -> dict[str, str]:
    """Parse and validate one question-search LLM response.

    The accepted JSON shape is exactly:

        {"answer_provided": "fully|partially|never", "summary": "..."}

    Markdown code fences and surrounding explanatory text are tolerated, but
    the extracted JSON object and its fields must still be valid.
    """
    if not isinstance(response, str) or not response.strip():
      raise ValueError("The LLM response must be a non-empty string.")

    response_text = response.strip()
    payload: Any

    try:
      payload = json.loads(response_text)
    except json.JSONDecodeError:
      decoder = json.JSONDecoder()
      payload = None
      for character_index, character in enumerate(response_text):
        if character != "{":
          continue
        try:
          payload, _ = decoder.raw_decode(response_text[character_index:])
          break
        except json.JSONDecodeError:
          continue

      if payload is None:
        raise ValueError("The LLM response does not contain a valid JSON object.")

    if not isinstance(payload, dict):
      raise ValueError("The LLM response must be a JSON object.")

    expected_keys = {"answer_provided", "summary"}
    actual_keys = set(payload)
    if actual_keys != expected_keys:
      missing_keys = expected_keys - actual_keys
      extra_keys = actual_keys - expected_keys
      details: list[str] = []
      if missing_keys:
        details.append(f"missing keys: {sorted(missing_keys)}")
      if extra_keys:
        details.append(f"unexpected keys: {sorted(extra_keys)}")
      raise ValueError("Invalid question-search JSON format (" + "; ".join(details) + ").")

    answer_provided = payload["answer_provided"]
    summary = payload["summary"]

    if answer_provided not in {"fully", "partially", "never"}:
      raise ValueError(
        "answer_provided must be one of: 'fully', 'partially', or 'never'."
      )
    if not isinstance(summary, str) or not summary.strip():
      raise ValueError("summary must be a non-empty string.")

    return {
      "answer_provided": answer_provided,
      "summary": summary.strip(),
    }
