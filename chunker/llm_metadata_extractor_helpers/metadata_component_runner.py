from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from .metadata_models import MetadataDecision
from .response_parsing_validation import MetadataResponseValidator


class MetadataComponentRunner:
  """Run one metadata prompt and validate its structured response."""

  def __init__(
      self,
      *,
      generate: Callable[[list[dict[str, str]]], str],
      prompt_builders: dict[str, Callable[[dict[str, Any]], str]],
      validator: MetadataResponseValidator | None = None,
  ) -> None:
    self.generate = generate
    self.prompt_builders = prompt_builders
    self.validator = validator or MetadataResponseValidator()

  def extract(
      self,
      component: str,
      compact_json: dict[str, Any],
  ) -> MetadataDecision:
    prompt_builder = self.prompt_builders.get(component)
    if prompt_builder is None:
      raise ValueError(f"Unsupported metadata component: {component}")

    raw_response = self.generate([
        {"role": "user", "content": prompt_builder(compact_json)}
    ])
    payload = self.validator.parse_json_response(raw_response)
    decision = self.validator.validate_decision(payload, component)
    source_pages = [
        int(page["page_number"])
        for page in compact_json.get("pages", [])
        if isinstance(page, dict) and isinstance(page.get("page_number"), int)
    ]
    evidence_sources = ["gemma"]
    if "doi_metadata" in compact_json:
      evidence_sources.append("crossref")

    if component == "journal" and isinstance(decision.value, dict):
      provenance = {
          field: list(evidence_sources)
          for field, value in decision.value.items()
          if value is not None
      }
    else:
      provenance = {"value": evidence_sources} if decision.value is not None else {}

    return replace(
        decision,
        source_pages=source_pages,
        provenance=provenance,
    )
