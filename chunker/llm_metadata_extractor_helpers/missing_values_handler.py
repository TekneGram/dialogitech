from __future__ import annotations

from typing import Any, Callable

from .metadata_models import MetadataDecision


class MetaDataMissingValuesHandler:
  JOURNAL_FIELDS = ("name", "volume", "issue", "year", "doi", "issn")

  def __init__(self, input_fn: Callable[[str], str] = input) -> None:
    self.input_fn = input_fn

  def missing_fields(
      self,
      decision: MetadataDecision,
      component: str,
  ) -> list[str]:
    if component == "title":
      return ["value"] if decision.value is None else []

    if component == "authors":
      return ["value"] if not decision.value else []

    if component == "journal":
      if decision.value is None:
        return list(self.JOURNAL_FIELDS)
      if not isinstance(decision.value, dict):
        return list(self.JOURNAL_FIELDS)
      return [
          field
          for field in self.JOURNAL_FIELDS
          if decision.value.get(field) is None
      ]

    raise ValueError(f"Unsupported metadata component: {component}")

  def needs_more_evidence(
      self,
      decision: MetadataDecision,
      component: str,
  ) -> bool:
    return bool(self.missing_fields(decision, component))

  def merge(
      self,
      previous: MetadataDecision,
      current: MetadataDecision,
      component: str,
  ) -> MetadataDecision:
    if component == "journal":
      previous_value = previous.value if isinstance(previous.value, dict) else {}
      current_value = current.value if isinstance(current.value, dict) else {}
      merged_value = dict(previous_value)
      merged_value.update({
          key: value
          for key, value in current_value.items()
          if value is not None
      })
      value: Any | None = merged_value or None
    else:
      value = current.value if current.value is not None else previous.value

    return MetadataDecision(
        value=value,
        confidence=current.confidence,
        reason=current.reason,
        source_pages=sorted(set(previous.source_pages + current.source_pages)),
    )

  def prompt_for_manual_entry(
      self,
      decision: MetadataDecision,
      component: str,
  ) -> MetadataDecision:
    if component == "title":
      value = self._prompt_non_empty("Enter title")
    elif component == "authors":
      authors_text = self._prompt_non_empty(
          "Enter authors as a semicolon-separated list"
      )
      value = [author.strip() for author in authors_text.split(";") if author.strip()]
    elif component == "journal":
      value = dict(decision.value or {})
      for field in self.JOURNAL_FIELDS:
        if value.get(field) is None:
          value[field] = self._prompt_non_empty(f"Enter journal {field}")
    else:
      raise ValueError(f"Unsupported metadata component: {component}")

    return MetadataDecision(
        value=value,
        confidence="high",
        reason="Completed by manual user input.",
        source_pages=decision.source_pages,
    )

  def _prompt_non_empty(self, prompt: str) -> str:
    while True:
      value = self.input_fn(f"{prompt}: ").strip()
      if value:
        return value
      print("Value cannot be empty.", flush=True)
