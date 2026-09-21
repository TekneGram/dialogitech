from __future__ import annotations

from typing import Any, Callable

from .metadata_evidence import MetadataEvidence
from .metadata_models import MetadataDecision
from .missing_values_handler import MetaDataMissingValuesHandler


class MetadataExtractionFlow:
  """Coordinate bounded page expansion and manual metadata fallback."""

  def __init__(
      self,
      *,
      evidence: MetadataEvidence,
      component_extractor: Callable[[str, dict[str, Any]], MetadataDecision],
      missing_values_handler: MetaDataMissingValuesHandler,
      event_logger: Callable[[str], None] | None = None,
  ) -> None:
    self.evidence = evidence
    self.component_extractor = component_extractor
    self.missing_values_handler = missing_values_handler
    self.event_logger = event_logger or (lambda message: None)

  def extract(
      self,
      document: dict[str, Any],
      *,
      components: tuple[str, ...],
      allow_manual: bool,
  ) -> dict[str, MetadataDecision]:
    initial_pages = self.evidence.select_available_pages(document, [0, 1])
    if not initial_pages:
      raise RuntimeError("Marker document does not contain pages 0 or 1.")

    initial_json = self.evidence.compact_pages(initial_pages)
    decisions = {
        component: self.component_extractor(component, initial_json)
        for component in components
    }

    unresolved = [
        component
        for component, decision in decisions.items()
        if self.missing_values_handler.needs_more_evidence(decision, component)
    ]

    if unresolved:
      additional_pages = self.evidence.select_available_pages(document, [2, 3])
      if additional_pages:
        expanded_json = self.evidence.compact_pages(initial_pages + additional_pages)
        for component in unresolved:
          additional_decision = self.component_extractor(component, expanded_json)
          decisions[component] = self.missing_values_handler.merge(
              decisions[component],
              additional_decision,
              component,
          )

    for component, decision in list(decisions.items()):
      if not self.missing_values_handler.needs_more_evidence(decision, component):
        continue

      missing = self.missing_values_handler.missing_fields(decision, component)
      message = (
          f"Metadata component {component!r} is missing values after pages 0–3: "
          f"{', '.join(missing)}"
      )
      if not allow_manual:
        raise RuntimeError(message)

      self.event_logger(message)
      decisions[component] = self.missing_values_handler.prompt_for_manual_entry(
          decision,
          component,
      )

    return decisions
