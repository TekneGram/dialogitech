from __future__ import annotations

from typing import Any, Callable

from .doi_metadata_client import DoiMetadataClient, DoiMetadataError
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
      doi_metadata_client: DoiMetadataClient | None = None,
      event_logger: Callable[[str], None] | None = None,
  ) -> None:
    self.evidence = evidence
    self.component_extractor = component_extractor
    self.missing_values_handler = missing_values_handler
    self.doi_metadata_client = doi_metadata_client
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

    doi_metadata = self._lookup_doi_metadata(decisions, unresolved)
    if doi_metadata:
      doi_json = dict(initial_json)
      doi_json["doi_metadata"] = doi_metadata
      for component in unresolved:
        doi_decision = self.component_extractor(component, doi_json)
        decisions[component] = self.missing_values_handler.merge(
            decisions[component],
            doi_decision,
            component,
        )

      unresolved = [
          component
          for component, decision in decisions.items()
          if self.missing_values_handler.needs_more_evidence(decision, component)
      ]

    if unresolved:
      additional_pages = self.evidence.select_available_pages(document, [2, 3])
      if additional_pages:
        expanded_json = self.evidence.compact_pages(initial_pages + additional_pages)
        if doi_metadata:
          expanded_json["doi_metadata"] = doi_metadata
        for component in unresolved:
          additional_decision = self.component_extractor(component, expanded_json)
          decisions[component] = self.missing_values_handler.merge(
              decisions[component],
              additional_decision,
              component,
          )

        unresolved = [
            component
            for component, decision in decisions.items()
            if self.missing_values_handler.needs_more_evidence(decision, component)
        ]

    if unresolved and allow_manual and "journal" in decisions:
      if not self._has_usable_doi(decisions["journal"]):
        manual_doi = self.missing_values_handler.prompt_for_doi()
        decisions["journal"] = self._with_doi(decisions["journal"], manual_doi)

        if manual_doi != "unknown":
          doi_metadata = self._lookup_doi_metadata(decisions, unresolved)
          if doi_metadata:
            doi_json = dict(initial_json)
            doi_json["doi_metadata"] = doi_metadata
            for component in unresolved:
              doi_decision = self.component_extractor(component, doi_json)
              decisions[component] = self.missing_values_handler.merge(
                  decisions[component],
                  doi_decision,
                  component,
              )

            unresolved = [
                component
                for component, decision in decisions.items()
                if self.missing_values_handler.needs_more_evidence(decision, component)
            ]

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

  def _lookup_doi_metadata(
      self,
      decisions: dict[str, MetadataDecision],
      unresolved: list[str],
  ) -> dict[str, Any] | None:
    if self.doi_metadata_client is None or not unresolved:
      return None

    journal = decisions.get("journal")
    journal_value = journal.value if journal is not None else None
    doi = journal_value.get("doi") if isinstance(journal_value, dict) else None
    if not isinstance(doi, str) or not doi.strip() or doi.strip().lower() == "unknown":
      return None

    try:
      metadata = self.doi_metadata_client.lookup(doi)
    except (DoiMetadataError, ValueError) as exc:
      self.event_logger(f"DOI metadata lookup failed for {doi}: {exc}")
      return None

    self.event_logger(f"Loaded DOI metadata for {doi}.")
    return metadata

  def _has_usable_doi(self, decision: MetadataDecision) -> bool:
    value = decision.value
    doi = value.get("doi") if isinstance(value, dict) else None
    return isinstance(doi, str) and bool(doi.strip()) and doi.strip().lower() != "unknown"

  def _with_doi(
      self,
      decision: MetadataDecision,
      doi: str,
  ) -> MetadataDecision:
    value = dict(decision.value) if isinstance(decision.value, dict) else {}
    value["doi"] = doi
    return MetadataDecision(
        value=value,
        confidence=decision.confidence,
        reason=decision.reason,
        source_pages=decision.source_pages,
        provenance={
            **decision.provenance,
            "doi": ["manual"],
        },
    )
