from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from .doi_candidate_extractor import DoiCandidateExtractor
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
      doi_candidate_extractor: DoiCandidateExtractor | None = None,
      event_logger: Callable[[str], None] | None = None,
  ) -> None:
    self.evidence = evidence
    self.component_extractor = component_extractor
    self.missing_values_handler = missing_values_handler
    self.doi_metadata_client = doi_metadata_client
    self.doi_candidate_extractor = doi_candidate_extractor or DoiCandidateExtractor()
    self.event_logger = event_logger or (lambda message: None)
    self._doi_confirmation_rejected = False

  def extract(
      self,
      document: dict[str, Any],
      *,
      components: tuple[str, ...],
      allow_manual: bool,
  ) -> dict[str, MetadataDecision]:
    self._doi_confirmation_rejected = False
    initial_pages = self.evidence.select_available_pages(document, [0, 1])
    if not initial_pages:
      raise RuntimeError("Marker document does not contain pages 0 or 1.")

    initial_json = self.evidence.compact_pages(initial_pages)
    decisions = {
        component: self.component_extractor(component, initial_json)
        for component in components
    }
    self._validate_marker_doi(decisions, initial_json)

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
          if component == "journal":
            additional_decision = self._validated_journal_decision(
                additional_decision,
                expanded_json,
            )
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

        # Page expansion may reveal a DOI that was not available, or was
        # rejected, during the initial pages. Try Crossref again before asking
        # the user for manual metadata.
        if unresolved:
          expanded_doi_metadata = self._lookup_doi_metadata(decisions, unresolved)
          if expanded_doi_metadata:
            expanded_doi_json = dict(expanded_json)
            expanded_doi_json["doi_metadata"] = expanded_doi_metadata
            for component in unresolved:
              doi_decision = self.component_extractor(component, expanded_doi_json)
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
    if self.doi_metadata_client is None or not unresolved or self._doi_confirmation_rejected:
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

    if not self._has_comparison_evidence(decisions):
      if not self.missing_values_handler.confirm_crossref_metadata(metadata):
        self._doi_confirmation_rejected = True
        self.event_logger(
            f"Rejected DOI metadata for {doi}: user confirmation was declined."
        )
        decisions["journal"] = self._mark_doi_unresolved(journal)
        return None

    if not self._doi_matches_paper(metadata, decisions):
      self.event_logger(
          f"Rejected DOI metadata for {doi}: title/author validation failed."
      )
      decisions["journal"] = self._mark_doi_unresolved(journal)
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
      *,
      provenance_source: str = "manual",
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
            "doi": [provenance_source],
        },
    )

  def _validate_marker_doi(
      self,
      decisions: dict[str, MetadataDecision],
      compact_json: dict[str, Any],
  ) -> None:
    journal = decisions.get("journal")
    if journal is None:
      return
    decisions["journal"] = self._validated_journal_decision(journal, compact_json)

  def _validated_journal_decision(
      self,
      decision: MetadataDecision,
      compact_json: dict[str, Any],
  ) -> MetadataDecision:
    if not isinstance(decision.value, dict):
      return decision

    candidates = self.doi_candidate_extractor.extract(compact_json)
    if not candidates:
      return decision

    candidate_dois = {candidate.doi.lower() for candidate in candidates if candidate.score > 0}
    value = dict(decision.value)
    doi = value.get("doi")

    if isinstance(doi, str) and doi.strip() and doi.strip().lower() != "unknown":
      normalized = doi.strip().lower()
      if normalized not in candidate_dois:
        self.event_logger(
            f"Rejected DOI {doi}: it does not occur in the Marker page HTML."
        )
        value["doi"] = None
        return replace(
            decision,
            value=value,
            provenance={**decision.provenance, "doi": ["unresolved"]},
        )
      value["doi"] = normalized
      return replace(
          decision,
          value=value,
          provenance={
              **decision.provenance,
              "doi": sorted(set(decision.provenance.get("doi", []) + ["marker_html"])),
          },
      )

    canonical = self.doi_candidate_extractor.canonical(compact_json)
    if canonical is None:
      return decision

    value["doi"] = canonical.doi
    return replace(
        decision,
        value=value,
        provenance={**decision.provenance, "doi": ["marker_html"]},
    )

  def _doi_matches_paper(
      self,
      metadata: dict[str, Any],
      decisions: dict[str, MetadataDecision],
  ) -> bool:
    matcher = getattr(self.doi_metadata_client, "matches_paper", None)
    if matcher is None:
      return True

    title_decision = decisions.get("title")
    authors_decision = decisions.get("authors")
    title = title_decision.value if title_decision is not None else None
    authors = authors_decision.value if authors_decision is not None else None
    return matcher(metadata, title=title, authors=authors)

  def _has_comparison_evidence(
      self,
      decisions: dict[str, MetadataDecision],
  ) -> bool:
    title = decisions.get("title")
    authors = decisions.get("authors")
    return bool(
        (title is not None and isinstance(title.value, str) and title.value.strip())
        or (authors is not None and isinstance(authors.value, list) and authors.value)
    )

  def _mark_doi_unresolved(self, decision: MetadataDecision) -> MetadataDecision:
    value = dict(decision.value) if isinstance(decision.value, dict) else {}
    value["doi"] = "unknown"
    for field in self.missing_values_handler.JOURNAL_REQUIRED_FIELDS:
      value[field] = None
    return replace(
        decision,
        value=value,
        provenance={
            **decision.provenance,
            "doi": ["unresolved"],
        },
    )
