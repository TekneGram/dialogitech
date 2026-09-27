from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from chunker.llm_metadata_extractor_helpers.metadata_models import MetadataDecision
from chunker.llm_metadata_extractor_helpers.metadata_models import MetadataExtractionResult
from chunker.llm_worker import LLMWorker
from chunker.llm_metadata_extractor_helpers.metadata_component_runner import MetadataComponentRunner
from chunker.llm_metadata_extractor_helpers.doi_metadata_client import DoiMetadataClient
from chunker.llm_metadata_extractor_helpers.metadata_evidence import MetadataEvidence
from chunker.llm_metadata_extractor_helpers.metadata_flow import MetadataExtractionFlow
from chunker.llm_metadata_extractor_helpers.missing_values_handler import MetaDataMissingValuesHandler
from chunker.llm_metadata_extractor_helpers.response_parsing_validation import MetadataResponseValidator
from chunker.llm_metadata_extractor_helpers.reference_extractor import DeterministicReferenceExtractor

DEFAULT_MODEL_PATH = "unsloth/gemma-4-E4B-it-UD-MLX-4bit"
DEFAULT_PYTHON_EXECUTABLE = (
  Path.home() / ".unsloth" / "unsloth_gemma4_mlx" / "bin" / "python"
)

class LLMMetadataExtractor:
  """
  Gemma-backed extraction of the following metadata
  from the json file returned after extraction by marker:
    - title
    - authors: [ "...", "..." ]
    - keywords: [ "..." ]
    - references: [ "...", "..." ] (deterministically extracted from the Marker references section)
    - journal: {
        "name" : "...",
        "volume": "...",
        "issue": "...",
        "year": "...",
        "doi": "...",
        "issn": "..."
    }
  """
  MODEL_MAX_TOKENS = 400
  def __init__(
      self,
      *,
      model_path: str | Path = DEFAULT_MODEL_PATH,
      python_executable: str | Path = DEFAULT_PYTHON_EXECUTABLE,
      max_tokens: int | None = None,
      temperature: float = 0.0,
      request_timeout_seconds: float = 180.0,
      event_logger: Callable[[str], None] | None = None,
      input_fn: Callable[[str], str] | None = None,
      doi_metadata_client: DoiMetadataClient | None = None,
  ) -> None:
    if request_timeout_seconds <= 0:
      raise ValueError("request_timeout_seconds must be a positive number.")

    self.model_path = str(model_path)
    self.python_executable = str(python_executable)
    self.max_tokens = max_tokens or self.MODEL_MAX_TOKENS
    self.temperature = temperature
    self.request_timeout_seconds = request_timeout_seconds
    self.event_logger = event_logger
    self.metadata_handler = MetaDataMissingValuesHandler(input_fn or input)
    self.response_validator = MetadataResponseValidator()
    self.doi_metadata_client = doi_metadata_client or DoiMetadataClient()
    self.evidence = MetadataEvidence()
    self.reference_extractor = DeterministicReferenceExtractor()
    self.component_runner = MetadataComponentRunner(
        generate=self._generate,
        prompt_builders={
            "title": self.build_title_prompt,
            "journal": self.build_journal_prompt,
            "authors": self.build_authors_prompt,
        },
        validator=self.response_validator,
    )
    self.extraction_flow = MetadataExtractionFlow(
        evidence=self.evidence,
        component_extractor=self._extract_component,
        missing_values_handler=self.metadata_handler,
        doi_metadata_client=self.doi_metadata_client,
        event_logger=self._log_event,
    )

    # Start lazily when the first request is made.
    self._worker: LLMWorker | None = None

  def extract_metadata(
      self,
      source: str | Path | dict[str, Any],
      *,
      allow_manual: bool = True,
  ) -> MetadataExtractionResult:
    document = self.load_marker_json(source)
    decisions = self.extraction_flow.extract(
        document,
        components=("title", "journal", "authors"),
        allow_manual=allow_manual,
    )
    return MetadataExtractionResult(
        title=decisions["title"],
        journal=decisions["journal"],
        authors=decisions["authors"],
        keywords=self._extract_keywords(document),
        references=self.reference_extractor.extract(document),
    )

  def extract_all(
      self,
      source: str | Path | dict[str, Any],
      *,
      allow_manual: bool = True,
  ) -> dict[str, Any]:
    """Return the legacy dictionary shape for pipeline consumers."""
    result = self.extract_metadata(source, allow_manual=allow_manual)
    return {
        "title": result.title.value,
        "journal": result.journal.value,
        "authors": result.authors.value or [],
        "keywords": list(result.keywords.value or []),
        "references": result.references,
    }

  def extract_component(
      self,
      source: str | Path | dict[str, Any],
      component: str,
      *,
      allow_manual: bool = True,
  ) -> MetadataDecision:
    if component == "keywords":
      document = self.load_marker_json(source)
      return self._extract_keywords(document)

    if component not in {"title", "journal", "authors"}:
      raise ValueError(f"Unsupported metadata component: {component}")

    document = self.load_marker_json(source)
    return self.extraction_flow.extract(
        document,
        components=(component,),
        allow_manual=allow_manual,
    )[component]

  # Handle the json metadata
  def load_marker_json(self, source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """
    Load a marker JSON document from path or existing dictionary
    """
    if isinstance(source, dict):
      return source

    path = Path(source)

    if not path.is_file():
      raise FileNotFoundError(f"Marker JSON file not found: {path}")

    try:
      document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
      raise ValueError(f"Invalid JSON in Marker file: {path}") from exc

    if not isinstance(document, dict):
      raise ValueError("Marker JSON root must be an object")

    document.setdefault("__source_path__", str(path))
    return document

  def select_pages(
      self,
      document: dict[str, Any],
      page_numbers: list[int],
  ) -> list[dict[str, Any]]:
    return self.evidence.select_pages(document, page_numbers)

  def compact_page_json(
      self,
      pages: list[dict[str, Any]],
  ) -> dict[str, list[dict[str, Any]]]:
    return self.evidence.compact_pages(pages)

  def build_keywords_prompt(
      self,
      compact_json: dict[str, Any],
      *,
      mode: str = "explicit",
  ) -> str:
    evidence_json = json.dumps(
        {
            key: value
            for key, value in compact_json.items()
            if not key.startswith("__")
        },
        ensure_ascii=False,
        indent=2,
    )
    if mode == "explicit":
      instructions = """
      Extract only keywords explicitly presented by the authors or publisher.
      Recognize labels such as Keywords, Key words, Key terms, Index terms, and
      Descriptors, even when formatting is irregular. Do not infer topical
      keywords in this pass. If no explicit keyword list is present, return an
      empty list.
      """
      reason_example = "The terms appear in an explicit keyword list."
    else:
      instructions = """
      No explicit keyword list was found in the opening pages. Infer a small
      set of useful topical keywords from the title, abstract, introduction,
      and early substantive discussion. Prefer central, repeated terms useful
      for research retrieval. Do not invent unsupported methods, findings,
      populations, or claims. Return at most eight keywords.
      """
      reason_example = "No explicit list was found; these terms are supported by the early article text."

    return f"""
      Identify keywords from the supplied Marker page data.

      Rules:
      - Use only the supplied JSON evidence.
      - Normalize whitespace and remove duplicate keywords.
      - Return a list of non-empty strings.
      - If no keywords can be found, return an empty list rather than null.
      - Confidence must be exactly one of "high", "medium", or "low".
      - Return JSON only. Do not include Markdown or commentary.

      {instructions.strip()}

      Required JSON format:
      {{
        "value": ["keyword one", "keyword two"],
        "confidence": "medium",
        "reason": "{reason_example}"
      }}

      Marker page data:
      {evidence_json}
    """.strip()

  # Prompts for the individual metadata fields
  def build_title_prompt(self, compact_json) -> str:
    """
    Returns
    {
      "value": "Paper title",
      "confidence": "High",
      "reason": "This is the largest title-like heading before the abstract"
    }
    If the information is absent, value is null and confidence is high with reason that metadata is absent.
    """
    evidence_json = json.dumps(
          compact_json,
          ensure_ascii=False,
          indent=2
        )
    
    return f"""
    Identify the academic paper title from the supplied Marker page data.

    Rules:
    - Use only the supplied JSON evidence.
    - Prefer a prominent title-like SectionHeader near the beginning
    - Do not return the journal name, article type, abstract heading, or author names.
    - If the title is not present, return null
    - Confidence must be exactly one of "high", "medium" or "low"
    - Return JSON only. Do not include Markdown or commentary.

    Required JSON format: 
    {{
      "value": "Paper title or null",
      "confidence": "high",
      "reason": "Concise explanation for the decision"
    }}

    Marker page data:
    {evidence_json}
    """.strip()

  def build_journal_prompt(self, compact_json: dict[str, Any]) -> str:
    """
    Returns
    {
      "value": {
        "name": "journal name",
        "volume": "12",
        "issue": "2",
        "year": "2025",
        "doi": "10.xxx/example",
        "issn": "...",
      },
      "confidence":"high",
      "reason": "The journal and publication details appear in the front matter."
    }
    If the information is absent, value is null and confidence is high with reason that metadata is absent.
    """
    evidence_json = json.dumps(
      compact_json,
      ensure_ascii=False,
      indent=2,
    )

    return f"""
      Identify the journal metadata from the supplied Marker page data.

      Rules:
      - Use only the supplied JSON evidence.
      - Extract the journal name, volume, issue, year, DOI, and ISSN when present.
      - Do not infer values that are not explicitly present.
      - If no journal metadata is present, return null for "value".
      - Confidence must be exactly one of "high", "medium", or "low".
      - Return JSON only. Do not include Markdown or commentary.

      Required JSON format:
      {{
        "value": {{
          "name": "Journal name or null",
          "volume": "Volume or null",
          "issue": "Issue or null",
          "year": "Year or null",
          "doi": "DOI or null",
          "issn": "ISSN or null"
        }},
        "confidence": "high",
        "reason": "Concise explanation for the decision"
      }}

      Marker page data:
      {evidence_json}
      """.strip()

  def build_authors_prompt(self, compact_json: dict[str, Any]) -> str:
    """
    Returns
    {
      "value": ["Author one", "Author two"],
        "confidence": "high",
        "reasons": "These names appear directly below the title and before the abstract"
    }
    If the information is absent, value is null and confidence is high with reason that metadata is absent.
    """
    evidence_json = json.dumps(
      compact_json,
      ensure_ascii=False,
      indent=2,
    )

    return f"""
      Identify the academic paper authors from the supplied Marker page data.

      Rules:
      - Use only the supplied JSON evidence.
      - Prefer names appearing directly below the title and before the abstract.
      - Exclude affiliations, institutions, publishers, copyright holders, editors, and journal names.
      - Return authors in their displayed order.
      - If no authors are present, return null for "value".
      - Confidence must be exactly one of "high", "medium", or "low".
      - Return JSON only. Do not include Markdown or commentary.

      Required JSON format:
      {{
        "value": ["Author One", "Author Two"],
        "confidence": "high",
        "reason": "Concise explanation for the decision"
      }}

      Marker page data:
      {evidence_json}
      """.strip()

  # LLM execution and caching
  # Consider updating mlx_llm_runner to handle KV Cache
  def _generate(self, messages: list[dict[str, str]]) -> str:
    if self._worker is None:
      runner_path = Path(__file__).with_name("mlx_llm_runner.py")

      self._worker = LLMWorker(
        python_executable=self.python_executable,
        runner_path=runner_path,
        model_path=self.model_path,
        request_timeout_seconds=self.request_timeout_seconds,
        event_logger=self._log_event
      )

    try:
      return self._worker.generate(
        messages=messages,
        max_tokens=self.max_tokens,
        temperature=self.temperature
      )
    except RuntimeError:
      self._worker.close()
      self._worker = None
      raise

  def _log_event(self, message: str) -> None:
    if self.event_logger is not None:
      self.event_logger(message)

  def close(self) -> None:
    if self._worker is not None:
      self._worker.close()
      self._worker = None

  def __enter__(self) -> "LLMMetadataExtractor":
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    self.close()

  def _extract_component(
      self,
      component: str,
      compact_json: dict[str, Any],
  ) -> MetadataDecision:
    if component == "keywords":
      mode = str(compact_json.get("__keyword_mode", "explicit"))
      return self.component_runner.extract(
          component,
          compact_json,
          prompt_builder=lambda data: self.build_keywords_prompt(data, mode=mode),
      )
    return self.component_runner.extract(component, compact_json)

  def _extract_keywords(self, document: dict[str, Any]) -> MetadataDecision:
    """Extract explicit keywords, then infer them if the opening pages have none."""
    initial_pages = self.evidence.select_available_pages(document, [0, 1])
    if not initial_pages:
      return MetadataDecision(
          value=[],
          confidence="low",
          reason="No Marker pages were available for keyword extraction.",
          source_pages=[],
          provenance={},
      )

    initial_json = self.evidence.compact_pages(initial_pages)
    try:
      explicit_json = dict(initial_json)
      explicit_json["__keyword_mode"] = "explicit"
      explicit = self._extract_component("keywords", explicit_json)
      explicit_values = self._normalize_keywords(explicit.value)
      if explicit_values:
        return MetadataDecision(
            value=explicit_values,
            confidence=explicit.confidence,
            reason=explicit.reason,
            source_pages=explicit.source_pages,
            provenance={"value": ["gemma", "explicit_keyword_list"]},
        )
    except Exception as exc:
      self._log_event(f"Explicit keyword extraction failed: {type(exc).__name__}: {exc}")

    additional_pages = self.evidence.select_available_pages(document, [2, 3, 4])
    expanded_pages = initial_pages + [
        page for page in additional_pages if page not in initial_pages
    ]
    expanded_json = self.evidence.compact_pages(expanded_pages)
    try:
      expanded_json["__keyword_mode"] = "inferred"
      inferred = self._extract_component("keywords", expanded_json)
      inferred_values = self._normalize_keywords(inferred.value)
      return MetadataDecision(
          value=inferred_values,
          confidence=inferred.confidence,
          reason=(
              inferred.reason
              if inferred_values
              else "No explicit or inferable keywords were found."
          ),
          source_pages=inferred.source_pages or [
              page["page_number"] for page in expanded_pages
          ],
          provenance=(
              {"value": ["gemma", "inferred_from_article"]}
              if inferred_values else {}
          ),
      )
    except Exception as exc:
      self._log_event(f"Inferred keyword extraction failed: {type(exc).__name__}: {exc}")
      return MetadataDecision(
          value=[],
          confidence="low",
          reason="Keyword extraction did not produce a usable result.",
          source_pages=[page["page_number"] for page in expanded_pages],
          provenance={},
      )

  @staticmethod
  def _normalize_keywords(value: Any) -> list[str]:
    if not isinstance(value, list):
      return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
      if not isinstance(item, str):
        continue
      keyword = " ".join(item.split()).strip()
      key = keyword.casefold()
      if keyword and key not in seen:
        normalized.append(keyword)
        seen.add(key)
    return normalized
