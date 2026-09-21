from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from chunker.llm_metadata_extractor_helpers.metadata_models import MetadataDecision
from chunker.llm_metadata_extractor_helpers.metadata_models import MetadataExtractionResult
from chunker.llm_metadata_extractor_helpers.gemma_worker import MetadataGemmaWorker
from chunker.llm_metadata_extractor_helpers.metadata_component_runner import MetadataComponentRunner
from chunker.llm_metadata_extractor_helpers.metadata_evidence import MetadataEvidence
from chunker.llm_metadata_extractor_helpers.metadata_flow import MetadataExtractionFlow
from chunker.llm_metadata_extractor_helpers.missing_values_handler import MetaDataMissingValuesHandler
from chunker.llm_metadata_extractor_helpers.response_parsing_validation import MetadataResponseValidator

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
    self.evidence = MetadataEvidence()
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
        event_logger=self._log_event,
    )

    # Start lazily when the first request is made.
    self._worker: Any = None

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
    )

  def extract_component(
      self,
      source: str | Path | dict[str, Any],
      component: str,
      *,
      allow_manual: bool = True,
  ) -> MetadataDecision:
    if component not in {"title", "journal", "authors"}:
      raise ValueError(f"Unsupported metadata component: {component}")

    document = self.load_marker_json(source)
    return self.extraction_flow.extract(
        document,
        components=(component,),
        allow_manual=allow_manual,
    )[component]

  # NOTES
  #  - run_full_pipeline_folder.py catches exceptions per PDF.
  # - It records the failure, prints the error and traceback, then continues to the
  #   next PDF.

  # - It prints a final success/failure summary and exits with status 1 if any
  #   failed.

  # What is not yet covered:

  # - Metadata errors are not currently written to a dedicated per-paper log.
  # - The single-file runner only exposes a normal traceback.
  # - The LLM metadata extractor must send errors through the pipeline’s logger and
  #   close its worker cleanly.

  # - Interactive metadata prompts may block batch processing; batch mode needs a
  #   policy such as automatic failure or explicit manual-entry mode.

  # When integrating, add a metadata log such as:

  # marker/conversion_results/<paper_id>/<paper_id>_metadata.log

  # Then let metadata exceptions propagate from process_pdf; the batch runner will
  # catch them, print them, log them in its failure summary, and continue processing
  # the next file.


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

  # Three separate prompts for the three metadata fields
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

      self._worker = MetadataGemmaWorker(
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
        temperature=self.temperature,
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
    return self.component_runner.extract(component, compact_json)
