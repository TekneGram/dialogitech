from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, dataclass

# Confidence should be either "high", "medium" or "low"
@dataclass
class MetadataDecision:
  value: Any | None
  confidence: str
  reason: str
  source_pages: list[int]

@dataclass
class MetadataExtractionResult:
  title: MetadataDecision
  journal: MetadataDecision
  authors: MetadataDecision

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
  def __init__(self) -> None:
    return

  # Handle the json metadata
  def load_marker_json(self, source) -> dict:
    return

  def select_pages(self, document, page_numbers) -> list[dict]:
    return

  def compact_page_json(self, pages) -> dict:
    return

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
    return

  def build_journal_prompt(self, compact_json) -> str:
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
    return

  def build_authors_prompt(self, compact_json) -> str:
    """
    Returns
    {
      "value": ["Author one", "Author two"],
        "confidence": "high",
        "reasons": "These names appear directly below the title and before the abstract"
    }
    If the information is absent, value is null and confidence is high with reason that metadata is absent.
    """
    return

  # LLM execution and caching
  # Consider updating mlx_llm_runner to handle KV Cache
  def _generate(messages) -> str:
    return

  def _extract_component(component, compact_json) -> MetadataDecision:
    return

