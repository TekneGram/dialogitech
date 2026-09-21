from dataclasses import dataclass
from typing import Any

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