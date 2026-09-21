from dataclasses import dataclass, field
from typing import Any

# Confidence should be either "high", "medium" or "low"
@dataclass
class MetadataDecision:
  value: Any | None
  confidence: str
  reason: str
  source_pages: list[int]
  provenance: dict[str, list[str]] = field(default_factory=dict)

@dataclass
class MetadataExtractionResult:
  title: MetadataDecision
  journal: MetadataDecision
  authors: MetadataDecision
  references: list[str] = field(default_factory=list)
