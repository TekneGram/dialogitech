from __future__ import annotations

import html
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any


DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


@dataclass
class DoiCandidate:
  doi: str
  score: int
  occurrences: int = 0
  pages: list[int] = field(default_factory=list)
  contexts: list[str] = field(default_factory=list)


class DoiCandidateExtractor:
  """Extract and rank DOI candidates from compact Marker HTML evidence."""

  def extract(self, compact_json: dict[str, Any]) -> list[DoiCandidate]:
    candidates: dict[str, DoiCandidate] = {}
    for page in compact_json.get("pages", []):
      if not isinstance(page, dict):
        continue
      page_number = page.get("page_number")
      for block in page.get("blocks", []):
        if not isinstance(block, dict) or not isinstance(block.get("html"), str):
          continue
        for doi in self._dois_from_html(block["html"]):
          context = self._context(block["html"])
          score = self._score(doi, context, page_number)
          candidate = candidates.setdefault(doi, DoiCandidate(doi=doi, score=0))
          candidate.score += score
          candidate.occurrences += 1
          if isinstance(page_number, int) and page_number not in candidate.pages:
            candidate.pages.append(page_number)
          if context and context not in candidate.contexts:
            candidate.contexts.append(context)

    return sorted(
        candidates.values(),
        key=lambda candidate: (candidate.score, candidate.occurrences),
        reverse=True,
    )

  def canonical(self, compact_json: dict[str, Any]) -> DoiCandidate | None:
    candidates = self.extract(compact_json)
    if not candidates:
      return None
    if candidates[0].score <= 0:
      return None
    return candidates[0]

  def _dois_from_html(self, html_text: str) -> set[str]:
    decoded = urllib.parse.unquote(html.unescape(html_text))
    values = set(DOI_PATTERN.findall(decoded))
    return {self._normalize(doi) for doi in values}

  def _normalize(self, doi: str) -> str:
    return doi.rstrip(".,;)]}").lower()

  def _context(self, html_text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(html_text))
    return re.sub(r"\s+", " ", text).strip()

  def _score(self, doi: str, context: str, page_number: Any) -> int:
    lowered = context.lower()
    score = 10 if page_number in {0, 1} else 1
    if "cite this article" in lowered or "doi.org" in lowered:
      score += 50
    if "doi" in lowered:
      score += 20
    if re.search(r"\b(previous|next)\b", lowered):
      score -= 100
    if "download" in lowered or "table of contents" in lowered:
      score -= 30
    return score
