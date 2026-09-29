from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from typing import Any, Callable


class ArxivMetadataError(RuntimeError):
  """Raised when the arXiv metadata search fails."""


class ArxivMetadataClient:
  API_URL = "https://export.arxiv.org/api/query"
  ATOM_NS = "http://www.w3.org/2005/Atom"
  ARXIV_NS = "http://arxiv.org/schemas/atom"

  def __init__(
      self,
      *,
      timeout_seconds: float = 15.0,
      urlopen: Callable[..., Any] | None = None,
  ) -> None:
    if timeout_seconds <= 0:
      raise ValueError("timeout_seconds must be positive.")
    self.timeout_seconds = timeout_seconds
    self._urlopen = urlopen or urllib.request.urlopen

  def search(
      self,
      *,
      title: str,
      authors: Any = None,
  ) -> dict[str, Any] | None:
    """Search arXiv first with authors, then with title alone."""
    if not isinstance(title, str) or not title.strip():
      return None

    author_surnames = self._author_surnames(authors)
    if author_surnames:
      result = self._search_once(
          title=title,
          author_surnames=author_surnames,
          require_author_match=True,
      )
      if result is not None:
        return result

    return self._search_once(
        title=title,
        author_surnames=set(),
        require_author_match=False,
    )

  def _search_once(
      self,
      *,
      title: str,
      author_surnames: set[str],
      require_author_match: bool,
  ) -> dict[str, Any] | None:
    title_query = f'ti:"{title.strip()}"'
    if author_surnames:
      authors_query = " OR ".join(
          f'au:"{surname}"' for surname in sorted(author_surnames)
      )
      search_query = f"{title_query} AND ({authors_query})"
    else:
      search_query = title_query

    params = urllib.parse.urlencode({
        "search_query": search_query,
        "start": "0",
        "max_results": "10",
    })
    request = urllib.request.Request(
        f"{self.API_URL}?{params}",
        headers={"User-Agent": "DialogiTech arXiv metadata client"},
    )

    try:
      with self._urlopen(request, timeout=self.timeout_seconds) as response:
        payload = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
      raise ArxivMetadataError(
          f"arXiv search failed for {title!r}: {exc}"
      ) from exc

    try:
      root = ET.fromstring(payload)
    except (ET.ParseError, UnicodeDecodeError) as exc:
      raise ArxivMetadataError("arXiv returned invalid XML.") from exc

    ranked: list[tuple[float, dict[str, Any]]] = []
    for entry in root.findall(f"{{{self.ATOM_NS}}}entry"):
      metadata = self._parse_entry(entry)
      if metadata is None:
        continue

      title_score = self._title_similarity(title, metadata["title"])
      matched_authors = author_surnames.intersection(
          self._author_surnames(metadata["authors"])
      )
      if require_author_match:
        if title_score < 0.85 or not matched_authors:
          continue
        score = title_score + min(0.15, 0.05 * len(matched_authors))
      else:
        if title_score < 0.92:
          continue
        score = title_score
      ranked.append((score, metadata))

    if not ranked:
      return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]

  def _parse_entry(self, entry: ET.Element) -> dict[str, Any] | None:
    title = self._element_text(entry, self.ATOM_NS, "title")
    entry_id = self._element_text(entry, self.ATOM_NS, "id")
    if not title or not entry_id:
      return None

    arxiv_id_match = re.search(r"arxiv\.org/abs/([^?]+)", entry_id, re.IGNORECASE)
    if not arxiv_id_match:
      return None
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id_match.group(1))
    authors = []
    for author in entry.findall(f"{{{self.ATOM_NS}}}author"):
      name = self._element_text(author, self.ATOM_NS, "name")
      if name:
        authors.append(name)

    doi = self._element_text(entry, self.ARXIV_NS, "doi")
    published = self._element_text(entry, self.ATOM_NS, "published")
    return {
        "doi": doi or "unknown",
        "title": " ".join(title.split()),
        "authors": authors,
        "journal": "arXiv-preprint",
        "year": published[:4] if published and published[:4].isdigit() else None,
        "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
    }

  def _element_text(
      self,
      parent: ET.Element,
      namespace: str,
      name: str,
  ) -> str | None:
    element = parent.find(f"{{{namespace}}}{name}")
    if element is None or element.text is None:
      return None
    value = " ".join(element.text.split())
    return value or None

  def _author_surnames(self, authors: Any) -> set[str]:
    if not isinstance(authors, list):
      authors = [authors]
    surnames: set[str] = set()
    for author in authors:
      if not isinstance(author, str):
        continue
      tokens = re.findall(r"[a-z0-9]+", author.lower())
      if tokens:
        surnames.add(tokens[-1])
    return surnames

  def _title_similarity(self, left: str, right: str) -> float:
    normalize = lambda value: " ".join(re.findall(r"[a-z0-9]+", value.lower()))
    return SequenceMatcher(None, normalize(left), normalize(right)).ratio()
