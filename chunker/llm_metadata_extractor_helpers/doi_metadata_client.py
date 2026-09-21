from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


class DoiMetadataError(RuntimeError):
  """Raised when DOI metadata cannot be retrieved or parsed."""


class DoiMetadataClient:
  CROSSREF_BASE_URL = "https://api.crossref.org/v1/works/"
  DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)

  def __init__(
      self,
      *,
      cache_dir: str | Path = "data/doi_metadata_cache",
      timeout_seconds: float = 15.0,
      mailto: str | None = None,
      urlopen: Callable[..., Any] | None = None,
  ) -> None:
    if timeout_seconds <= 0:
      raise ValueError("timeout_seconds must be positive.")

    self.cache_dir = Path(cache_dir)
    self.timeout_seconds = timeout_seconds
    self.mailto = mailto
    self._urlopen = urlopen or urllib.request.urlopen

  def normalize_doi(self, doi: str) -> str:
    if not isinstance(doi, str):
      raise ValueError("DOI must be a string.")

    normalized = doi.strip()
    normalized = re.sub(r"^https?://(dx\.)?doi\.org/", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^doi:\s*", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.strip().rstrip(".,;)").strip()

    if not self.DOI_PATTERN.match(normalized):
      raise ValueError(f"Invalid DOI: {doi!r}")
    return normalized

  def lookup(self, doi: str) -> dict[str, Any]:
    normalized_doi = self.normalize_doi(doi)
    cache_path = self._cache_path(normalized_doi)

    cached = self._read_cache(cache_path)
    if cached is not None:
      return cached

    encoded_doi = urllib.parse.quote(normalized_doi, safe="/")
    url = f"{self.CROSSREF_BASE_URL}{encoded_doi}"
    if self.mailto:
      url = f"{url}?mailto={urllib.parse.quote(self.mailto)}"

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "DialogiTech DOI metadata client",
        },
    )

    try:
      with self._urlopen(request, timeout=self.timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
      raise DoiMetadataError(
          f"Failed to retrieve Crossref metadata for DOI {normalized_doi}: {exc}"
      ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
      raise DoiMetadataError(
          f"Crossref returned invalid JSON for DOI {normalized_doi}."
      ) from exc

    try:
      normalized = self._normalize_record(normalized_doi, payload)
    except (KeyError, TypeError, ValueError) as exc:
      raise DoiMetadataError(
          f"Crossref response did not contain usable metadata for DOI {normalized_doi}."
      ) from exc

    self.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return normalized

  def _cache_path(self, doi: str) -> Path:
    key = hashlib.sha256(doi.lower().encode("utf-8")).hexdigest()
    return self.cache_dir / f"{key}.json"

  def _read_cache(self, path: Path) -> dict[str, Any] | None:
    if not path.is_file():
      return None
    try:
      payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      return None
    return payload if isinstance(payload, dict) else None

  def _normalize_record(
      self,
      doi: str,
      payload: dict[str, Any],
  ) -> dict[str, Any]:
    message = payload["message"]
    if not isinstance(message, dict):
      raise TypeError("Crossref message must be an object.")

    return {
        "doi": doi,
        "title": self._first_string(message.get("title")),
        "authors": self._authors(message.get("author")),
        "journal": self._first_string(message.get("container-title")),
        "year": self._year(message),
        "volume": self._string_value(message.get("volume")),
        "issue": self._string_value(message.get("issue")),
        "issn": self._first_string(message.get("ISSN")),
    }

  def _first_string(self, value: Any) -> str | None:
    if isinstance(value, list):
      value = value[0] if value else None
    return value.strip() if isinstance(value, str) and value.strip() else None

  def _string_value(self, value: Any) -> str | None:
    if value is None:
      return None
    text = str(value).strip()
    return text or None

  def _authors(self, authors: Any) -> list[str]:
    if not isinstance(authors, list):
      return []

    names: list[str] = []
    for author in authors:
      if not isinstance(author, dict):
        continue
      name = " ".join(
          part.strip()
          for part in (author.get("given"), author.get("family"))
          if isinstance(part, str) and part.strip()
      )
      if name:
        names.append(name)
    return names

  def _year(self, message: dict[str, Any]) -> str | None:
    for field in ("published-print", "published-online", "issued"):
      date = message.get(field)
      if not isinstance(date, dict):
        continue
      parts = date.get("date-parts")
      if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
        year = parts[0][0]
        if isinstance(year, int):
          return str(year)
    return None
