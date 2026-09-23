from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

class CacheManager:
  def __init__(self) -> None:
    self.search_results_directory = Path(__file__).resolve().parents[2] / "search_results"
    self.search_results_csv = self.search_results_directory / "search_results.csv"

  def record_search(self, *, search_results: list[str], query: str) -> tuple[int, str]:
    """Record a search and return its conversation number and data filename."""
    self.search_results_directory.mkdir(parents=True, exist_ok=True)

    fieldnames = [
      "conversation_number",
      "chunk_ids",
      "query",
      "date",
      "time",
      "data_file",
    ]

    existing_rows: list[dict[str, str]] = []
    if self.search_results_csv.exists():
      with self.search_results_csv.open("r", newline="", encoding="utf-8") as csv_file:
        existing_rows = list(csv.DictReader(csv_file))

    conversation_number = len(existing_rows) + 1
    now = datetime.now()
    data_file = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.md"

    row = {
      "conversation_number": str(conversation_number),
      "chunk_ids": json.dumps(search_results),
      "query": query,
      "date": now.strftime("%Y-%m-%d"),
      "time": now.strftime("%H:%M:%S"),
      "data_file": data_file,
    }

    write_header = not self.search_results_csv.exists() or self.search_results_csv.stat().st_size == 0
    with self.search_results_csv.open("a", newline="", encoding="utf-8") as csv_file:
      writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
      if write_header:
        writer.writeheader()
      writer.writerow(row)

    return conversation_number, data_file

  def append_search(
    self,
    *,
    conversation_number: int,
    data_file: str,
    search_results: list[str],
  ) -> tuple[int, str]:
    """Append chunk IDs to an existing conversation and return its identifiers."""
    if not self.search_results_csv.exists():
      raise FileNotFoundError(f"Search results CSV does not exist: {self.search_results_csv}")

    fieldnames = [
      "conversation_number",
      "chunk_ids",
      "query",
      "date",
      "time",
      "data_file",
    ]

    with self.search_results_csv.open("r", newline="", encoding="utf-8") as csv_file:
      rows = list(csv.DictReader(csv_file))

    matching_row = next(
      (
        row for row in rows
        if row.get("conversation_number") == str(conversation_number)
      ),
      None,
    )
    if matching_row is None:
      raise ValueError(f"Conversation number not found: {conversation_number}")
    if matching_row.get("data_file") != data_file:
      raise ValueError(
        f"Data file does not match conversation {conversation_number}: {data_file}"
      )

    try:
      existing_chunk_ids = json.loads(matching_row.get("chunk_ids", "[]"))
    except json.JSONDecodeError as exc:
      raise ValueError(
        f"Invalid chunk_ids JSON for conversation {conversation_number}"
      ) from exc

    if not isinstance(existing_chunk_ids, list) or not all(
      isinstance(chunk_id, str) for chunk_id in existing_chunk_ids
    ):
      raise ValueError(
        f"chunk_ids must be a list of strings for conversation {conversation_number}"
      )

    combined_chunk_ids = list(existing_chunk_ids)
    for chunk_id in search_results:
      if chunk_id not in combined_chunk_ids:
        combined_chunk_ids.append(chunk_id)

    now = datetime.now()
    matching_row["chunk_ids"] = json.dumps(combined_chunk_ids)
    matching_row["date"] = now.strftime("%Y-%m-%d")
    matching_row["time"] = now.strftime("%H:%M:%S")

    with self.search_results_csv.open("w", newline="", encoding="utf-8") as csv_file:
      writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(rows)

    return conversation_number, data_file

  def read_record(self, *, conversation_number: int) -> dict[str, str]:
    """Return one cached search record by conversation number."""
    if not self.search_results_csv.exists():
      raise FileNotFoundError(f"Search results CSV does not exist: {self.search_results_csv}")

    with self.search_results_csv.open("r", newline="", encoding="utf-8") as csv_file:
      rows = csv.DictReader(csv_file)
      for row in rows:
        if row.get("conversation_number") == str(conversation_number):
          return dict(row)

    raise ValueError(f"Conversation number not found: {conversation_number}")

  def extract_summaries(self, conversation_number: int) -> list[str]:
    """Return only the per-paper summary text for one conversation."""
    record = self.read_record(conversation_number=conversation_number)
    document = self.read_results(record["data_file"])

    section_match = re.search(
      r"^## Per[- ]paper summaries\s*$",
      document,
      flags=re.MULTILINE | re.IGNORECASE,
    )
    if section_match is None:
      return []

    section = document[section_match.end():]
    section = re.split(r"^## (?!#)", section, maxsplit=1, flags=re.MULTILINE)[0]
    summaries = []
    for match in re.finditer(
      r"^### Summary\s*\n(?P<summary>.*?)(?=^### |\Z)",
      section,
      flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    ):
      summary = match.group("summary").strip()
      if summary:
        summaries.append(summary)
    return summaries

  def extract_queries(self, conversation_numbers: list[int]) -> dict[int, str]:
    """Return the original query for each requested conversation."""
    return {
      conversation_number: self.read_record(
        conversation_number=conversation_number
      )["query"]
      for conversation_number in conversation_numbers
    }

  def append_to_data_file(
    self,
    *,
    data_file: str,
    query_to_embed: str,
    question_for_llm: str,
    row: dict[str, object],
    parsed_response: dict[str, str],
  ) -> bool:
    """Append one accepted LLM result to its conversation Markdown file."""
    if parsed_response.get("answer_provided") == "never":
      return False

    if parsed_response.get("answer_provided") not in {"fully", "partially"}:
      raise ValueError("parsed_response has an invalid answer_provided value")

    self.search_results_directory.mkdir(parents=True, exist_ok=True)
    data_path = self.search_results_directory / Path(data_file).name
    file_exists = data_path.exists()
    file_is_empty = not file_exists or data_path.stat().st_size == 0

    if file_is_empty:
      document = (
        "# Query\n\n"
        f"Query to embed: {query_to_embed}\n\n"
        f"Question for LLM: {question_for_llm}\n\n"
      )
      result_number = 1
    else:
      document = data_path.read_text(encoding="utf-8")
      result_number = sum(
        line.startswith("## Search result ")
        for line in document.splitlines()
      ) + 1
      if not document.endswith("\n"):
        document += "\n"

    authors = row.get("authors", [])
    if isinstance(authors, list):
      authors_text = ", ".join(str(author) for author in authors)
    else:
      authors_text = str(authors) if authors else ""

    citation_values = [
      authors_text,
      str(row.get("year")) if row.get("year") is not None else "",
      str(row.get("paper_title", "")),
      str(row.get("journal", "")),
      str(row.get("volume", "")),
      str(row.get("issue", "")),
    ]
    citation = ", ".join(value for value in citation_values if value)

    doi = row.get("doi")
    if doi:
      doi_text = str(doi)
      doi_url = doi_text if doi_text.startswith("http") else f"https://doi.org/{doi_text}"
      citation = f"{citation}, [DOI]({doi_url})" if citation else f"[DOI]({doi_url})"

    rhetorical_moves = row.get("rhetorical_moves", "")
    if isinstance(rhetorical_moves, list):
      rhetorical_moves_text = "; ".join(
        ", ".join(
          f"{key}: {value}"
          for key, value in move.items()
        ) if isinstance(move, dict) else str(move)
        for move in rhetorical_moves
      )
    else:
      rhetorical_moves_text = str(rhetorical_moves) if rhetorical_moves else ""

    document += (
      f"## Search result {result_number}\n\n"
      "### Paper citation\n\n"
      f"{citation}\n\n"
      "### Summary\n\n"
      f"{parsed_response['summary']}\n\n"
      "### Retrieved text\n\n"
      f"{row.get('text', '')}\n\n"
      "### Other information\n\n"
      f"section_title: {row.get('section_title', '')}\n"
      f"heading_level: {row.get('heading_level', '')}\n"
      f"classification_label: {row.get('classification_label', '')}\n"
      f"paper_type: {row.get('paper_type', '')}\n"
      f"rhetorical_moves: {rhetorical_moves_text}\n"
      f"markdown_path: {row.get('markdown_path', '')}\n"
      f"pdf_path: {row.get('pdf_path', '')}\n\n"
    )

    data_path.write_text(document, encoding="utf-8")
    return True

  def read_results(self, data_file: Path | str) -> str:
    """Read and return the Markdown text for a cached data file."""
    data_path = self.search_results_directory / Path(data_file).name
    if not data_path.exists():
      raise FileNotFoundError(f"Search results data file does not exist: {data_path}")
    if not data_path.is_file():
      raise ValueError(f"Search results data path is not a file: {data_path}")

    return data_path.read_text(encoding="utf-8")

  def append_per_paper_summary(
    self,
    *,
    data_file: str,
    citation: str,
    summary: str,
  ) -> None:
    """Append a generated per-paper summary to a cached Markdown file."""
    data_path = self.search_results_directory / Path(data_file).name
    if not data_path.exists():
      raise FileNotFoundError(f"Search results data file does not exist: {data_path}")

    document = data_path.read_text(encoding="utf-8")
    if document and not document.endswith("\n"):
      document += "\n"

    if "## Per-paper summaries" not in document:
      document += "\n## Per-paper summaries\n\n"

    document += (
      "### Paper citation\n\n"
      f"{citation}\n\n"
      "### Summary\n\n"
      f"{summary.strip()}\n\n"
    )

    data_path.write_text(document, encoding="utf-8")

  def save_summaries_synthesis(
    self,
    *,
    conversation_number: int,
    synthesis: str,
  ) -> None:
    """Save or replace the synthesis section for a conversation."""
    record = self.read_record(conversation_number=conversation_number)
    data_path = self.search_results_directory / Path(record["data_file"]).name
    document = self.read_results(record["data_file"])
    section = f"\n## Summaries synthesis\n\n{synthesis.strip()}\n"
    section_pattern = re.compile(
      r"\n## Summaries synthesis\s*\n.*?(?=\n## (?!#)|\Z)",
      flags=re.DOTALL | re.IGNORECASE,
    )
    if section_pattern.search(document):
      document = section_pattern.sub(section, document, count=1)
    else:
      document = document.rstrip() + "\n" + section
    data_path.write_text(document, encoding="utf-8")

  def save_review(self, review: str) -> Path:
    """Save a generated review and return its path."""
    self.search_results_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    review_path = self.search_results_directory / f"review_{timestamp}.md"
    review_path.write_text(review.strip() + "\n", encoding="utf-8")
    return review_path
