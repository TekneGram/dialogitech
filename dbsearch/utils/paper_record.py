from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class PaperRecord:
  rank: int
  paper_title: str
  authors: list[str]
  year: int | None
  journal: str | None
  paper_type: str | None
  markdown_path: str | None
  pdf_path: str | None
  count: int


def split_markdown_row(line: str) -> list[str]:
  """Split a Markdown table row while preserving escaped pipes."""
  line = line.strip()

  if line.startswith("|"):
    line = line[1:]
  if line.endswith("|"):
    line = line[:-1]

  cells: list[str] = []
  current: list[str] = []
  escaped = False

  for character in line:
    if character == "|" and not escaped:
      cells.append("".join(current).strip())
      current = []
      continue

    current.append(character)

    if character == "\\" and not escaped:
      escaped = True
    else:
      escaped = False

  cells.append("".join(current).strip())

  return [
    cell.replace(r"\|", "|")
    for cell in cells
  ]


def is_separator_row(cells: list[str]) -> bool:
  return all(
    re.fullmatch(r":?-+:?", cell.replace(" ", ""))
    for cell in cells
  )


def parse_paper_results(path: str | Path) -> list[PaperRecord]:
  path = Path(path)
  lines = path.read_text(encoding="utf-8").splitlines()

  table_header_index: int | None = None

  for index, line in enumerate(lines):
    if line.startswith("| Rank |"):
      table_header_index = index
      break

  if table_header_index is None:
    raise ValueError("Could not find the paper-results table.")

  header = split_markdown_row(lines[table_header_index])

  if table_header_index + 1 >= len(lines):
    raise ValueError("The table has no data rows.")

  separator = split_markdown_row(lines[table_header_index + 1])

  if not is_separator_row(separator):
    raise ValueError("The Markdown table has no valid separator row.")

  expected_columns = [
    "Rank",
    "Paper title",
    "Authors",
    "Year",
    "Journal",
    "Paper type",
    "Markdown path",
    "PDF path",
    "Count",
  ]

  if header != expected_columns:
    raise ValueError(
      f"Unexpected table columns: {header!r}"
    )

  papers: list[PaperRecord] = []

  for line in lines[table_header_index + 2:]:
    if not line.strip().startswith("|"):
      continue

    cells = split_markdown_row(line)

    if len(cells) != len(expected_columns):
      continue

    authors = [
      author.strip()
      for author in cells[2].split(",")
      if author.strip()
    ]

    papers.append(
      PaperRecord(
        rank=int(cells[0]),
        paper_title=cells[1],
        authors=authors,
        year=int(cells[3]) if cells[3] else None,
        journal=cells[4] or None,
        paper_type=cells[5] or None,
        markdown_path=cells[6] or None,
        pdf_path=cells[7] or None,
        count=int(cells[8]),
      )
    )

  return papers
