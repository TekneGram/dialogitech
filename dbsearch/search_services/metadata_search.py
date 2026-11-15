import lancedb
from dataclasses import dataclass
from pathlib import Path

@dataclass(slots=True)
class PaperRecord:
  exists: bool
  paper_title: str | None
  paper_id: str | None
  markdown_path: Path | str | None
  classified_chunks_path: Path | str | None

class MetadataSearch:
  def __init__(self) -> None:
    return

  def get_paper_by_title(
      self,
      paper_title
  ) -> PaperRecord:
    db = lancedb.connect("data/lancedb")
    table = db.open_table("paper_chunks")

    escaped_title = paper_title.replace("'", "''")

    rows = (
      table.search()
      .where(
        f"paper_title = '{escaped_title}'",
        prefilter=True,
      )
      .limit(1)
      .to_list()
    )

    if not rows:
      print(f"No exact paper-title match found for {paper_title}")
      return PaperRecord(
        exists = False,
        paper_title="No paper with that title exists",
        paper_id="xxx - no file",
        markdown_path="No file",
        classified_chunks_path="No file"
      )

    row = rows[0]

    filtered_path = Path(row["markdown_path"])
    classified_chunks_path = filtered_path.with_name(
      filtered_path.name.replace(
        "_filtered.md",
        "_classified_chunks.json"
      )
    )

    paper_record = PaperRecord(
      exists = True,
      paper_title=row['paper_title'],
      paper_id=row['paper_id'],
      markdown_path=filtered_path,
      classified_chunks_path=classified_chunks_path if classified_chunks_path.exists() else None
    )

    return paper_record