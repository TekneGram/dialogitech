
def papers_to_markdown_format(
    papers_by_id: dict[str, dict],
    topics: list[str]
) -> str:

  def cell(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("|", r"\|")
    text = text.replace("\n", " ")
    return text

  lines = [
    "# Paper search results",
    "",
    f"Topics searched: {', '.join(topics)}",
    "",
    "| Rank | Paper title | Authors | Year | Journal | Paper type | | Markdown path | PDF Path | Count |",
  ]

  for rank, (paper_id, paper) in enumerate(
    papers_by_id.items(),
    start=1
  ):
    authors = ", ".join(
      str(author)
      for author in paper.get("authors", [])
    )

    lines.append(
      "| "
      f"{rank} | "
      f"{cell(paper.get('paper_title'))} | "
      f"{cell(authors)} |"
      f"{cell(paper.get('year'))} |"
      f"{cell(paper.get('journal'))} | "
      f"{cell(paper.get('paper_type'))} | "
      f"{cell(paper.get('markdown_path'))} | "
      f"{cell(paper.get('pdf_path'))} | "
      f"{paper.get('count', 0)} |"
    )

  return "\n".join(lines) + "\n"