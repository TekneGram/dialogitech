
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
    "| Rank | Paper title | Authors | Year | Journal | Paper type | Markdown path | PDF path | Count |",
    "| ---: | --- | --- | ---: | --- | --- | --- | --- | ---: |",
  ]

  for rank, (paper_id, paper) in enumerate(
    papers_by_id.items(),
    start=1
  ):
    authors = ", ".join(
      str(author)
      for author in paper.get("authors", [])
    )

    cells = [
      str(rank),
      cell(paper.get("paper_title")),
      cell(authors),
      cell(paper.get("year")),
      cell(paper.get("journal")),
      cell(paper.get("paper_type")),
      cell(paper.get("markdown_path")),
      cell(paper.get("pdf_path")),
      str(paper.get("count", 0)),
    ]
    lines.append(f"| {' | '.join(cells)} |")

  return "\n".join(lines) + "\n"
