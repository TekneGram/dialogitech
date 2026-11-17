from .llm_services.llm_worker import LLMWorker
from .search_services.search import SearchLanceDB

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Paper:
  paper_title: str | None
  authors: list[str] = field(default_factory=list)
  year: int | None = None
  journal: str | None = None
  paper_type: str | None = None
  markdown_path: str | None = None
  pdf_path: str | None = None
  count: int = 0

class FindPapersByTopic:

  def __init__(
      self
  ) -> None:
    self.llm_worker = LLMWorker(
      python_executable="/Users/danielmikaleola/.unsloth/unsloth_gemma4_mlx/bin/python",
      runner_path=Path(__file__).resolve().parents[1] / "chunker" / "mlx_llm_runner.py",
      model_path="unsloth/gemma-4-E4B-it-UD-MLX-4bit",
      request_timeout_seconds=120.0
    )
    self.lancedb = SearchLanceDB()

    # If this class is started with a conversation number, add the papers_by_id data from the relevant data source
    self.papers_by_id: dict[str, dict[str, Any]] = {}

  def find_by_topics(self, topics: list[str]) -> dict[str, dict[str, Any]]:

    for topic in topics:
      results = self.lancedb.basic_search(text_to_search=topic, limit=100)

      for row in results:
        paper_id = row["paper_id"]

        if paper_id not in self.papers_by_id:
          self.papers_by_id[paper_id] = {
            "paper_title": row.get("paper_title"),
            "authors": row.get("authors", []),
            "year": row.get("year"),
            "journal": row.get("journal"),
            "paper_type": row.get("paper_type"),
            "markdown_path": row.get("markdown_path"),
            "pdf_path": row.get("pdf_path"),
            "count": 0
          }

        self.papers_by_id[paper_id]["count"] += 1

    # Send papers_by_id to an md file and format it nicely.
    # Save topic with a conversation number and a conversation name
    # Save papers_by_id data to a conversation-number linked item

    # Sort in descending order
    self.papers_by_id = dict(
      sorted(
        self.papers_by_id.items(),
        key=lambda item: item[1].get("count", 0),
        reverse = True
      )
    )
    print(self.papers_by_id)
    return self.papers_by_id

  def get_an_abstract(self, paper_title: str) -> None:
    """
    Uses the paper title to find the file and return the abstract
    We need to develop a function or class that can handle finding specific papers based on paper titles since
    this is a recurring request.
    """

    # Get the markdown_path for the paper title
      # If the paper title doesn't turn up a markdown path because e.g., the title is not an exact match, then use an edit distance formula to match all the 
      # titles in the self.papers_by_id for the most likely paper title

    # Search for the ..._classified_chunks.json file
    # Inside the json file search for the Headings, Abstract and get content (This should be the whole abstract)
    # If there is no json file or the Headings Abstract does not exist, search for the md file
    # Inside the md file look for the heading #Abstract, and slice all the text from #Abstract up to the next title heading. Send to the llm and ask the llm to return only the abstract after checking it is the abstract
    # If that doesn't exist, slice the top 1000 words from the md file, send it to the llm and ask the llm to return the abstract part only.

  def query_a_paper(self, query: str, paper_title: str) -> None:
    """
    Use the paper title to query a paper.
    The llm should look at each section in the paper and summarize it. After summarizing it, it should comment on
    the relevance of that section of the paper to the query
    Both of these should be parsed computationally into a json object as follows:
    {
      "query": "how does this section address trust in LLMs?"
      "paper_info":  {self.papers_by_id.item}
      "summaries": [
        {
          "section": "Method",
          "summary": "...",
          "comment": "...",
          "relevance_to_query": "high/medium/low"
        },
        {
          ...
        }
      ]
    }
    If there is only an md file and no json file, then use the headings to determine the section and slice by headings.
    If there are no headings in the md file, then slice every 1000 words and have the llm determine the section.
    """
