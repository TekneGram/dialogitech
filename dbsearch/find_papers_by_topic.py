from .llm_services.llm_worker import LLMWorker
from .search_services.search import SearchLanceDB
from .conversation_tracker import ConversationTracker

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
import csv
import uuid
from datetime import datetime

from .utils.papers_to_md_format import papers_to_markdown_format

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
      self,
      conversation_tracker: ConversationTracker
  ) -> None:

    # Set up directories and files in the class
    self.tracking_directory = (
          Path(__file__).resolve().parents[1] / "data" / "conversation_tracking"
        )
    self.find_papers_metadata: Path = self.tracking_directory / "find_papers_metadata.csv"
    self.find_papers_directory: Path = self.tracking_directory / "find_papers"
    self.ensure_conversation_metadata_file()

    # Set up RAG
    self.lancedb = SearchLanceDB()

    # Use this to manage the conversation
    self.conversation_tracker = conversation_tracker

    # If this class is started with a conversation number, add the papers_by_id data from the relevant data source
    self.papers_by_id: dict[str, dict[str, Any]] = {}

  def ensure_conversation_metadata_file(self) -> None:
    """
    Make sure the find_papers metadata file exists
    If it doesn't, then create it.
    """

    self.tracking_directory.mkdir(
      parents=True,
      exist_ok=True,
    )

    if not self.find_papers_metadata.exists():
      fieldnames=[
        "find_papers_id",
        "conversation_id",
        "created",
        "papers_list_md_files"
      ]

      with self.find_papers_metadata.open(
        "w",
        newline="",
        encoding="utf-8",
      ) as csv_file:
        writer=csv.DictWriter(
          csv_file,
          fieldnames=fieldnames
        )
        writer.writeheader()

  def find_by_topics(self, topics: list[str]) -> None:
    """
    Get a list of topics / queries by the researcher
    Search for papers that match those topics
    """
    
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

    # Sort in descending order
    self.papers_by_id = dict(
      sorted(
        self.papers_by_id.items(),
        key=lambda item: item[1].get("count", 0),
        reverse = True
      )
    )

    # Manage the conversation up to now
    if self.conversation_tracker.get_current_conversation_id() is None:
      # If there if no conversation yet, start a new one
      all_queries = " ".join(
        f"query {index}: {value}"
        for index, value in enumerate(topics)
      )
      title = self.conversation_tracker.create_conversation_title(context=all_queries)
      conversation_id, conversation_file = self.conversation_tracker.start_new_conversation(title)
      print(f"You started a new conversation. \n ID: {conversation_id} \n FILE: {conversation_file}")

    # Save a new find_papers_list md file with metadata
    find_papers_id = uuid.uuid4().hex
    papers_list_md_file = self.find_papers_directory / f"{find_papers_id}.md"
    papers_list_md_file.touch()
    now = datetime.now().isoformat(timespec="seconds")

    fieldnames = [
      "find_papers_id",
      "conversation_id",
      "created",
      "papers_list_md_file"
    ]

    file_is_empty = (
      not self.find_papers_metadata.exists()
      or self.find_papers_metadata.stat().st_size == 0
    )

    with self.find_papers_metadata.open(
      "a",
      newline="",
      encoding="utf-8"
    ) as csv_file:
      writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
      )

      if file_is_empty:
        writer.writeheader()

      writer.writerow(
        {
          "find_papers_id": find_papers_id,
          "conversation_id": self.conversation_tracker.get_current_conversation_id(),
          "created": now,
          "papers_list_md_file": papers_list_md_file
        }
      )

    # Record the conversation
    self.conversation_tracker.update_conversation(f"New LanceDB search for files by topics, with information at {papers_list_md_file}")
    
    # Send papers_by_id to an md file and format it nicely.
    md_format = papers_to_markdown_format(papers_by_id=self.papers_by_id, topics=topics)
    papers_list_md_file.write_text(md_format, encoding="utf-8")


    

    
    print(f"Done! Check the file {papers_list_md_file} for information about each file found.")

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
