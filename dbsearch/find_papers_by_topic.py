from .llm_services.llm_worker import LLMWorker
from .search_services.search import SearchLanceDB
from .conversation_tracker import ConversationTracker

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
import csv
import uuid
from datetime import datetime
import json

from .utils.papers_to_md_format import papers_to_markdown_format
from .utils.paper_record import parse_paper_results, PaperRecord
from .utils.paper_matching import find_best_paper_match

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
    self.summaries_metadata: Path = self.tracking_directory / "summaries_metadata.csv"
    self.ensure_conversation_metadata_file()

    # Set up RAG
    self.lancedb = SearchLanceDB()

    # Use this to manage the conversation
    self.conversation_tracker = conversation_tracker

    # If this class is started with a conversation number, add the papers_by_id data from the relevant data source
    self.papers_by_id: dict[str, dict[str, Any]] = {}

    # find_papers_ids - there may be multiple ids associated with a given conversation
    # This is taken from the find_paper_metadata.csv which contains the find_papers_id
    # The find_papers_id associated a query to find papers with the results of the papers found
    # The results are stored in the md file listed under papers_list_md_file in the find_papers_metadata.csv
    self.find_papers_ids: list[str] | None = None

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

  def set_find_papers_ids_by_conversation_id(self) -> None:
    """
    A helper function
    Returns the find_paper_id values from find_papers_metadata.csv file in conversation_tracking
    """
    conv_id = self.conversation_tracker.get_current_conversation_id()
    self.find_papers_ids = []
    with self.find_papers_metadata.open(
      "r",
      newline="",
      encoding="utf-8",
    ) as csv_file:
      reader = csv.DictReader(
        csv_file
      )
      self.find_papers_ids = [
        row["find_papers_id"]
        for row in reader
        if row.get("conversation_id") == conv_id
      ]

  def get_find_papers_ids(self) -> list[str]:
    """A helper function"""
    return self.find_papers_ids

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
    The llm reads each section of the paper and determines the extent to which it contains an answer to the
    user's query, returning a result "high", "medium", "low" and "none".
    Any query with a relevance of "high", "medium" or "low" gets summarized by the llm.
    The summarized metadata is saved to summaries_metadata.csv and the summary itself is saved
    as an md file in the summaries folder.
    """
    # This function below uses the current conversation_id to set the find_papers_ids
    # that are associated with that conversation_id
    self.set_find_papers_ids_by_conversation_id()

    # Get the md file for each of the find_papers_ids
    papers: list[PaperRecord] = self.load_papers_for_find_ids()

    # Now use the find_best_paper_match for the title provided by comparing them with the papers
    best = find_best_paper_match(paper_title, papers)

    # Get the specific paper's path
    print(f"The paper matched from your provided title is titled: {best.paper.paper_title}, published in {best.paper.year} and the authors are {best.paper.authors}")
    best_paper_md_path = best.paper.markdown_path

    # Check that the classified_chunks.json file exists.
    if not best_paper_md_path:
      raise FileNotFoundError(
        f"No Markdown path found for {best.paper.paper_title!r}"
      )

    # Resolve relative paths from the repository root
    repo_root = Path(__file__).resolve().parents[1]
    markdown_path = Path(best_paper_md_path)
    if not markdown_path.is_absolute():
      markdown_path = repo_root / markdown_path

    classified_json_name = (
      markdown_path.name.removesuffix("_filtered.md")
      + "_classified_chunks.json"
    )

    classified_json_path = markdown_path.with_name(classified_json_name)

    summary_metadata_fieldnames = [
      "summary_id",
      "conversation_id",
      "content_path",
      "paper_title",
      "query",
      "section_summarized",
      "relevance",
      "summary_md"
    ]

    llm = LLMWorker(
      python_executable="/Users/danielmikaleola/.unsloth/unsloth_gemma4_mlx/bin/python",
      runner_path=Path(__file__).resolve().parents[1] / "chunker" / "mlx_llm_runner.py",
      model_path="unsloth/gemma-4-E4B-it-UD-MLX-4bit",
      request_timeout_seconds=120.0
    )

    system_prompt = (
      "You read a section from an academic paper. "
      "How well does the section answers the user's query? "
      "Judge how well the section answers the user's query with 'well', 'reasonable', 'irrelevant' "
      "Output only those labels. Do not output any other comments."
      )

    user_prompt = "Here is a section of the academic paper: \n"

    if classified_json_path.exists():
      classified_payload = json.loads(
        classified_json_path.read_text(encoding="utf-8")
      )
      headings = classified_payload.get("headings", [])

      # Create markdown file for summaries
      now = datetime.now().isoformat(timespec="seconds")
      summaries_directory = (
        Path(__file__).resolve().parents[1] / "data" / "conversation_tracking" / "summaries"
      )  
      summaries_directory.mkdir(parents=True, exist_ok=True)
      summary_path = (
        summaries_directory / f"{now}_{best.paper.paper_title}.md"
      )
      summary_markdown = (
        f"# Summaries of sections relevant to the query: {query} \n"
        f"**Paper:** {best.paper.paper_title}\n"
        f"**Authors:** {', '.join(best.paper.authors)}\n"
        f"**Year:** {best.paper.year}\n"
      )

      summary_path.write_text(
        summary_markdown,
        encoding="utf-8"
      )


      for heading in headings:
        section_title = heading.get("title")
        content = heading.get("content")
        updated_user_prompt = user_prompt + content + "\n\n" + f"The user's question about the paper is: {query}. Does the section answer the user's query?"
        messages = [
          {
            "role": "system",
            "content": system_prompt
          },
          {
            "role": "user",
            "content": updated_user_prompt
          }
        ]
        relevance = llm.generate(
          messages=messages,
          max_tokens=8,
          temperature=0.0
        )

        if relevance.strip().lower() in {"well", "reasonable"}:
          system_prompt_summarize = (
            "You summarize a section from an academic paper. "
            "Your summary should be focused on answering the user's query using only the information in the section."
            "However, do not comment on the relevance of the section to the user's query."
            "You write in an academic style."
            "When summarizing any information containing a citation, write 'as cited by...'."
          )

          authors = ", ".join(best.paper.authors)
          user_prompt_summarize = (
            "Here is the section of the text: \n"
            f"{content} \n\n"
            f"The user's query about the text is: {query} \n"
            f"Write the summary to in response to the query. \n"
            f"The authors of the paper are: {authors}.\n"
            f"The publication year was: {best.paper.year}"
          )
          messages_summarize = [
            {
              "role": "system",
              "content": system_prompt_summarize
            },
            {
              "role": "user",
              "content": user_prompt_summarize
            }
          ]
          summary = llm.generate(
            messages=messages_summarize,
            max_tokens=512,
            temperature=0.7
          )

          # Now save this summary information into the summaries_metadata.csv
          # And save the summary to a specific md file
          summary_id = uuid.uuid4().hex
          file_is_empty = (
            not self.summaries_metadata.exists()
            or self.summaries_metadata.stat().st_size == 0
          )
          with self.summaries_metadata.open(
            "a",
            newline="",
            encoding="utf-8"
          ) as csv_file:
            writer = csv.DictWriter(
              csv_file,
              fieldnames=summary_metadata_fieldnames
            )
            if file_is_empty:
              writer.writeheader()
            writer.writerow(
              {
                "summary_id": summary_id,
                "conversation_id": self.conversation_tracker.get_current_conversation_id(),
                "content_path": str(classified_json_path),
                "paper_title": best.paper.paper_title,
                "query": query,
                "section_summarized": section_title,
                "relevance": relevance.strip().lower(),
                "summary_md": f"{now}_{best.paper.paper_title}.md"
              }
            )
          # Append to the paper's summary
          with summary_path.open(
            "a",
            encoding="utf-8"
          ) as summary_file:
            summary_file.write(
              f"\n## {section_title}\n\n"
              f"{summary.strip()}\n"
            )
          
          


    else:
      classified_payload = None
      # Use md file and slice and summarize every 1000

    

  def load_papers_for_find_ids(self) -> list[PaperRecord]:
    """A helper function"""
    requested_ids = set(self.find_papers_ids)
    papers: list[PaperRecord] = []

    with self.find_papers_metadata.open(
      "r",
      newline="",
      encoding="utf-8", 
    ) as csv_file:
      reader = csv.DictReader(csv_file)

      for row in reader:
        if row.get("find_papers_id") not in requested_ids:
          continue

        markdown_path = row.get("papers_list_md_file")

        if not markdown_path:
          continue

        papers.extend(parse_paper_results(markdown_path))

    return papers



