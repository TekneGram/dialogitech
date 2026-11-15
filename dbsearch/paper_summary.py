from .llm_services.llm_worker import LLMWorker
from pathlib import Path
from .response_parser.response_parser import ResponseParser
from .search_services.search import SearchLanceDB
from .cache_manager.cache_manager import CacheManager
from chunker.llm_section_type_classifier_helpers.section_taxonomy import SectionLabel
from collections.abc import Collection
from .search_services.metadata_search import MetadataSearch

import re
from collections import Counter
import json
from datetime import datetime
import csv

from dataclasses import dataclass

@dataclass(slots=True)
class SectionRecord:
  title: str
  label: str
  label_counts: dict
  text: str

class SummarizePaper:

  def __init__(self) -> None:
    # Start the llm immediately
    self.llm_worker = LLMWorker(
      python_executable="/Users/danielmikaleola/.unsloth/unsloth_gemma4_mlx/bin/python",
      runner_path=Path(__file__).resolve().parents[1] / "chunker" / "mlx_llm_runner.py",
      model_path="unsloth/gemma-4-E4B-it-UD-MLX-4bit",
      request_timeout_seconds=120.0
    )

  def summarize_paper(
      self,
      paper_title: str
  ) -> None:
    """
    Parses the paper section by section
    summarizes each section
    saves the summary to a summary file
    provides the name of the file where the summary was saved
    """

    md_search = MetadataSearch()
    md_search_results = md_search.get_paper_by_title(paper_title)
    if not md_search_results.exists:
      print("Unfortunately, your paper title returned nothing!")
      return

    # Check if the summary already exists for this paper
    if self.summary_exists(md_search_results.paper_title) == True:
      summary_filename = f"{md_search_results.markdown_path.stem}_summary{md_search_results.markdown_path.suffix}"
      print(f"That paper already has a summary, probably: {summary_filename}")
      return

    if md_search_results.classified_chunks_path is not None:
      # This is the case where sections have been properly chunked
      summary_results_directory = (
          Path(__file__).resolve().parents[1] / "summary_results"
        )
      summary_results_directory.mkdir(parents=True, exist_ok=True)
      summary_filename = f"{md_search_results.markdown_path.stem}_summary{md_search_results.markdown_path.suffix}"
      summary_path = summary_results_directory / summary_filename
    
      # Extract the sections texts
      sections_text = self.get_sections_text(json_path=md_search_results.classified_chunks_path)
      for section_text in sections_text:
        summary = self.summarize_section(section_text)
        with summary_path.open(
          "a",
          encoding="utf-8"
        ) as summary_file:
          summary_file.write("\n\n")
          summary_file.write(summary)
      print(f"Paper summarized at {summary_path}")
      
      # Record the data to the summary_results.csv
      self.record_to_csv(paper_title=paper_title, markdown_path=md_search_results.markdown_path, summary_path=summary_path)

    else:
      # In this case, sections have not been properly chunked meaning only the md file exists, so summarize the md file.
      self.summarize_full_paper(md_search_results.markdown_path)
  
  def summary_exists(self, paper_title: str) -> bool:
    """
    Returns True if a summary for a paper already exists
    Returns False if the summary has not been recorded in the csv file.
    """
    summary_results_directory = (
      Path(__file__).resolve().parents[1] / "summary_results"
    )
    summary_csv_path = summary_results_directory / "summary_results.csv"

    with summary_csv_path.open(
      "r",
      newline="",
      encoding="utf-8"
    ) as csv_file:
      reader = csv.DictReader(csv_file)

      already_summarized = any(
        row.get("paper_title", "").strip() == paper_title.strip()
        for row in reader
      )

      if already_summarized:
        return True
      else:
        return False

  def get_sections_text(
      self,
      json_path: Path,
  ) -> list[SectionRecord]:
    """
    Loop through each section in the json and find the section heading
    Get the section heading from the
    """
    json_data = json.loads(
      json_path.read_text(encoding="utf-8")
    )

    results: list[SectionRecord] = []

    for heading in json_data.get("headings", []):
      title = heading.get("title")
      chunks = heading.get("chunks")
      content = heading.get("content")

      if not isinstance(title, str):
        continue

      if not isinstance(chunks, list) or len(chunks) == 0:
        continue

      labels: list[str] = []

      for chunk in chunks:
        classification = chunk.get("classification", {})

        if not isinstance(classification, dict):
          continue

        label = classification.get("label")

        if isinstance(label, str) and label.strip():
          labels.append(label)

      if not labels:
        continue

      label_counts = Counter(labels)
      selected_label = label_counts.most_common(1)[0][0]


      results.append(
        SectionRecord(
          title=title,
          label=selected_label,
          label_counts=dict(label_counts),
          text=content
        )
      )
    
    return results


  #### SUMMARIZING SECTIONS #####

  def summarize_section(
      self,
      section: SectionRecord,
  ) -> str:
    """
    Call the LLM and summarize that section depending on the type of section it is.
    Save the summary metadata to summary_results/summary_results.csv and save the summary md file to summary_results folder
    """

    # Chunk the text into 500 word chunks to be summarized
    chunk_size=500
    words = re.findall(r"\S+", section.text)
    text_chunks = [
      " ".join(words[start:start + chunk_size])
      for start in range(0, len(words), chunk_size)
    ]

    summary = ""
    for chunk in text_chunks:
      new_summary = self.llm_summarizer_sections(chunk, section.title, section.label, summary)
      summary = summary + " " + new_summary
    
    return summary


  def llm_summarizer_sections(self, text_to_summarize: str, section_title: str, section_label: str, previous_summary: str) -> str:
    

    if previous_summary == "":
      if section_label != "":
        system_prompt_new = f"You summarize from part of an academic paper. The specific part has the title '{section_label}: {section_title}'. Start the summary with the title '{section_label}: {section_title}'. Focus only on writing a summary. The summary must be much shorter than the original. Do not add any extra comments."
      else:
        system_prompt_new = f"You summarize from part of an academic paper. The specific part has the title '{section_title}'. Start the summary with the title '{section_title}'. Focus only on writing a summary. The summary must be much shorter than the original. Do not add any extra comments."
      messages = [
        {
          "role": "system",
          "content": system_prompt_new
        }
      ]
    else:
      if section_label != "":
        system_prompt_continue = f"You summarize from part of an academic paper. The specific part has the title '{section_label}: {section_title}'. You have already summarized part of it as follows: {previous_summary} \n\n Now you must continue the summary. Focus on the user's text only and use the part already summarized just to help. Summarize only. The summary should be much shorter than the original Do not add extra comments."
      else:
        system_prompt_continue = f"You summarize from part of an academic paper. The specific part has the title '{section_title}'. You have already summarized part of it as follows: {previous_summary} \n\n Now you must continue the summary. Focus on the user's text only and use the part already summarized just to help. Summarize only. The summary should be much shorter than the original. Do not add extra comments."
      messages = [
        {
          "role": "system",
          "content": system_prompt_continue
        }
      ]

    # Set up the user role in messages
    messages.append(
      {
        "role": "user",
        "content": text_to_summarize
      }
    )


    # Set up the LLM Worker and send the processing request
    

    summary = self.llm_worker.generate(
      messages=messages,
      max_tokens=500,
      temperature=0.7
    )

    return summary



  ##### SUMMARIZE THE FULL PAPER WITHOUT SECTIONS #####
  def summarize_full_paper(
      self,
      markdown_path: Path,
      paper_title: str,
  ) -> None:
    """
    This function assumes that we do not have the paper chunked
    So we summarize every 500 words
    Each summary is fed into the next 500 words to keep the thread
    Save the summary metadata to summary_results/summary_results.csv and save the summary md file to summary_results folder
    """
    # Open the markdown file using the markdown_path and read the text
    if not markdown_path.exists():
      raise FileNotFoundError(
        f"Markdown file not found: {markdown_path}"
      )

    markdown_text = markdown_path.read_text(
      encoding="utf-8"
    )

    # Chunk the text into 500 word chunks
    words = re.findall(r"\S+", markdown_text)
    chunk_size = 500
    chunked_texts = [
      " ".join(words[start:start + chunk_size])
      for start in range(0, len(words), chunk_size)
    ]

    # Loop through each chunk of text and summarize it
    summary = ""
    for text in chunked_texts:
      new_summary = self.llm_summarizer_full_paper(text, summary)
      summary = summary + "\n\n" + new_summary

    # Save the details about the summary in the summary_results.csv file and save the full summary to its own file
    summary_results_directory = (
      Path(__file__).resolve().parents[1] / "summary_results"
    )
    summary_results_directory.mkdir(parents=True, exist_ok=True)
    summary_filename = f"{markdown_path.stem}_summary{markdown_path.suffix}"
    summary_path = summary_results_directory / summary_filename
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Paper summarized at {summary_path}")

    # Record the data to the summary_results.csv
    self.record_to_csv(paper_title=paper_title, markdown_path=markdown_path, summary_path=summary_path)


  def llm_summarizer_full_paper(
      self,
      text_to_summarize: str,
      previous_section_summary: str | None
  ) -> str:

    if previous_section_summary != "":
      messages = [
        {
          "role": "system",
          "content": f"You summarize a chunk of text from an academic paper. The previous chunks were summarized as follows: {previous_section_summary}. \n Focus only on summarizing the text that follows and use only the previous chunk summary as helpful context. Keep the summary to around 10 sentences. Write only the summary. Do not add extra comments."
        }
      ]
    else:
      messages = [
        {
          "role": "system",
          "content": f"You summarize a chunk of text from the beginning of an academic paper. If the paper contains abstract and front matter, ignore them and focus only on the introduction part. Focus only on summarizing the introduction. Keep the summary brief. Write only the summary. Do not add extra comments."
        }
      ]

    # Set up the user role in messages
    messages.append(
      {
        "role": "user",
        "content": text_to_summarize
      }
    )

    summary = self.llm_worker.generate(
      messages=messages,
      max_tokens=500,
      temperature=0.7
    )

    return summary

  def record_to_csv(self, paper_title: str, markdown_path: str, summary_path: str) -> None:
    summary_results_directory = (
          Path(__file__).resolve().parents[1] / "summary_results"
        )
    summary_csv_path = summary_results_directory / "summary_results.csv"
    
    fieldnames = [
      "paper_title",
      "markdown_path",
      "summary_path",
      "date",
      "time"
    ]

    file_exists = summary_csv_path.exists()
    now = datetime.now()

    with summary_csv_path.open(
      "a",
      newline="\n",
      encoding="utf-8",
    ) as csv_file:
      writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
      )

      if not file_exists:
        writer.writeheader()

      writer.writerow(
        {
          "paper_title": paper_title,
          "markdown_path": str(markdown_path),
          "summary_path": str(summary_path),
          "date": now.strftime("%Y-%m-%d"),
          "time": now.strftime("%H:%M:%S")
        }
      )