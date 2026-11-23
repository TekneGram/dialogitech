from .llm_services.llm_worker import LLMWorker
from .conversation_tracker import ConversationTracker

from pathlib import Path
from dataclasses import dataclass, field
import csv
import uuid
from datetime import datetime
import json

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


class SynthesizeSummaries:

  def __init__(
      self,
      conversation_tracker: ConversationTracker
  ) -> None:

    self.tracking_directory = (
      Path(__file__).resolve().parents[1] / "data" / "conversation_tracking"
    )

    self.summaries_metadata: Path = self.tracking_directory / "summaries_metadata.csv"
    self.summaries_md_path: Path = self.tracking_directory / "summaries"
    self.syntheses_path: Path = self.tracking_directory / "syntheses"
    self.synthesis_metadata: Path = self.tracking_directory / "synthesis_metadata.csv"

    self.conversation_tracker = conversation_tracker

  def synthesize_summaries(self, user_request: str) -> None:

    # For the current conversation id, get the set of summary files
    conversation_id = self.conversation_tracker.get_current_conversation_id()
    summary_paths = self.get_summary_paths_for_conversation(conversation_id)

    # Start the LLM worker
    llm = LLMWorker(
      python_executable="/Users/danielmikaleola/.unsloth/unsloth_gemma4_mlx/bin/python",
      runner_path=Path(__file__).resolve().parents[1] / "chunker" / "mlx_llm_runner.py",
      model_path="unsloth/gemma-4-E4B-it-UD-MLX-4bit",
      request_timeout_seconds=120.0
    )

    # STEP 1: Understand the requested information
    # First, use the LLM to rewrite the user's request for information extraction
    # because we need to know what aspect of the summary we need to focus on

    # STEP 2: Extract the requested information from each summary
    # Since paper summaries could be quite extensive, we should extract each 
    # the relevant information from the summary first, along with the full citation.
    # Then collate extracted information.

    # STEP 3: Synthesize the collated information
    # Once all the information that is relevant has been collated
    # we can synthesize it in plain English

    # STEP 1
    system_prompt_1 = (
      "You extract information from summaries of academic papers. "
      "Read the user's request and understand the information the user is looking for. "
      "Rewrite the user's request to focus on the specific information that the user "
      "would like to extract from the summaries. When re-writing the request, ensure "
      "details about how to summarize are excluded. Focus just on information the user wishes to extract as if it is requested from just one paper. "
      "Do not comment on the user's request. \n\n"
      "Example user request 1: Write a synthesis of the summaries focusing on the key points of "
      "trust in LLMs. Focus first on malicious uses of LLMs. Then focus on hallucinations. Finally, focus on comparing trust in humans and trust in LLMs. \n\n"
      "Example paraphrase from user request 1: Where available, briefly summarize key points related to malicious use of LLMS, hallucination and comparing trust in humans and trust in LLMs. \n\n"
      "Example user request 2: Synthesis the summaries, but make sure to focus on the experimental approaches used in each paper. \n\n"
      "Example paragraph from user request 2: Where available, briefly summarize the experimental approach used in the paper. \n\n"
    )

    messages_1 = [
      {
        "role": "system",
        "content": system_prompt_1
      },
      {
        "role": "user",
        "content": user_request
      }
    ]
    information_request = llm.generate(
      messages=messages_1,
      max_tokens=128,
      temperature=0.7
    )

    print(f"Each summary will be queried according to the following information request: {information_request}")

    # STEP 2
    system_prompt_2 = (
      "You extract information from summaries according to information requested by the user. "
      "You must extract only the information from the summary. "
      "Do not add new information. Do not comment on or refer to the summary itself. "
      "Do not include any secondary citations (e.g., as cited by). "
      "Balance being concise with thorough coverage, focusing on the information requested. "
      "After the summary, write the full APA reference for the paper. "
      "For example: \n\n"
      "Jones, T., & Lakoff, R. (2023). The use of data in language analysis. *Language Analysis Journal 3*(2), pp. 12-33."
    )

    extracted_information_list: list[str] = []

    for summary_path in summary_paths:
      summary_text = summary_path.read_text(encoding="utf-8")
      messages_2 = [
        {
          "role": "system",
          "content": system_prompt_2
        },
        {
          "role": "user",
          "content": information_request + "\n\nHere is the summary: \n\n" + summary_text
        }
      ]
      extracted_information = llm.generate(
        messages=messages_2,
        max_tokens=1024,
        temperature=0.7
      )
      extracted_information_list.append(f"\n\nEXTRACTED INFO: {extracted_information}")

    # STEP 3
    extracted_information_text = "\n\n".join(extracted_information_list)
    print(extracted_information_text)
    system_prompt_3 = (
      "You synthesize a set of summaries according to the user's request. "
      "You write in an academic style. "
      "You must cite the specific paper you are summarizing using the reference provided. "
      "For each paper summarized, add its reference as provided in APA style at the end of the synthesis."
    )
    messages_3 = [
      {
        "role": "system",
        "content": system_prompt_3
      },
      {
        "role": "user",
        "content": user_request + "\n\nHere is the information to synthesize:\n\n" + extracted_information_text
      }
    ]

    synthesis = llm.generate(
      messages=messages_3,
      max_tokens=2048,
      temperature=0.7
    )
    print(f"\n\nSYNTHESIS: {synthesis}")

    # Save synthesis to file
    synthesis_id = uuid.uuid4().hex
    now = datetime.now().isoformat(timespec="seconds")
    self.syntheses_path.mkdir(parents=True, exist_ok=True)
    synthesis_path = (
      self.syntheses_path / f"{now}_{synthesis_id}.md"
    )
    synthesis_path.write_text(f"# SYNTHESIS \n\n ## Initial Request \n {user_request} \n\n ## Synthesis: \n\n")
    with synthesis_path.open(
      "a",
      encoding="utf-8"
    ) as synthesis_file:
      synthesis_file.write(
        f"{synthesis.strip()}\n"
      )
    
    # Save metadata
    synthesis_fieldnames = [
      "synthesis_id",
      "conversation_id",
      "user_request",
      "path_to_synthesis"
    ]
    file_is_empty = (
      not self.synthesis_metadata.exists()
      or self.synthesis_metadata.stat().st_size == 0
    )
    with self.synthesis_metadata.open(
      "a",
      newline="",
      encoding="utf-8"
    ) as csv_file:
      writer = csv.DictWriter(
        csv_file,
        fieldnames=synthesis_fieldnames
      )
      if file_is_empty:
        writer.writeheader()
      writer.writerow(
        {
          "synthesis_id": synthesis_id,
          "conversation_id": self.conversation_tracker.get_current_conversation_id(),
          "user_request": user_request,
          "path_to_synthesis": synthesis_path
        }
      )

  def get_summary_paths_for_conversation(
      self, conversation_id: str
  ) -> set[Path]:

    summary_paths: set[Path] = set()

    with self.summaries_metadata.open(
      "r",
      newline="",
      encoding="utf-8"
    ) as csv_file:
      reader = csv.DictReader(csv_file)

      for row in reader:
        if row.get("conversation_id") != conversation_id:
          continue

        summary_md = row.get("summary_md")

        if summary_md:
          summary_md_path = Path(summary_md)
          summary_paths.add(self.summaries_md_path / summary_md_path)
    
    return summary_paths