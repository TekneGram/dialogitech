from .llm_services.llm_worker import LLMWorker

from pathlib import Path
import csv
import uuid
from datetime import datetime

class ConversationTracker:

  def __init__(
      self,
      conversation_id: str | None = None
    ) -> None:
    # Set up directories and files in the class
    self.tracking_directory = (
          Path(__file__).resolve().parents[1] / "data" / "conversation_tracking"
        )
    self.conversation_metadata: Path = self.tracking_directory / "conversation_metadata.csv"
    self.conversations_directory: Path = self.tracking_directory / "conversations"
    self.ensure_conversation_metadata_file()

    # Set up the conversations context
    self.conv_titles_ids: dict[str, str] = {}
    self.current_conversation_id: str | None = conversation_id
    self.aggregate_conversation_titles()

    # Set up LLM
    self.llm_worker = LLMWorker(
      python_executable="/Users/danielmikaleola/.unsloth/unsloth_gemma4_mlx/bin/python",
      runner_path=Path(__file__).resolve().parents[1] / "chunker" / "mlx_llm_runner.py",
      model_path="unsloth/gemma-4-E4B-it-UD-MLX-4bit",
      request_timeout_seconds=120.0
    )

  def get_current_conversation_id(self) -> str | None:
    if self.current_conversation_id is not None:
      return self.current_conversation_id
    else:
      return None

  def ensure_conversation_metadata_file(self) -> None:
    """
    Make sure the conversation metadata file exists
    If it doesn't, then create it.
    """
    self.tracking_directory.mkdir(
      parents=True,
      exist_ok=True,
    )

    if not self.conversation_metadata.exists():
      fieldnames=[
        "conversation_id",
        "started",
        "updated",
        "conversation_title",
        "conversation_history_md_file"
      ]

      with self.conversation_metadata.open(
        "w",
        newline="",
        encoding="utf-8",
      ) as csv_file:
        writer = csv.DictWriter(
          csv_file,
          fieldnames=fieldnames
        )
        writer.writeheader()

  def aggregate_conversation_titles(self) -> None:
    """
    Called on _init_ which means
    to get the conversation titles we just pull them
    """
    with self.conversation_metadata.open(
      "r",
      newline="",
      encoding="utf-8",
    ) as csv_file:
      reader = csv.DictReader(csv_file)

      for row in reader:
        conversation_title = row["conversation_title"]
        conversation_id = row["conversation_id"]
        self.conv_titles_ids[conversation_title] = conversation_id

  def get_conversation_titles(self) -> dict[str, str]:
    return self.conv_titles_ids

      
  def get_conversation_history_path_by_title(self, conversation_title: str) -> Path | None:
    with self.conversation_metadata.open(
      "r",
      newline="",
      encoding="utf-8"
    ) as csv_file:
      reader = csv.DictReader(csv_file)

      for row in reader:
        if row.get("conversation_title") == str(conversation_title):
          return Path(row.get("conversation_history_md_file"))

    raise ValueError(
      f"No row found for conversation title {conversation_title}"
    )

  def get_conversation_history_path_by_id(self, conversation_id: str) -> Path | None:

    with self.conversation_metadata.open(
      "r",
      newline="",
      encoding="utf-8",
    ) as csv_file:
      reader = csv.DictReader(csv_file)

      for row in reader:
        if row.get("conversation_id") == str(conversation_id):
          self.current_conversation_id = conversation_id
          return Path(row.get("conversation_history_md_file"))

    raise ValueError(
      f"No row found for conversation_id={conversation_id}"
    )

  def start_new_conversation(self, conversation_title: str) -> tuple[str, Path]:
    """
    Starts a new line in the csv file.
    The conversation_title should be passed in by the caller, not handled here.
    Return the conversation_id and the Path of the md file where the conversation is held
    """
    conversation_id = uuid.uuid4().hex

    # Create a new md file with the conversation_id
    conversation_file = self.conversations_directory / f"{conversation_id}.md"
    conversation_file.touch()

    now = datetime.now().isoformat(timespec="seconds")

    fieldnames = [
      "conversation_id",
      "started",
      "updated",
      "conversation_title",
      "conversation_history_md_file"
    ]

    file_is_empty = (
      not self.conversation_metadata.exists()
      or self.conversation_metadata.stat().st_size == 0
    )

    # Update csv metadata
    with self.conversation_metadata.open(
      "a",
      newline="",
      encoding="utf-8",
    ) as csv_file:
      writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
      )

      if file_is_empty:
        writer.writeheader()

      writer.writerow(
        {
          "conversation_id": conversation_id,
          "started": now,
          "updated": now,
          "conversation_title": conversation_title,
          "conversation_history_md_file": str(conversation_file)
        }
      )

    self.current_conversation_id = conversation_id
    
    return self.current_conversation_id, conversation_file

  def continue_conversation(self, conversation_id) -> tuple[str, Path]:
    self.current_conversation_id = conversation_id
    conversation_history_path = self.get_conversation_history_path_by_id(conversation_id)

    with self.conversation_metadata.open(
      "r",
      newline="",
      encoding="utf-8"
    ) as csv_file:
      reader = csv.DictReader(csv_file)
      fieldnames=reader.fieldnames
      rows = list(reader)

    found = False
    updated_time = datetime.now().isoformat(timespec="seconds")

    for row in rows:
      if row.get("conversation_id") == conversation_id:
        row["updated"] = updated_time
        found = True
        break

    if not found:
      raise ValueError(
        f"Conversation not found {conversation_id}"
      )

    with self.conversation_metadata.open(
      "w",
      newline="",
      encoding="utf-8"
    ) as csv_file:
      writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
      )
      writer.writeheader()
      writer.writerows(rows)

    return self.current_conversation_id, conversation_history_path

  def update_conversation(self, latest: str) -> None:
    # Open the md file
    # Append the latest with the current data and time
    md_path = self.get_conversation_history_path_by_id(self.current_conversation_id)
    now = datetime.now().isoformat(timespec="seconds")

    with md_path.open("a", encoding="utf-8") as md_file:
      md_file.write(f"\n\n[{now}]\n{latest}")

  def create_conversation_title(self, context: str) -> str:
    system_prompt = f"You create a short title for a conversation, around five to ten words. Return only the title. Do not make extra comments."

    messages = [
      {
        "role": "system",
        "content": system_prompt
      }
    ]

    messages.append(
      {
        "role": "user",
        "content": f"The user has made the following queries: {context}. Decide the best title based on these queries."
      }
    )

    title = self.llm_worker.generate(
      messages=messages,
      max_tokens=20,
      temperature=0.7
    )

    return title