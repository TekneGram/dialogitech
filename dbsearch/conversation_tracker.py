
from pathlib import Path
import csv
import uuid
from datetime import datetime

class ConversationTracker:

  def __init__(self) -> None:
    # Set up directories and files in the class
    self.tracking_directory = (
          Path(__file__).resolve().parents[1] / "data" / "conversation_tracking"
        )
    self.conversation_metadata: Path = self.tracking_directory / "conversation_metadata.csv"
    self.conversations_directory: Path = self.tracking_directory / "conversations"
    self.ensure_conversation_metadata_file()

    # Set up the conversations context
    self.conv_titles_ids: dict[str, str] = {}
    self.current_conversation_id: str | None = None
    self.get_conversation_titles()

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

  def get_conversation_titles(self) -> None:
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

      
  def get_conversation_metadata_by_title(self, conversation_title: str) -> Path | None:
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

  def get_conversation_metadata_by_id(self, conversation_id: str) -> Path | None:

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

    fieldnames = [
      "conversation_id",
      "started",
      "updated",
      "conversation_title",
      "conversation_history_md_file"
    ]

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
  
