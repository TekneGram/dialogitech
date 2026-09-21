from llm_metadata_extractor import MetadataDecision

class MetadataResponseValidator:

  def __init__(self) -> None:
    return

  # Response parsing and validation
  # Handle fenched JSON, extra explanatory text, malformed JSON recovery,
  # missing value
  # invalid confidence values
  # wrong data types, such as a string instead of an author list.
  # Invalid responses should raise an error, log the error, and call a correction prompt to the LLM
  # Continuous errors (say 3 consecutive fails) should result prompting the user for manual entry
  def parse_json_response(raw_response) -> dict:
    return

  def validate_decision(payload, component) -> MetadataDecision:
    return