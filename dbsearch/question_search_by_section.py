# Question search by section does the following
# 1. Embed a search request and search in the lance database, filtering for just the section selected
# 2. Return the top 10 closest items
# 3. Send each of the 10 retrieved texts to a language model with a focused question (provided by the user)
# 4. The language model determines whether the retrieved text answers the focused question and provides a short description of the answer.

from .llm_services.llm_worker import LLMWorker
from pathlib import Path
from .response_parser.response_parser import ResponseParser
from .search_services.search import SearchLanceDB
from .cache_manager.cache_manager import CacheManager
from chunker.llm_section_type_classifier_helpers.section_taxonomy import SectionLabel
from collections.abc import Collection
import json

class QuestionSearchBySection:
  def __init__(self) -> None:
    # Start the llm lazily
    self.llm_worker: LLMWorker | None = None

  def question_search_by_section(
      self,
      *,
      question_to_embed: str,
      question_for_llm: str,
      search_limit: int = 10,
      sections: SectionLabel | Collection[SectionLabel],
      conversation_number: int | None = None
  ) -> None:
    # Start the database
    db = SearchLanceDB()
    search_results = None

    # Get the cache manager
    cm = CacheManager()

    # Search the vector database
    # If there is a conversation number then we are deepening the search by excluding previously found chunks
    if conversation_number is not None:
      current_conversation = cm.read_record(conversation_number=conversation_number)
      excluded_chunks_ids = json.loads(current_conversation["chunk_ids"])
      data_file = current_conversation["data_file"]
      search_results = db.search_by_section(
        text_to_search=question_to_embed,
        sections=sections,
        limit=search_limit,
        excluded_chunk_ids=excluded_chunks_ids
      )
      search_results_chunk_ids = [
        row["chunk_id"] for row in search_results
      ]
      current_conversation_number, current_data_file = cm.append_search(conversation_number=conversation_number, data_file=data_file, search_results=search_results_chunk_ids)
    else:
      # This starts a new conversation because a conversation_number was not provided
      search_results = db.search_by_section(
        text_to_search=question_to_embed,
        sections=sections,
        limit=search_limit
      )
      search_results_chunk_ids = [
        row["chunk_id"] for row in search_results
      ]
      current_conversation_number, current_data_file = cm.record_search(search_results=search_results_chunk_ids, query=question_to_embed)

    print("--------------------------CONVERSATION DETAILS---------------------------")
    print(f"Your current conversation number is: {current_conversation_number}")
    print(f"The data file where search results will be saved is: {current_data_file}")
    print("-------------------------------------------------------------------------")

    ### LLM WORKER EVALUATES EACH RESULT ###
    # Start the llm worker and prepare to analyze the search results.
    self.llm_worker = LLMWorker(
      python_executable="/Users/danielmikaleola/.unsloth/unsloth_gemma4_mlx/bin/python",
      runner_path=Path(__file__).resolve().parents[1] / "chunker" / "mlx_llm_runner.py",
      model_path="unsloth/gemma-4-E4B-it-UD-MLX-4bit",
      request_timeout_seconds=120.0
    )

    system_prompt = """
      Determine whether the text fully, partially or never answers the question to be asked. 
      If the text answers the question fully or partially, provide a one or two sentence summary of what the text says in response to the question. 
      If the text never answers the question, provide just a one or two sentence summary of the text.\n\n
      Output the text as json. For example: {"answer_provided": "never", "summary": "..."} or {"answer_provided": "fully", "summary": "..."}
    """

    parser = ResponseParser()
    for row in search_results:
      # Send the results to the LLM and retrieve the results back.
      messages=[
        {
          "role": "system",
          "content": system_prompt
        },
        {
          "role": "user",
          "content": (
            f"Question:\n\n{question_for_llm}\n\n"
            f"Text:\n\n{row['text']}"
          )
        }
      ]

      parsed_response = None
      for attempt in range(3):
        try:
          raw_response = self.llm_worker.generate(
            messages=messages,
            max_tokens=300,
            temperature=0.0,
          )

          parsed_response = parser.question_search_response(raw_response)
          break
        except (ValueError, RuntimeError) as error:
          if attempt == 2:
            print(
              f"Skipping chunk {row['chunk_id']} after "
              f"3 failed attempts: {error}"
            )
            break

          messages.append(
            {
              "role": "user",
              "content": (
                "Your previous response was invalid. Try again. Return only valud JSON with exactly these fields:"
                '`{"answer_provided": "fully|partially|never", "summary": "..."}`'
              ),
            }
          )
      if parsed_response is None:
        continue
      print(parsed_response)
      cm.append_to_data_file(
        data_file=current_data_file,
        query_to_embed=question_to_embed,
        question_for_llm=question_for_llm,
        row=row,
        parsed_response=parsed_response
      )