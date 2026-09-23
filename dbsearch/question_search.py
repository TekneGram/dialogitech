# Question search does the following
# 1. Embed a question and search for that embedding in the lance database.
# 2. Return the top 10 closest items
# 3. Send each of the 10 retrieved text to a language model with a focused question (provided by the user)
# 4. The language model determines whether the retrieved text answers the focused question and provides a short description of the answer. If the LLM determines that the answer is not contained, it can say so, but still provide a short description of the text retrieved.
# 5. Each language model response is then processed for the user to return the chunk_id, the paper_title, authors, journal, volume, issue, year, the section_title, heading_level, text, paper_type, rhetorical_moves and pdf_path.
# 6. All this data is cached.
# 7. In order to cache the results, we need a hash, and this hash needs to be stored somewhere (search_results folder)
# 8. We should have a function that looks up all the titles of conversations (truncated forms of the original question) and allows us to continue the conversation

from .llm_services.llm_worker import LLMWorker
from pathlib import Path
from .response_parser.response_parser import ResponseParser
from .search_services.search import SearchLanceDB
from .cache_manager.cache_manager import CacheManager
import json

class QuestionSearch:
  def __init__(self) -> None:
    # Start the llm lazily
    self.llm_worker: LLMWorker | None = None

  def question_search(self, *, question_to_embed: str, question_for_llm: str, search_limit: int = 10, conversation_number: int | None) -> None:
    # Start the database
    db = SearchLanceDB()
    search_results = None

    # Get the cache manager
    cm = CacheManager()

    # Search the vector database
    if conversation_number is not None:
      # Deepening the search
      current_conversation = cm.read_record(conversation_number=conversation_number)
      excluded_chunk_ids = json.loads(current_conversation["chunk_ids"])
      data_file = current_conversation["data_file"]
      search_results = db.basic_search_exclude_chunk_ids(question_to_embed, search_limit, excluded_chunk_ids)
      search_results_chunk_ids = [
        row["chunk_id"] for row in search_results
      ]
      # Update the cache
      current_conversation_number, current_data_file = cm.append_search(conversation_number=conversation_number, data_file=data_file, search_results=search_results_chunk_ids)
    else:
      # A new search
      search_results = db.basic_search(question_to_embed, search_limit)
      # Create the cache
      search_results_chunk_ids = [
        row["chunk_id"] for row in search_results
      ]
      current_conversation_number, current_data_file = cm.record_search(search_results=search_results_chunk_ids, query=question_to_embed)


    print("--------------------------CONVERSATION DETAILS---------------------------")
    print(f"Your current conversation number is: {current_conversation_number}")
    print(f"The data file where search results will be saved is: {current_data_file}")
    print("-------------------------------------------------------------------------")


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