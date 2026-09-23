from __future__ import annotations

from pathlib import Path

from .cache_manager.cache_manager import CacheManager
from .llm_services.llm_worker import LLMWorker


class ReviewWriter:
  """Build a literature review from cached conversation summaries."""

  def __init__(self) -> None:
    self.llm_worker: LLMWorker | None = None
    self.cm = CacheManager()

  def write_review(self, conversation_numbers: list[int]) -> Path:
    summaries = {
      conversation_number: self.cm.extract_summaries(conversation_number)
      for conversation_number in conversation_numbers
    }
    queries = self.cm.extract_queries(conversation_numbers)
    review_plan = self.build_review_plan(queries, summaries)

    syntheses: dict[int, str] = {}
    for conversation_number, query in queries.items():
      synthesis = self.synthesize_summaries(
        query=query,
        summaries=summaries[conversation_number],
      )
      self.cm.save_summaries_synthesis(
        conversation_number=conversation_number,
        synthesis=synthesis,
      )
      syntheses[conversation_number] = synthesis

    synthesis_text = "\n\n".join(
      f"Query {conversation_number}: {syntheses[conversation_number]}"
      for conversation_number in conversation_numbers
    )
    review = self._generate(
      system_prompt=(
        "You write a literature review from syntheses of summaries. "
        "You keep all the citations provided and you keep all the references "
        "provided. You follow the research plan and write in an academic prose"
      ),
      user_prompt=(
        f"This is the writing plan: {review_plan}\n\n"
        f"Here are the syntheses:\n{synthesis_text}"
      ),
      max_tokens=2000,
    )
    review_path = self.cm.save_review(review)
    print(f"Review saved to {review_path}")
    return review_path

  def build_review_plan(
    self,
    queries: dict[int, str],
    summaries: dict[int, list[str]],
  ) -> str:
    """Ask the LLM for a writing plan using one example per query."""
    examples = []
    for conversation_number, query in queries.items():
      conversation_summaries = summaries.get(conversation_number, [])
      first_summary = conversation_summaries[0] if conversation_summaries else ""
      examples.append(
        f"query {conversation_number}: {query}; "
        f"example response: {first_summary}"
      )

    return self._generate(
      system_prompt=(
        "You will receive some queries asked by a user and examples of "
        "responses. Your task is to make a writing plan. Here is an example "
        "of a plan: ‘write a review which contrasts the advantages and "
        "disadvantages of X. Write one paragraph on the advantages and one "
        "paragraph on the disadvantages.’"
      ),
      user_prompt="; ".join(examples),
      max_tokens=500,
    )

  def synthesize_summaries(
    self,
    *,
    query: str,
    summaries: list[str],
  ) -> str:
    """Synthesize the per-paper summaries associated with one query."""
    del query
    summaries_text = "\n\n".join(summaries)
    return self._generate(
      system_prompt=(
        "You write a synthesis of multiple summaries. The synthesis should "
        "do two things. First, highlight common points across multiple "
        "summaries. Second, add additional points that are mentioned by only "
        "one or two summaries. Your writing style is academic prose. Include "
        "citations and write your references at the end of the synthesis "
        "using APA style as provided in the summaries. Here are the summaries."
      ),
      user_prompt=summaries_text,
      max_tokens=1200,
    )

  def _generate(
    self,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
  ) -> str:
    if self.llm_worker is None:
      self.llm_worker = LLMWorker(
        python_executable="/Users/danielmikaleola/.unsloth/unsloth_gemma4_mlx/bin/python",
        runner_path=Path(__file__).resolve().parents[1] / "chunker" / "mlx_llm_runner.py",
        model_path="unsloth/gemma-4-E4B-it-UD-MLX-4bit",
        request_timeout_seconds=120.0,
      )
    return self.llm_worker.generate(
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
      ],
      max_tokens=max_tokens,
      temperature=0.0,
    )
