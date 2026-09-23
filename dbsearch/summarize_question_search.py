import re

from .cache_manager.cache_manager import CacheManager
from .llm_services.llm_worker import LLMWorker
from pathlib import Path

class SummarizeQuestionSearch:
  def __init__(self):
    self.cm = CacheManager()
    self.llm_worker = None
    return

  def create_per_paper_summary(self, conversation_number: int) -> None:
    record = self.cm.read_record(conversation_number=conversation_number)
    search_results = self.cm.read_results(record["data_file"])
    original_query = record["query"]

    summary_by_citation: dict[str, list[str]] = {}
    result_pattern = re.compile(
      r"### Paper citation\s*\n"
      r"(?P<citation>.*?)\s*\n"
      r"### Summary\s*\n"
      r"(?P<summary>.*?)"
      r"(?=\n### Retrieved text|\n## Search result|\Z)",
      re.DOTALL,
    )

    for match in result_pattern.finditer(search_results):
      citation = " ".join(match.group("citation").split())
      summary = match.group("summary").strip()
      if not citation or not summary:
        continue
      summary_by_citation.setdefault(citation, []).append(summary)

    if self.llm_worker is None:
      self.llm_worker = LLMWorker(
        python_executable="/Users/danielmikaleola/.unsloth/unsloth_gemma4_mlx/bin/python",
        runner_path=Path(__file__).resolve().parents[1] / "chunker" / "mlx_llm_runner.py",
        model_path="unsloth/gemma-4-E4B-it-UD-MLX-4bit",
        request_timeout_seconds=120.0
      )

    system_prompt = """You summarize the findings of a search in an academic style.
    You will receive three key pieces of information.
    1. A question asked by the user.
    2. A citation that includes authors, paper title, journal information and other metadata
    3. One or more summaries in response to the question asked by the user.
    You must collect the summaries together into academic prose.
    You have three ways to collect the summaries and you may mix them up as you need.
    According to <CITATION> (year), + details
    <CITATION> (year) note that + details (you can change the reporting verb)
    Details + (<CITATION>, year).
    You must return academic English prose as plain text.
    You must ensure all the key details in the supplied summaries that answer the user's question are included in the prose.
    You must not add any extra information to the summaries.
    At the end, include a bibliographic reference in APA style, e.g.,
    Surname, N., & Parsons, D. (2026) Article name. Journal Name. Volume. Issue. doi.

    Here are the three key pieces of information:
    """

    for citation, summaries in summary_by_citation.items():
      summaries_text = "\n". join(
        f"- {summary}"
        for summary in summaries
      )

      response = self.llm_worker.generate(
        messages=[
          {
            "role": "system",
            "content": system_prompt
          },
          {
            "role": "user",
            "content": (
              f"Original user question:\n {original_query}\n\n"
              f"Paper citation:\n {citation}\n\n"
              f"Summaries:\n{summaries_text}"
            )
          }
        ],
        max_tokens=500,
        temperature=0.0
      )

      self.cm.append_per_paper_summary(
        data_file=record["data_file"],
        citation=citation,
        summary=response,
      )
      print(response)


    #return summary_by_citation
