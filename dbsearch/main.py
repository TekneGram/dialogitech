import argparse

from .question_search import QuestionSearch

def question_search(
    question_to_embed: str, 
    question_for_llm: str,
    search_limit: int,
    conversation_number: int | None
  ) -> None:
  search = QuestionSearch()
  search.question_search(
    question_to_embed=question_to_embed,
    question_for_llm=question_for_llm,
    search_limit=search_limit,
    conversation_number=conversation_number
  )

def main() -> None:
  parser = argparse.ArgumentParser(description="DialogiTech database search commands.")
  subparsers = parser.add_subparsers(dest="command", required=True)

  question_search_parser = subparsers.add_parser(
    "question_search",
    help="Search paper chunks and ask an LLM about each result.",
  )

  question_search_parser.add_argument(
    "--question-to-embed",
    required=True,
    help="Question used to search the database by embedding similarity.",
  )
  question_search_parser.add_argument(
    "--question-for-llm",
    default="Does the text answer the question?",
    help="Focused question sent to the LLM for each retrieved text.",
  )
  question_search_parser.add_argument(
    "--search-limit",
    type=int,
    default=10,
    help="Set this to retrieve more results."
  )
  question_search_parser.add_argument(
    "--conversation-number",
    default=None,
    help="Supply this to deepen a previous search."
  )

  args = parser.parse_args()
  if args.command == "question_search":
    question_search(
      question_to_embed=args.question_to_embed,
      question_for_llm=args.question_for_llm,
      search_limit=args.search_limit,
      conversation_number=args.conversation_number
    )

if __name__ == "__main__":
  main()
