import argparse

from .question_search import QuestionSearch
from .review_writer import ReviewWriter
from .summarize_question_search import SummarizeQuestionSearch

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

def summarize_question_search(
    conversation_number: int
) -> None:
  search = SummarizeQuestionSearch()
  search.create_per_paper_summary(conversation_number)

def write_review(conversation_numbers: list[int]) -> None:
  ReviewWriter().write_review(conversation_numbers)

def main() -> None:
  parser = argparse.ArgumentParser(description="DialogiTech database search commands.")
  subparsers = parser.add_subparsers(dest="command", required=True)

  # Create subparser commands
  question_search_parser = subparsers.add_parser(
    "question_search",
    help="Search paper chunks and ask an LLM about each result.",
  )
  summarize_question_search_parser = subparsers.add_parser(
    "summarize_question_search",
    help="Create a summary of the findings after the question search."
  )
  write_review_parser = subparsers.add_parser(
    "write_review",
    help="Write a literature review from cached conversation summaries.",
  )

  # Create flags for question_search
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

  # Create flags for summarize_question_search
  summarize_question_search_parser.add_argument(
    "--conversation-number",
    default=None,
    help="Supply the conversation number to summarize results for that conversation"
  )

  write_review_parser.add_argument(
    "--conversation-numbers",
    nargs="+",
    type=int,
    required=True,
    help="Conversation numbers whose summaries should be used in the review.",
  )

  args = parser.parse_args()
  if args.command == "question_search":
    question_search(
      question_to_embed=args.question_to_embed,
      question_for_llm=args.question_for_llm,
      search_limit=args.search_limit,
      conversation_number=args.conversation_number
    )

  if args.command == "summarize_question_search":
    summarize_question_search(
      conversation_number=args.conversation_number
    )

  if args.command == "write_review":
    write_review(conversation_numbers=args.conversation_numbers)


if __name__ == "__main__":
  main()
