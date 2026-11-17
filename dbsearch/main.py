import argparse

from .question_search import QuestionSearch
from .question_search_by_section import QuestionSearchBySection
from .review_writer import ReviewWriter
from .summarize_question_search import SummarizeQuestionSearch
from .paper_summary import SummarizePaper
from .find_papers_by_topic import FindPapersByTopic

# Make a straightforward search
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

# Search a specified section
def question_search_by_section(
    question_to_embed: str,
    question_for_llm: str,
    search_limit: int,
    sections: str | list[str],
    conversation_number: int | None
) -> None:
  search = QuestionSearchBySection()
  search.question_search_by_section(
    question_to_embed=question_to_embed,
    question_for_llm=question_for_llm,
    search_limit=search_limit,
    sections=sections,
    conversation_number=conversation_number
  )

# Write a summary of a search based on a specific conversation number.
def summarize_question_search(
    conversation_number: int
) -> None:
  search = SummarizeQuestionSearch()
  search.create_per_paper_summary(conversation_number)

# Write a review based on a number of summaries across multiple conversations.
def write_review(conversation_numbers: list[int]) -> None:
  ReviewWriter().write_review(conversation_numbers)

# Summarize a specific paper.
def summarize_a_paper(paper_title: str) -> None:
  SummarizePaper().summarize_paper(paper_title)


# List papers based on topics covered
def find_papers_by_topics(topics: list[str]) -> None:
  FindPapersByTopic().find_by_topics(topics)


def main() -> None:
  parser = argparse.ArgumentParser(description="DialogiTech database search commands.")
  subparsers = parser.add_subparsers(dest="command", required=True)

  # Create subparser commands
  question_search_parser = subparsers.add_parser(
    "question_search",
    help="Search paper chunks and ask an LLM about each result.",
  )

  question_search_by_section_parser = subparsers.add_parser(
    "question_search_by_section",
    help="Search paper chunks by section and ask an LLM about each result."
  )

  summarize_question_search_parser = subparsers.add_parser(
    "summarize_question_search",
    help="Create a summary of the findings after the question search."
  )

  write_review_parser = subparsers.add_parser(
    "write_review",
    help="Write a literature review from cached conversation summaries.",
  )

  summarize_a_paper_parser = subparsers.add_parser(
    "summarize_a_paper",
    help="Read each section of a paper and summarize each section; return the path to the full summary file."
  )

  find_papers_by_topics_parser = subparsers.add_parser(
    "find_papers_by_topics",
    help="Search for papers where a list of topics has matches and investigate those papers."
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
    help="Focused question sent to the LLM for each retrieved text so LLM can evaluate whether the retrieved text is related to the query.",
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

  # Create flags for question_search_by_section
  question_search_by_section_parser.add_argument(
    "--question-to-embed",
    required=True,
    help="Question used to search the database by embedding similarity.",
  )

  question_search_by_section_parser.add_argument(
    "--question-for-llm",
    default="Does the text answer the question?",
    help="Focused question sent to the LLM for each retrieved text so LLM can evaluate whether the retrieved text is related to the query."
  )

  question_search_by_section_parser.add_argument(
    "--search-limit",
    type=int,
    default=10,
    help="Set this to retrieve more results."
  )

  question_search_by_section_parser.add_argument(
    "--sections",
    nargs="+",
    type=str,
    required=True,
    choices=[
      "abstract",
      "introduction",
      "method",
      "results",
      "discussion",
      "review_method",
      "literature_synthesis",
      "conceptual_framework",
      "argument_or_analysis",
      "evaluation_or_example",
      "practice_description",
      "background_or_review",
      "method_or_approach",
      "analysis_or_argument",
      "discussion_or_conclusion",
      "front_matter",
    ],
    help="One or more section labels. Items include 'abstract', 'introduction', 'method', 'results', 'discussion', 'review_method', 'literature_synthesis', 'conceptual_framework', 'argument_or_analysis', 'evaluation_or_example', 'practice_description', 'background_or_review', 'method_or_approach', 'analysis_or_argument', 'discussion_or_conclusion'",
  )

  question_search_by_section_parser.add_argument(
    "--conversation-number",
    default=None,
    type=int,
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

  # Create flags for summarize_a_paper
  summarize_a_paper_parser.add_argument(
    "--paper-title",
    default=None,
    help="Supply the title of the paper you wish to summarize."
  )

  # Create flags for find_papers_by_topics
  find_papers_by_topics_parser.add_argument(
    "--topics",
    nargs="+",
    type=str,
    required=True,
    help="a list of topics or ideas that you want searched for in papers"
  )

  args = parser.parse_args()
  if args.command == "question_search":
    question_search(
      question_to_embed=args.question_to_embed,
      question_for_llm=args.question_for_llm,
      search_limit=args.search_limit,
      conversation_number=args.conversation_number
    )

  if args.command == "question_search_by_section":
    question_search_by_section(
      question_to_embed=args.question_to_embed,
      question_for_llm=args.question_for_llm,
      search_limit=args.search_limit,
      sections=args.sections,
      conversation_number=args.conversation_number
    )

  if args.command == "summarize_question_search":
    summarize_question_search(
      conversation_number=args.conversation_number
    )

  if args.command == "write_review":
    write_review(conversation_numbers=args.conversation_numbers)

  if args.command == "summarize_a_paper":
    summarize_a_paper(paper_title=args.paper_title)

  if args.command == "find_papers_by_topics":
    find_papers_by_topics(topics=args.topics)


if __name__ == "__main__":
  main()
