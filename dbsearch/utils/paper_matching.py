from __future__ import annotations
import csv
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from .paper_record import PaperRecord

@dataclass
class PaperMatch:
  paper: PaperRecord
  score: float

def normalize_title(title: str) -> str:
  """Normalize a title for comparison"""
  title = unicodedata.normalize("NFKC", title)
  title = title.lower()

  # Remove punctuation but preserve word boundaries
  title = re.sub(r"[^\w\s]", " ", title)

  # Collapse repeated whitespace
  return " ".join(title.split())

def title_similarity(left: str, right: str) -> float:
  """Return a similarity score between 0 and 1"""
  left_normalized = normalize_title(left)
  right_normalized = normalize_title(right)

  if left_normalized == right_normalized:
    return 1.0

  left_words = set(left_normalized.split())
  right_words = set(right_normalized.split())

  if not left_words or not right_words:
    return 0.0

  intersection = left_words & right_words
  union = left_words | right_words

  token_score = len(intersection) / len(union)
  sequence_score = SequenceMatcher(
    None,
    left_normalized,
    right_normalized
  ).ratio()

  return (token_score * 0.6) + (sequence_score * 0.4)

def find_best_paper_match(
    paper_title: str, 
    papers: list[PaperRecord], 
    *, 
    minimum_score: float = 0.55, 
    ambiguity_margin: float = 0.05
  ) -> PaperMatch:
    if not papers:
      raise ValueError("No paper records were provided!")

    matches = sorted(
      (
        PaperMatch(
          paper=paper,
          score=title_similarity(
            paper_title,
            paper.paper_title
          ),
        )
        for paper in papers
      ),
      key=lambda match: match.score,
      reverse=True,
    )

    best = matches[0]
    second_best = matches[1] if len(matches) > 1 else None

    if best.score < minimum_score:
      raise LookupError(
        f"No sufficiently similar paper title found for"
        f"{paper_title!r}. Best score: {best.score:.3f}"
      )

    if (
      second_best is not None
      and best.score - second_best.score < ambiguity_margin
    ):
      raise LookupError(
        "The title is ambiguous. Candidates were: \n"
        f"1. {best.paper.paper_title} ({best.score:.3f})\n"
        f"2. {second_best.paper.paper_title} "
        f"({second_best.score:.3f})"
      )

    return best