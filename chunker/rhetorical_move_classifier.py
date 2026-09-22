"""Backward-compatible exports for rhetorical-move classification types."""

from .llm_rhetorical_move_classifier_helpers import (
    ALL_RHETORICAL_MOVES,
    SECTION_ALLOWED_MOVES,
    RhetoricalMoveClassification,
    RhetoricalMoveLabel,
    RhetoricalMoveResult,
    allowed_moves,
    validate_rhetorical_move_result,
)

__all__ = [
    "ALL_RHETORICAL_MOVES",
    "SECTION_ALLOWED_MOVES",
    "RhetoricalMoveClassification",
    "RhetoricalMoveLabel",
    "RhetoricalMoveResult",
    "allowed_moves",
    "validate_rhetorical_move_result",
]
