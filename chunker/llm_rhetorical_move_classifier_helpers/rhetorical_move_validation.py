from __future__ import annotations

from .rhetorical_move_models import RhetoricalMoveResult
from .rhetorical_move_taxonomy import allowed_moves


def validate_rhetorical_move_result(
    result: RhetoricalMoveResult,
    *,
    section_label: str,
) -> None:
    if len(result.moves) > 3:
        raise RuntimeError("Rhetorical move classification returned more than three moves.")
    allowed = set(allowed_moves(section_label))
    labels = [move.label for move in result.moves]
    if len(labels) != len(set(labels)):
        raise RuntimeError("Rhetorical move classification returned duplicate move labels.")
    invalid = set(labels) - allowed
    invalid.discard("unclassified")
    if invalid:
        raise RuntimeError(
            f"Rhetorical move classification returned moves not allowed for {section_label}: {sorted(invalid)}"
        )
