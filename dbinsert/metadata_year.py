from __future__ import annotations

import re
from pathlib import Path


def infer_year_from_filename(path: str | Path | None) -> str | None:
    """Return a four-digit year when it prefixes a source filename."""
    if path is None:
        return None
    match = re.match(r"^(\d{4})(?:\D|$)", Path(path).stem)
    return match.group(1) if match else None


def apply_year_fallback(metadata: dict, source_path: str | Path | None) -> str:
    """Fill a missing journal year from the source filename or ``unknown``."""
    journal = metadata.setdefault("journal", {})
    if not isinstance(journal, dict):
        journal = {}
        metadata["journal"] = journal

    current_year = journal.get("year")
    if isinstance(current_year, str) and current_year.strip() and current_year.strip().lower() != "unknown":
        return current_year.strip()

    inferred_year = infer_year_from_filename(source_path)
    journal["year"] = inferred_year or "unknown"
    return journal["year"]
