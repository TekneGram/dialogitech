from __future__ import annotations

from typing import Any

from dbquery.models import QueryFilters


def build_where_clause(filters: QueryFilters | None) -> str | None:
    if filters is None:
        return None

    clauses: list[str] = []
    if filters.paper_id:
        clauses.append(f"paper_id = '{_escape(filters.paper_id)}'")
    if filters.paper_id_in:
        clauses.append(f"paper_id IN ({_render_string_list(filters.paper_id_in)})")
    if filters.year is not None:
        clauses.append(f"year = {int(filters.year)}")
    if filters.year_min is not None:
        clauses.append(f"year >= {int(filters.year_min)}")
    if filters.year_max is not None:
        clauses.append(f"year <= {int(filters.year_max)}")
    if filters.classification_label:
        clauses.append(
            f"classification_label = '{_escape(filters.classification_label)}'"
        )
    if filters.classification_label_in:
        clauses.append(
            f"classification_label IN ({_render_string_list(filters.classification_label_in)})"
        )
    return " AND ".join(clauses) or None


def expanded_limit_for_filters(limit: int, filters: QueryFilters | None) -> int:
    if filters is None:
        return limit
    if filters.paper_title_contains or filters.section_title_contains or filters.author or filters.authors_any:
        return limit * 5
    return limit


def row_matches_filters(row: dict[str, Any], filters: QueryFilters | None) -> bool:
    if filters is None:
        return True

    if filters.paper_id and str(row.get("paper_id")) != filters.paper_id:
        return False
    if filters.paper_id_in and str(row.get("paper_id")) not in filters.paper_id_in:
        return False
    if filters.year is not None and _as_int(row.get("year")) != filters.year:
        return False
    year = _as_int(row.get("year"))
    if filters.year_min is not None and (year is None or year < filters.year_min):
        return False
    if filters.year_max is not None and (year is None or year > filters.year_max):
        return False
    if filters.classification_label:
        if str(row.get("classification_label") or "") != filters.classification_label:
            return False
    if filters.classification_label_in:
        classification_label = str(row.get("classification_label") or "")
        if classification_label not in filters.classification_label_in:
            return False
    if filters.paper_title_contains:
        paper_title = str(row.get("paper_title") or "")
        if filters.paper_title_contains.lower() not in paper_title.lower():
            return False
    if filters.section_title_contains:
        section_title = str(row.get("section_title") or "")
        if filters.section_title_contains.lower() not in section_title.lower():
            return False
    if filters.author:
        authors = [str(author) for author in row.get("authors", [])]
        if not any(filters.author.lower() in author.lower() for author in authors):
            return False
    if filters.authors_any:
        authors = [str(author) for author in row.get("authors", [])]
        normalized_authors = [author.lower() for author in authors]
        if not any(
            any(requested.lower() in author for author in normalized_authors)
            for requested in filters.authors_any
        ):
            return False
    return True


def _escape(value: str) -> str:
    return value.replace("'", "''")


def _render_string_list(values: list[str]) -> str:
    return ", ".join(f"'{_escape(value)}'" for value in values)


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
