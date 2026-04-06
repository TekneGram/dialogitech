from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetadataPromptContext:
    paper_id: str
    pdf_path: Path
    marker_json_path: Path


@dataclass(frozen=True)
class MetadataValidationIssue:
    field_name: str
    prompt_label: str


class MetadataValidationError(RuntimeError):
    pass


class MetadataPromptAbortedError(MetadataValidationError):
    pass


class MetadataPromptUnavailableError(MetadataValidationError):
    pass


class MetadataCompletenessChecker:
    def find_missing_fields(self, extracted_metadata: dict[str, Any]) -> list[MetadataValidationIssue]:
        issues: list[MetadataValidationIssue] = []

        title = extracted_metadata.get("title")
        if not isinstance(title, str) or not title.strip():
            issues.append(MetadataValidationIssue(field_name="title", prompt_label="Title"))

        authors = extracted_metadata.get("authors")
        if not isinstance(authors, list) or not any(
            isinstance(author, str) and author.strip() for author in authors
        ):
            issues.append(
                MetadataValidationIssue(
                    field_name="authors",
                    prompt_label="Authors",
                )
            )

        journal = extracted_metadata.get("journal")
        journal_name = journal.get("name") if isinstance(journal, dict) else None
        if not isinstance(journal_name, str) or not journal_name.strip():
            issues.append(
                MetadataValidationIssue(
                    field_name="journal.name",
                    prompt_label="Journal name",
                )
            )

        year = journal.get("year") if isinstance(journal, dict) else None
        if not isinstance(year, str) or not year.strip().isdigit():
            issues.append(MetadataValidationIssue(field_name="year", prompt_label="Publication year"))

        return issues


class InteractiveMetadataPrompter:
    def complete_metadata(
        self,
        extracted_metadata: dict[str, Any],
        *,
        context: MetadataPromptContext,
        issues: list[MetadataValidationIssue],
    ) -> dict[str, Any]:
        if not sys.stdin.isatty():
            raise MetadataPromptUnavailableError(
                f"[{context.paper_id}] Missing required metadata but terminal input is unavailable."
            )

        title = extracted_metadata.get("title")
        authors = list(extracted_metadata.get("authors") or [])
        journal = dict(extracted_metadata.get("journal") or {})

        self._print_header(context=context, extracted_title=title, issues=issues)

        if self._has_issue(issues, "title"):
            title = self._prompt_non_empty("Enter title")

        if self._has_issue(issues, "authors"):
            authors_text = self._prompt_non_empty("Enter authors as a semicolon-separated list")
            authors = [author.strip() for author in authors_text.split(";") if author.strip()]
            if not authors:
                raise MetadataValidationError(
                    f"[{context.paper_id}] Manual authors input did not contain any valid author names."
                )

        if self._has_issue(issues, "journal.name"):
            journal["name"] = self._prompt_non_empty("Enter journal name")

        if self._has_issue(issues, "year"):
            journal["year"] = self._prompt_year("Enter publication year")

        completed = dict(extracted_metadata)
        completed["title"] = title
        completed["authors"] = authors
        completed["journal"] = journal
        return completed

    def _print_header(
        self,
        *,
        context: MetadataPromptContext,
        extracted_title: Any,
        issues: list[MetadataValidationIssue],
    ) -> None:
        missing_labels = ", ".join(issue.prompt_label for issue in issues)
        print("", flush=True)
        print(f"[{context.paper_id}] Required metadata is missing: {missing_labels}", flush=True)
        print(f"PDF path: {context.pdf_path.resolve()}", flush=True)
        print(f"Marker JSON path: {context.marker_json_path.resolve()}", flush=True)
        if isinstance(extracted_title, str) and extracted_title.strip():
            print(f"Current extracted title: {extracted_title.strip()}", flush=True)
        print("Enter the missing values to continue processing this paper.", flush=True)

    def _has_issue(self, issues: list[MetadataValidationIssue], field_name: str) -> bool:
        return any(issue.field_name == field_name for issue in issues)

    def _prompt_non_empty(self, prompt: str) -> str:
        while True:
            try:
                value = input(f"{prompt}: ").strip()
            except (EOFError, KeyboardInterrupt) as exc:
                raise MetadataPromptAbortedError("Metadata entry aborted by user.") from exc
            if value:
                return value
            print("Value cannot be empty.", flush=True)

    def _prompt_year(self, prompt: str) -> str:
        while True:
            value = self._prompt_non_empty(prompt)
            if value.isdigit():
                return value
            print("Year must be a numeric value such as 2025.", flush=True)
