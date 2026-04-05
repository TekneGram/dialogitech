from __future__ import annotations

import json
import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metadata_extractor import MetadataExtractor


@dataclass(slots=True)
class RemovedBlock:
    block_id: str
    block_type: str
    text: str
    reason: str
    page_id: int | None = None


@dataclass(slots=True)
class RenderedItem:
    kind: str
    text: str
    page_id: int | None = None
    block_id: str | None = None


class BoilerplateFilter:
    REFERENCE_SECTION_TITLE = "references"
    EXCLUDED_SECTION_TITLES = {
        REFERENCE_SECTION_TITLE,
        "acknowledgment",
        "acknowledgement",
        "declaration of use of ai",
    }
    TEXT_BLOCK_TYPES = {
        "Text",
        "TextInlineMath",
        "SectionHeader",
        "ListItem",
        "ListGroup",
        "Code",
        "Equation",
        "Footnote",
        "Handwriting",
    }
    SKIPPED_BLOCK_TYPES = {
        "PageHeader",
        "PageFooter",
        "Picture",
        "PictureGroup",
        "Figure",
        "FigureGroup",
        "Caption",
        "Table",
        "TableGroup",
        "TableCell",
        "TableOfContents",
        "Form",
        "Reference",
    }
    COPYRIGHT_PATTERNS = [
        re.compile(r"\bcopyright\b", re.IGNORECASE),
        re.compile(r"©"),
        re.compile(r"\bcreative\s+commons\b", re.IGNORECASE),
        re.compile(r"\bopen\s+access\b", re.IGNORECASE),
        re.compile(r"\bdistributed\s+under\s+the\s+terms\b", re.IGNORECASE),
        re.compile(r"\bpermits\s+unrestricted\s+use\b", re.IGNORECASE),
        re.compile(r"\boriginal\s+author\s+and\s+source\s+are\s+credited\b", re.IGNORECASE),
        re.compile(r"\battribution(?:[- ]non[- ]commercial)?\b", re.IGNORECASE),
        re.compile(r"\binternational\s+license\b", re.IGNORECASE),
        re.compile(r"\blicense\b", re.IGNORECASE),
        re.compile(r"\bissn\b", re.IGNORECASE),
        re.compile(r"\bdoi\b", re.IGNORECASE),
        re.compile(r"https?://doi\.org/", re.IGNORECASE),
    ]
    REFERENCE_AUTHOR_START = re.compile(
        r"^(?:-+\s*)?(?:[A-Z][A-Za-z'`\-]+,\s+(?:[A-Z]\.\s*)+)(?:,\s*&\s*[A-Z][A-Za-z'`\-]+,\s+(?:[A-Z]\.\s*)+)?"
    )
    REFERENCE_SIGNAL_PATTERNS = [
        re.compile(r"\(\d{4}[a-z]?\)"),
        re.compile(r"\b(?:journal|proceedings|conference|review|corpus|studies)\b", re.IGNORECASE),
        re.compile(r"https?://"),
        re.compile(r"\bdoi\b", re.IGNORECASE),
        re.compile(r"\b\d+\s*[,:]\s*\d+(?:[–-]\d+)?\b"),
    ]

    def __init__(self) -> None:
        self.removed_blocks: list[RemovedBlock] = []
        self._skip_section_active = False
        self._drop_remaining_blocks = False
        self._metadata: dict[str, Any] = {}
        self._normalized_title_to_remove: str | None = None
        self._normalized_authors_to_remove: set[str] = set()
        self._front_matter_author_window = False

    def load_json(self, source: str | Path | dict[str, Any]) -> dict[str, Any]:
        if isinstance(source, dict):
            return source
        path = Path(source)
        return json.loads(path.read_text(encoding="utf-8"))

    def convert_file(
        self,
        source: str | Path,
        output_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        document = self.load_json(source)
        markdown = self.convert_json_to_markdown(document, metadata=metadata)
        if output_path is not None:
            Path(output_path).write_text(markdown, encoding="utf-8")
        return markdown

    def convert_json_to_markdown(
        self,
        document: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.removed_blocks = []
        self._skip_section_active = False
        self._drop_remaining_blocks = False
        self._prepare_metadata(document, metadata)
        items: list[RenderedItem] = []

        for page in document.get("children", []):
            page_items = self._render_page(page)
            items.extend(page_items)

        markdown = self._assemble_markdown(items)
        markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
        return markdown + "\n" if markdown else ""

    def _render_page(self, page: dict[str, Any]) -> list[RenderedItem]:
        blocks = page.get("children", []) or []
        suppress_heading_ids = self._find_title_page_heading_ids(blocks)
        items: list[RenderedItem] = []
        for block in blocks:
            if self._drop_remaining_blocks:
                self._record_removed(block, "after_references")
                continue

            block_type = block.get("block_type", "")
            block_page_id = self._page_id_from_block_id(block.get("id", ""))
            if block_type == "SectionHeader":
                heading_text = self._clean_text(self._html_to_text(block.get("html", "")))
                normalized_heading = self._normalize_title(heading_text)
                if normalized_heading == self.REFERENCE_SECTION_TITLE:
                    self._drop_remaining_blocks = True
                self._skip_section_active = self._is_excluded_section_title(heading_text)
                if block_page_id == 0:
                    self._front_matter_author_window = normalized_heading != "abstract"
                else:
                    self._front_matter_author_window = False

            if self._skip_section_active:
                reason = "references_cutoff" if self._drop_remaining_blocks else "excluded_section"
                self._record_removed(block, reason)
                continue

            rendered = self._render_block(block, suppress_heading_ids=suppress_heading_ids)
            if rendered:
                items.extend(rendered)
        return items

    def _render_block(
        self,
        block: dict[str, Any],
        suppress_heading_ids: set[str] | None = None,
    ) -> list[RenderedItem]:
        block_type = block.get("block_type", "")
        block_id = block.get("id", "")
        page_id = self._page_id_from_block_id(block_id)
        suppress_heading_ids = suppress_heading_ids or set()

        if block_type in self.SKIPPED_BLOCK_TYPES:
            self._record_removed(block, "skipped_block_type")
            return []

        if block_type == "SectionHeader":
            if block_id in suppress_heading_ids:
                self._record_removed(block, "title_page_heading")
                return []
            text = self._html_to_text(block.get("html", ""))
            text = self._clean_text(text)
            if not text or self._is_boilerplate_text(text):
                self._record_removed(block, "boilerplate_heading")
                return []
            if self._should_remove_title(text):
                self._record_removed(block, "metadata_title")
                return []
            return [RenderedItem(kind="heading", text=f"# {text}", page_id=page_id, block_id=block_id)]

        if block_type == "ListGroup":
            text = self._render_list_group(block)
            text = self._clean_text_block(text)
            if not text:
                return []
            return [RenderedItem(kind="list", text=text, page_id=page_id, block_id=block_id)]

        if block_type not in self.TEXT_BLOCK_TYPES:
            self._record_removed(block, "unsupported_block_type")
            return []

        text = self._html_to_text(block.get("html", ""))
        text = self._clean_text_block(text)
        if not text:
            return []
        if self._is_boilerplate_text(text):
            self._record_removed(block, "boilerplate_text")
            return []
        if self._should_remove_author_or_affiliation(text):
            self._record_removed(block, "metadata_author")
            return []
        if self._looks_like_reference_entry(text):
            self._record_removed(block, "reference_entry")
            return []
        return [RenderedItem(kind="text", text=text, page_id=page_id, block_id=block_id)]

    def _render_list_group(self, block: dict[str, Any]) -> str:
        children = block.get("children") or []
        items: list[str] = []
        for child in children:
            text = self._html_to_text(child.get("html", ""))
            text = self._clean_text_block(text)
            if text and not self._is_boilerplate_text(text):
                items.append(f"- {text}")
        return "\n".join(items)

    def _html_to_text(self, html_text: str) -> str:
        if not html_text:
            return ""
        text = re.sub(r"<br\s*/?>", "\n", html_text, flags=re.IGNORECASE)
        text = re.sub(r"</(p|div|h\d|li|tr)>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        return text

    def _clean_text(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _normalize_title(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def _clean_text_block(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _assemble_markdown(self, items: list[RenderedItem]) -> str:
        lines: list[str] = []
        previous_kind: str | None = None

        for item in items:
            text = item.text.strip()
            if not text:
                continue

            if item.kind == "heading":
                if lines and lines[-1] != "":
                    lines.append("")
                lines.append(text)
                lines.append("")
            elif item.kind == "list":
                if previous_kind not in {None, "heading"} and lines and lines[-1] != "":
                    lines.append("")
                lines.append(text)
                lines.append("")
            else:
                if previous_kind == "text" and lines:
                    lines[-1] = self._merge_text_fragments(lines[-1], text)
                else:
                    if lines and lines[-1] != "":
                        lines.append("")
                    lines.append(text)

            previous_kind = item.kind

        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)

    def _merge_text_fragments(self, left: str, right: str) -> str:
        left = left.rstrip()
        right = right.lstrip()
        if not left:
            return right
        if not right:
            return left

        if left.endswith((".", "?", "!", ":", ";")):
            return f"{left}\n\n{right}"

        if right[:1].islower() or right[:1] in {"(", "[", '"', "'"}:
            return f"{left} {right}"

        return f"{left}\n\n{right}"

    def _find_title_page_heading_ids(self, blocks: list[dict[str, Any]]) -> set[str]:
        if not blocks:
            return set()

        first_page_id = self._page_id_from_block_id(blocks[0].get("id", ""))
        if first_page_id != 0:
            return set()

        headings = [block for block in blocks if block.get("block_type") == "SectionHeader"]
        if len(headings) < 2:
            return set()

        first_heading = headings[0]
        next_heading = headings[1]
        first_index = blocks.index(first_heading)
        next_index = blocks.index(next_heading)
        between_blocks = blocks[first_index + 1:next_index]

        meaningful_between = []
        for block in between_blocks:
            if block.get("block_type") in self.SKIPPED_BLOCK_TYPES:
                continue
            text = self._clean_text(self._html_to_text(block.get("html", "")))
            if not text:
                continue
            if self._is_boilerplate_text(text):
                continue
            meaningful_between.append(text)

        first_title = self._clean_text(self._html_to_text(first_heading.get("html", "")))
        next_title = self._clean_text(self._html_to_text(next_heading.get("html", "")))
        is_shorter_than_next = len(first_title.split()) < len(next_title.split())

        if not meaningful_between and is_shorter_than_next:
            return {first_heading.get("id", "")}

        return set()

    def _is_excluded_section_title(self, title: str) -> bool:
        normalized = re.sub(r"\s+", " ", title).strip().lower()
        return normalized in self.EXCLUDED_SECTION_TITLES

    def _prepare_metadata(self, document: dict[str, Any], metadata: dict[str, Any] | None) -> None:
        if metadata is None:
            metadata = MetadataExtractor().extract_all(document)
        self._metadata = metadata
        title = metadata.get("title")
        authors = metadata.get("authors", [])
        self._normalized_title_to_remove = self._normalize_title(title) if title else None
        self._normalized_authors_to_remove = {
            self._normalize_title(author) for author in authors if self._normalize_title(author)
        }
        self._front_matter_author_window = True

    def _should_remove_title(self, text: str) -> bool:
        return bool(self._normalized_title_to_remove and self._normalize_title(text) == self._normalized_title_to_remove)

    def _should_remove_author_or_affiliation(self, text: str) -> bool:
        normalized = self._normalize_title(text)
        if not normalized:
            return False
        if normalized in self._normalized_authors_to_remove:
            return True
        if self._front_matter_author_window and self._looks_like_affiliation(text):
            return True
        return False

    def _looks_like_affiliation(self, text: str) -> bool:
        lowered = text.lower()
        return any(
            keyword in lowered
            for keyword in {
                "university",
                "college",
                "department",
                "faculty",
                "school",
                "institute",
                "center",
                "centre",
                "laboratory",
                "lab",
                "hospital",
            }
        )

    def _is_boilerplate_text(self, text: str) -> bool:
        lowered = text.lower()
        pattern_hits = sum(1 for pattern in self.COPYRIGHT_PATTERNS if pattern.search(text))

        if pattern_hits >= 2:
            return True

        if pattern_hits >= 1 and len(text.split()) <= 40:
            return True

        # Short front-matter publication strings are usually not useful for chunking.
        if len(text.split()) <= 20 and ("issn" in lowered or "doi" in lowered):
            return True

        return False

    def _looks_like_reference_entry(self, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False

        matched_lines = 0
        for line in lines:
            if not self.REFERENCE_AUTHOR_START.search(line):
                continue

            signal_hits = sum(1 for pattern in self.REFERENCE_SIGNAL_PATTERNS if pattern.search(line))
            if signal_hits >= 2:
                matched_lines += 1

        if matched_lines == 0:
            return False

        # For multi-line lists, one strong match is enough. For plain text paragraphs,
        # require a majority of non-empty lines to look like citations.
        return matched_lines == len(lines) or matched_lines / len(lines) >= 0.5

    def _record_removed(self, block: dict[str, Any], reason: str) -> None:
        text = self._clean_text(self._html_to_text(block.get("html", "")))
        page_id = self._page_id_from_block_id(block.get("id", ""))
        self.removed_blocks.append(
            RemovedBlock(
                block_id=block.get("id", ""),
                block_type=block.get("block_type", ""),
                text=text,
                reason=reason,
                page_id=page_id,
            )
        )

    def _page_id_from_block_id(self, block_id: str) -> int | None:
        match = re.match(r"^/page/(\d+)", block_id)
        if not match:
            return None
        return int(match.group(1))
