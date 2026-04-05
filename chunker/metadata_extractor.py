from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any


class MetadataExtractor:
    EXCLUDED_FRONT_MATTER_HEADINGS = {
        "abstract",
        "background",
        "introduction",
        "methods",
        "references",
        "acknowledgment",
        "acknowledgement",
        "declaration of use of ai",
    }
    AFFILIATION_KEYWORDS = {
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
    REFERENCE_AUTHOR_START = re.compile(
        r"^(?:-+\s*)?(?:[A-Z][A-Za-z'`\-]+,\s+(?:[A-Z]\.\s*)+)(?:,\s*&\s*[A-Z][A-Za-z'`\-]+,\s+(?:[A-Z]\.\s*)+)?"
    )
    YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
    VOLUME_PATTERN = re.compile(r"\bVolume\s+(\d+)\b", re.IGNORECASE)
    ISSUE_PATTERN = re.compile(r"\b(?:Number|Issue)\s+(\d+)\b", re.IGNORECASE)
    DOI_PATTERN = re.compile(r"https?://doi\.org/\S+|\b10\.\d{4,9}/\S+", re.IGNORECASE)
    ISSN_PATTERN = re.compile(r"\bISSN\s+([0-9Xx\-]+)\b")

    def load_json(self, source: str | Path | dict[str, Any]) -> dict[str, Any]:
        if isinstance(source, dict):
            return source
        path = Path(source)
        return json.loads(path.read_text(encoding="utf-8"))

    def extract_all(self, source: str | Path | dict[str, Any]) -> dict[str, Any]:
        document = self.load_json(source)
        return {
            "title": self.extract_title(document),
            "journal": self.extract_journal_metadata(document),
            "authors": self.extract_authors(document),
            "references": self.extract_references(document),
        }

    def extract_title(self, document: dict[str, Any]) -> str | None:
        page0_blocks = self._page_blocks(document, 0)
        abstract_index = self._find_heading_index(page0_blocks, "Abstract")
        heading_candidates: list[str] = []

        for index, block in enumerate(page0_blocks):
            if abstract_index is not None and index >= abstract_index:
                break
            if block.get("block_type") != "SectionHeader":
                continue
            title = self._clean_text(self._html_to_text(block.get("html", "")))
            if not title:
                continue
            if self._normalize_title(title) in {
                "vocabulary learning and instruction",
                "abstract",
            }:
                continue
            heading_candidates.append(title)

        if not heading_candidates:
            return None
        return max(heading_candidates, key=lambda item: len(item.split()))

    def extract_journal_metadata(self, document: dict[str, Any]) -> dict[str, str | None]:
        front_blocks = self._front_matter_blocks(document)
        front_texts = [self._clean_text(self._html_to_text(block.get("html", ""))) for block in front_blocks]
        front_text = "\n".join(text for text in front_texts if text)

        journal_name = None
        page0_blocks = self._page_blocks(document, 0)
        for block in page0_blocks:
            if block.get("block_type") != "SectionHeader":
                continue
            text = self._clean_text(self._html_to_text(block.get("html", "")))
            normalized = self._normalize_title(text)
            if normalized and normalized not in self.EXCLUDED_FRONT_MATTER_HEADINGS:
                if normalized != self._normalize_title(self.extract_title(document) or ""):
                    journal_name = text
                    break

        volume = self._first_group(self.VOLUME_PATTERN, front_text)
        issue = self._first_group(self.ISSUE_PATTERN, front_text)
        year = self._first_group(self.YEAR_PATTERN, front_text, group=0)
        doi = self._first_group(self.DOI_PATTERN, front_text, group=0)
        issn = self._first_group(self.ISSN_PATTERN, front_text)

        return {
            "name": journal_name,
            "volume": volume,
            "issue": issue,
            "year": year,
            "doi": doi,
            "issn": issn,
        }

    def extract_authors(self, document: dict[str, Any]) -> list[str]:
        title = self.extract_title(document)
        if title is None:
            return []

        page0_blocks = self._page_blocks(document, 0)
        title_seen = False
        authors: list[str] = []

        for block in page0_blocks:
            block_type = block.get("block_type")
            text = self._clean_text(self._html_to_text(block.get("html", "")))
            if not text:
                continue

            if block_type == "SectionHeader":
                if self._normalize_title(text) == self._normalize_title(title):
                    title_seen = True
                    continue
                if self._normalize_title(text) == "abstract":
                    break
                continue

            if not title_seen:
                continue

            if self._looks_like_affiliation(text):
                continue
            if self._looks_like_boilerplate_front_matter(text):
                continue
            if self._looks_like_author_line(text):
                authors.extend(self._split_authors(text))

        return self._dedupe_preserve_order(authors)

    def extract_references(self, document: dict[str, Any]) -> list[str]:
        references: list[str] = []
        in_references = False

        for page in document.get("children", []):
            for block in page.get("children", []) or []:
                block_type = block.get("block_type")
                text = self._clean_text_block(self._html_to_text(block.get("html", "")))
                if block_type == "SectionHeader":
                    normalized = self._normalize_title(text)
                    if normalized == "references":
                        in_references = True
                        continue
                    if in_references and normalized:
                        return self._dedupe_preserve_order(references)

                if not in_references:
                    continue

                if block_type == "ListGroup":
                    references.extend(self._extract_list_items(block))
                elif block_type in {"Text", "ListItem"} and text:
                    if self._looks_like_reference_entry(text):
                        references.append(text)

        return self._dedupe_preserve_order(references)

    def _front_matter_blocks(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for page in document.get("children", [])[:2]:
            for block in page.get("children", []) or []:
                if block.get("block_type") == "SectionHeader":
                    heading = self._clean_text(self._html_to_text(block.get("html", "")))
                    if self._normalize_title(heading) == "abstract":
                        return blocks
                blocks.append(block)
        return blocks

    def _page_blocks(self, document: dict[str, Any], page_id: int) -> list[dict[str, Any]]:
        for page in document.get("children", []):
            if self._page_id_from_block_id(page.get("id", "")) == page_id:
                return page.get("children", []) or []
        return []

    def _find_heading_index(self, blocks: list[dict[str, Any]], title: str) -> int | None:
        normalized_title = self._normalize_title(title)
        for index, block in enumerate(blocks):
            if block.get("block_type") != "SectionHeader":
                continue
            text = self._clean_text(self._html_to_text(block.get("html", "")))
            if self._normalize_title(text) == normalized_title:
                return index
        return None

    def _extract_list_items(self, block: dict[str, Any]) -> list[str]:
        items: list[str] = []
        for child in block.get("children") or []:
            text = self._clean_text_block(self._html_to_text(child.get("html", "")))
            if text:
                items.append(text)
        return items

    def _html_to_text(self, html_text: str) -> str:
        if not html_text:
            return ""
        text = re.sub(r"<br\s*/?>", "\n", html_text, flags=re.IGNORECASE)
        text = re.sub(r"</(p|div|h\d|li|tr|blockquote|ul|ol)>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        return text

    def _clean_text(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _clean_text_block(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _normalize_title(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def _looks_like_affiliation(self, text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in self.AFFILIATION_KEYWORDS)

    def _looks_like_boilerplate_front_matter(self, text: str) -> bool:
        lowered = text.lower()
        return "issn" in lowered or "doi" in lowered or "copyright" in lowered

    def _looks_like_author_line(self, text: str) -> bool:
        if len(text.split()) > 8:
            return False
        if self._looks_like_affiliation(text) or self._looks_like_boilerplate_front_matter(text):
            return False
        return bool(re.match(r"^[A-Z][A-Za-z.\-'\s,&]+$", text))

    def _split_authors(self, text: str) -> list[str]:
        parts = re.split(r"\s+(?:and|&)\s+|,\s*(?=[A-Z][a-z])", text)
        return [part.strip() for part in parts if part.strip()]

    def _looks_like_reference_entry(self, text: str) -> bool:
        first_line = text.splitlines()[0].strip()
        if not self.REFERENCE_AUTHOR_START.search(first_line):
            return False
        signal_count = 0
        if re.search(r"\(\d{4}[a-z]?\)", text):
            signal_count += 1
        if re.search(r"https?://|\bdoi\b", text, re.IGNORECASE):
            signal_count += 1
        if re.search(r"\b(?:journal|proceedings|conference|studies|review|corpus)\b", text, re.IGNORECASE):
            signal_count += 1
        if re.search(r"\b\d+\s*[,:]\s*\d+(?:[–-]\d+)?\b", text):
            signal_count += 1
        return signal_count >= 2

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            key = self._normalize_title(item)
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _page_id_from_block_id(self, block_id: str) -> int | None:
        match = re.match(r"^/page/(\d+)", block_id)
        if not match:
            return None
        return int(match.group(1))

    def _first_group(self, pattern: re.Pattern[str], text: str, group: int = 1) -> str | None:
        match = pattern.search(text)
        if not match:
            return None
        return match.group(group).strip()
