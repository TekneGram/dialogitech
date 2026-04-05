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
        "tech",
        "research",
        "academy",
        "sciences",
        "engineering",
        "laboratories",
    }
    AUTHOR_MARKER_PATTERN = re.compile(
        r"(?:\^\{[^}]+\}|\[[^\]]+\]|[\*†‡§¶‖]+|\b\d+\b(?=\s*[A-Z]))"
    )
    NUMBERED_SECTION_HEADING_PATTERN = re.compile(r"^\d+(?:\.\d+)*\s+[A-Z]")
    PERSON_NAME_PATTERN = re.compile(
        r"^[A-Z][A-Za-z'`\-]+(?:\s+[A-Z](?:\.)?)?(?:\s+[A-Z][A-Za-z'`\-]+){0,3}$"
    )
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
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("__source_path__", str(path))
        return payload

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
        title = self.extract_title(document)
        for block in page0_blocks:
            if block.get("block_type") != "SectionHeader":
                continue
            text = self._clean_text(self._html_to_text(block.get("html", "")))
            if not self._looks_like_journal_candidate(text, title=title):
                continue
            journal_name = text
            break

        volume = self._first_group(self.VOLUME_PATTERN, front_text)
        issue = self._first_group(self.ISSUE_PATTERN, front_text)
        year = self._extract_year(front_text, document)
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

        authors: list[str] = []
        for block in self._candidate_front_matter_blocks(document, title):
            text = self._clean_text(self._html_to_text(block.get("html", "")))
            if not text or self._looks_like_boilerplate_front_matter(text):
                continue
            cleaned_text = self._strip_author_markers(text)
            author_segment, _ = self._split_mixed_author_affiliation_block(cleaned_text)
            if not author_segment:
                continue
            authors.extend(self._extract_author_candidates_from_text(author_segment))

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

    def _candidate_front_matter_blocks(
        self,
        document: dict[str, Any],
        title: str,
    ) -> list[dict[str, Any]]:
        page0_blocks = self._page_blocks(document, 0)
        title_seen = False
        candidates: list[dict[str, Any]] = []

        for block in page0_blocks:
            block_type = block.get("block_type")
            text = self._clean_text(self._html_to_text(block.get("html", "")))
            if not text:
                continue

            if block_type == "SectionHeader":
                normalized = self._normalize_title(text)
                if normalized == self._normalize_title(title):
                    title_seen = True
                    continue
                if normalized == "abstract":
                    break
                if title_seen:
                    continue

            if title_seen:
                candidates.append(block)

        return candidates

    def _strip_author_markers(self, text: str) -> str:
        text = self.AUTHOR_MARKER_PATTERN.sub(" ", text)
        text = re.sub(r"\s*[,;:/]\s*", ", ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip(" ,")

    def _split_mixed_author_affiliation_block(self, text: str) -> tuple[str, str | None]:
        normalized_text = self._clean_text(text)
        if not normalized_text:
            return "", None
        if not self._looks_like_affiliation(normalized_text):
            return normalized_text, None

        tokens = normalized_text.split()
        for index, token in enumerate(tokens):
            if self._looks_like_affiliation(token):
                prefix = " ".join(tokens[:index]).strip(" ,;")
                suffix = " ".join(tokens[index:]).strip(" ,;")
                return prefix, suffix or None

        for match in re.finditer(r"[,;]", normalized_text):
            prefix = normalized_text[: match.start()].strip(" ,;")
            suffix = normalized_text[match.end() :].strip(" ,;")
            if prefix and suffix and self._looks_like_affiliation(suffix):
                return prefix, suffix

        return "", normalized_text

    def _extract_author_candidates_from_text(self, text: str) -> list[str]:
        if not text:
            return []
        if self._looks_like_author_line(text):
            return [candidate for candidate in self._split_authors(text) if self._looks_like_person_name(candidate)]

        normalized = text.replace(" and ", ", ").replace(" & ", ", ")
        raw_parts = [part.strip(" ,;") for part in normalized.split(",") if part.strip(" ,;")]
        if len(raw_parts) > 1:
            return [part for part in raw_parts if self._looks_like_person_name(part)]

        name_pattern = re.compile(
            r"[A-Z][A-Za-z'`\-]+(?:\s+[A-Z](?:\.)?)?(?:\s+[A-Z][A-Za-z'`\-]+)+"
        )
        candidates = [match.group(0).strip() for match in name_pattern.finditer(text)]
        filtered_candidates = [candidate for candidate in candidates if self._looks_like_person_name(candidate)]
        if filtered_candidates:
            return filtered_candidates
        return self._extract_sequential_author_names(text)

    def _looks_like_person_name(self, text: str) -> bool:
        cleaned = self._clean_text(text).strip(" ,;")
        if not cleaned:
            return False
        if len(cleaned.split()) < 2 or len(cleaned.split()) > 4:
            return False
        if self._looks_like_affiliation(cleaned):
            return False
        if self._looks_like_numbered_section_heading(cleaned):
            return False
        return bool(self.PERSON_NAME_PATTERN.match(cleaned))

    def _looks_like_numbered_section_heading(self, text: str) -> bool:
        return bool(self.NUMBERED_SECTION_HEADING_PATTERN.match(text.strip()))

    def _looks_like_journal_candidate(self, text: str, *, title: str | None) -> bool:
        normalized = self._normalize_title(text)
        if not normalized:
            return False
        if title is not None and normalized == self._normalize_title(title):
            return False
        if normalized in self.EXCLUDED_FRONT_MATTER_HEADINGS:
            return False
        if self._looks_like_numbered_section_heading(text):
            return False
        if re.search(r"\b(?:introduction|methods?|results?|discussion|conclusion|references)\b", normalized):
            return False
        return True

    def _extract_year(self, front_text: str, document: dict[str, Any]) -> str | None:
        year = self._first_group(self.YEAR_PATTERN, front_text, group=0)
        if year:
            return year
        return self._extract_year_from_document(document)

    def _extract_year_from_document(self, document: dict[str, Any]) -> str | None:
        source_path = str(document.get("__source_path__", ""))
        year = self._extract_year_from_identifier(source_path)
        if year:
            return year
        first_page_blocks = self._page_blocks(document, 0)
        for block in first_page_blocks:
            block_id = str(block.get("id", ""))
            year = self._extract_year_from_identifier(block_id)
            if year:
                return year
        document_id = str(document.get("id", ""))
        return self._extract_year_from_identifier(document_id)

    def _extract_year_from_identifier(self, identifier: str) -> str | None:
        match = re.search(r"\b((?:19|20)\d{2})[_-][A-Za-z]", identifier)
        if match:
            return match.group(1)
        path_match = re.search(r"/((?:19|20)\d{2})_[^/]+", identifier)
        if path_match:
            return path_match.group(1)
        return None

    def _extract_sequential_author_names(self, text: str) -> list[str]:
        tokens = [token.strip(" ,;") for token in text.split() if token.strip(" ,;")]
        names: list[str] = []
        current: list[str] = []

        for token in tokens:
            candidate_text = " ".join(current + [token]).strip()
            if self._looks_like_affiliation(candidate_text):
                break
            if not re.match(r"^[A-Z][A-Za-z'`\-]*\.?$", token):
                if current:
                    current = []
                continue
            current.append(token)
            if len(current) >= 2 and self._looks_like_person_name(" ".join(current)):
                names.append(" ".join(current))
                current = []
            elif len(current) >= 4:
                current = []

        return names

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
