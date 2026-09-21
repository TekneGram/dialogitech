from __future__ import annotations

import html
import re
from typing import Any


class DeterministicReferenceExtractor:
    """Extract the references section from Marker blocks without an LLM."""

    REFERENCE_AUTHOR_START = re.compile(
        r"^(?:-+\s*)?(?:[A-Z][A-Za-z'`\-]+,\s+(?:[A-Z]\.\s*)+)"
        r"(?:,\s*&\s*[A-Z][A-Za-z'`\-]+,\s+(?:[A-Z]\.\s*)+)?"
    )

    def extract(self, document: dict[str, Any]) -> list[str]:
        references: list[str] = []
        in_references = False

        for page in document.get("children", []):
            for block in page.get("children", []) or []:
                block_type = block.get("block_type")
                text = self._clean_block_text(self._html_to_text(block.get("html", "")))

                if block_type == "SectionHeader":
                    normalized = self._normalize(text)
                    if normalized == "references":
                        in_references = True
                        continue
                    if in_references and normalized:
                        return self._dedupe(references)

                if not in_references:
                    continue

                if block_type == "ListGroup":
                    references.extend(self._list_items(block))
                elif block_type in {"Text", "ListItem"} and text:
                    if self._looks_like_reference_entry(text):
                        references.append(text)

        return self._dedupe(references)

    def _list_items(self, block: dict[str, Any]) -> list[str]:
        items: list[str] = []
        for child in block.get("children") or []:
            text = self._clean_block_text(self._html_to_text(child.get("html", "")))
            if text:
                items.append(text)
        return items

    def _looks_like_reference_entry(self, text: str) -> bool:
        first_line = text.splitlines()[0].strip()
        if not self.REFERENCE_AUTHOR_START.search(first_line):
            return False

        signal_count = sum(
            (
                bool(re.search(r"\(\d{4}[a-z]?\)", text)),
                bool(re.search(r"https?://|\bdoi\b", text, re.IGNORECASE)),
                bool(
                    re.search(
                        r"\b(?:journal|proceedings|conference|studies|review|corpus)\b",
                        text,
                        re.IGNORECASE,
                    )
                ),
                bool(re.search(r"\b\d+\s*[,:]\s*\d+(?:[–-]\d+)?\b", text)),
            )
        )
        return signal_count >= 2

    def _html_to_text(self, html_text: str) -> str:
        if not html_text:
            return ""
        text = re.sub(r"<br\s*/?>", "\n", html_text, flags=re.IGNORECASE)
        text = re.sub(r"</(p|div|h\d|li|tr|blockquote|ul|ol)>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        return html.unescape(text)

    def _clean_block_text(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def _dedupe(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            key = self._normalize(item)
            if key and key not in seen:
                seen.add(key)
                result.append(item)
        return result
