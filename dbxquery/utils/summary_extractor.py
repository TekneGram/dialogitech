from __future__ import annotations

import re


class PriorSummaryExtractor:
    def extract(self, document_text: str) -> str:
        synthesized = self._extract_synthesized_summary(document_text)
        if synthesized is not None:
            return synthesized
        return document_text.strip()

    def _extract_synthesized_summary(self, document_text: str) -> str | None:
        pattern = re.compile(
            r"^## Synthesized Summary\s*$\n+(.*?)(?=^##\s|\Z)",
            flags=re.MULTILINE | re.DOTALL,
        )
        match = pattern.search(document_text)
        if match is None:
            return None
        summary = match.group(1).strip()
        return summary or None
