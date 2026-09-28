from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_TEXT_QUALITY_CONFIG_PATH = Path(__file__).with_name("text_quality_config.json")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_WORD_PATTERN = re.compile(r"\S+")
_NON_TEXT_BLOCK_TYPES = {
    "PageHeader",
    "PageFooter",
    "Picture",
    "PictureGroup",
    "Figure",
    "FigureGroup",
    "Caption",
    "Reference",
}


@dataclass(frozen=True, slots=True)
class TextQualityThresholds:
    minimum_pages_for_sparse_check: int
    minimum_total_words: int
    maximum_text_page_ratio: float
    maximum_words_on_page: int
    maximum_words_per_page_ratio: float

    @classmethod
    def from_path(cls, path: str | Path | None = None) -> "TextQualityThresholds":
        config_path = Path(path) if path is not None else DEFAULT_TEXT_QUALITY_CONFIG_PATH
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Text quality configuration must be a JSON object: {config_path}")
        try:
            thresholds = cls(
                minimum_pages_for_sparse_check=int(payload["minimum_pages_for_sparse_check"]),
                minimum_total_words=int(payload["minimum_total_words"]),
                maximum_text_page_ratio=float(payload["maximum_text_page_ratio"]),
                maximum_words_on_page=int(payload["maximum_words_on_page"]),
                maximum_words_per_page_ratio=float(payload["maximum_words_per_page_ratio"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid text quality configuration: {config_path}") from exc
        thresholds.validate()
        return thresholds

    def validate(self) -> None:
        if self.minimum_pages_for_sparse_check < 1:
            raise ValueError("minimum_pages_for_sparse_check must be positive")
        if self.minimum_total_words < 1:
            raise ValueError("minimum_total_words must be positive")
        if not 0 < self.maximum_text_page_ratio <= 1:
            raise ValueError("maximum_text_page_ratio must be greater than 0 and at most 1")
        if self.maximum_words_on_page < 1:
            raise ValueError("maximum_words_on_page must be positive")
        if self.maximum_words_per_page_ratio <= 0:
            raise ValueError("maximum_words_per_page_ratio must be positive")


@dataclass(frozen=True, slots=True)
class TextExtractionQuality:
    page_count: int
    total_words: int
    total_characters: int
    text_pages: int
    text_page_ratio: float
    max_words_on_page: int
    likely_requires_ocr: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class OcrRequiredError(RuntimeError):
    def __init__(self, message: str, quality: TextExtractionQuality) -> None:
        super().__init__(message)
        self.quality = quality


class MarkerTextQualityChecker:
    """Measure whether Marker recovered enough text for downstream processing."""

    def __init__(self, thresholds: TextQualityThresholds | None = None) -> None:
        self.thresholds = thresholds or TextQualityThresholds.from_path()

    def inspect(self, document: dict[str, Any]) -> TextExtractionQuality:
        pages = document.get("children")
        if not isinstance(pages, list):
            raise ValueError("Marker document does not contain a valid children list.")

        page_word_counts: list[int] = []
        page_character_counts: list[int] = []
        for page in pages:
            texts: list[str] = []
            self._collect_text(page.get("children", []) if isinstance(page, dict) else [], texts)
            page_text = " ".join(texts)
            page_character_counts.append(len(page_text))
            page_word_counts.append(len(_WORD_PATTERN.findall(page_text)))

        page_count = len(pages)
        total_words = sum(page_word_counts)
        total_characters = sum(page_character_counts)
        text_pages = sum(words > 0 for words in page_word_counts)
        text_page_ratio = text_pages / page_count if page_count else 0.0
        max_words_on_page = max(page_word_counts, default=0)

        likely_requires_ocr, reason = self._assess(
            page_count=page_count,
            total_words=total_words,
            text_page_ratio=text_page_ratio,
            max_words_on_page=max_words_on_page,
        )
        return TextExtractionQuality(
            page_count=page_count,
            total_words=total_words,
            total_characters=total_characters,
            text_pages=text_pages,
            text_page_ratio=text_page_ratio,
            max_words_on_page=max_words_on_page,
            likely_requires_ocr=likely_requires_ocr,
            reason=reason,
        )

    def _assess(
        self,
        *,
        page_count: int,
        total_words: int,
        text_page_ratio: float,
        max_words_on_page: int,
    ) -> tuple[bool, str]:
        if total_words == 0:
            return True, "Marker extracted no text from the document."
        if page_count == 0:
            return True, "Marker returned no pages."
        if page_count < self.thresholds.minimum_pages_for_sparse_check:
            return False, "The document is short but contains extractable text."

        expected_sparse_limit = max(
            self.thresholds.minimum_total_words,
            int(page_count * self.thresholds.maximum_words_per_page_ratio),
        )
        sparse_document = (
            total_words < expected_sparse_limit
            and (
                text_page_ratio < self.thresholds.maximum_text_page_ratio
                or max_words_on_page < self.thresholds.maximum_words_on_page
            )
        )
        if sparse_document:
            return True, (
                "Marker extracted very little text relative to the document size "
                "and page coverage."
            )
        return False, "Marker extracted enough text for downstream processing."

    def _collect_text(self, blocks: Any, texts: list[str]) -> None:
        if not isinstance(blocks, list):
            return
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("block_type")
            if block_type not in _NON_TEXT_BLOCK_TYPES:
                raw_html = block.get("html")
                if isinstance(raw_html, str):
                    text = self._html_to_text(raw_html)
                    if text:
                        texts.append(text)
            self._collect_text(block.get("children"), texts)

    @staticmethod
    def _html_to_text(raw_html: str) -> str:
        text = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
        text = _HTML_TAG_PATTERN.sub(" ", text)
        text = html.unescape(text)
        return re.sub(r"\s+", " ", text).strip()
