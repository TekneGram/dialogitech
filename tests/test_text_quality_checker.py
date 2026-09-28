from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dbinsert.text_quality_checker import MarkerTextQualityChecker, TextQualityThresholds


def marker_document(page_texts: list[str]) -> dict:
    return {
        "children": [
            {
                "id": f"/page/{index}/Page/{index}",
                "children": [
                    {"block_type": "Text", "html": f"<p>{text}</p>"}
                    for text in ([page_text] if page_text else [])
                ],
            }
            for index, page_text in enumerate(page_texts)
        ]
    }


class TestMarkerTextQualityChecker(unittest.TestCase):
    def test_image_only_document_requires_ocr(self) -> None:
        document = {
            "children": [
                {
                    "id": "/page/0/Page/0",
                    "children": [
                        {"block_type": "Picture", "html": ""},
                    ],
                }
            ]
        }

        quality = MarkerTextQualityChecker().inspect(document)

        self.assertTrue(quality.likely_requires_ocr)
        self.assertEqual(quality.total_words, 0)

    def test_short_single_page_with_text_is_not_rejected(self) -> None:
        quality = MarkerTextQualityChecker().inspect(marker_document(["A short abstract with text."]))

        self.assertFalse(quality.likely_requires_ocr)
        self.assertGreater(quality.total_words, 0)

    def test_sparse_multi_page_document_requires_ocr(self) -> None:
        quality = MarkerTextQualityChecker().inspect(
            marker_document(["Title only"] + [""] * 4)
        )

        self.assertTrue(quality.likely_requires_ocr)
        self.assertEqual(quality.text_pages, 1)

    def test_normal_multi_page_document_is_not_rejected(self) -> None:
        page_text = " ".join(f"word{i}" for i in range(120))
        quality = MarkerTextQualityChecker().inspect(marker_document([page_text] * 3))

        self.assertFalse(quality.likely_requires_ocr)
        self.assertEqual(quality.text_pages, 3)

    def test_thresholds_can_be_loaded_from_an_editable_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "thresholds.json"
            path.write_text(
                json.dumps(
                    {
                        "minimum_pages_for_sparse_check": 3,
                        "minimum_total_words": 25,
                        "maximum_text_page_ratio": 0.25,
                        "maximum_words_on_page": 50,
                        "maximum_words_per_page_ratio": 5.0,
                    }
                ),
                encoding="utf-8",
            )

            thresholds = TextQualityThresholds.from_path(path)

        self.assertEqual(thresholds.minimum_total_words, 25)
        self.assertEqual(thresholds.maximum_text_page_ratio, 0.25)


if __name__ == "__main__":
    unittest.main()
