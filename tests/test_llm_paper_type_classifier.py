from __future__ import annotations

import unittest

from chunker.llm_paper_type_classifier import PaperTypeClassificationLLM
from chunker.paper_type_classifier import PaperTypeEvidence


class TestPaperTypeClassificationLLM(unittest.TestCase):
    def test_invalid_output_falls_back_to_other_or_unclear(self) -> None:
        events: list[str] = []

        class InvalidClassifier(PaperTypeClassificationLLM):
            def _generate(self, messages):
                return "not json"

        classifier = InvalidClassifier(
            model_path="unused",
            event_logger=events.append,
        )
        result = classifier.classify(self._evidence())

        self.assertEqual(result.label, "other_or_unclear")
        self.assertEqual(result.confidence, "low")
        self.assertFalse(result.needs_llm)
        self.assertTrue(any("fallback applied" in event for event in events))

    def test_runtime_failure_falls_back_to_other_or_unclear(self) -> None:
        class FailedClassifier(PaperTypeClassificationLLM):
            def _generate(self, messages):
                raise RuntimeError("worker unavailable")

        result = FailedClassifier(model_path="unused").classify(self._evidence())

        self.assertEqual(result.label, "other_or_unclear")
        self.assertIn("worker unavailable", result.reason)

    def _evidence(self) -> PaperTypeEvidence:
        return PaperTypeEvidence(
            title="Example",
            abstract="Example abstract",
            headings=["Introduction"],
            opening_excerpt="Example",
            closing_excerpt="Example",
        )


if __name__ == "__main__":
    unittest.main()
