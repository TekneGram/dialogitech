from __future__ import annotations

import unittest

from chunker.llm_paper_type_classifier import PaperTypeClassificationLLM


class TestPaperTypeClassificationLLM(unittest.TestCase):
    def test_valid_response_is_returned(self) -> None:
        class ValidClassifier(PaperTypeClassificationLLM):
            def _generate(self, messages):
                return (
                    '{"label":"literature_review",'
                    '"confidence":"high",'
                    '"reason":"The paper synthesizes prior research."}'
                )

        result = ValidClassifier(model_path="unused").classify(
            self._markdown(),
            metadata={"title": "Example paper"},
        )

        self.assertEqual(result.label, "literature_review")
        self.assertEqual(result.source, "llm")
        self.assertEqual(result.confidence, "high")
        self.assertEqual(result.reason, "The paper synthesizes prior research.")

    def test_invalid_first_response_is_corrected(self) -> None:
        events: list[str] = []

        class CorrectedClassifier(PaperTypeClassificationLLM):
            responses = iter([
                "not json",
                '{"label":"methodological_paper",'
                '"confidence":"medium",'
                '"reason":"The paper focuses on method development."}',
            ])

            def _generate(self, messages):
                return next(self.responses)

        result = CorrectedClassifier(
            model_path="unused",
            event_logger=events.append,
        ).classify(self._markdown())

        self.assertEqual(result.label, "methodological_paper")
        self.assertTrue(any("correction" in event for event in events))

    def test_invalid_output_falls_back_to_other_or_unclear(self) -> None:
        events: list[str] = []

        class InvalidClassifier(PaperTypeClassificationLLM):
            def _generate(self, messages):
                return "not json"

        classifier = InvalidClassifier(
            model_path="unused",
            event_logger=events.append,
        )
        result = classifier.classify(self._markdown())

        self.assertEqual(result.label, "other_or_unclear")
        self.assertEqual(result.confidence, "low")
        self.assertTrue(any("fallback applied" in event for event in events))

    def test_invalid_correction_also_falls_back(self) -> None:
        class StillInvalidClassifier(PaperTypeClassificationLLM):
            def _generate(self, messages):
                return "not json"

        result = StillInvalidClassifier(model_path="unused").classify(self._markdown())

        self.assertEqual(result.label, "other_or_unclear")
        self.assertEqual(result.confidence, "low")
        self.assertIn("assigned other_or_unclear", result.reason)

    def test_runtime_failure_falls_back_to_other_or_unclear(self) -> None:
        class FailedClassifier(PaperTypeClassificationLLM):
            def _generate(self, messages):
                raise RuntimeError("worker unavailable")

        result = FailedClassifier(model_path="unused").classify(self._markdown())

        self.assertEqual(result.label, "other_or_unclear")
        self.assertIn("worker unavailable", result.reason)

    def _markdown(self) -> str:
        return "# ABSTRACT\nExample abstract.\n\n# Introduction\nExample."


if __name__ == "__main__":
    unittest.main()
