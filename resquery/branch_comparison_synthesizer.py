from __future__ import annotations

from dbquery.gemma_client import GemmaClient

from .models import BranchComparisonResult, BranchSynthesisResult


class BranchComparisonSynthesizer:
    SYSTEM_PROMPT = """You compare two research-branch summaries.

Rules:
- Use only the supplied branch summaries.
- Do not invent evidence or disagreements.
- Write concise sections named "Similarities" and "Points of Contention".
- Refer to branches using their branch IDs exactly.
- If there is little real disagreement, say so plainly.
"""

    def __init__(self, gemma_client: GemmaClient) -> None:
        self.gemma_client = gemma_client

    def compare(
        self,
        left: BranchSynthesisResult,
        right: BranchSynthesisResult,
    ) -> BranchComparisonResult:
        comparison_text = self.gemma_client.generate(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"Branch {left.branch_id} ({left.branch_label}) summary:",
                            left.summary_text,
                            f"Branch {right.branch_id} ({right.branch_label}) summary:",
                            right.summary_text,
                            "Compare these two branch summaries.",
                        ]
                    ),
                },
            ],
            max_tokens=700,
        ).strip()
        return BranchComparisonResult(
            left_branch_id=left.branch_id,
            right_branch_id=right.branch_id,
            comparison_text=comparison_text,
        )
