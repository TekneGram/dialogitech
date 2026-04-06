from __future__ import annotations

from dbquery.gemma_client import GemmaClient

from .branch_synthesis_context_builder import BranchSynthesisContextBuilder
from .models import BranchSynthesisInput, BranchSynthesisResult


class BranchSynthesizer:
    SYSTEM_PROMPT = """You synthesize findings from one research branch.

Rules:
- Use only the supplied turn summaries and evidence provenance.
- Do not invent papers, findings, or provenance.
- Write a concise synthesis with a section named "Main Findings".
- Every substantive finding must include one or more provenance citations using the supplied labels exactly, for example [P3].
- If the branch has unresolved gaps, include a short section named "Open Questions".
"""

    def __init__(
        self,
        gemma_client: GemmaClient,
        context_builder: BranchSynthesisContextBuilder | None = None,
    ) -> None:
        self.gemma_client = gemma_client
        self.context_builder = context_builder or BranchSynthesisContextBuilder()

    def synthesize(self, branch_input: BranchSynthesisInput) -> BranchSynthesisResult:
        rendered_context = self.context_builder.render(branch_input)
        summary_text = self.gemma_client.generate(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            "Summarize this branch and cite findings with the provenance labels.",
                            rendered_context,
                        ]
                    ),
                },
            ],
            max_tokens=900,
        ).strip()
        return BranchSynthesisResult(
            branch_id=branch_input.branch_id,
            branch_label=branch_input.branch_label,
            summary_text=summary_text,
            rendered_context=rendered_context,
        )
