from __future__ import annotations

from dbquery.gemma_client import GemmaClient

from .models import EvidencePlan, EvidenceResult, GroundedAnswer
from .utils.prompt_rendering import FollowupPromptRenderer


class GemmaGroundedWriter:
    SYSTEM_PROMPT = """You answer academic follow-up questions using only retrieved evidence.

Rules:
- Use only the supplied retrieved evidence.
- Do not use outside knowledge.
- If the evidence is insufficient, say so explicitly.
- If evidence conflicts, say so explicitly.
- Cite claims with paper identifiers in square brackets, for example [2025_Zhu].
- Keep the answer concise and directly responsive to the user question.
"""

    def __init__(
        self,
        gemma_client: GemmaClient,
        prompt_renderer: FollowupPromptRenderer | None = None,
    ) -> None:
        self.gemma_client = gemma_client
        self.prompt_renderer = prompt_renderer or FollowupPromptRenderer()

    def write_answer(
        self,
        *,
        plan: EvidencePlan,
        prior_summary: str,
        user_question: str,
        evidence_results: list[EvidenceResult],
    ) -> GroundedAnswer:
        prompt = self.prompt_renderer.render_grounded_answer_prompt(
            plan=plan,
            prior_summary=prior_summary,
            user_question=user_question,
            evidence_results=evidence_results,
        )
        answer_text = self.gemma_client.generate(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
        ).strip()
        return GroundedAnswer(
            answer_text=answer_text,
            used_chunk_ids=self._collect_chunk_ids(evidence_results),
            used_paper_ids=self._collect_paper_ids(evidence_results),
            insufficient_evidence=not any(result.fused_results for result in evidence_results),
        )

    def _collect_chunk_ids(self, evidence_results: list[EvidenceResult]) -> list[str]:
        seen: set[str] = set()
        chunk_ids: list[str] = []
        for evidence_result in evidence_results:
            for chunk in evidence_result.fused_results:
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                chunk_ids.append(chunk.chunk_id)
        return chunk_ids

    def _collect_paper_ids(self, evidence_results: list[EvidenceResult]) -> list[str]:
        seen: set[str] = set()
        paper_ids: list[str] = []
        for evidence_result in evidence_results:
            for chunk in evidence_result.fused_results:
                if chunk.paper_id in seen:
                    continue
                seen.add(chunk.paper_id)
                paper_ids.append(chunk.paper_id)
        return paper_ids
