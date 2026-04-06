from __future__ import annotations

from .models import ResearchSessionState, StateUpdateClaim


class ClaimDeduper:
    def dedupe(
        self,
        *,
        existing_state: ResearchSessionState,
        claims: list[StateUpdateClaim],
    ) -> list[StateUpdateClaim]:
        existing_texts = {self._normalize(claim.text) for claim in existing_state.claims.values()}
        deduped: list[StateUpdateClaim] = []
        seen_new: set[str] = set()
        for claim in claims:
            normalized = self._normalize(claim.text)
            if not normalized or normalized in existing_texts or normalized in seen_new:
                continue
            deduped.append(claim)
            seen_new.add(normalized)
        return deduped

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().split())
