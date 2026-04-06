from __future__ import annotations

from dbquery.models import BatchSummary

from .batch_summary_claims_extractor import GemmaBatchSummaryClaimsExtractor
from .claim_deduper import ClaimDeduper
from .followup_suggester import GemmaFollowupSuggester
from .models import ResearchSessionState, StateUpdate


class ResearchStateUpdater:
    def __init__(
        self,
        *,
        claims_extractor: GemmaBatchSummaryClaimsExtractor,
        claim_deduper: ClaimDeduper,
        followup_suggester: GemmaFollowupSuggester,
    ) -> None:
        self.claims_extractor = claims_extractor
        self.claim_deduper = claim_deduper
        self.followup_suggester = followup_suggester

    def update_state(
        self,
        *,
        existing_state: ResearchSessionState,
        user_question: str,
        synthesized_summary: str,
        summaries: list[BatchSummary],
    ) -> StateUpdate:
        claims_added = self._extract_claims(user_question=user_question, summaries=summaries)
        claims_added = self.claim_deduper.dedupe(existing_state=existing_state, claims=claims_added)
        followups_added = self.followup_suggester.suggest(
            user_question=user_question,
            synthesized_summary=synthesized_summary,
        )
        return StateUpdate(
            claims_added=claims_added,
            followups_added=followups_added,
        )

    def _extract_claims(self, *, user_question: str, summaries: list[BatchSummary]):
        claims = []
        for summary in summaries:
            claims.extend(
                self.claims_extractor.extract(
                    user_question=user_question,
                    summary=summary,
                )
            )
        return claims
