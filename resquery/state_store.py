from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from dbquery.models import QueryFilters, QueryRequest

from .models import (
    EvidenceIndexEntry,
    ResearchClaim,
    ResearchSessionState,
    ResearchTurn,
    SelectedClaim,
    SelectedEvidence,
    SelectedFollowup,
    SelectedStateView,
    SuggestedFollowup,
)


class ResearchStateStore:
    def load(self, path: str | Path) -> ResearchSessionState:
        payload = Path(path).read_text(encoding="utf-8")
        return self._from_dict(json.loads(payload))

    def save(self, state: ResearchSessionState, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
        return target

    def exists(self, path: str | Path) -> bool:
        return Path(path).exists()

    def _from_dict(self, payload: dict[str, object]) -> ResearchSessionState:
        turns_payload = self._require_dict(payload.get("turns"), "turns")
        claims_payload = self._require_dict(payload.get("claims"), "claims")
        followups_payload = self._require_dict(
            payload.get("followup_suggestions", payload.get("open_questions", {})),
            "followup_suggestions",
        )
        evidence_payload = self._require_dict(payload.get("evidence_index"), "evidence_index")
        return ResearchSessionState(
            session_id=str(payload["session_id"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            root_query=str(payload["root_query"]),
            turn_order=[str(item) for item in payload.get("turn_order", [])],
            turns={key: self._load_turn(key, value) for key, value in turns_payload.items()},
            claims={key: self._load_claim(key, value) for key, value in claims_payload.items()},
            followup_suggestions={
                key: self._load_followup(key, value) for key, value in followups_payload.items()
            },
            evidence_index={
                key: self._load_evidence_entry(key, value) for key, value in evidence_payload.items()
            },
        )

    def _load_turn(self, turn_id: str, payload: object) -> ResearchTurn:
        data = self._require_dict(payload, f"turn {turn_id}")
        query_request_payload = self._require_dict(data.get("query_request"), f"turn {turn_id} query_request")
        return ResearchTurn(
            turn_id=str(data["turn_id"]),
            user_question=str(data["user_question"]),
            timestamp=str(data["timestamp"]),
            selected_state_view=self._load_selected_state_view(
                self._require_dict(data.get("selected_state_view"), f"turn {turn_id} selected_state_view")
            ),
            query_request=self._load_query_request(query_request_payload),
            contextualized_query=str(data.get("contextualized_query", query_request_payload.get("query", ""))),
            query_output_path=self._optional_string(data.get("query_output_path")),
            synthesized_summary=str(data["synthesized_summary"]),
            retrieved_chunk_ids=[str(item) for item in data.get("retrieved_chunk_ids", [])],
            claims_added=[str(item) for item in data.get("claims_added", [])],
            followups_added=[str(item) for item in data.get("followups_added", data.get("open_questions_added", []))],
        )

    def _load_selected_state_view(self, payload: dict[str, object]) -> SelectedStateView:
        return SelectedStateView(
            root_query=str(payload["root_query"]),
            prior_claims=[
                SelectedClaim(
                    claim_id=str(item["claim_id"]),
                    text=str(item["text"]),
                    status=str(item["status"]),
                    confidence=str(item["confidence"]),
                )
                for item in payload.get("prior_claims", [])
                if isinstance(item, dict)
            ],
            followup_suggestions=[
                SelectedFollowup(
                    question_id=str(item["question_id"]),
                    text=str(item["text"]),
                )
                for item in payload.get("followup_suggestions", payload.get("open_questions", []))
                if isinstance(item, dict)
            ],
            recent_evidence=[
                SelectedEvidence(
                    chunk_id=str(item["chunk_id"]),
                    paper_id=str(item["paper_id"]),
                    section_label=self._optional_string(item.get("section_label")),
                    section_title=str(item["section_title"]),
                )
                for item in payload.get("recent_evidence", [])
                if isinstance(item, dict)
            ],
        )

    def _load_query_request(self, payload: dict[str, object]) -> QueryRequest:
        filters_payload = payload.get("filters")
        filters = None
        if isinstance(filters_payload, dict):
            filters = QueryFilters(
                paper_id=self._optional_string(filters_payload.get("paper_id")),
                paper_id_in=[str(item) for item in filters_payload.get("paper_id_in", [])],
                year=self._optional_int(filters_payload.get("year")),
                year_min=self._optional_int(filters_payload.get("year_min")),
                year_max=self._optional_int(filters_payload.get("year_max")),
                paper_title_contains=self._optional_string(filters_payload.get("paper_title_contains")),
                classification_label=self._optional_string(filters_payload.get("classification_label")),
                classification_label_in=[
                    str(item) for item in filters_payload.get("classification_label_in", [])
                ],
                section_title_contains=self._optional_string(filters_payload.get("section_title_contains")),
                author=self._optional_string(filters_payload.get("author")),
                authors_any=[str(item) for item in filters_payload.get("authors_any", [])],
            )
        return QueryRequest(
            query=str(payload["query"]),
            retrieval_mode=str(payload.get("retrieval_mode", "full")),
            retrieval_depth=int(payload.get("retrieval_depth", 10)),
            final_top_k=int(payload.get("final_top_k", 15)),
            batch_size=int(payload.get("batch_size", 5)),
            rewrite_count=int(payload.get("rewrite_count", 2)),
            include_hyde=bool(payload.get("include_hyde", True)),
            min_rrf_lists=int(payload.get("min_rrf_lists", 1)),
            rrf_k=int(payload.get("rrf_k", 60)),
            min_relevance_score=self._optional_float(payload.get("min_relevance_score")),
            filters=filters,
        )

    def _load_claim(self, claim_id: str, payload: object) -> ResearchClaim:
        data = self._require_dict(payload, f"claim {claim_id}")
        return ResearchClaim(
            claim_id=str(data["claim_id"]),
            text=str(data["text"]),
            status=str(data["status"]),
            confidence=str(data["confidence"]),
            evidence_chunk_ids=[str(item) for item in data.get("evidence_chunk_ids", [])],
            created_in_turn=str(data["created_in_turn"]),
        )

    def _load_followup(self, question_id: str, payload: object) -> SuggestedFollowup:
        data = self._require_dict(payload, f"followup {question_id}")
        return SuggestedFollowup(
            question_id=str(data["question_id"]),
            text=str(data["text"]),
            created_in_turn=str(data["created_in_turn"]),
        )

    def _load_evidence_entry(self, chunk_id: str, payload: object) -> EvidenceIndexEntry:
        data = self._require_dict(payload, f"evidence entry {chunk_id}")
        return EvidenceIndexEntry(
            chunk_id=str(data["chunk_id"]),
            paper_id=str(data["paper_id"]),
            section_label=self._optional_string(data.get("section_label")),
            section_title=str(data["section_title"]),
        )

    def _require_dict(self, payload: object, label: str) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise ValueError(f"Expected {label} to be an object.")
        return payload

    def _optional_string(self, value: object) -> str | None:
        return None if value is None else str(value)

    def _optional_int(self, value: object) -> int | None:
        return None if value is None else int(value)

    def _optional_float(self, value: object) -> float | None:
        return None if value is None else float(value)
