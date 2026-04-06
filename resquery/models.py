from __future__ import annotations

from dataclasses import dataclass, field

from dbquery.models import QueryPipelineResult, QueryRequest


@dataclass(slots=True)
class ResearchClaim:
    claim_id: str
    text: str
    status: str
    confidence: str
    evidence_chunk_ids: list[str] = field(default_factory=list)
    created_in_turn: str = ""


@dataclass(slots=True)
class SuggestedFollowup:
    question_id: str
    text: str
    created_in_turn: str = ""


@dataclass(slots=True)
class EvidenceIndexEntry:
    chunk_id: str
    paper_id: str
    section_label: str | None
    section_title: str


@dataclass(slots=True)
class SelectedClaim:
    claim_id: str
    text: str
    status: str
    confidence: str


@dataclass(slots=True)
class SelectedFollowup:
    question_id: str
    text: str


@dataclass(slots=True)
class SelectedEvidence:
    chunk_id: str
    paper_id: str
    section_label: str | None
    section_title: str


@dataclass(slots=True)
class SelectedStateView:
    root_query: str
    prior_claims: list[SelectedClaim] = field(default_factory=list)
    followup_suggestions: list[SelectedFollowup] = field(default_factory=list)
    recent_evidence: list[SelectedEvidence] = field(default_factory=list)


@dataclass(slots=True)
class ResearchTurn:
    turn_id: str
    user_question: str
    timestamp: str
    selected_state_view: SelectedStateView
    query_request: QueryRequest
    contextualized_query: str
    query_output_path: str | None
    synthesized_summary: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    claims_added: list[str] = field(default_factory=list)
    followups_added: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResearchSessionState:
    session_id: str
    created_at: str
    updated_at: str
    root_query: str
    turn_order: list[str] = field(default_factory=list)
    turns: dict[str, ResearchTurn] = field(default_factory=dict)
    claims: dict[str, ResearchClaim] = field(default_factory=dict)
    followup_suggestions: dict[str, SuggestedFollowup] = field(default_factory=dict)
    evidence_index: dict[str, EvidenceIndexEntry] = field(default_factory=dict)


@dataclass(slots=True)
class StateUpdateClaim:
    text: str
    status: str
    confidence: str
    evidence_chunk_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StateUpdateFollowup:
    text: str


@dataclass(slots=True)
class StateUpdate:
    claims_added: list[StateUpdateClaim] = field(default_factory=list)
    followups_added: list[StateUpdateFollowup] = field(default_factory=list)


@dataclass(slots=True)
class ResQueryRequest:
    session_path: str
    user_question: str
    query_request: QueryRequest
    query_output_path: str | None = None


@dataclass(slots=True)
class ResQueryResult:
    session_state: ResearchSessionState
    selected_state_view: SelectedStateView
    query_result: QueryPipelineResult
    query_output_path: str | None
    turn: ResearchTurn
    state_update: StateUpdate
    session_path: str
    turn_output_path: str | None = None
