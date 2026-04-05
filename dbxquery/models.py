from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from dbquery.models import FusedChunkResult


AnswerMode = Literal[
    "direct_answer",
    "compare",
    "critique",
    "recommend",
    "clarify_insufficient_evidence",
]
EvidenceTool = Literal["retrieve_evidence"]


@dataclass(slots=True)
class EvidenceFilters:
    paper_id_in: list[str] = field(default_factory=list)
    authors_any: list[str] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    classification_label_in: list[str] = field(default_factory=list)
    section_title_contains: str | None = None


@dataclass(slots=True)
class EvidenceToolCall:
    tool: EvidenceTool
    query: str
    filters: EvidenceFilters = field(default_factory=EvidenceFilters)
    top_k: int = 6


@dataclass(slots=True)
class EvidencePlan:
    intent: str
    answer_mode: AnswerMode
    reason: str
    tool_calls: list[EvidenceToolCall]


@dataclass(slots=True)
class FollowupRequest:
    prior_summary: str
    user_question: str
    db_path: str
    table_name: str


@dataclass(slots=True)
class EvidenceResult:
    tool_call: EvidenceToolCall
    fused_results: list[FusedChunkResult]


@dataclass(slots=True)
class GroundedAnswer:
    answer_text: str
    used_chunk_ids: list[str]
    used_paper_ids: list[str]
    insufficient_evidence: bool


@dataclass(slots=True)
class FollowupPipelineResult:
    request: FollowupRequest
    plan: EvidencePlan
    evidence_results: list[EvidenceResult]
    grounded_answer: GroundedAnswer
