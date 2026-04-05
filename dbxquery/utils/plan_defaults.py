from __future__ import annotations

ALLOWED_TOOLS = {"retrieve_evidence"}
ALLOWED_ANSWER_MODES = {
    "direct_answer",
    "compare",
    "critique",
    "recommend",
    "clarify_insufficient_evidence",
}
DEFAULT_TOP_K = 6
MAX_TOP_K = 12
MAX_TOOL_CALLS = 3
