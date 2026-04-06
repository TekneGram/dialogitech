from __future__ import annotations

import argparse

from dbquery.gemma_client import GemmaClient

from .branch_comparison_synthesizer import BranchComparisonSynthesizer
from .branch_synthesis_context_builder import BranchSynthesisContextBuilder
from .branch_synthesizer import BranchSynthesizer
from .final_report_writer import FinalReportWriter
from .run_session import DEFAULT_GEMMA_MODEL, DEFAULT_GEMMA_PYTHON
from .session_synthesizer import SessionSynthesizer
from .state_store import ResearchStateStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize branch findings and branch comparisons from a resquery session.")
    parser.add_argument("--session-path", required=True, help="Path to the persisted research session JSON.")
    parser.add_argument("--output-path", required=True, help="Path to write the final synthesis report.")
    parser.add_argument(
        "--branch-id",
        action="append",
        dest="branch_ids",
        help="Optional branch ID to include. Repeat to limit synthesis to selected branches.",
    )
    parser.add_argument("--model-path", default=DEFAULT_GEMMA_MODEL, help="Gemma model path.")
    parser.add_argument(
        "--python-executable",
        default=DEFAULT_GEMMA_PYTHON,
        help="Python executable for the external Gemma MLX runtime.",
    )
    args = parser.parse_args()

    gemma_client = GemmaClient(
        model_path=args.model_path,
        python_executable=args.python_executable,
    )
    synthesizer = SessionSynthesizer(
        state_store=ResearchStateStore(),
        branch_context_builder=BranchSynthesisContextBuilder(),
        branch_synthesizer=BranchSynthesizer(
            gemma_client=gemma_client,
            context_builder=BranchSynthesisContextBuilder(),
        ),
        branch_comparison_synthesizer=BranchComparisonSynthesizer(gemma_client),
    )
    result = synthesizer.synthesize(
        session_path=args.session_path,
        branch_ids=args.branch_ids,
    )
    output_path = FinalReportWriter().write(result, args.output_path)
    print(f"session_id={result.session_state.session_id}")
    print(f"branch_count={len(result.branch_summaries)}")
    print(f"comparison_count={len(result.branch_comparisons)}")
    print(f"output_path={output_path}")


if __name__ == "__main__":
    main()
