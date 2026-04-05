from __future__ import annotations

import argparse
from pathlib import Path

from dbinsert.lancedb_schema import DEFAULT_TABLE_NAME
from dbquery.gemma_client import GemmaClient

from .evidence_planner import GemmaEvidencePlanner
from .evidence_retriever import EvidenceRetriever
from .followup_pipeline import FollowupPipeline
from .grounded_writer import GemmaGroundedWriter
from .models import FollowupRequest
from .output_writer import FollowupOutputWriter
from .plan_validator import EvidencePlanValidator
from .utils import PriorSummaryExtractor

DEFAULT_GEMMA_MODEL = "unsloth/gemma-4-E4B-it-UD-MLX-4bit"
DEFAULT_GEMMA_PYTHON = "/Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a grounded single-turn follow-up query over LanceDB.")
    parser.add_argument("user_question", help="The follow-up question to answer.")
    parser.add_argument("--prior-summary-path", required=True, help="Path to a text file containing the prior summary.")
    parser.add_argument("--db-path", required=True, help="Path to the LanceDB database directory.")
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME, help="Name of the LanceDB table.")
    parser.add_argument("--output-path", help="Optional path to write the full follow-up trace as markdown.")
    parser.add_argument("--model-path", default=DEFAULT_GEMMA_MODEL, help="Gemma model path.")
    parser.add_argument(
        "--python-executable",
        default=DEFAULT_GEMMA_PYTHON,
        help="Python executable for the external Gemma MLX runtime.",
    )
    args = parser.parse_args()

    prior_summary_document = Path(args.prior_summary_path).read_text(encoding="utf-8")
    prior_summary = PriorSummaryExtractor().extract(prior_summary_document)
    gemma_client = GemmaClient(
        model_path=args.model_path,
        python_executable=args.python_executable,
    )
    pipeline = FollowupPipeline(
        planner=GemmaEvidencePlanner(gemma_client),
        validator=EvidencePlanValidator(),
        retriever=EvidenceRetriever(db_path=args.db_path, table_name=args.table_name),
        grounded_writer=GemmaGroundedWriter(gemma_client),
    )
    result = pipeline.run(
        FollowupRequest(
            prior_summary=prior_summary,
            user_question=args.user_question,
            db_path=args.db_path,
            table_name=args.table_name,
        )
    )
    output_path = None
    if args.output_path:
        output_path = FollowupOutputWriter().write(result, args.output_path)

    print(f"intent={result.plan.intent}")
    print(f"answer_mode={result.plan.answer_mode}")
    print(f"reason={result.plan.reason}")
    if output_path is not None:
        print(f"output_path={output_path}")
    for index, evidence_result in enumerate(result.evidence_results, start=1):
        print(f"-- evidence call {index} --")
        print(f"query={evidence_result.tool_call.query}")
        print(f"filters={evidence_result.tool_call.filters}")
        print(f"hits={len(evidence_result.fused_results)}")
    print("-- grounded answer --")
    print(result.grounded_answer.answer_text)


if __name__ == "__main__":
    main()
