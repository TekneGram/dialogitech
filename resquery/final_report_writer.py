from __future__ import annotations

from pathlib import Path

from .models import SessionSynthesisResult


class FinalReportWriter:
    def write(self, result: SessionSynthesisResult, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._render(result), encoding="utf-8")
        return path

    def _render(self, result: SessionSynthesisResult) -> str:
        lines: list[str] = []
        lines.append("# Research Session Final Report")
        lines.append("")
        lines.append(f"Session ID: {result.session_state.session_id}")
        lines.append(f"Root query: {result.session_state.root_query}")
        lines.append(f"Branch count: {len(result.branch_summaries)}")
        lines.append("")
        lines.append("## Branch Summaries")
        lines.append("")
        for branch_summary in result.branch_summaries:
            lines.append(f"### {branch_summary.branch_id} | {branch_summary.branch_label}")
            lines.append("")
            lines.append(branch_summary.summary_text)
            lines.append("")
            lines.append("#### Provenance Context")
            lines.append("")
            lines.append("```text")
            lines.append(branch_summary.rendered_context)
            lines.append("```")
            lines.append("")

        lines.append("## Branch Comparisons")
        lines.append("")
        if not result.branch_comparisons:
            lines.append("[none]")
            lines.append("")
        else:
            for comparison in result.branch_comparisons:
                lines.append(f"### {comparison.left_branch_id} vs {comparison.right_branch_id}")
                lines.append("")
                lines.append(comparison.comparison_text)
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"
