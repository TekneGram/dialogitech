from __future__ import annotations

from pathlib import Path

from .models import QueryPipelineResult


class QueryOutputWriter:
    def write(self, result: QueryPipelineResult, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._render(result), encoding="utf-8")
        return path

    def _render(self, result: QueryPipelineResult) -> str:
        lines: list[str] = []
        lines.append("# Query Output")
        lines.append("")
        lines.append(f"Query: {result.request.query}")
        lines.append(f"Retrieval mode: {result.request.retrieval_mode}")
        lines.append(f"Rewrites: {', '.join(result.rewrites)}")
        lines.append(f"HyDE enabled: {'yes' if result.hyde_text is not None else 'no'}")
        if result.hyde_text is not None:
            lines.append(f"HyDE text: {result.hyde_text}")
        lines.append(f"Raw hits: {len(result.retrieved_chunks)}")
        lines.append(f"Fused hits: {len(result.fused_results)}")
        lines.append("")
        lines.append("## Ranked Chunks")
        lines.append("")

        for index, chunk in enumerate(result.fused_results, start=1):
            lines.append(f"### Rank {index}")
            lines.append(f"chunk_id: {chunk.chunk_id}")
            lines.append(f"paper_id: {chunk.paper_id}")
            lines.append(f"paper_title: {chunk.paper_title}")
            lines.append(f"authors: {', '.join(chunk.authors) if chunk.authors else '[missing]'}")
            lines.append(f"year: {chunk.year if chunk.year is not None else '[missing]'}")
            lines.append(f"section_title: {chunk.section_title}")
            lines.append(f"chunk_index: {chunk.chunk_index}")
            lines.append(f"classification_label: {chunk.classification_label}")
            lines.append(f"classification_source: {chunk.classification_source}")
            lines.append(f"rrf_score: {chunk.rrf_score:.6f}")
            lines.append(f"retrieval_keys: {', '.join(chunk.retrieval_keys)}")
            lines.append(
                "retrieval_kinds: "
                + ", ".join(chunk.retrieval_kinds)
            )
            lines.append(
                "contributing_ranks: "
                + ", ".join(
                    f"{key}={rank}" for key, rank in sorted(chunk.contributing_ranks.items())
                )
            )
            lines.append(
                f"relevance_score: {chunk.relevance_score if chunk.relevance_score is not None else '[none]'}"
            )
            lines.append(f"distance: {chunk.distance if chunk.distance is not None else '[none]'}")
            lines.append(f"fts_score: {chunk.fts_score if chunk.fts_score is not None else '[none]'}")
            lines.append(f"markdown_path: {chunk.markdown_path if chunk.markdown_path is not None else '[none]'}")
            lines.append(
                f"marker_json_path: {chunk.marker_json_path if chunk.marker_json_path is not None else '[none]'}"
            )
            lines.append(f"pdf_path: {chunk.pdf_path if chunk.pdf_path is not None else '[none]'}")
            lines.append("text:")
            lines.append("```text")
            lines.append(chunk.text)
            lines.append("```")
            lines.append("")

        lines.append("## Summaries")
        lines.append("")
        for summary in result.summaries:
            lines.append(f"### Summary Batch {summary.batch_index}")
            lines.append(f"chunk_ids: {', '.join(summary.chunk_ids)}")
            lines.append(f"citation_labels: {', '.join(summary.citation_labels)}")
            lines.append(summary.summary_text)
            lines.append("")

        if result.synthesized_summary is not None:
            lines.append("## Synthesized Summary")
            lines.append("")
            lines.append(result.synthesized_summary)
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"
