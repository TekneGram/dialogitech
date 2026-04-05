from __future__ import annotations

import json
import select
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chunker.boilerplate_filter import BoilerplateFilter
from chunker.llm_section_classifier import ChunkClassificationLLM
from chunker.markdown_section_chunker import MarkdownSectionChunker
from chunker.metadata_extractor import MetadataExtractor
from chunker.section_classifier import ChunkClassificationEnricher, ClassifiedHeadingSplit

from .ingest_service import ChunkIngestionService
from .models import PaperMetadataRecord
from .pipeline_loader import load_classified_heading_splits


DEFAULT_GEMMA_MODEL_PATH = "unsloth/gemma-4-E4B-it-UD-MLX-4bit"
DEFAULT_GEMMA_PYTHON = "/Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python"
MARKER_PROGRESS_INTERVALS = (60, 180, 300)
MARKER_PROGRESS_REPEAT_SECONDS = 300


class PdfToLancePipeline:
    def __init__(
        self,
        ingestion_service: ChunkIngestionService,
        *,
        conversion_root: str | Path,
        min_words: int = 200,
        overlap_words: int = 50,
    ) -> None:
        self.ingestion_service = ingestion_service
        self.conversion_root = Path(conversion_root)
        self.min_words = min_words
        self.overlap_words = overlap_words

    def process_pdf(
        self,
        pdf_path: str | Path,
        *,
        model_path: str | None = None,
        python_executable: str | None = None,
        force_llm: bool = False,
        replace_existing: bool = False,
        create_indexes: bool = True,
        rerun_marker: bool = False,
        rerun_filtered_markdown: bool = False,
        rerun_classification: bool = False,
    ) -> dict[str, Any]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        paper_id = pdf_path.stem
        artifact_dir = self.conversion_root / paper_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        marker_json_path = artifact_dir / f"{paper_id}.json"
        filtered_markdown_path = artifact_dir / f"{paper_id}_filtered.md"
        classified_json_path = artifact_dir / f"{paper_id}_classified_chunks.json"
        marker_log_path = artifact_dir / f"{paper_id}_marker.log"
        classification_log_path = artifact_dir / f"{paper_id}_classification.log"

        resolved_model_path = model_path or DEFAULT_GEMMA_MODEL_PATH
        resolved_python_executable = python_executable or DEFAULT_GEMMA_PYTHON
        self._emit_stage(f"[{paper_id}] Pipeline started.")

        if rerun_marker or not marker_json_path.exists():
            self._emit_stage(
                f"[{paper_id}] Starting Marker conversion. Log: {marker_log_path}"
            )
            self._run_marker(
                pdf_path=pdf_path,
                output_dir=self.conversion_root,
                marker_log_path=marker_log_path,
            )
        else:
            self._emit_stage(f"[{paper_id}] Reusing existing Marker JSON: {marker_json_path}")

        if not marker_json_path.exists():
            raise RuntimeError(f"Marker did not produce the expected JSON output: {marker_json_path}")

        self._emit_stage(f"[{paper_id}] Loading Marker JSON.")
        metadata_extractor = MetadataExtractor()
        document = metadata_extractor.load_json(marker_json_path)
        extracted_metadata = metadata_extractor.extract_all(document)

        if rerun_filtered_markdown or not filtered_markdown_path.exists():
            self._emit_stage(f"[{paper_id}] Generating filtered markdown.")
            filtered_markdown = BoilerplateFilter().convert_json_to_markdown(
                document,
                metadata=extracted_metadata,
            )
            filtered_markdown_path.write_text(filtered_markdown, encoding="utf-8")
        else:
            self._emit_stage(f"[{paper_id}] Reusing existing filtered markdown: {filtered_markdown_path}")
            filtered_markdown = filtered_markdown_path.read_text(encoding="utf-8")

        if self._can_reuse_classified_json(
            classified_json_path=classified_json_path,
            filtered_markdown_path=filtered_markdown_path,
            model_path=resolved_model_path,
            python_executable=resolved_python_executable,
            force_llm=force_llm,
            rerun_classification=rerun_classification,
        ):
            self._emit_stage(f"[{paper_id}] Reusing existing classified chunks: {classified_json_path}")
            classified_splits, _ = load_classified_heading_splits(classified_json_path)
        else:
            self._emit_stage(f"[{paper_id}] Chunking filtered markdown.")
            heading_splits = MarkdownSectionChunker(
                min_words=self.min_words,
                overlap_words=self.overlap_words,
            ).process(filtered_markdown)

            self._emit_stage(f"[{paper_id}] Classifying chunks.")
            with classification_log_path.open("w", encoding="utf-8") as classification_log:
                llm_classifier = ChunkClassificationLLM(
                    filtered_markdown=filtered_markdown,
                    heading_splits=heading_splits,
                    model_path=resolved_model_path,
                    python_executable=resolved_python_executable,
                    event_logger=lambda message: self._emit_classification_event(
                        paper_id=paper_id,
                        message=message,
                        log_file=classification_log,
                    ),
                )

                classified_splits = ChunkClassificationEnricher(
                    llm_classifier=llm_classifier,
                    force_llm=force_llm,
                ).enrich_heading_splits(heading_splits)
                self._ensure_all_chunks_resolved(classified_splits, paper_id=paper_id)
                self._write_classified_json(
                    classified_json_path=classified_json_path,
                    classified_splits=classified_splits,
                    source_markdown=filtered_markdown_path,
                    model_path=resolved_model_path,
                    python_executable=resolved_python_executable,
                    force_llm=force_llm,
                )

        self._ensure_all_chunks_resolved(classified_splits, paper_id=paper_id)

        paper_metadata = self._build_paper_metadata(
            paper_id=paper_id,
            extracted_metadata=extracted_metadata,
            pdf_path=pdf_path,
            marker_json_path=marker_json_path,
            markdown_path=filtered_markdown_path,
        )

        self._emit_stage(f"[{paper_id}] Generating embeddings and inserting into LanceDB.")
        inserted_count = self.ingestion_service.ingest_paper(
            paper_metadata,
            classified_splits,
            create_indexes=create_indexes,
            replace_existing=replace_existing,
        )
        self._emit_stage(f"[{paper_id}] Ingestion complete. Inserted {inserted_count} chunks.")

        return {
            "paper_id": paper_id,
            "paper_title": paper_metadata.paper_title,
            "marker_json_path": str(marker_json_path),
            "filtered_markdown_path": str(filtered_markdown_path),
            "classified_json_path": str(classified_json_path),
            "marker_log_path": str(marker_log_path),
            "classification_log_path": str(classification_log_path),
            "inserted_chunks": inserted_count,
        }

    def _run_marker(self, *, pdf_path: Path, output_dir: Path, marker_log_path: Path) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        marker_script = repo_root / "marker" / "convert_single.py"
        command = [
            sys.executable,
            str(marker_script),
            str(pdf_path),
            "--output_dir",
            str(output_dir),
            "--output_format",
            "json",
        ]
        marker_log_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.monotonic()
        report_schedule = list(MARKER_PROGRESS_INTERVALS)
        next_repeat_report = MARKER_PROGRESS_REPEAT_SECONDS

        with marker_log_path.open("w", encoding="utf-8") as marker_log:
            process = subprocess.Popen(
                command,
                stdout=marker_log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                cwd=repo_root,
            )
            try:
                while True:
                    returncode = process.poll()
                    if returncode is not None:
                        break

                    elapsed_seconds = int(time.monotonic() - start_time)
                    if report_schedule and elapsed_seconds >= report_schedule[0]:
                        self._emit_marker_progress(
                            pdf_path=pdf_path,
                            elapsed_seconds=elapsed_seconds,
                            marker_log_path=marker_log_path,
                        )
                        report_schedule.pop(0)
                    elif elapsed_seconds >= next_repeat_report:
                        self._emit_marker_progress(
                            pdf_path=pdf_path,
                            elapsed_seconds=elapsed_seconds,
                            marker_log_path=marker_log_path,
                        )
                        next_repeat_report += MARKER_PROGRESS_REPEAT_SECONDS

                    if self._stop_requested():
                        self._terminate_process(process)
                        raise RuntimeError(
                            f"Marker conversion stopped by user for {pdf_path.name}. "
                            f"Partial log: {marker_log_path}"
                        )

                    time.sleep(1)
            except KeyboardInterrupt as exc:
                self._terminate_process(process)
                raise RuntimeError(
                    f"Marker conversion interrupted for {pdf_path.name}. Partial log: {marker_log_path}"
                ) from exc

        if process.returncode != 0:
            raise RuntimeError(
                f"Marker conversion failed with exit code {process.returncode}. "
                f"See log: {marker_log_path}"
            )

        total_seconds = int(time.monotonic() - start_time)
        self._emit_stage(
            f"[{pdf_path.stem}] Marker conversion finished in {self._format_elapsed(total_seconds)}."
        )

    def _build_paper_metadata(
        self,
        *,
        paper_id: str,
        extracted_metadata: dict[str, Any],
        pdf_path: Path,
        marker_json_path: Path,
        markdown_path: Path,
    ) -> PaperMetadataRecord:
        journal_payload = extracted_metadata.get("journal") or {}
        title = extracted_metadata.get("title") or paper_id
        authors = list(extracted_metadata.get("authors") or [])
        references = list(extracted_metadata.get("references") or [])

        year = None
        raw_year = journal_payload.get("year") if isinstance(journal_payload, dict) else None
        if isinstance(raw_year, str) and raw_year.isdigit():
            year = int(raw_year)

        return PaperMetadataRecord(
            paper_id=paper_id,
            paper_title=title,
            authors=authors,
            journal=journal_payload.get("name") if isinstance(journal_payload, dict) else None,
            volume=journal_payload.get("volume") if isinstance(journal_payload, dict) else None,
            issue=journal_payload.get("issue") if isinstance(journal_payload, dict) else None,
            year=year,
            doi=journal_payload.get("doi") if isinstance(journal_payload, dict) else None,
            issn=journal_payload.get("issn") if isinstance(journal_payload, dict) else None,
            references=references,
            markdown_path=str(markdown_path),
            marker_json_path=str(marker_json_path),
            pdf_path=str(pdf_path),
        )

    def _write_classified_json(
        self,
        *,
        classified_json_path: Path,
        classified_splits: list[ClassifiedHeadingSplit],
        source_markdown: Path,
        model_path: str | None,
        python_executable: str | None,
        force_llm: bool,
    ) -> None:
        payload = {
            "source_markdown": str(source_markdown),
            "model": model_path,
            "python_executable": python_executable,
            "force_llm": force_llm,
            "total_heading_splits": len(classified_splits),
            "total_chunks": sum(len(split.chunks) for split in classified_splits),
            "headings": [self._serialize_heading_split(split) for split in classified_splits],
        }
        classified_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _serialize_heading_split(self, split: ClassifiedHeadingSplit) -> dict[str, Any]:
        serialized = asdict(split)
        for chunk in serialized["chunks"]:
            classification = chunk["classification"]
            classification["needs_llm"] = bool(classification["needs_llm"])
            classification["used_context"] = bool(classification["used_context"])
        return serialized

    def _can_reuse_classified_json(
        self,
        *,
        classified_json_path: Path,
        filtered_markdown_path: Path,
        model_path: str | None,
        python_executable: str | None,
        force_llm: bool,
        rerun_classification: bool,
    ) -> bool:
        if rerun_classification or not classified_json_path.exists():
            return False

        payload = json.loads(classified_json_path.read_text(encoding="utf-8"))
        stored_markdown = payload.get("source_markdown")
        stored_model = payload.get("model")
        stored_python = payload.get("python_executable")
        stored_force_llm = bool(payload.get("force_llm", False))

        return (
            stored_markdown == str(filtered_markdown_path)
            and stored_model == model_path
            and stored_python == python_executable
            and stored_force_llm == force_llm
            and not self._payload_has_unresolved_chunks(payload)
        )

    def _emit_stage(self, message: str) -> None:
        print(message, flush=True)

    def _emit_classification_event(
        self,
        *,
        paper_id: str,
        message: str,
        log_file: Any,
    ) -> None:
        line = f"[{paper_id}] {message}"
        self._emit_stage(line)
        print(line, file=log_file, flush=True)

    def _emit_marker_progress(
        self,
        *,
        pdf_path: Path,
        elapsed_seconds: int,
        marker_log_path: Path,
    ) -> None:
        self._emit_stage(
            f"[{pdf_path.stem}] Marker still running after {self._format_elapsed(elapsed_seconds)}. "
            f"Type 'stop' then Enter to cancel, or press Ctrl-C. Log: {marker_log_path}"
        )

    def _stop_requested(self) -> bool:
        if not sys.stdin.isatty():
            return False
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return False

        response = sys.stdin.readline().strip().lower()
        return response in {"stop", "quit", "q"}

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _format_elapsed(self, elapsed_seconds: int) -> str:
        minutes, seconds = divmod(elapsed_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def _ensure_all_chunks_resolved(
        self,
        classified_splits: list[ClassifiedHeadingSplit],
        *,
        paper_id: str,
    ) -> None:
        unresolved: list[str] = []
        for split in classified_splits:
            for chunk in split.chunks:
                if chunk.classification.needs_llm or chunk.classification.label is None:
                    unresolved.append(
                        f"{split.title} [chunk {chunk.chunk_index}] "
                        f"source={chunk.classification.source} "
                        f"label={chunk.classification.label} "
                        f"reason={chunk.classification.reason}"
                    )
        if unresolved:
            preview = "\n".join(unresolved[:10])
            raise RuntimeError(
                f"[{paper_id}] Classification left unresolved chunks. "
                f"Refusing to write incomplete data to LanceDB.\n{preview}"
            )

    def _payload_has_unresolved_chunks(self, payload: dict[str, Any]) -> bool:
        headings = payload.get("headings")
        if not isinstance(headings, list):
            return True
        for heading in headings:
            chunks = heading.get("chunks", [])
            if not isinstance(chunks, list):
                return True
            for chunk in chunks:
                classification = chunk.get("classification", {})
                if classification.get("needs_llm") or classification.get("label") is None:
                    return True
        return False
