from __future__ import annotations

import json
import select
import subprocess
import signal
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chunker.boilerplate_filter import BoilerplateFilter
from chunker.llm_section_classifier import SectionClassificationLLM
from chunker.llm_paper_type_classifier import PaperTypeClassificationLLM
from chunker.markdown_section_chunker import MarkdownSectionChunker
from chunker.llm_rhetorical_move_classifier import RhetoricalMoveClassificationLLM
from chunker.llm_rhetorical_move_classifier_helpers import validate_rhetorical_move_result
from chunker.llm_paper_type_classifier_helpers.paper_type_models import (
    PAPER_TYPES,
    PaperTypeClassification,
)
from chunker.llm_section_type_classifier_helpers.section_taxonomy import allowed_sections
from chunker.section_classifier import ClassifiedHeadingSplit, ClassifiedSectionChunk

from chunker.llm_metadata_extractor import LLMMetadataExtractor
from chunker.llm_metadata_extractor_helpers.metadata_models import (MetadataExtractionResult)

from .ingest_service import ChunkIngestionService
from .metadata_checker import (
    InteractiveMetadataPrompter,
    MetadataCompletenessChecker,
    MetadataPromptContext,
)
from .models import PaperMetadataRecord
from .pipeline_loader import load_classified_heading_splits


DEFAULT_GEMMA_MODEL_PATH = "unsloth/gemma-4-E4B-it-UD-MLX-4bit"
DEFAULT_GEMMA_PYTHON = str(Path.home() / ".unsloth" / "unsloth_gemma4_mlx" / "bin" / "python")
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
        self.metadata_checker = MetadataCompletenessChecker()
        self.metadata_prompter = InteractiveMetadataPrompter()

    def process_pdf(
        self,
        pdf_path: str | Path,
        *,
        model_path: str | None = None,
        python_executable: str | None = None,
        llm_timeout_seconds: float = 180.0,
        replace_existing: bool = False,
        create_indexes: bool = True,
        rerun_marker: bool = False,
        rerun_filtered_markdown: bool = False,
        rerun_classification: bool = False,
        rerun_rhetorical_moves: bool = False,
        rerun_paper_type: bool = False,
    ) -> dict[str, Any]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Prepare artifact paths
        paper_id = pdf_path.stem
        artifact_dir = self.conversion_root / paper_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        marker_json_path = artifact_dir / f"{paper_id}.json"
        filtered_markdown_path = artifact_dir / f"{paper_id}_filtered.md"
        classified_json_path = artifact_dir / f"{paper_id}_classified_chunks.json"
        marker_log_path = artifact_dir / f"{paper_id}_marker.log"
        classification_log_path = artifact_dir / f"{paper_id}_classification.log"
        metadata_trace_log_path = artifact_dir / f"{paper_id}_metadata_trace.log"

        # Prepare model paths and print to the screen that the pipeline has started
        resolved_model_path = model_path or DEFAULT_GEMMA_MODEL_PATH
        resolved_python_executable = python_executable or DEFAULT_GEMMA_PYTHON
        self._emit_stage(f"[{paper_id}] Pipeline started.")

        # When artifacts are saved, there is no need to re-run marker; instead reuse the existing json artifacts from a previous marker run.
        if rerun_marker or not marker_json_path.exists():
            self._emit_stage(
                f"[{paper_id}] Starting Marker conversion. Log: {marker_log_path}"
            )
            # Run marker because current artifacts do not exist yet.
            self._run_marker(
                pdf_path=pdf_path,
                output_dir=self.conversion_root,
                marker_log_path=marker_log_path,
            )
        else:
            self._emit_stage(f"[{paper_id}] Reusing existing Marker JSON: {marker_json_path}")

        if not marker_json_path.exists():
            raise RuntimeError(f"Marker did not produce the expected JSON output: {marker_json_path}")

        # Start the analysis and extraction of data from the Marker artifacts
        self._emit_stage(f"[{paper_id}] Loading Marker JSON.")
        metadata_log_path = artifact_dir / f"{paper_id}_metadata.log"
        with metadata_log_path.open("a", encoding="utf-8") as metadata_log:
            metadata_extractor = LLMMetadataExtractor(
                model_path=resolved_model_path,
                python_executable=resolved_python_executable,
                request_timeout_seconds=llm_timeout_seconds,
                event_logger=lambda message: self._emit_metadata_event(
                    paper_id=paper_id,
                    message=message,
                    log_file=metadata_log,
                ),
            )
            try:
                document = metadata_extractor.load_marker_json(marker_json_path)
                extracted_result = metadata_extractor.extract_metadata(document)
                extracted_metadata = self._metadata_result_to_dict(extracted_result)
            finally:
                metadata_extractor.close()

        # Check all the metadata is present - if not can still insert into LanceDB but issue a warning
        # and a trace log.
        extracted_metadata = self._ensure_required_metadata(
            paper_id=paper_id,
            pdf_path=pdf_path,
            marker_json_path=marker_json_path,
            extracted_metadata=extracted_metadata,
            metadata_trace_log_path=metadata_trace_log_path,
        )

        # Create the raw text markdown file from the pdf text data
        # Recreate it if rerun_filtered_markdown is set to True (e.g., we are re-running the data extraction from just the json files after an earlier marker pdf extraction)
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

        # Determine whether the existing classified-chunks.json in conversion_results is still valid
        # Returns True or False
        reuse_classification = self._can_reuse_classified_json(
            classified_json_path=classified_json_path,
            filtered_markdown_path=filtered_markdown_path,
            model_path=resolved_model_path,
            python_executable=resolved_python_executable,
            rerun_classification=rerun_classification,
            rerun_paper_type=rerun_paper_type,
        )

        # If a _classified_chunks.json exists in conversion_results, then reuse it.
        # If not, then create it by classifying the type of paper.
        if reuse_classification:
            self._emit_stage(f"[{paper_id}] Reusing existing classified chunks: {classified_json_path}")
            classified_splits, _ = load_classified_heading_splits(classified_json_path)
            paper_type = self._load_paper_type(classified_json_path)
        else:
            self._emit_stage(f"[{paper_id}] Classifying paper type.")
            with classification_log_path.open("w", encoding="utf-8") as classification_log:
                # Instantiate the class using Python's "with" statement, so no need to use try: ... finally: ...
                with PaperTypeClassificationLLM(
                    model_path=resolved_model_path,
                    python_executable=resolved_python_executable,
                    request_timeout_seconds=llm_timeout_seconds,
                    event_logger=lambda message: self._emit_classification_event(
                        paper_id=paper_id,
                        message=f"paper type: {message}",
                        log_file=classification_log
                    ),
                ) as paper_type_classifier:
                    paper_type = paper_type_classifier.classify(
                        filtered_markdown=filtered_markdown,
                        metadata=extracted_metadata
                    )

            # After paper type has been classified, emit the current status to the user.
            self._emit_paper_type_status(paper_type, paper_id=paper_id)

            # Begin the chunking process
            # This chunks the text into sizes of min_words
            # Each chunk will then be classified.
            self._emit_stage(f"[{paper_id}] Chunking filtered markdown.")
            heading_splits = MarkdownSectionChunker(
                min_words=self.min_words,
                overlap_words=self.overlap_words,
            ).process(filtered_markdown)

            # Begin the classification of the chunks into sections.
            self._emit_stage(f"[{paper_id}] Classifying chunks.")
            with classification_log_path.open("a", encoding="utf-8") as classification_log:
                section_classifier = SectionClassificationLLM(
                    filtered_markdown=filtered_markdown,
                    heading_splits=heading_splits,
                    model_path=resolved_model_path,
                    python_executable=resolved_python_executable,
                    request_timeout_seconds=llm_timeout_seconds,
                    event_logger=lambda message: self._emit_classification_event(
                        paper_id=paper_id,
                        message=message,
                        log_file=classification_log,
                    ),
                )

                try:
                    previous_resolved_label = None
                    classified_splits: list[ClassifiedHeadingSplit] = []
                    for heading_split in heading_splits:
                        classified_chunks: list[ClassifiedSectionChunk] = []
                        for section_chunk in heading_split.chunks:
                            classification = section_classifier.classify_section_chunk(
                                section_chunk=section_chunk,
                                heading_split=heading_split,
                                paper_type=paper_type.label,
                                previous_label=previous_resolved_label,
                            )
                            classified_chunks.append(
                                ClassifiedSectionChunk(
                                    title=section_chunk.title,
                                    heading_level=section_chunk.heading_level,
                                    chunk_index=section_chunk.chunk_index,
                                    text=section_chunk.text,
                                    word_count=section_chunk.word_count,
                                    classification=classification,
                                )
                            )
                            if classification.label is not None:
                                previous_resolved_label = classification.label
                        classified_splits.append(
                            ClassifiedHeadingSplit(
                                title=heading_split.title,
                                heading_level=heading_split.heading_level,
                                raw_heading=heading_split.raw_heading,
                                content=heading_split.content,
                                chunks=classified_chunks,
                            )
                        )
                finally:
                    section_classifier.close()

        # Continue to classify the chunks into rhetorical moves
        self._emit_paper_type_status(paper_type, paper_id=paper_id)
        self._ensure_all_chunks_resolved(classified_splits, paper_id=paper_id, paper_type=paper_type.label)
        if rerun_rhetorical_moves or not self._has_rhetorical_moves(classified_splits):
            self._emit_stage(f"[{paper_id}] Classifying rhetorical moves.")
            with classification_log_path.open("a", encoding="utf-8") as classification_log:
                rhetorical_classifier = RhetoricalMoveClassificationLLM(
                    filtered_markdown=filtered_markdown,
                    heading_splits=classified_splits,
                    model_path=resolved_model_path,
                    python_executable=resolved_python_executable,
                    request_timeout_seconds=llm_timeout_seconds,
                    event_logger=lambda message: self._emit_classification_event(
                        paper_id=paper_id,
                        message=f"rhetorical moves: {message}",
                        log_file=classification_log,
                    ),
                )
                try:
                    for split in classified_splits:
                        for chunk in split.chunks:
                            if chunk.classification.label is None:
                                chunk.classification.label = "unclassified"
                                self._emit_classification_event(
                                    paper_id=paper_id,
                                    message=(
                                        "rhetorical moves: section classification label was missing; "
                                        f"assigned 'unclassified' for {split.title} "
                                        f"[chunk {chunk.chunk_index}] before rhetorical classification."
                                    ),
                                    log_file=classification_log,
                                )
                            chunk.rhetorical_move_result = rhetorical_classifier.classify(
                                chunk=chunk,
                                heading_split=split,
                                section_label=chunk.classification.label,
                            )
                finally:
                    rhetorical_classifier.close()
            self._ensure_all_rhetorical_moves_resolved(classified_splits, paper_id=paper_id)
            self._write_classified_json(
                classified_json_path=classified_json_path,
                classified_splits=classified_splits,
                source_markdown=filtered_markdown_path,
                model_path=resolved_model_path,
                python_executable=resolved_python_executable,
                paper_type=paper_type,
            )
        else:
            self._ensure_all_rhetorical_moves_resolved(classified_splits, paper_id=paper_id)

        paper_metadata = self._build_paper_metadata(
            paper_id=paper_id,
            extracted_metadata=extracted_metadata,
            pdf_path=pdf_path,
            marker_json_path=marker_json_path,
            markdown_path=filtered_markdown_path,
            paper_type=paper_type,
        )

        # Insert data into LanceDB here
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
            "paper_type": paper_type.label,
            "marker_json_path": str(marker_json_path),
            "filtered_markdown_path": str(filtered_markdown_path),
            "classified_json_path": str(classified_json_path),
            "marker_log_path": str(marker_log_path),
            "classification_log_path": str(classification_log_path),
            "metadata_log_path": str(metadata_log_path),
            "metadata_trace_log_path": str(metadata_trace_log_path),
            "inserted_chunks": inserted_count,
        }

    def _ensure_required_metadata(
        self,
        *,
        paper_id: str,
        pdf_path: Path,
        marker_json_path: Path,
        extracted_metadata: dict[str, Any],
        metadata_trace_log_path: Path,
    ) -> dict[str, Any]:
        issues = self.metadata_checker.find_missing_fields(extracted_metadata)
        if not issues:
            return extracted_metadata

        self._record_metadata_issues(
            paper_id=paper_id,
            issues=issues,
            metadata_trace_log_path=metadata_trace_log_path,
            stage="before manual completion",
        )

        self._emit_stage(
            f"[{paper_id}] Missing required metadata detected. Prompting for manual entry."
        )
        completed_metadata = self.metadata_prompter.complete_metadata(
            extracted_metadata,
            context=MetadataPromptContext(
                paper_id=paper_id,
                pdf_path=pdf_path,
                marker_json_path=marker_json_path,
            ),
            issues=issues,
        )

        remaining_issues = self.metadata_checker.find_missing_fields(completed_metadata)
        if remaining_issues:
            missing = ", ".join(issue.field_name for issue in remaining_issues)
            message = (
                f"[{paper_id}] Metadata remains incomplete after prompting: {missing}. "
                "Continuing with available metadata; see the metadata trace log."
            )
            self._emit_stage(f"WARNING: {message}")
            self._record_metadata_issues(
                paper_id=paper_id,
                issues=remaining_issues,
                metadata_trace_log_path=metadata_trace_log_path,
                stage="after manual completion",
            )

        return completed_metadata

    def _record_metadata_issues(
        self,
        *,
        paper_id: str,
        issues: list[Any],
        metadata_trace_log_path: Path,
        stage: str,
    ) -> None:
        metadata_trace_log_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_trace_log_path.open("a", encoding="utf-8") as trace_log:
            trace_log.write(f"[{paper_id}] Missing metadata {stage}:\n")
            for issue in issues:
                trace_log.write(f"- {issue.field_name} ({issue.prompt_label})\n")

    def _metadata_result_to_dict(self, result: MetadataExtractionResult) -> dict[str, Any]:
        return {
            "title": result.title.value,
            "journal": result.journal.value,
            "authors": list(result.authors.value or []),
            "references": list(result.references),
        }

    def _emit_metadata_event(self, *, paper_id: str, message: str, log_file: Any) -> None:
        log_file.write(f"[{paper_id}] {message}\n")
        log_file.flush()

    # _run_marker calls a subProcess to run the conversion of the pdf to 
    def _run_marker(self, *, pdf_path: Path, output_dir: Path, marker_log_path: Path) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        marker_script = repo_root / "marker" / "convert_single.py"

        base_command =[
            sys.executable,
            str(marker_script),
            str(pdf_path),
            "--output_dir",
            str(output_dir),
            "--output_format",
            "json",
        ]

        marker_log_path.parent.mkdir(parents=True, exist_ok=True)

        def run_once(command: list[str], log_mode: str) -> tuple[bool, str | None]:
            with marker_log_path.open(log_mode, encoding="utf-8") as marker_log:
                process = subprocess.Popen(
                    command,
                    stdout=marker_log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    cwd=repo_root,
                    start_new_session=True,
                )

                log_position = marker_log_path.stat().st_size
                start_time = time.monotonic()

                try:
                    while True:
                        returncode = process.poll()
                        if returncode is not None:
                            return returncode == 0, None

                        # Read newly written log output.
                        try:
                            with marker_log_path.open("r", encoding="utf-8") as log:
                                log.seek(log_position)
                                new_output = log.read()
                                log_position = log.tell()
                        except FileNotFoundError:
                            new_output = ""

                        if "Inference error: Request timed out." in new_output:
                            self._emit_stage(
                                f"[{pdf_path.stem}] VLM timeout detected. "
                                "Stopping Marker and retrying without OCR."
                            )

                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()

                            return False, "VLM timeout"

                        if self._stop_requested():
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait(timeout=10)
                            raise RuntimeError(
                                f"Marker conversion stopped by user for {pdf_path.name}."
                            )

                        elapsed_seconds = int(time.monotonic() - start_time)
                        self._emit_marker_progress(
                            pdf_path=pdf_path,
                            elapsed_seconds=elapsed_seconds,
                            marker_log_path=marker_log_path
                        )

                        time.sleep(1)

                except KeyboardInterrupt as exc:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                    raise RuntimeError(
                        f"Marker conversion interrupted for {pdf_path.name}."
                    ) from exc

        # First attempt: normal Marker processing with VLM/OCR enabled
        succeeded, failure_reason = run_once(base_command, "w")

        if succeeded:
            return

        if failure_reason == "VLM timeout":
            # Fallback: use only the PDF text layer.
            fallback_command = [
                *base_command,
                "--disable_ocr"
            ]

            succeeded, _ = run_once(fallback_command, "a")

            if succeeded:
                self._emit_stage(
                    f"[{pdf_path.stem}] Marker fallback completed without OCR."
                )
                return

        raise RuntimeError(
            f"Marker conversion failed. See log: {marker_log_path}."
        )
                            

        # command = [
        #     sys.executable,
        #     str(marker_script),
        #     str(pdf_path),
        #     "--output_dir",
        #     str(output_dir),
        #     "--output_format",
        #     "json",
        # ]
        # marker_log_path.parent.mkdir(parents=True, exist_ok=True)
        # start_time = time.monotonic()
        # report_schedule = list(MARKER_PROGRESS_INTERVALS)
        # next_repeat_report = MARKER_PROGRESS_REPEAT_SECONDS

        # with marker_log_path.open("w", encoding="utf-8") as marker_log:
        #     process = subprocess.Popen(
        #         command,
        #         stdout=marker_log,
        #         stderr=subprocess.STDOUT,
        #         stdin=subprocess.DEVNULL,
        #         text=True,
        #         cwd=repo_root,
        #     )
        #     try:
        #         while True:
        #             returncode = process.poll()
        #             if returncode is not None:
        #                 break

        #             elapsed_seconds = int(time.monotonic() - start_time)
        #             if report_schedule and elapsed_seconds >= report_schedule[0]:
        #                 self._emit_marker_progress(
        #                     pdf_path=pdf_path,
        #                     elapsed_seconds=elapsed_seconds,
        #                     marker_log_path=marker_log_path,
        #                 )
        #                 report_schedule.pop(0)
        #             elif elapsed_seconds >= next_repeat_report:
        #                 self._emit_marker_progress(
        #                     pdf_path=pdf_path,
        #                     elapsed_seconds=elapsed_seconds,
        #                     marker_log_path=marker_log_path,
        #                 )
        #                 next_repeat_report += MARKER_PROGRESS_REPEAT_SECONDS

        #             if self._stop_requested():
        #                 self._terminate_process(process)
        #                 raise RuntimeError(
        #                     f"Marker conversion stopped by user for {pdf_path.name}. "
        #                     f"Partial log: {marker_log_path}"
        #                 )

        #             time.sleep(1)
        #     except KeyboardInterrupt as exc:
        #         self._terminate_process(process)
        #         raise RuntimeError(
        #             f"Marker conversion interrupted for {pdf_path.name}. Partial log: {marker_log_path}"
        #         ) from exc

        # if process.returncode != 0:
        #     raise RuntimeError(
        #         f"Marker conversion failed with exit code {process.returncode}. "
        #         f"See log: {marker_log_path}"
        #     )

        # total_seconds = int(time.monotonic() - start_time)
        # self._emit_stage(
        #     f"[{pdf_path.stem}] Marker conversion finished in {self._format_elapsed(total_seconds)}."
        # )

    def _build_paper_metadata(
        self,
        *,
        paper_id: str,
        extracted_metadata: dict[str, Any],
        pdf_path: Path,
        marker_json_path: Path,
        markdown_path: Path,
        paper_type: PaperTypeClassification,
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
            paper_type=paper_type.label,
            paper_type_source=paper_type.source,
            paper_type_confidence=paper_type.confidence,
            paper_type_used_context=paper_type.used_context,
            paper_type_reason=paper_type.reason,
        )

    def _write_classified_json(
        self,
        *,
        classified_json_path: Path,
        classified_splits: list[ClassifiedHeadingSplit],
        source_markdown: Path,
        model_path: str | None,
        python_executable: str | None,
        paper_type: PaperTypeClassification,
    ) -> None:
        payload = {
            "source_markdown": str(source_markdown),
            "model": model_path,
            "python_executable": python_executable,
            "paper_type": asdict(paper_type),
            "total_heading_splits": len(classified_splits),
            "total_chunks": sum(len(split.chunks) for split in classified_splits),
            "headings": [self._serialize_heading_split(split) for split in classified_splits],
        }
        classified_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _serialize_heading_split(self, split: ClassifiedHeadingSplit) -> dict[str, Any]:
        serialized = asdict(split)
        for chunk in serialized["chunks"]:
            classification = chunk["classification"]
            classification["used_context"] = bool(classification["used_context"])
        return serialized

    def _can_reuse_classified_json(
        self,
        *,
        classified_json_path: Path,
        filtered_markdown_path: Path,
        model_path: str | None,
        python_executable: str | None,
        rerun_classification: bool,
        rerun_paper_type: bool,
    ) -> bool:
        if rerun_classification or rerun_paper_type or not classified_json_path.exists():
            return False

        payload = json.loads(classified_json_path.read_text(encoding="utf-8"))
        stored_markdown = payload.get("source_markdown")
        stored_model = payload.get("model")
        stored_python = payload.get("python_executable")
        return (
            stored_markdown == str(filtered_markdown_path)
            and stored_model == model_path
            and stored_python == python_executable
            and self._payload_has_resolved_paper_type(payload)
            and not self._payload_has_unresolved_or_non_llm_chunks(payload)
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
        paper_type: str,
    ) -> None:
        unresolved: list[str] = []
        for split in classified_splits:
            for chunk in split.chunks:
                if chunk.classification.label is None:
                    unresolved.append(
                        f"{split.title} [chunk {chunk.chunk_index}] "
                        f"source={chunk.classification.source} "
                        f"label={chunk.classification.label} "
                        f"reason={chunk.classification.reason}"
                        )
                elif chunk.classification.label == "unclassified":
                    continue
                elif chunk.classification.label not in allowed_sections(paper_type):
                    unresolved.append(f"{split.title} [chunk {chunk.chunk_index}] label={chunk.classification.label} is invalid for paper type {paper_type}")
        if unresolved:
            preview = "\n".join(unresolved[:10])
            raise RuntimeError(
                f"[{paper_id}] Classification left unresolved chunks. "
                f"Refusing to write incomplete data to LanceDB.\n{preview}"
            )

    def _payload_has_unresolved_or_non_llm_chunks(self, payload: dict[str, Any]) -> bool:
        headings = payload.get("headings")
        if not isinstance(headings, list):
            return True
        for heading in headings:
            chunks = heading.get("chunks", [])
            if not isinstance(chunks, list):
                return True
            for chunk in chunks:
                classification = chunk.get("classification", {})
                if (
                    classification.get("label") is None
                    or classification.get("source") not in {"llm", "llm_fallback"}
                ):
                    return True
        return False

    def _has_rhetorical_moves(self, classified_splits: list[ClassifiedHeadingSplit]) -> bool:
        return all(
            chunk.rhetorical_move_result is not None
            for split in classified_splits
            for chunk in split.chunks
        )

    def _ensure_all_rhetorical_moves_resolved(
        self,
        classified_splits: list[ClassifiedHeadingSplit],
        *,
        paper_id: str,
    ) -> None:
        missing = [
            f"{split.title} [chunk {chunk.chunk_index}]"
            for split in classified_splits
            for chunk in split.chunks
            if chunk.rhetorical_move_result is None
        ]
        if missing:
            raise RuntimeError(
                f"[{paper_id}] Rhetorical move classification left unresolved chunks. "
                f"Refusing to write incomplete data to LanceDB.\n" + "\n".join(missing[:10])
            )
        for split in classified_splits:
            for chunk in split.chunks:
                assert chunk.rhetorical_move_result is not None
                assert chunk.classification.label is not None
                validate_rhetorical_move_result(
                    chunk.rhetorical_move_result,
                    section_label=chunk.classification.label,
                )

    def _emit_paper_type_status(self, paper_type: PaperTypeClassification, *, paper_id: str) -> None:
        if paper_type.label not in PAPER_TYPES:
            self._emit_stage(
                f"[{paper_id}] Paper type emitted by LLM was determined not to be valid, so continuing with "
                f"other_or_unclear as the classification: {paper_type.reason}"
            )
        else:
            self._emit_stage(
                f"[{paper_id}] Paper type emitted by LLM was determined to be valid: "
                f"{paper_type.reason}"
            )

    def _load_paper_type(self, classified_json_path: Path) -> PaperTypeClassification:
        payload = json.loads(classified_json_path.read_text(encoding="utf-8"))
        value = payload.get("paper_type")
        if not isinstance(value, dict):
            raise RuntimeError("Classified artifact is missing paper_type.")
        return PaperTypeClassification(
            label=value.get("label"), source=value["source"], reason=value["reason"],
            confidence=value.get("confidence"), used_context=bool(value.get("used_context", False)),
        )

    def _payload_has_resolved_paper_type(self, payload: dict[str, Any]) -> bool:
        paper_type = payload.get("paper_type")
        return (
            isinstance(paper_type, dict)
            and paper_type.get("label") in PAPER_TYPES
        )
