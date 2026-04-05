from __future__ import annotations

import hashlib

from chunker.section_classifier import ClassifiedHeadingSplit

from .models import ChunkRecord, PaperMetadataRecord


class PaperChunkSerializer:
    def serialize_paper(
        self,
        paper_metadata: PaperMetadataRecord,
        classified_splits: list[ClassifiedHeadingSplit],
    ) -> list[ChunkRecord]:
        chunk_records: list[ChunkRecord] = []

        for split in classified_splits:
            for chunk in split.chunks:
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=self.make_chunk_id(
                            paper_id=paper_metadata.paper_id,
                            section_title=chunk.title,
                            chunk_index=chunk.chunk_index,
                        ),
                        paper_id=paper_metadata.paper_id,
                        paper_title=paper_metadata.paper_title,
                        authors=list(paper_metadata.authors),
                        journal=paper_metadata.journal,
                        volume=paper_metadata.volume,
                        issue=paper_metadata.issue,
                        year=paper_metadata.year,
                        doi=paper_metadata.doi,
                        issn=paper_metadata.issn,
                        references=list(paper_metadata.references or []),
                        section_title=chunk.title,
                        heading_level=chunk.heading_level,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        word_count=chunk.word_count,
                        classification_label=chunk.classification.label,
                        classification_source=chunk.classification.source,
                        classification_confidence=chunk.classification.confidence,
                        used_context=chunk.classification.used_context,
                        reason=chunk.classification.reason,
                        markdown_path=paper_metadata.markdown_path,
                        marker_json_path=paper_metadata.marker_json_path,
                        pdf_path=paper_metadata.pdf_path,
                    )
                )

        return chunk_records

    def make_chunk_id(
        self,
        paper_id: str,
        section_title: str,
        chunk_index: int,
    ) -> str:
        payload = f"{paper_id}\n{section_title}\n{chunk_index}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
