from __future__ import annotations

from .models import FusedChunkResult, SummaryBatch


class ChunkBatcher:
    def batch(self, chunks: list[FusedChunkResult], *, batch_size: int) -> list[SummaryBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        batches: list[SummaryBatch] = []
        for start in range(0, len(chunks), batch_size):
            batches.append(
                SummaryBatch(
                    batch_index=len(batches),
                    chunks=chunks[start : start + batch_size],
                )
            )
        return batches
