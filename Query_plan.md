Yes. HyDE fits cleanly into this design and is worth adding.

In your case, HyDE would mean asking Gemma to generate a short hypothetical answer paragraph for the user query, embedding that paragraph, and running an additional vector search from it. That often improves recall for conceptual questions where the database text may not share the user’s phrasing.

**Updated Plan**

**Files**
Create these files under `/Users/danielparsons/Documents/Development/Dialogitech/dbquery`:

- `[dbquery/__init__.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/__init__.py)`
  Public exports for the query package.

- `[dbquery/models.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/models.py)`
  Shared dataclasses for query requests, search variants, retrieval hits, fused results, batches, and summaries.

- `[dbquery/query_rewriter.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/query_rewriter.py)`
  Gemma-based rewrite generation for two short retrieval-oriented alternatives.

- `[dbquery/hyde_generator.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/hyde_generator.py)`
  Gemma-based hypothetical answer generation for HyDE.

- `[dbquery/query_embedder.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/query_embedder.py)`
  Embeds the original query, rewrites, and HyDE text using the same embedding backend family used at ingest.

- `[dbquery/lancedb_retriever.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/lancedb_retriever.py)`
  Runs hybrid, FTS, and vector retrieval against LanceDB.

- `[dbquery/result_fuser.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/result_fuser.py)`
  Deduplicates by `chunk_id` and computes reciprocal rank fusion scores across all result lists.

- `[dbquery/chunk_batcher.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/chunk_batcher.py)`
  Splits top fused chunks into summarization groups.

- `[dbquery/citation_formatter.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/citation_formatter.py)`
  Formats author-year citation labels and prompt metadata blocks.

- `[dbquery/gemma_summarizer.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/gemma_summarizer.py)`
  Summarizes chunk batches with required author-year citations.

- `[dbquery/query_pipeline.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/query_pipeline.py)`
  Orchestrates rewrite generation, HyDE generation, retrieval, fusion, batching, and summarization.

- `[dbquery/run_query.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/run_query.py)`
  CLI entry point for retrieval and summarization.

Optional helpers if needed:
- `[dbquery/utils/rrf.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/utils/rrf.py)`
- `[dbquery/utils/text_budget.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/utils/text_budget.py)`

**Classes**
- `QueryRequest` in `[dbquery/models.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/models.py)`
  Holds the user query and runtime settings.

- `SearchVariant` in `[dbquery/models.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/models.py)`
  Represents one retrieval input, such as original, rewrite-1, rewrite-2, or HyDE.

- `RetrievedChunk` in `[dbquery/models.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/models.py)`
  Represents one raw DB hit with its originating search list and rank.

- `FusedChunkResult` in `[dbquery/models.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/models.py)`
  Represents one deduped chunk with final RRF score and provenance.

- `SummaryBatch` in `[dbquery/models.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/models.py)`
  Represents one group of chunks for summarization.

- `BatchSummary` in `[dbquery/models.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/models.py)`
  Represents one citation-aware summary plus source provenance.

- `GemmaQueryRewriter` in `[dbquery/query_rewriter.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/query_rewriter.py)`
  Produces exactly two concise retrieval rewrites.

- `GemmaHyDEGenerator` in `[dbquery/hyde_generator.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/hyde_generator.py)`
  Produces a short hypothetical answer passage optimized for semantic retrieval.

- `QueryEmbedder` in `[dbquery/query_embedder.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/query_embedder.py)`
  Embeds all search variants.

- `LanceDBRetriever` in `[dbquery/lancedb_retriever.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/lancedb_retriever.py)`
  Runs:
  - original-query hybrid
  - original-query FTS
  - rewrite-1 vector
  - rewrite-2 vector
  - HyDE vector

- `ReciprocalRankFuser` in `[dbquery/result_fuser.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/result_fuser.py)`
  Merges all lists, dedupes by `chunk_id`, and computes final rank.

- `ChunkBatcher` in `[dbquery/chunk_batcher.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/chunk_batcher.py)`
  Creates summarization batches, default `8-10` chunks.

- `CitationFormatter` in `[dbquery/citation_formatter.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/citation_formatter.py)`
  Builds grounded `(Author, Year)` labels for prompts.

- `GemmaBatchSummarizer` in `[dbquery/gemma_summarizer.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/gemma_summarizer.py)`
  Summarizes batches while requiring citations drawn only from supplied metadata.

- `QueryPipeline` in `[dbquery/query_pipeline.py](/Users/danielparsons/Documents/Development/Dialogitech/dbquery/query_pipeline.py)`
  Coordinates the entire process through dependency injection.

**Retrieval Flow**
1. Accept the original user query.
2. Generate two short rewrites with Gemma.
3. Generate one HyDE paragraph with Gemma.
4. Run:
   - original-query hybrid
   - original-query FTS
   - rewrite-1 vector
   - rewrite-2 vector
   - HyDE vector
5. Deduplicate by `chunk_id`.
6. Sort by RRF score.
7. Batch top results.
8. Summarize each batch with citations.

**Why HyDE Helps**
- rewrites improve phrasing coverage
- FTS improves exact-term matching
- HyDE improves conceptual recall when the corpus expresses the answer differently than the user asked it

That combination is strong and still operationally simple.