# Dialogitech

Dialogitech is a small academic-paper processing pipeline built around filtered Markdown.

The workflow is:

1. extract metadata from Marker JSON
2. convert the document to filtered Markdown
3. remove references and everything after references
4. split the Markdown into heading-based chunks
5. classify each chunk as `abstract`, `introduction`, `method`, `results`, or `discussion`
6. send unresolved chunks to Gemma 4 for fallback classification
7. embed the chunks locally with Ollama and store them in LanceDB

## Quick Start

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
ollama pull qwen3-embedding:0.6b
./.venv/bin/python -m compileall chunker dbinsert
./.venv/bin/python -m dbinsert.run_full_pipeline \
  pdfs/2025_Uchida.pdf \
  --db-path data/lancedb \
  --replace-existing
```

## Repository Layout

- `chunker/`: core pipeline code
- `dbquery/`: LanceDB query rewriting, retrieval, fusion, and summarization
- `dbxquery/`: grounded follow-up querying over prior synthesized summaries
- `resquery/`: iterative research sessions with branch-local history, exploration modes, and end-of-session synthesis
- `marker/`: local Marker checkout and conversion outputs
- `pdfs/`: source PDFs

- `chunker/metadata_extractor.py`: title, journal, author, and reference extraction
- `chunker/boilerplate_filter.py`: converts Marker JSON into filtered Markdown and drops references plus all trailing appendix/supplement content
- `chunker/markdown_section_chunker.py`: splits filtered Markdown into section chunks
- `chunker/chunk_models.py`: classified chunk and heading-split data models
- `chunker/llm_section_classifier.py`: LLM fallback classification, quintiles, and context retrieval
- `chunker/run_section_classification.py`: CLI runner for classifying a filtered Markdown file
- `dbinsert/`: LanceDB ingestion, indexing, full-pipeline orchestration, and inspection CLIs
- `dbquery/`: query rewriting with Gemma, hybrid/vector/FTS retrieval, reciprocal-rank fusion, batch summaries, and synthesized summaries
- `dbxquery/`: single-turn grounded follow-up planning, filtered evidence retrieval, and answer writing over prior summary outputs
- `resquery/`: multi-turn research sessions that reuse `dbquery`, track branch-local exploration history, extract new claims from batch summaries, suggest follow-up questions, and synthesize branch findings

## Requirements

- Python 3.12
- macOS on Apple Silicon for the MLX Gemma 4 path
- a virtual environment at `.venv`

## Setup

### 1. Clone the repo and create the repo virtual environment

```bash
python3.12 -m venv .venv
```

### 2. Install the repo dependencies

Install the Marker-compatible repo environment plus local LanceDB ingestion dependencies:

```bash
./.venv/bin/pip install -r requirements.txt
```

This installs:

- the local `marker/` package
- LanceDB
- PyArrow

The repo `.venv` intentionally does not install `mlx-lm`.

### 3. Install Ollama

Install Ollama from:

- https://ollama.com/download

After installation, confirm it is available:

```bash
ollama --version
```

### 4. Pull the local embedding model

For vector ingestion, pull the embedding model used by `dbinsert`:

```bash
ollama pull qwen3-embedding:0.6b
```

Confirm it is installed:

```bash
ollama list
```

### 5. Set up the separate Unsloth MLX environment for Gemma 4 classification

The working Gemma 4 path for this repo is intentionally separate from `.venv`.

Working environment:

- Python executable:
  `/Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python`
- model:
  `unsloth/gemma-4-E4B-it-UD-MLX-4bit`

If you need the optional in-process `mlx-lm` package in another environment, install:

```bash
python -m pip install -r requirements-mlx.txt
```

You will also need the Gemma model available through the Hugging Face / MLX setup used by that environment.

### 6. Verify the local services and models

Check Ollama:

```bash
curl http://localhost:11434/api/version
ollama list
```

Check the repo environment:

```bash
./.venv/bin/python -m compileall chunker dbinsert dbquery dbxquery resquery
```

## Basic Flow

There are two entry points:

1. start from an existing filtered Markdown file
2. start from Marker JSON, then convert it to filtered Markdown before chunking/classifying

## Run Deterministic Classification

This uses heading-based deterministic rules only:

```bash
./.venv/bin/python -m chunker.run_section_classification \
  marker/conversion_results/2025_Uchida/2025_Uchida_filtered.md
```

Stdout output contains one row per chunk with:

- heading
- chunk index
- word count
- label
- source
- confidence
- whether context was used
- reason

## Run With Gemma 4 Fallback

The working setup uses a separate Unsloth MLX environment:

- Python executable:
  `/Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python`
- model:
  `unsloth/gemma-4-E4B-it-UD-MLX-4bit`

Run the classifier with LLM fallback enabled for unresolved chunks:

```bash
./.venv/bin/python -m chunker.run_section_classification \
  marker/conversion_results/2025_Uchida/2025_Uchida_filtered.md \
  --model-path unsloth/gemma-4-E4B-it-UD-MLX-4bit \
  --python-executable /Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python
```

Force the LLM to classify every chunk:

```bash
./.venv/bin/python -m chunker.run_section_classification \
  marker/conversion_results/2025_Uchida/2025_Uchida_filtered.md \
  --model-path unsloth/gemma-4-E4B-it-UD-MLX-4bit \
  --python-executable /Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python \
  --force-llm
```

## LLM Behavior

The LLM path is controller-based:

- the model sees the chunk text and which quintile of the article it comes from
- if uncertain, it can request one round of context
- context is: previous heading section + current chunk + next heading section
- if the model requests context a second time, the classifier sends one final confirmation prompt using the previous resolved chunk label
- the final response is normalized into the same classification metadata used by the deterministic pipeline

The pipeline still refuses to insert unresolved chunks into LanceDB. If Gemma does not return a concrete label but a previous resolved chunk label exists, the run logs the LLM failure and deterministically inherits the previous label. If there is no previous resolved label available, the run still fails instead of writing incomplete data.

## Example Pipeline From Marker JSON

If you already have a Marker JSON document in `marker/conversion_results/...`, the Python flow is:

```python
from pathlib import Path

from chunker import BoilerplateFilter, MetadataExtractor, MarkdownSectionChunker
from chunker.llm_section_classifier import SectionClassificationLLM
from chunker.chunk_models import ClassifiedHeadingSplit, ClassifiedSectionChunk

json_path = Path("marker/conversion_results/2025_Uchida/2025_Uchida.json")
document = MetadataExtractor().load_json(json_path)
metadata = MetadataExtractor().extract_all(document)
markdown = BoilerplateFilter().convert_json_to_markdown(document, metadata=metadata)
heading_splits = MarkdownSectionChunker().process(markdown)

llm = SectionClassificationLLM(
    filtered_markdown=markdown,
    heading_splits=heading_splits,
    model_path="unsloth/gemma-4-E4B-it-UD-MLX-4bit",
    python_executable="/Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python",
)

classified = []
with llm:
    for heading_split in heading_splits:
        chunks = []
        previous_label = None
        for chunk in heading_split.chunks:
            classification = llm.classify_section_chunk(
                section_chunk=chunk,
                heading_split=heading_split,
                paper_type="empirical_research",
                previous_label=previous_label,
            )
            if classification.label is not None:
                previous_label = classification.label
            chunks.append(ClassifiedSectionChunk(
                title=chunk.title,
                heading_level=chunk.heading_level,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                word_count=chunk.word_count,
                classification=classification,
            ))
        classified.append(ClassifiedHeadingSplit(
            title=heading_split.title,
            heading_level=heading_split.heading_level,
            raw_heading=heading_split.raw_heading,
            content=heading_split.content,
            chunks=chunks,
        ))
```

## Notes

- The repo `.venv` is kept Marker-compatible.
- The Gemma 4 runtime is intentionally isolated in a separate Unsloth environment.
- The large downloaded Gemma model files live in the Hugging Face cache, not in this repo.
- Generated outputs such as classified JSON files should be treated as derived artifacts.

## LanceDB Ingestion

Run the full pipeline from a single PDF:

```bash
./.venv/bin/python -m dbinsert.run_full_pipeline \
  pdfs/2025_Uchida.pdf \
  --db-path data/lancedb \
  --replace-existing
```

Run the full pipeline over a folder of PDFs:

```bash
./.venv/bin/python -m dbinsert.run_full_pipeline_folder \
  pdfs \
  --db-path data/lancedb \
  --replace-existing
```

This will:

- derive `paper_id` from the PDF filename
- reuse or create Marker JSON in `marker/conversion_results/<paper_id>/`
- extract full paper metadata
- write filtered Markdown
- remove references and all content after the references section
- classify chunks with deterministic rules plus Gemma fallback for unresolved chunks
- generate local Ollama embeddings with `qwen3-embedding:0.6b`
- insert chunk rows plus metadata into LanceDB

The folder runner processes PDFs sequentially. If one PDF fails, the error is printed and recorded in the final batch summary, and the run continues with the remaining files. The command exits non-zero at the end if any file failed.

If required metadata is missing after Marker extraction, the pipeline pauses and asks for manual terminal input for:

- title
- authors
- journal name
- publication year

The prompt includes the current `paper_id`, PDF path, and Marker JSON path so you can identify the file being processed. In the folder runner, if metadata entry is aborted or unavailable for a file, that file is logged as failed and the batch continues.

The full pipeline defaults to the working external Gemma 4 setup:

- model: `unsloth/gemma-4-E4B-it-UD-MLX-4bit`
- python: `/Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python`

You normally do not need to pass these explicitly.

## Query The Database

After data has been ingested into LanceDB, query it with the `dbquery` pipeline:

```bash
./.venv/bin/python -m dbquery.run_query \
  "How well do LLMs handle text generation at different CEFR levels?" \
  --db-path data/lancedb \
  --output-path outputs/llms_cefr_levels_query.md
```

This pipeline currently does the following:

1. takes the original user query
2. asks Gemma for two retrieval rewrites
3. generates one HyDE paragraph for semantic retrieval
4. runs five retrieval branches:
   - original-query hybrid
   - original-query full-text search
   - rewrite-1 vector
   - rewrite-2 vector
   - HyDE vector
5. deduplicates retrieved rows by `chunk_id`
6. ranks the deduplicated rows with reciprocal rank fusion
7. summarizes the top-ranked chunks in small batches
8. synthesizes the batch summaries into a final summary

The output artifact written by `--output-path` contains:

- the ranked deduplicated chunks in final rank order
- chunk metadata for inspection
- the full chunk text for each ranked hit
- the batch summaries
- the synthesized summary appended at the end

Useful query flags:

- `--retrieval-mode`: retrieval branch selection. Available modes are `full`, `fts_only`, `hybrid_only`, and `fts_hybrid`
- `--retrieval-depth`: number of results to keep from each retrieval branch before fusion
- `--final-top-k`: number of fused ranked chunks to keep after dedupe and RRF
- `--batch-size`: number of chunks to summarize per batch
- `--rewrite-count`: number of Gemma rewrites to generate
- `--rrf-k`: reciprocal-rank-fusion smoothing constant
- `--min-rrf-lists`: minimum number of retrieval lists that must contain a chunk before it can survive fusion
- `--min-relevance-score`: optional LanceDB relevance-score cutoff where applicable
- `--paper-id`: restrict retrieval to a single paper
- `--year`: restrict retrieval to a publication year
- `--paper-title-contains`: restrict retrieval to papers whose title contains a substring
- `--classification-label`: restrict retrieval to chunks with a specific section label
- `--author` or `--authors`: restrict retrieval to chunks whose author list contains a string
- `--no-hyde`: disable the HyDE retrieval branch

Retrieval mode guidance:

- `full`: run the complete pipeline with hybrid, FTS, rewrites, and HyDE. This is the default and is best for broad conceptual questions.
- `fts_only`: run only original-query full-text search. Use this for very specific terms, acronyms, variable names, or exact phrases.
- `hybrid_only`: run only original-query hybrid retrieval. Use this when you want semantic plus lexical retrieval without rewrites or HyDE.
- `fts_hybrid`: run original-query FTS and original-query hybrid only. Use this for focused questions where exact wording matters, but semantic backup is still useful.

Example with explicit tuning:

```bash
./.venv/bin/python -m dbquery.run_query \
  "Why do LLMs not generate good quality CEFR texts?" \
  --db-path data/lancedb \
  --output-path outputs/llms_cefr_quality_query.md \
  --retrieval-depth 12 \
  --final-top-k 10 \
  --batch-size 5
```

Example for an exact-term lookup:

```bash
./.venv/bin/python -m dbquery.run_query \
  "What is vpersent?" \
  --db-path data/lancedb \
  --retrieval-mode fts_only \
  --output-path outputs/vpersent_query.md
```

Example with metadata filtering:

```bash
./.venv/bin/python -m dbquery.run_query \
  "What is vpersent?" \
  --db-path data/lancedb \
  --retrieval-mode fts_only \
  --paper-id 2025_Uchida \
  --year 2025 \
  --paper-title-contains CEFR \
  --author Uchida \
  --output-path outputs/vpersent_filtered_query.md
```

Operational note:

- Like the classification pipeline, the Gemma query path may fail inside the sandbox because MLX/Metal initialization is machine-dependent.
- On this machine, real Gemma query runs work reliably outside the sandbox with the external Unsloth MLX environment.

## Research Sessions

For iterative multi-turn research, use the `resquery` pipeline.

```bash
./.venv/bin/python -m resquery.run_session \
  "What are the weaknesses of LLMs?" \
  --session-path ressessions/llm_weaknesses.json \
  --db-path data/lancedb \
  --run-mode continue \
  --branch-id b1 \
  --query-output-path resoutputs/llm_weaknesses_t1_dbquery.md \
  --output-path resoutputs/llm_weaknesses_t1_resquery.md
```

Continue the same branch with a focused follow-up:

```bash
./.venv/bin/python -m resquery.run_session \
  "How does chain of thought impact the accuracy of CEFR generated text?" \
  --session-path ressessions/llm_weaknesses.json \
  --db-path data/lancedb \
  --run-mode continue \
  --branch-id b1 \
  --query-output-path resoutputs/llm_weaknesses_t2_dbquery.md \
  --output-path resoutputs/llm_weaknesses_t2_resquery.md
```

Go deeper on the same branch while excluding already seen chunks:

```bash
./.venv/bin/python -m resquery.run_session \
  "Explain more about why there are these limitations in generating CEFR texts." \
  --session-path ressessions/llm_weaknesses.json \
  --db-path data/lancedb \
  --run-mode deepen \
  --branch-id b1 \
  --query-output-path resoutputs/llm_weaknesses_t3_dbquery.md \
  --output-path resoutputs/llm_weaknesses_t3_resquery.md
```

Expand outward from the same branch while excluding already seen papers:

```bash
./.venv/bin/python -m resquery.run_session \
  "Find adjacent literature on this question." \
  --session-path ressessions/llm_weaknesses.json \
  --db-path data/lancedb \
  --run-mode expand \
  --branch-id b1 \
  --query-output-path resoutputs/llm_weaknesses_t4_dbquery.md \
  --output-path resoutputs/llm_weaknesses_t4_resquery.md
```

Start a separate line of inquiry in a new branch:

```bash
./.venv/bin/python -m resquery.run_session \
  "What are the weaknesses of LLMs in assessment design?" \
  --session-path ressessions/llm_weaknesses.json \
  --db-path data/lancedb \
  --run-mode new \
  --branch-id b2 \
  --branch-label assessment \
  --query-output-path resoutputs/llm_weaknesses_b2_t1_dbquery.md \
  --output-path resoutputs/llm_weaknesses_b2_t1_resquery.md
```

The current `resquery` design does the following:

1. loads the persisted research session and branch state
2. resolves the branch for the current turn
3. chooses a run mode:
   - `new`: create a new branch with empty seen-history
   - `continue`: use the current question as-is on the selected branch
   - `deepen`: inject compact branch context and exclude already seen `chunk_ids`
   - `expand`: inject compact branch context and exclude already seen `paper_ids`
4. runs `dbquery` as the retrieval and synthesis engine
5. extracts compact claims from each individual `dbquery` batch summary
6. attaches claim evidence programmatically from that batch summary's `chunk_ids`
7. stores branch-local seen `chunk_ids` and `paper_ids` for later `deepen` and `expand` runs
8. asks Gemma for user-facing follow-up suggestions from the new synthesized summary
9. writes the updated session JSON and a markdown turn artifact

`resquery` stores:

- branch metadata and branch-local turn order
- branch-local seen `chunk_ids` and `paper_ids`
- prior claims
- follow-up suggestions
- evidence index metadata
- per-turn contextualized query text
- per-turn synthesized summaries

Important design notes:

- a brand-new session starts with default branch `b1`; use `continue --branch-id b1` for the first turn, then `new` when you want an additional branch such as `b2`
- claims come from batch summaries, not from the final synthesized summary
- claim evidence is not generated by Gemma; it comes from known `dbquery` batch-summary `chunk_ids`
- follow-up suggestions are suggestions for the user, not resolved/unresolved state
- `deepen` and `expand` use compact branch context instead of injecting the full claim history
- for `deepen` and `expand`, branch claims are selected by semantic similarity to the current question, with status/confidence used as fallback ordering

## Research Synthesis

After you finish one or more `resquery` branches, synthesize them into a final report:

```bash
./.venv/bin/python -m resquery.run_synthesis \
  --session-path ressessions/llm_weaknesses.json \
  --output-path resoutputs/llm_weaknesses_final_report.md
```

Limit synthesis to selected branches by repeating `--branch-id`:

```bash
./.venv/bin/python -m resquery.run_synthesis \
  --session-path ressessions/llm_weaknesses.json \
  --branch-id b1 \
  --branch-id b2 \
  --output-path resoutputs/llm_weaknesses_final_report.md
```

The synthesis pipeline does the following:

1. reads the saved `resquery` session state
2. builds one branch-local synthesis input per selected branch using:
   - ordered turn summaries
   - claim-level evidence provenance
3. asks Gemma for one summary per branch with provenance citations
4. compares branch summaries pairwise and asks Gemma for similarities and points of contention
5. writes one final markdown report containing:
   - one section per branch summary
   - one section per branch comparison
   - the provenance context used for each branch summary

## Follow-up Queries

After a `dbquery` run has written a markdown artifact with a synthesized summary, query that summary with the `dbxquery` follow-up pipeline:

```bash
./.venv/bin/python -m dbxquery.run_followup \
  "Tell me how Zhu and Uchida differ on this issue?" \
  --prior-summary-path outputs/llms_cefr_quality_query.md \
  --db-path data/lancedb \
  --output-path xoutputs/zhu_vs_uchida_differences_followup.md
```

This first version currently does the following:

1. reads the prior query artifact
2. extracts only the `## Synthesized Summary` section when present
3. asks Gemma to emit a strict JSON evidence plan from `prior summary + user question`
4. validates the plan against a closed schema
5. runs one to three filtered evidence retrieval calls against LanceDB
6. asks Gemma to answer using only the retrieved evidence
7. writes a trace containing the prior summary, plan, tool calls, evidence, and final answer

The current `dbxquery` scope is intentionally narrow:

- single-turn only
- synchronous only
- one fixed tool: `retrieve_evidence`
- closed-set filters only: `paper_id_in`, `authors_any`, `year_min`, `year_max`, `classification_label_in`, `section_title_contains`

The output artifact written by `--output-path` contains:

- the extracted synthesized summary used as context
- the evidence plan produced by Gemma
- each retrieval tool call
- the retrieved evidence chunks
- the final grounded answer
- the chunk IDs and paper IDs used in the answer

Operational note:

- Like the classification and `dbquery` paths, `dbxquery` may fail inside the sandbox because the external Gemma MLX runtime depends on Metal.
- On this machine, real `dbxquery` runs work reliably outside the sandbox with the external Unsloth MLX environment.

## Full Pipeline Reruns

The full runner is resumable by artifact:

- it reuses existing Marker JSON unless you pass `--rerun-marker`
- it reuses existing filtered markdown unless you pass `--rerun-filtered-markdown`
- it reuses existing classified chunks unless you pass `--rerun-classification`

Useful examples:

Rerun classification only:

```bash
./.venv/bin/python -m dbinsert.run_full_pipeline \
  pdfs/2025_Zhu.pdf \
  --db-path data/lancedb \
  --replace-existing \
  --rerun-classification
```

Regenerate filtered markdown and classification after filter changes:

```bash
./.venv/bin/python -m dbinsert.run_full_pipeline \
  pdfs/2025_Zhu.pdf \
  --db-path data/lancedb \
  --replace-existing \
  --rerun-filtered-markdown \
  --rerun-classification
```

## Logs

The full pipeline writes per-paper logs under `marker/conversion_results/<paper_id>/`:

- `<paper_id>_marker.log`: Marker conversion output
- `<paper_id>_classification.log`: chunk-level Gemma send/response log

During long Marker runs, the pipeline prints elapsed-time progress messages and you can stop the run by typing `stop` and pressing Enter, or by pressing `Ctrl-C`.

Ingest one classified paper:

```bash
./.venv/bin/python -m dbinsert.run_ingest \
  --db-path data/lancedb \
  --classified-json marker/conversion_results/2025_Uchida/2025_Uchida_classified_chunks.json \
  --marker-json marker/conversion_results/2025_Uchida/2025_Uchida.json \
  --replace-existing
```

Inspect the stored table:

```bash
./.venv/bin/python -m dbinsert.inspect_table \
  --db-path data/lancedb \
  --paper-id 2025_Uchida
```
