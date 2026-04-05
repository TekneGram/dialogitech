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
- `marker/`: local Marker checkout and conversion outputs
- `pdfs/`: source PDFs

- `chunker/metadata_extractor.py`: title, journal, author, and reference extraction
- `chunker/boilerplate_filter.py`: converts Marker JSON into filtered Markdown and drops references plus all trailing appendix/supplement content
- `chunker/markdown_section_chunker.py`: splits filtered Markdown into section chunks
- `chunker/section_classifier.py`: deterministic section classification and enrichment
- `chunker/llm_section_classifier.py`: LLM fallback classification, quintiles, and context retrieval
- `chunker/run_section_classification.py`: CLI runner for classifying a filtered Markdown file
- `dbinsert/`: LanceDB ingestion, indexing, full-pipeline orchestration, and inspection CLIs

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
./.venv/bin/python -m compileall chunker dbinsert
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
from chunker.llm_section_classifier import ChunkClassificationLLM
from chunker.section_classifier import ChunkClassificationEnricher

json_path = Path("marker/conversion_results/2025_Uchida/2025_Uchida.json")
document = MetadataExtractor().load_json(json_path)
metadata = MetadataExtractor().extract_all(document)
markdown = BoilerplateFilter().convert_json_to_markdown(document, metadata=metadata)
heading_splits = MarkdownSectionChunker().process(markdown)

llm = ChunkClassificationLLM(
    filtered_markdown=markdown,
    heading_splits=heading_splits,
    model_path="unsloth/gemma-4-E4B-it-UD-MLX-4bit",
    python_executable="/Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python",
)

classified = ChunkClassificationEnricher(llm_classifier=llm).enrich_heading_splits(heading_splits)
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

This will:

- derive `paper_id` from the PDF filename
- reuse or create Marker JSON in `marker/conversion_results/<paper_id>/`
- extract full paper metadata
- write filtered Markdown
- remove references and all content after the references section
- classify chunks with deterministic rules plus Gemma fallback for unresolved chunks
- generate local Ollama embeddings with `qwen3-embedding:0.6b`
- insert chunk rows plus metadata into LanceDB

The full pipeline defaults to the working external Gemma 4 setup:

- model: `unsloth/gemma-4-E4B-it-UD-MLX-4bit`
- python: `/Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python`

You normally do not need to pass these explicitly.

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
