# AGENTS.md

## Purpose

This repository processes academic papers into structured chunk data.

Pipeline:

1. read Marker JSON or filtered Markdown
2. extract metadata
3. convert Marker JSON to filtered Markdown
4. chunk by heading
5. classify chunks as `abstract`, `introduction`, `method`, `results`, or `discussion`
6. use Gemma 4 only for unresolved chunks unless LLM mode is forced

## Key Directories

- `chunker/`: active project code for filtering, chunking, and classification
- `marker/`: local Marker checkout plus conversion outputs
- `pdfs/`: source PDFs

## Core Modules

### `chunker/metadata_extractor.py`

Extracts title, journal metadata, authors, and references from Marker JSON.

### `chunker/boilerplate_filter.py`

Converts Marker JSON into filtered Markdown by removing front matter noise, figures, tables, page furniture, and references while preserving heading structure.

### `chunker/markdown_section_chunker.py`

Splits filtered Markdown into heading-based chunks using:

- `MarkdownHeading`
- `SectionChunk`
- `HeadingSplit`
- `MarkdownSectionChunker`

Chunks are word-count based with overlap and sentence-aware boundaries.

### `chunker/section_classifier.py`

Pipeline-facing classification layer. It should contain only:

- `ChunkClassification`
- `ClassifiedSectionChunk`
- `ClassifiedHeadingSplit`
- `DeterministicSectionClassifier`
- `ChunkClassificationEnricher`
- `classify_filtered_markdown(...)`

Keep this file focused on deterministic rules and orchestration. Do not move prompt logic, markdown anchoring, or MLX runtime code back into it.

### `chunker/llm_section_classifier.py`

LLM-specific classification layer. It should contain:

- `ChunkClassificationLLM`
- `ChunkLocation`
- `ChunkContext`
- `LLMDecision`
- `article_quintile(...)`

It owns prompt construction, controller-loop behavior, quintiles, Markdown anchoring, context retrieval, response parsing, and external MLX integration.

The LLM class should be injectable into `ChunkClassificationEnricher`. The rest of the pipeline should not need to know prompt details.

### `chunker/mlx_llm_runner.py`

Thin external runner for Gemma 4 when using a separate Python environment. Reads JSON from stdin and writes raw model output to stdout.

### `chunker/run_section_classification.py`

CLI runner for classifying a filtered Markdown file from the terminal.

## Runtime Model Setup

The repo `.venv` is for the repo itself and should stay Marker-compatible.

Do not install Gemma 4 support directly into the repo `.venv` unless there is a strong reason and you understand the dependency impact.

The working Gemma 4 setup is currently:

- external Python:
  `/Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python`
- model identifier:
  `unsloth/gemma-4-E4B-it-UD-MLX-4bit`

The downloaded model weights are not stored in this repo. They live in the Hugging Face cache under the user home directory.

## Important Output Semantics

Each classified chunk must preserve:

- original chunk text
- chunk index
- word count
- final label
- source
- confidence
- used_context
- reason

This metadata is downstream pipeline input, not just display output. Preserve alignment with the original chunk objects.

## Working Rules

- Prefer filtered Markdown as the source of truth for chunk context.
- Avoid re-chunking unnecessarily if chunk data already exists in the current pipeline step.
- Keep deterministic classification conservative.
- Prefer passing unresolved cases to the LLM rather than adding weak heuristic text rules.
- Preserve repo `.venv` dependency integrity.
- Keep new code modular; avoid putting orchestration, prompting, parsing, and runtime control into one file.

## Validation

When changing classification code, verify at least:

1. `./.venv/bin/python -m compileall chunker`
2. deterministic classification on `marker/conversion_results/2025_Uchida/2025_Uchida_filtered.md`
3. if touching the LLM path, one real fallback chunk through the external Gemma runtime

## Useful Commands

Deterministic run:

```bash
./.venv/bin/python -m chunker.run_section_classification \
  marker/conversion_results/2025_Uchida/2025_Uchida_filtered.md
```

Run with Gemma fallback:

```bash
./.venv/bin/python -m chunker.run_section_classification \
  marker/conversion_results/2025_Uchida/2025_Uchida_filtered.md \
  --model-path unsloth/gemma-4-E4B-it-UD-MLX-4bit \
  --python-executable /Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python
```
